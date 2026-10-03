"""Creator K01 - the development ledger (C77 secs 7, 22, 30, 35, 37, 58-61; package CR01) - IMPLEMENTED, NOT VALIDATED.

The single source of truth for development state. An append-only JSONL file in which every line is one typed record
(creator.model) wrapped in an envelope that carries:

    seq, id, rtype, version, data, provenance, prev, hash

`hash` is sha256 over the canonical JSON of everything else in the envelope, and `prev` is the previous line's hash, so any edit,
deletion, insertion, reorder or truncation of the history is detected by `verify()`. Records are validated when they are built
(model.py) and again against the ledger's state when they are appended: referenced records must exist and have an allowed type,
lineage rules hold, status changes follow the legal state machine from the subject's ACTUAL current state, VALIDATED needs an
independent role and verifiable evidence, an ImprovementClaim's verdict is recomputed and must match, an ADOPT decision needs an
IMPROVEMENT claim, cited evidence must hash to what is cited. Anything else is refused (fail closed) and never written.

Nothing here is ever updated in place: the current state is FOLDED from the history (C77 sec 61 - no silent state mutation)."""
from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import fnmatch
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from creator import model as M

CREATOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CREATOR_DIR.parent
GENESIS = "0" * 64
LOCK_TIMEOUT_S = 30.0


def _take_over_dead_lock(path: Path) -> bool:
    """Remove the lock file only when the process it names is DEAD (an unknown or unreadable holder counts as alive: never
    steal). The name is re-read just before removing, so a lock another writer has just re-taken is left alone."""
    try:
        holder = int(path.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return False
    if holder <= 0 or holder == os.getpid():
        return False
    try:
        import psutil
        if psutil.pid_exists(holder) and psutil.Process(holder).status() != psutil.STATUS_ZOMBIE:
            return False
    except Exception:                                   # noqa: BLE001 - cannot tell: treat as alive
        return False
    try:
        if path.read_text(encoding="utf-8").strip() != str(holder):
            return False
        path.unlink()
    except OSError:
        return False
    return True


class LedgerError(RuntimeError):
    """A write was refused or the stored history is not trustworthy. Never caught-and-continued by callers."""


# ------------------------------------------------------------------------------------------------ canonical hashing

def canonical(obj: Any) -> str:
    """The one serialisation that is hashed: sorted keys, no whitespace, UTF-8, no NaN/Infinity (they are not JSON)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


_RACY_NS = 2_000_000_000          # git's 'racily clean' rule: a file changed within this window of now may be rewritten inside
                                  # the same filesystem timestamp tick with the same size, so its stat key proves nothing


def _racy(mtime_ns: int) -> bool:
    return mtime_ns >= time.time_ns() - _RACY_NS


_TREE_FILE_CACHE: dict[str, tuple[tuple[int, int], bytes]] = {}          # abs path -> ((mtime_ns, size), CRLF-folded bytes)
_TREE_RESULT_CACHE: dict[tuple[str, str], tuple[tuple[Any, ...], str]] = {}   # (root, pattern) -> (snapshot, digest)


def _scan_files(root: Path) -> list[tuple[str, int, int]]:
    """(absolute path, mtime_ns, size) of every file under root, one scandir pass (much cheaper than rglob + Path.stat on
    Windows). __pycache__ is included; tree_hash filters it out, the git fingerprint keeps it."""
    out: list[tuple[str, int, int]] = []
    stack = [str(root)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    else:
                        st = e.stat()
                        out.append((e.path, st.st_mtime_ns, st.st_size))
        except OSError:
            continue
    return out


def _tree_hash_from(root: Path, pattern: str, scan: Sequence[tuple[str, int, int]]) -> str:
    base = str(root)
    cut = len(base.rstrip("/" + chr(92))) + 1
    rows = []
    for ap, mt, sz in scan:
        if fnmatch.fnmatch(os.path.basename(ap), pattern):
            rel = ap[cut:].replace(chr(92), "/")
            if "__pycache__" not in rel.split("/"):
                rows.append((rel, ap, mt, sz))
    rows.sort()
    snapshot = tuple((rel, mt, sz) for rel, _ap, mt, sz in rows)                # cheap string-sorted cache key
    key = (base, pattern)
    racy = any(_racy(mt) for _rel, _ap, mt, _sz in rows)
    hit = None if racy else _TREE_RESULT_CACHE.get(key)
    if hit is not None and hit[0] == snapshot:
        return hit[1]
    h = hashlib.sha256()
    for rel, ap, mt, sz in sorted(rows, key=lambda r: Path(r[0])):             # the original order: sorted(root.rglob(...)) paths
        cached = _TREE_FILE_CACHE.get(ap)
        if cached is not None and cached[0] == (mt, sz) and not _racy(mt):
            data = cached[1]
        else:
            data = Path(ap).read_bytes().replace(b"\r\n", b"\n")
            if not _racy(mt):
                _TREE_FILE_CACHE[ap] = ((mt, sz), data)
        h.update(rel.encode())
        h.update(data)
    digest = h.hexdigest()[:16]
    if not racy:
        _TREE_RESULT_CACHE[key] = (snapshot, digest)
    return digest


def tree_hash(root: Path, pattern: str = "*.py") -> str:
    """sha256 over the sorted (relative path, bytes) of every matching file under root; CRLF folded so a Windows checkout and a
    Linux one agree. Speed: file bytes are cached on (path, mtime_ns, size) and the whole digest on the full snapshot of those
    keys, so any edit, add, remove or rename changes a key and is seen on the next call (a file touched within _RACY_NS of now is
    always re-read: a same-size rewrite inside one timestamp tick keeps its key); the value is byte-identical to the
    uncached computation (tests/test_creator_ledger.py::test_tree_hash_cache_*)."""
    return _tree_hash_from(root, pattern, _scan_files(root))


def _git_dirs(repo: Path) -> tuple[Path, Path]:
    """(gitdir, common dir) for repo, reading the .git file of a worktree; (.git, .git) for a plain checkout."""
    dot = repo / ".git"
    if dot.is_file():
        gd = Path(dot.read_text(encoding="utf-8").strip().split(":", 1)[1].strip())
        if not gd.is_absolute():
            gd = (repo / gd).resolve()
        cd = gd
        cf = gd / "commondir"
        if cf.is_file():
            cd = (gd / cf.read_text(encoding="utf-8").strip()).resolve()
        return gd, cd
    return dot, dot


def _stat_key(p: Path) -> Optional[tuple[int, int]]:
    try:
        st = p.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _git_state_key(repo: Path, scans: Optional[Mapping[str, Sequence[tuple[str, int, int]]]] = None) -> Optional[tuple[Any, ...]]:
    """A cheap fingerprint of everything `git rev-parse HEAD` + `git status --porcelain -- creator engine` depend on: HEAD, the
    ref it names, packed-refs, the index, and the (path, mtime_ns, size) of every file under creator/ and engine/ (all kinds,
    not only .py). None = cannot fingerprint, so the caller runs git. `scans` maps str(dir) to an already-taken scan."""
    try:
        gd, cd = _git_dirs(repo)
        head = (gd / "HEAD").read_text(encoding="utf-8").strip()
        ref_key = None
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            ref_key = _stat_key(gd / ref) or _stat_key(cd / ref)
        files: list[tuple[str, int, int]] = []
        for sub in ("creator", "engine"):
            base = repo / sub
            if base.is_dir():
                files.extend(scans[str(base)] if scans is not None and str(base) in scans else _scan_files(base))
        key = (head, ref_key, _stat_key(cd / "packed-refs"), _stat_key(gd / "index"), tuple(sorted(files)))
        stamps = [k[0] for k in key[1:4] if k is not None] + [mt for _p, mt, _s in files]
        return None if any(_racy(mt) for mt in stamps) else key        # racily clean: run git
    except (OSError, IndexError, ValueError):
        return None


_GIT_CACHE: dict[str, tuple[tuple[Any, ...], str]] = {}


def _git_commit(repo: Path, scans: Optional[Mapping[str, Sequence[tuple[str, int, int]]]] = None) -> str:
    key = _git_state_key(repo, scans)
    if key is not None:
        hit = _GIT_CACHE.get(str(repo))
        if hit is not None and hit[0] == key:
            return hit[1]
    result = _git_commit_uncached(repo)
    if key is not None and result != "unknown":
        _GIT_CACHE[str(repo)] = (key, result)
    return result


def _git_commit_uncached(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=20)
        head = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "creator", "engine"], cwd=repo, capture_output=True,
                               text=True, timeout=30).stdout.strip()
        return (head or "unknown") + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def current_provenance(seed: Optional[int] = None, config: Optional[Mapping[str, Any]] = None,
                       repo: Path = REPO_ROOT) -> M.Provenance:
    """C77 sec 35, computed - never typed by a caller."""
    try:
        from engine import provenance as P
        engine_hash = P.engine_tree_hash()
    except Exception:                                        # noqa: BLE001 - the Creator must also run outside Weekly7
        engine_hash = tree_hash(repo / "engine") if (repo / "engine").is_dir() else "absent"
    scans = {str(CREATOR_DIR): _scan_files(CREATOR_DIR)}      # one directory pass feeds both the tree hash and the git fingerprint
    return M.Provenance(engine_tree_hash=engine_hash, creator_tree_hash=_tree_hash_from(CREATOR_DIR, "*.py", scans[str(CREATOR_DIR)]),
                        git_commit=_git_commit(repo, scans),
                        config_hash=sha256_text(canonical(dict(config)))[:16] if config is not None else None, seed=seed,
                        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))


# ------------------------------------------------------------------------------------------------ envelopes

@dataclasses.dataclass(frozen=True)
class Entry:
    """One verified line of the ledger."""
    seq: int
    id: str
    rtype: str
    version: int
    record: M.Record
    provenance: M.Provenance
    prev: str
    hash: str

    def envelope(self) -> dict[str, Any]:
        return _envelope(self.seq, self.id, self.rtype, self.version, self.record.to_dict(), self.provenance, self.prev)


def _envelope(seq: int, rid: str, rtype: str, version: int, data: Mapping[str, Any], prov: M.Provenance,
              prev: str) -> dict[str, Any]:
    return {"seq": seq, "id": rid, "rtype": rtype, "version": version, "data": dict(data),
            "provenance": dataclasses.asdict(prov), "prev": prev}


def _line_hash(env: Mapping[str, Any]) -> str:
    return sha256_text(canonical(env))


def record_id(rec: M.Record, prev: str, seq: int) -> str:
    """PREFIX-<20 hex>: content + position addressed, so the same content appended twice gets two ids and an id never repeats."""
    return f"{rec.PREFIX}-{sha256_text(canonical({'d': rec.to_dict(), 'p': prev, 's': seq, 't': rec.RTYPE}))[:20]}"


# ------------------------------------------------------------------------------------------------ the folded view

@dataclasses.dataclass
class View:
    """The current state, folded from the history. Built only by Ledger._fold; never written to disk as authority."""
    entries: list[Entry] = dataclasses.field(default_factory=list)
    by_id: dict[str, Entry] = dataclasses.field(default_factory=dict)
    status: dict[str, M.Status] = dataclasses.field(default_factory=dict)
    status_history: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    unique: dict[tuple[str, str], str] = dataclasses.field(default_factory=dict)
    children: dict[str, list[str]] = dataclasses.field(default_factory=dict)

    @property
    def head(self) -> str:
        return self.entries[-1].hash if self.entries else GENESIS

    def apply(self, e: Entry) -> None:
        self.entries.append(e)
        self.by_id[e.id] = e
        rec = e.record
        for p in rec.parents:
            self.children.setdefault(p, []).append(e.id)
        if type(rec).STATEFUL:
            self.status[e.id] = M.Status.NOT_STARTED
            self.status_history[e.id] = []
        key_field = M.UNIQUE_KEYS.get(e.rtype)
        if key_field:
            self.unique[(e.rtype, str(getattr(rec, key_field)))] = e.id
        if isinstance(rec, M.Transition):
            self.status[rec.subject_id] = rec.to_state
            self.status_history.setdefault(rec.subject_id, []).append(e.id)

    def digest(self) -> str:
        """Digest of the folded state (ids, types, statuses): what a CheckpointMarker pins (C77 sec 59)."""
        state = sorted((i, self.by_id[i].rtype, self.status.get(i, M.Status.NOT_STARTED).value if i in self.status else "")
                       for i in self.by_id)
        return sha256_text(canonical(state))


# ------------------------------------------------------------------------------------------------ the ledger

class Ledger:
    """Append-only, hash-chained, validated development ledger.

        led = Ledger(path, evidence_root=repo)
        rid = led.append(M.Objective(...), seed=0, config={...})
        led.transition(rid_of_requirement, M.Status.IN_PROGRESS, reason="...", created_by=M.Role.KERNEL)
        led.verify()                       # raises LedgerError on any tampering
        led.view.status[rid]               # current status, folded from transitions
    """

    def __init__(self, path: str | os.PathLike, evidence_root: str | os.PathLike = REPO_ROOT, repair_torn: bool = False):
        self.path = Path(path)
        self.evidence_root = Path(evidence_root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.torn: Optional[str] = None
        self.view = self._load(repair_torn)

    # ---------------------------------------------------------------- reading

    def _read_lines(self) -> tuple[list[str], Optional[str]]:
        if not self.path.exists():
            return [], None
        raw = self.path.read_bytes().decode("utf-8")
        if not raw:
            return [], None
        lines = raw.split("\n")
        torn = None
        if lines[-1] != "":                                   # the last write did not finish its newline: a torn record
            torn = lines[-1]
            lines = lines[:-1]
        else:
            lines = lines[:-1]
        return lines, torn

    def _load(self, repair_torn: bool) -> View:
        lines, torn = self._read_lines()
        if torn is not None:
            if not repair_torn:
                raise LedgerError(f"{self.path}: the last line is torn (an interrupted append); open with repair_torn=True to "
                                  f"drop it - an interrupted write is never assumed to have succeeded (C77 sec 60)")
            self.torn = torn
            with self._locked():
                self._rewrite_without_torn(lines)
        return self._fold(lines)

    def _rewrite_without_torn(self, lines: list[str]) -> None:
        body = "".join(line + "\n" for line in lines)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_bytes(body.encode("utf-8"))
        os.replace(tmp, self.path)
        side = self.path.with_suffix(self.path.suffix + ".torn")
        with side.open("a", encoding="utf-8") as fh:            # the dropped bytes are kept as evidence, never silently lost
            fh.write(json.dumps({"dropped_at": dt.datetime.now(dt.timezone.utc).isoformat(), "bytes": self.torn}) + "\n")

    def _fold(self, lines: Sequence[str]) -> View:
        view = View()
        for n, line in enumerate(lines):
            try:
                env = json.loads(line)
            except json.JSONDecodeError as e:
                raise LedgerError(f"{self.path}:{n + 1}: not JSON ({e})") from e
            stored_hash = env.pop("hash", None)
            if stored_hash != _line_hash(env):
                raise LedgerError(f"{self.path}:{n + 1}: hash mismatch - the line was edited")
            if env.get("seq") != n:
                raise LedgerError(f"{self.path}:{n + 1}: seq {env.get('seq')} at position {n} - lines inserted, deleted or reordered")
            if env.get("prev") != view.head:
                raise LedgerError(f"{self.path}:{n + 1}: chain broken (prev does not match the previous line's hash)")
            try:
                rec = M.record_from_dict(env["rtype"], env["version"], env["data"])
                prov = M.Provenance(**env["provenance"])
            except (M.ModelError, TypeError, KeyError) as e:
                raise LedgerError(f"{self.path}:{n + 1}: stored record no longer validates: {e}") from e
            if record_id(rec, env["prev"], n) != env["id"]:
                raise LedgerError(f"{self.path}:{n + 1}: id does not match its content")
            view.apply(Entry(n, env["id"], env["rtype"], env["version"], rec, prov, env["prev"], stored_hash))
        return view

    def verify(self) -> int:
        """Re-read and re-check the whole chain from disk; returns the number of records. Raises LedgerError on tampering."""
        lines, torn = self._read_lines()
        if torn is not None:
            raise LedgerError(f"{self.path}: torn final line")
        fresh = self._fold(lines)
        if fresh.head != self.view.head and len(fresh.entries) == len(self.view.entries):
            raise LedgerError(f"{self.path}: the file no longer matches what this process wrote")
        return len(fresh.entries)

    # ---------------------------------------------------------------- locking

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        t0 = time.monotonic()
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                break
            except FileExistsError:
                if _take_over_dead_lock(self.lock_path):        # 2 Oct: a writer killed mid-append left the lock forever and
                    continue                                    # the swarm crash-looped on 'held for > 30s by another writer'
                if time.monotonic() - t0 > LOCK_TIMEOUT_S:
                    raise LedgerError(f"{self.lock_path}: held for > {LOCK_TIMEOUT_S}s by another writer") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.lock_path)

    # ---------------------------------------------------------------- validation against the ledger state

    def _check_refs(self, rec: M.Record) -> list[str]:
        errs = []
        cls = type(rec)
        if cls.PARENTS is not None:
            if not rec.parents:
                errs.append(f"{cls.RTYPE} needs at least one parent")
            elif cls.PARENTS and not any(p in self.view.by_id and self.view.by_id[p].rtype in cls.PARENTS for p in rec.parents):
                errs.append(f"{cls.RTYPE} needs a parent of type {cls.PARENTS}")
        for p in rec.parents:
            if p not in self.view.by_id:
                errs.append(f"parent {p} does not exist")
        for name, allowed in cls.REFS.items():
            for r in rec.ref_values(name):
                e = self.view.by_id.get(r)
                if e is None:
                    errs.append(f"{name} -> {r} does not exist")
                elif allowed and e.rtype not in allowed:
                    errs.append(f"{name} -> {r} is a {e.rtype}, expected one of {allowed}")
        return errs

    def _check_evidence(self, rec: M.Record) -> list[str]:
        return [p for p in (ev.problem(self.evidence_root) for ev in rec.evidence) if p]

    def _check_unique(self, rec: M.Record) -> list[str]:
        f = M.UNIQUE_KEYS.get(rec.RTYPE)
        if f and (rec.RTYPE, str(getattr(rec, f))) in self.view.unique:
            return [f"{rec.RTYPE} {f}={getattr(rec, f)!r} already exists ({self.view.unique[(rec.RTYPE, str(getattr(rec, f)))]})"]
        return []

    def _check_transition(self, rec: M.Transition) -> list[str]:
        subj = self.view.by_id.get(rec.subject_id)
        if subj is None:
            return [f"transition subject {rec.subject_id} does not exist"]
        if not type(subj.record).STATEFUL:
            return [f"{subj.rtype} {rec.subject_id} has no status to change"]
        cur = self.view.status[rec.subject_id]
        if rec.from_state is not cur:
            return [f"{rec.subject_id} is {cur.value}, not {rec.from_state.value} (stale or forged transition)"]
        errs = []
        if rec.to_state in (M.Status.TESTED, M.Status.INTENDED_BEHAVIOR_VERIFIED, M.Status.VALIDATED):
            just = [self.view.by_id.get(j) for j in rec.justification_ids]
            if not just or any(j is None for j in just):
                errs.append(f"{rec.to_state.value} needs existing justification records")
            elif not any(j.record.evidence or isinstance(j.record, (M.TestRun, M.ImprovementClaim, M.Measurement))
                         for j in just if j is not None):
                errs.append(f"{rec.to_state.value} needs a justification that carries evidence or a computed result")
            if rec.to_state is M.Status.TESTED and not any(j is not None and isinstance(j.record, M.TestRun) for j in just):
                errs.append("TESTED needs a TestRun among the justifications (code exists is not tested, C77 sec 7)")
            if rec.to_state is M.Status.TESTED:
                for j in just:
                    if j is not None and isinstance(j.record, M.TestRun) and j.record.outcome is not M.TestOutcome.PASS:
                        errs.append(f"TestRun {j.id} outcome is {j.record.outcome.value}, not PASS")
            if rec.to_state is M.Status.VALIDATED and not rec.evidence:
                errs.append("VALIDATED needs evidence on the transition itself")
        return errs

    def _measurements(self, ids: Sequence[str]) -> list[M.Measurement]:
        out = []
        for i in ids:
            rec = self.view.by_id[i].record
            if not isinstance(rec, M.Measurement):
                raise LedgerError(f"{i} is not a Measurement")
            out.append(rec)
        return out

    def _check_claim(self, rec: M.ImprovementClaim) -> list[str]:
        try:
            base, cand = self._measurements(rec.baseline_ids), self._measurements(rec.candidate_ids)
            regs = list(zip(self._measurements(rec.regression_baseline_ids), self._measurements(rec.regression_candidate_ids)))
            hold = None
            if rec.holdout_baseline_id is not None and rec.holdout_candidate_id is not None:
                hb, hc = self._measurements([rec.holdout_baseline_id, rec.holdout_candidate_id])
                hold = (hb, hc)
        except (KeyError, LedgerError) as e:
            return [f"claim measurements unreadable: {e}"]
        if not base or not cand:
            return ["claim needs baseline and candidate measurements"]
        computed, detail = M.improvement_verdict(base, cand, regs, hold, rec.min_effect, rec.z)
        if computed is not rec.verdict:
            return [f"claimed {rec.verdict.value} but the rule computes {computed.value} ({detail.get('why', '')})"]
        return []

    def _check_decision(self, rec: M.Decision) -> list[str]:
        if rec.verdict is M.DecisionVerdict.ADOPT:
            claim = self.view.by_id.get(rec.claim_id or "")
            if claim is None or not isinstance(claim.record, M.ImprovementClaim):
                return ["ADOPT must cite an existing ImprovementClaim"]
            if claim.record.verdict is not M.Verdict.IMPROVEMENT:
                return [f"ADOPT rests on a claim whose computed verdict is {claim.record.verdict.value}, not IMPROVEMENT"]
            if claim.record.subject_id != rec.subject_id:
                return ["ADOPT cites a claim about a different subject"]
        return []

    def _check_checkpoint(self, rec: M.CheckpointMarker) -> list[str]:
        errs = []
        if rec.chain_head != self.view.head:
            errs.append("checkpoint chain_head is not the current head")
        if rec.chain_records != len(self.view.entries):
            errs.append("checkpoint chain_records does not match")
        if rec.view_digest != self.view.digest():
            errs.append("checkpoint view_digest does not match the folded state")
        return errs

    def problems(self, rec: M.Record) -> list[str]:
        """Everything that stops `rec` from being appended now (empty = acceptable)."""
        errs = self._check_refs(rec) + self._check_evidence(rec) + self._check_unique(rec)
        if isinstance(rec, M.Transition):
            errs += self._check_transition(rec)
        elif isinstance(rec, M.ImprovementClaim):
            errs += self._check_claim(rec)
        elif isinstance(rec, M.Decision):
            errs += self._check_decision(rec)
        elif isinstance(rec, M.CheckpointMarker):
            errs += self._check_checkpoint(rec)
        return errs

    # ---------------------------------------------------------------- writing

    def append(self, rec: M.Record, seed: Optional[int] = None, config: Optional[Mapping[str, Any]] = None,
               provenance: Optional[M.Provenance] = None) -> str:
        """Validate `rec` against the ledger and append it; returns its id. Refused records raise LedgerError and are not written."""
        with self._locked():
            self._catch_up()
            errs = self.problems(rec)
            if errs:
                raise LedgerError(f"refused {rec.RTYPE}: " + "; ".join(errs))
            prov = provenance or current_provenance(seed, config)
            seq, prev = len(self.view.entries), self.view.head
            rid = record_id(rec, prev, seq)
            env = _envelope(seq, rid, rec.RTYPE, type(rec).VERSION, rec.to_dict(), prov, prev)
            h = _line_hash(env)
            line = canonical({**env, "hash": h}) + "\n"
            with self.path.open("ab") as fh:
                fh.write(line.encode("utf-8"))
                fh.flush()
                os.fsync(fh.fileno())
            self.view.apply(Entry(seq, rid, rec.RTYPE, type(rec).VERSION, rec, prov, prev, h))
            return rid

    def _catch_up(self) -> None:
        """Another process may have appended since we loaded: re-fold from disk under the lock before validating."""
        lines, torn = self._read_lines()
        if torn is not None:
            raise LedgerError(f"{self.path}: torn final line written by another process; reopen with repair_torn=True")
        if len(lines) != len(self.view.entries):
            self.view = self._fold(lines)

    def transition(self, subject_id: str, to: M.Status, reason: str, created_by: M.Role,
                   justification_ids: Iterable[str] = (), evidence: Iterable[M.EvidenceRef] = ()) -> str:
        """Move a stateful record to `to` from its CURRENT state (read from the ledger, never supplied by the caller)."""
        cur = self.view.status.get(subject_id)
        if cur is None:
            raise LedgerError(f"{subject_id} has no status (missing or not a stateful record)")
        if to not in M.LEGAL[cur]:
            path = M.reachable(cur, to)
            raise LedgerError(f"illegal transition {cur.value} -> {to.value}" +
                              (f"; legal path: {' -> '.join(s.value for s in path)}" if path else "; unreachable"))
        return self.append(M.Transition(created_by=created_by, subject_id=subject_id, from_state=cur, to_state=to, reason=reason,
                                        justification_ids=tuple(justification_ids), evidence=tuple(evidence)))

    def checkpoint(self, label: str, created_by: M.Role = M.Role.KERNEL, note: str = "") -> str:
        """Pin the current chain head and folded-state digest (C77 sec 59)."""
        with self._locked():
            self._catch_up()
        rec = M.CheckpointMarker(created_by=created_by, label=label, chain_head=self.view.head,
                                 chain_records=len(self.view.entries), view_digest=self.view.digest(),
                                 source_rev=_git_commit(REPO_ROOT), note=note)
        return self.append(rec)

    # ---------------------------------------------------------------- queries

    def get(self, rid: str) -> M.Record:
        return self.view.by_id[rid].record

    def of_type(self, rtype: str) -> list[Entry]:
        return [e for e in self.view.entries if e.rtype == rtype]

    def with_status(self, *states: M.Status) -> list[Entry]:
        want = set(states)
        return [self.view.by_id[i] for i, s in self.view.status.items() if s in want]

    def lineage(self, rid: str) -> list[str]:
        """Ancestors of rid, nearest first (breadth-first over parents)."""
        out, frontier, seen = [], list(self.get(rid).parents), set()
        while frontier:
            nxt: list[str] = []
            for p in frontier:
                if p in seen or p not in self.view.by_id:
                    continue
                seen.add(p)
                out.append(p)
                nxt.extend(self.view.by_id[p].record.parents)
            frontier = nxt
        return out

    def descendants(self, rid: str) -> list[str]:
        out, frontier, seen = [], list(self.view.children.get(rid, ())), set()
        while frontier:
            nxt: list[str] = []
            for c in frontier:
                if c in seen:
                    continue
                seen.add(c)
                out.append(c)
                nxt.extend(self.view.children.get(c, ()))
            frontier = nxt
        return out

    def about(self, subject_id: str) -> list[Entry]:
        """Every record that speaks about subject_id (as parent or subject)."""
        return [e for e in self.view.entries if e.record.covers(subject_id)]

    def snapshot(self) -> dict[str, Any]:
        """Observability export (C77 sec 58): counts by type and status, open work, the head, and the latest checkpoint."""
        by_type: dict[str, int] = {}
        for e in self.view.entries:
            by_type[e.rtype] = by_type.get(e.rtype, 0) + 1
        by_status: dict[str, int] = {}
        for s in self.view.status.values():
            by_status[s.value] = by_status.get(s.value, 0) + 1
        ckps = self.of_type("CheckpointMarker")
        return {"records": len(self.view.entries), "head": self.view.head, "by_type": by_type, "by_status": by_status,
                "open": sorted(i for i, s in self.view.status.items() if s in M.OPEN_STATES),
                "last_checkpoint": ckps[-1].id if ckps else None, "digest": self.view.digest()}
