"""Bible Phase 0.3: checkpoint bundle for every major run.

A bundle is a directory holding config.json, metrics.json, seeds.json, summary.json, the logs, any artifact files
and MANIFEST.json (sha256 + size of every file, plus a hash of the manifest body itself). Bundles are write-once:
an existing run id is never overwritten, files are made read-only, and `verify` re-hashes everything so a changed,
deleted or added file is reported (fail closed: a missing manifest is a failure, not a pass)."""
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

from . import config as K

CORE = ("config.json", "metrics.json", "seeds.json", "summary.json")
MANIFEST = "MANIFEST.json"


class CheckpointError(RuntimeError):
    pass


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _dump(o):
    return json.dumps(o, indent=1, sort_keys=True, default=str, allow_nan=False)


def _safe_id(run_id):
    s = str(run_id)
    if not s or s != Path(s).name or s in (".", "..") or any(c in s for c in '<>:"/\\|?*'):
        raise CheckpointError(f"unsafe run id {run_id!r}")
    return s


def _body_hash(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def write_checkpoint(root, run_id, config, metrics, seeds, summary, now, logs=None, artifacts=None, provenance=None):
    """Create root/run_id. logs: {name: text}; artifacts: {name: existing file path to copy in}.
    Raises if the bundle exists, if a core input is missing/NaN (allow_nan=False), or if seeds is empty:
    an experiment without its seeds is not reproducible. Half-written bundles are removed on failure."""
    if config is None or metrics is None or summary is None:
        raise CheckpointError("config, metrics and summary are required")
    if not seeds:
        raise CheckpointError("seeds are required")
    d = Path(root) / _safe_id(run_id)
    if d.exists():
        raise CheckpointError(f"checkpoint {run_id} exists; prior experiments are never overwritten")
    d.mkdir(parents=True)
    try:
        payload = {"config.json": config, "metrics.json": metrics, "seeds.json": seeds, "summary.json": summary}
        for n, o in payload.items():
            (d / n).write_text(_dump(o), encoding="utf-8")
        for n, text in (logs or {}).items():
            (d / "logs").mkdir(exist_ok=True)
            (d / "logs" / _safe_id(n)).write_text(text, encoding="utf-8")
        for n, src in (artifacts or {}).items():
            src = Path(src)
            if not src.is_file():
                raise CheckpointError(f"artifact {n} not found: {src}")
            (d / "artifacts").mkdir(exist_ok=True)
            shutil.copyfile(src, d / "artifacts" / _safe_id(n))
        files = {}
        for p in sorted(d.rglob("*")):
            if p.is_file():
                files[p.relative_to(d).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        man = {"run_id": str(run_id), "created": str(now), "files": files, "files_hash": _body_hash(files),
               "provenance": provenance or {}, "config_sha256": files["config.json"]["sha256"]}
        (d / MANIFEST).write_text(_dump(man), encoding="utf-8")
        for p in d.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IREAD)
    except BaseException:
        _rmtree(d)
        raise
    return d


def _rmtree(d):
    def unlock(fn, path, _):
        os.chmod(path, stat.S_IWRITE)
        fn(path)
    shutil.rmtree(d, onerror=unlock)


def verify(bundle):
    """Return {'ok': bool, 'problems': [...]}. Checks: manifest present and self-consistent, every listed file present
    with the recorded hash and size, no unlisted file, core files present, seeds non-empty."""
    d = Path(bundle)
    probs = []
    mp = d / MANIFEST
    if not d.is_dir() or not mp.exists():
        return {"ok": False, "problems": ["no manifest"]}
    try:
        man = json.loads(mp.read_text(encoding="utf-8"))
        files = man["files"]
    except (ValueError, KeyError):
        return {"ok": False, "problems": ["manifest unreadable"]}
    if man.get("files_hash") != _body_hash(files):
        probs.append("manifest body altered")
    for n in CORE:
        if n not in files:
            probs.append(f"core file not in manifest: {n}")
    for n, meta in files.items():
        p = d / n
        if not p.is_file():
            probs.append(f"missing: {n}")
        elif p.stat().st_size != meta["bytes"] or sha256_file(p) != meta["sha256"]:
            probs.append(f"changed: {n}")
    on_disk = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()} - {MANIFEST}
    probs += [f"unlisted: {n}" for n in sorted(on_disk - set(files))]
    if "seeds.json" in files and (d / "seeds.json").is_file():
        try:
            if not json.loads((d / "seeds.json").read_text(encoding="utf-8")):
                probs.append("seeds empty")
        except ValueError:
            probs.append("seeds.json unreadable")
    return {"ok": not probs, "problems": probs}


def load(bundle):
    """Read a verified bundle back; refuses a bundle that fails verify."""
    v = verify(bundle)
    if not v["ok"]:
        raise CheckpointError(f"bundle failed verification: {v['problems']}")
    d = Path(bundle)
    out = {n[:-5]: json.loads((d / n).read_text(encoding="utf-8")) for n in CORE}
    out["manifest"] = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    return out


def list_checkpoints(root):
    r = Path(root)
    return sorted(p.name for p in r.iterdir() if (p / MANIFEST).exists()) if r.is_dir() else []


def verify_all(root):
    """Verify every bundle under root; a directory without a manifest counts as a failure (a half-made bundle)."""
    r = Path(root)
    res = {p.name: verify(p) for p in sorted(r.iterdir()) if p.is_dir()} if r.is_dir() else {}
    return {"ok": all(v["ok"] for v in res.values()), "bundles": res}


def same_config(a, b):
    """True when two bundles were run with byte-identical configuration."""
    return load(a)["manifest"]["config_sha256"] == load(b)["manifest"]["config_sha256"]


def default_root():
    return K.STATE / "checkpoints"


def _finite(o):
    """NaN/inf -> None throughout (bundles are strict JSON: allow_nan=False)."""
    import math
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def checkpoint_run(kind, config, metrics, seeds, summary, logs=None, artifacts=None, root=None):
    """Bible P0.3 in one call for a major run: bundle config, metrics, seeds, summary, logs and artifacts under
    state/checkpoints/<kind>_<UTC timestamp>, stamped with provenance. Never overwrites; returns the bundle path."""
    from datetime import datetime, timezone
    from . import provenance
    now = datetime.now(timezone.utc)
    run_id = f"{kind}_{now:%Y%m%dT%H%M%S}"
    return write_checkpoint(root or default_root(), run_id, _finite(config), _finite(metrics), _finite(seeds),
                            _finite(summary), now.isoformat(timespec="seconds"), logs=logs,
                            artifacts={k: v for k, v in (artifacts or {}).items() if Path(v).exists()},
                            provenance=_finite(provenance.stamp(config, seeds)))
