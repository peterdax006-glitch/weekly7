"""Disk-persisted, content-keyed cache for pure per-file analyses (import graph, component scan).

Every kernel cycle / swarm worker is a NEW process, so in-process caches never survive; this one does. Entry key =
sha256(file bytes) + a code-version SALT (hash of the source of the analysis code), so editing the analysis invalidates all
entries and editing a file invalidates only that file. Writers are concurrent-safe (temp file + os.replace); an unreadable,
truncated or wrong-shaped entry is ignored and recomputed. Set WEEKLY7_CREATOR_CACHE to move the dir, or to "off" to disable."""
from __future__ import annotations

import hashlib
import inspect
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

_MAGIC = "w7c1"


def cache_dir() -> Path | None:
    v = os.environ.get("WEEKLY7_CREATOR_CACHE")
    if v is not None and v.strip().lower() in ("off", "0", "none"):
        return None
    return Path(v) if v else Path.home() / ".weekly7_creator_cache"


def salt_of(funcs: Iterable[Callable[..., Any]], extra: Iterable[bytes] = ()) -> str:
    """Hash of the analysis code: the source text of each function (+ extra bytes such as an external ruler file)."""
    h = hashlib.sha256()
    for f in funcs:
        try:
            h.update(inspect.getsource(f).encode("utf-8"))
        except (OSError, TypeError):
            h.update(repr(f).encode())            # source unavailable: fall back to identity (still never stale across edits)
        h.update(b"\0")
    for b in extra:
        h.update(b)
        h.update(b"\0")
    return h.hexdigest()[:24]


def _path(kind: str, salt: str, key: str) -> Path | None:
    d = cache_dir()
    if d is None:
        return None
    return d / kind / salt / key[:2] / (key + ".pkl")


def key_of(*parts: str | bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def get(kind: str, salt: str, key: str) -> Any:
    """The stored value, or None (miss / corrupt / disabled)."""
    p = _path(kind, salt, key)
    if p is None:
        return None
    try:
        magic, k, value = pickle.loads(p.read_bytes())
        return value if magic == _MAGIC and k == key else None
    except Exception:                              # corrupt, truncated, missing, unpicklable: all mean "recompute"
        return None


def put(kind: str, salt: str, key: str, value: Any) -> None:
    p = _path(kind, salt, key)
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(pickle.dumps((_MAGIC, key, value), protocol=4))
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:
        pass                                       # a cache that cannot write is just a miss
