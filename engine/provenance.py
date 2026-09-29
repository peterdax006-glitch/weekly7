"""Bible Phase 0: provenance for every experiment - git commit, canon hash, Bible hash, blueprint version, config hash,
data snapshot, seed - plus a canon/Bible integrity check that fails closed."""
import hashlib, json, subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLUEPRINT_VERSION = "4.0"


def _sha(path):
    p = ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


@lru_cache(maxsize=1)
def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "engine", "scripts"], cwd=ROOT, capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


class IntegrityError(RuntimeError):
    pass


def verify_integrity():
    """Canon directives and the Bible must match their locks; otherwise nothing may run (fail closed)."""
    lock = json.loads((ROOT / "canon" / "canon.lock.json").read_text(encoding="utf-8"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("bc", ROOT / "canon" / "build_canon.py")
    bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
    bad = [c for c, _, _, t in bc.DIRECTIVES if lock.get(c) != bc.sha(t)]
    if bad:
        raise IntegrityError(f"canon altered: {bad}")
    bl = ROOT / "canon" / "bible.lock.json"
    if bl.exists():
        want = json.loads(bl.read_text())["sha256"]
        got = hashlib.sha256((ROOT / "BIBLE.md").read_text(encoding="utf-8").rstrip("\n").encode("utf-8")).hexdigest()
        if got != want:
            raise IntegrityError("BIBLE.md altered")
    return True


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def data_snapshot():
    """Fingerprint of the cached data the experiment read (sizes + mtimes of the main caches)."""
    cache = ROOT / "data" / "cache"
    parts = []
    for n in ("stocks_close.parquet", "stocks_pre2000_close.parquet", "events.parquet", "insider.parquet",
              "macro.parquet", "panel.parquet"):
        p = cache / n
        if p.exists():
            st = p.stat()
            parts.append(f"{n}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


CODE_FILES = ("engine/*.py", "scripts/livesim_loop2.py")


def code_hash():
    """Hash of the code that decides results, read from disk NOW (engine + the Test loop)."""
    h = hashlib.sha256()
    for pat in CODE_FILES:
        for f in sorted(ROOT.glob(pat)):
            h.update(f.name.encode()); h.update(f.read_bytes().replace(bytes([13, 10]), bytes([10])))
    return h.hexdigest()[:16]


# the code THIS process imported. A background run that outlives an edit keeps running the old code; comparing this
# with code_hash() later tells a stale-code result from a genuine parity failure (T9, w01c, 2026-09-28).
CODE_HASH_AT_IMPORT = code_hash()


def stale(recorded):
    """True when a result was produced by code other than what is on disk now."""
    return recorded is None or recorded != code_hash()


def stamp(cfg=None, seed=None):
    return {"code_hash": CODE_HASH_AT_IMPORT, "git_commit": git_commit(), "canon_sha": _sha("canon/canon.lock.json")[:16] if _sha("canon/canon.lock.json") else None,
            "bible_sha": _sha("canon/bible.lock.json")[:16] if _sha("canon/bible.lock.json") else None,
            "blueprint_version": BLUEPRINT_VERSION, "config_hash": config_hash(cfg) if cfg is not None else None,
            "data_snapshot": data_snapshot(), "seed": seed}
