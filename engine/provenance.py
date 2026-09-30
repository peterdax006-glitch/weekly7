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
    for lock_name in ("ten_hour_checklist.lock.json", "ultimate_master.lock.json"):        # later owner documents saved by scripts/save_owner_doc.py
        lk = ROOT / "canon" / lock_name
        if lk.exists():
            meta = json.loads(lk.read_text(encoding="utf-8"))
            f = ROOT / meta["file"]
            if not f.exists():
                raise IntegrityError(f"{meta['file']} missing while its lock exists")
            if hashlib.sha256(f.read_text(encoding="utf-8").rstrip("\n").encode("utf-8")).hexdigest() != meta["sha256"]:
                raise IntegrityError(f"{meta['file']} altered")
    pl = ROOT / "canon" / "prediction_error_addition.lock.json"   # the owner's C68 checklist addition, same rule
    if pl.exists():
        pf = ROOT / "PREDICTION_ERROR_ADDITION.md"
        if not pf.exists():
            raise IntegrityError("PREDICTION_ERROR_ADDITION.md missing while its lock exists")
        want = json.loads(pl.read_text(encoding="utf-8"))["sha256"]
        if hashlib.sha256(pf.read_text(encoding="utf-8").rstrip("\n").encode("utf-8")).hexdigest() != want:
            raise IntegrityError("PREDICTION_ERROR_ADDITION.md altered")
    rl = ROOT / "canon" / "research_contract.lock.json"      # the owner's Research-Brain build prompt (C66), same rule
    if rl.exists():
        rf = ROOT / "RESEARCH_BRAIN_CONTRACT.md"
        if not rf.exists():
            raise IntegrityError("RESEARCH_BRAIN_CONTRACT.md missing while its lock exists")
        want = json.loads(rl.read_text(encoding="utf-8"))["sha256"]
        if hashlib.sha256(rf.read_text(encoding="utf-8").rstrip("\n").encode("utf-8")).hexdigest() != want:
            raise IntegrityError("RESEARCH_BRAIN_CONTRACT.md altered")
    cl = ROOT / "canon" / "contract.lock.json"               # the owner's Self-Learning contract (C62), same rule
    if cl.exists():
        want = json.loads(cl.read_text(encoding="utf-8"))["sha256"]
        cf = ROOT / "SELF_LEARNING_CONTRACT.md"
        if not cf.exists():
            raise IntegrityError("SELF_LEARNING_CONTRACT.md missing while its lock exists")
        got = hashlib.sha256(cf.read_text(encoding="utf-8").rstrip("\n").encode("utf-8")).hexdigest()
        if got != want:
            raise IntegrityError("SELF_LEARNING_CONTRACT.md altered")
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


PROCESS_START = __import__("time").time()


_NORM_CACHE: dict = {}


def _norm(path):
    """File bytes with CRLF folded to LF. Cached by (mtime_ns, size): an edited file is re-read on the very next call, so the
    stale-code guard is unaffected, but unchanged files are not re-read on every stamp (the whole-engine re-read cost
    0.3-40 s per call across several builders on 29 Sep)."""
    p = Path(path)
    st = p.stat()
    key = str(p)
    hit = _NORM_CACHE.get(key)
    # "racy" guard (as git does): an equal-size rewrite inside one timestamp tick keeps (mtime, size) unchanged, so a file
    # modified in the last RACY_S seconds is never served from the cache (test_hash_changes_when_a_loaded_file_changes).
    fresh = __import__("time").time() - st.st_mtime < RACY_S
    if hit and not fresh and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2]
    data = p.read_bytes().replace(bytes([13, 10]), bytes([10]))
    if not fresh:
        _NORM_CACHE[key] = (st.st_mtime_ns, st.st_size, data)
    return data


RACY_S = 2.0


def loaded_code():
    """Repo source files this process has actually imported (engine.*, scripts), as repo-relative paths. A result
    depends on these and only these - a builder adding an unrelated engine file must not make it stale."""
    out = set()
    for m in list(__import__("sys").modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        rel = _REL_CACHE.get(f, _MISS)
        if rel is _MISS:                                 # Path.resolve() on ~1,600 modules cost ~0.18 s per stamp (29 Sep);
            try:                                          # a module's file never moves within a process, so resolve it once
                r = Path(f).resolve().relative_to(ROOT)
                rel = r.as_posix() if r.suffix == ".py" and r.parts[0] in ("engine", "scripts") else None
            except ValueError:
                rel = None
            _REL_CACHE[f] = rel
        if rel:
            out.add(rel)
    return sorted(out)


_REL_CACHE: dict = {}
_MISS = object()


def code_hash(files=None):
    """Hash of the given repo files as they are on disk NOW (default: the modules this process loaded)."""
    h = __import__("hashlib").sha256()
    for rel in (files if files is not None else loaded_code()):
        f = ROOT / rel
        h.update(rel.encode())
        h.update(_norm(f) if f.exists() else b"<missing>")
    return h.hexdigest()[:16]


def engine_tree_hash():
    """Hash of EVERY engine source file on disk (engine/**/*.py), independent of what this process has imported. C75 Phase 0
    (30 Sep): current_code_hash() used the loaded-module hash, so two identical results computed before and after another
    import carried different code hashes (test_learning_credit::test_determinism_and_seed_sensitivity). Files are read through
    the (mtime, size) cache, so repeated calls are cheap and an edit is seen on the next call."""
    files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "engine").rglob("*.py") if "__pycache__" not in p.parts)
    return code_hash(files)


def code_mixed(files=None):
    """Loaded files edited after this process started: the process may be running code that is no longer on disk
    (w01c, 2026-09-28: memory.py was edited mid-run and the result could not be reproduced)."""
    return [rel for rel in (files if files is not None else loaded_code())
            if (ROOT / rel).exists() and (ROOT / rel).stat().st_mtime > PROCESS_START]


def code_stamp():
    files = loaded_code()
    return {"code_hash": code_hash(files), "code_files": files, "code_mixed": code_mixed(files)}


def stale(recorded):
    """True when a result cannot be trusted to come from the code on disk now. `recorded` is a provenance dict
    (or a bare legacy code_hash string, which is always treated as stale: it covered unrelated files)."""
    if not isinstance(recorded, dict) or "code_files" not in recorded:
        return True
    if recorded.get("code_mixed"):
        return True
    return recorded.get("code_hash") != code_hash(recorded["code_files"])


def stamp(cfg=None, seed=None):
    return {**code_stamp(), "git_commit": git_commit(), "canon_sha": _sha("canon/canon.lock.json")[:16] if _sha("canon/canon.lock.json") else None,
            "bible_sha": _sha("canon/bible.lock.json")[:16] if _sha("canon/bible.lock.json") else None,
            "blueprint_version": BLUEPRINT_VERSION, "config_hash": config_hash(cfg) if cfg is not None else None,
            "data_snapshot": data_snapshot(), "seed": seed}
