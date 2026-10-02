"""Test results keyed by the whole git tree they ran on.

One development cycle used to run the same unchanged main tree through pytest three times: the round's main assessment, the
sandbox BASE run and (next round) the main assessment again. A passing per-file result is reusable exactly when the code it
ran on is byte-identical, and that is what a clean git tree hash says - for the test file AND everything it reads, imported or
not (the audit tests scan the whole repository, so an import-graph digest alone would be unsound for them).

Reuse is deliberately narrow: only a CLEAN tree (no uncommitted or untracked file, before and after the run), only files that
PASSED, only the same python, only entries younger than MAX_AGE_S. Anything else runs fresh. The candidate replicates are never
served from here: a sandbox tree is dirty by construction.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Mapping, Optional

from creator import sandbox as S

MAX_AGE_S = 6 * 3600.0
KEEP_TREES = 4
STATE_DIR = "state"
_LOCK = threading.Lock()                                   # swarm workers (threads) share one index


def tree_key(repo: str | Path, python: str, rev: str = "HEAD", require_clean: bool = True) -> Optional[str]:
    """`<hash of every top-level entry of rev except state/>:<python>`, or None when the tree cannot be vouched for (dirty, not a
    repository). state/ is left out on purpose: the kernel itself rewrites its tracked evidence there every cycle, no test imports
    it (testrun.py checks that), and keying on it would make main look dirty forever."""
    try:
        if require_clean and S.git(repo, "status", "--porcelain", "--", ".", f":(exclude){STATE_DIR}", check=False).stdout.strip():
            return None
        r = S.git(repo, "ls-tree", rev, check=False)
    except Exception:                                                   # noqa: BLE001 - no key means "run fresh"
        return None
    rows = [ln for ln in r.stdout.splitlines() if ln.split("	", 1)[-1] != STATE_DIR]
    if r.returncode != 0 or not rows:
        return None
    return f"{hashlib.sha256(chr(10).join(rows).encode()).hexdigest()}:{python}"


class TreeCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _index(self) -> dict[str, dict]:
        try:
            return json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _dir(self, key: str) -> Path:
        return self.root / hashlib.sha256(key.encode()).hexdigest()[:16]

    def lookup(self, key: Optional[str], now: Optional[float] = None) -> dict[str, tuple[str, str]]:
        """test file -> (reach digest, junit path) for every passing file recorded for exactly this tree."""
        ent = self._index().get(key or "")
        if not ent or (now if now is not None else time.time()) - float(ent.get("at", 0)) > MAX_AGE_S:
            return {}
        d = self._dir(key or "")
        return {tf: (rec["digest"], str(d / rec["xml"])) for tf, rec in ent["tests"].items() if (d / rec["xml"]).is_file()}

    def store(self, key: Optional[str], evidence: Mapping[str, object]) -> None:
        """Remember the PASS results of an assessment (TestEvidence objects) under `key`; older trees beyond KEEP_TREES go."""
        if not key:
            return
        with _LOCK:
            self._store(key, evidence)

    def _store(self, key: str, evidence: Mapping[str, object]) -> None:
        d = self._dir(key)
        d.mkdir(parents=True, exist_ok=True)
        tests: dict[str, dict[str, str]] = {}
        for tf, ev in evidence.items():
            src = Path(str(getattr(ev, "where", "")))
            if getattr(ev, "outcome", "") != "PASS" or not src.is_file():
                continue
            name = tf.replace("/", "__") + ".xml"
            shutil.copyfile(src, d / name)
            tests[tf] = {"digest": str(getattr(ev, "source_digest", "")), "xml": name}
        idx = self._index()
        idx[key] = {"at": time.time(), "tests": tests}
        for old in sorted(idx, key=lambda k: float(idx[k].get("at", 0)))[:-KEEP_TREES]:
            shutil.rmtree(self._dir(old), ignore_errors=True)
            del idx[old]
        tmp = self.root / "index.json.tmp"
        tmp.write_text(json.dumps(idx, indent=1), encoding="utf-8")
        tmp.replace(self.root / "index.json")
