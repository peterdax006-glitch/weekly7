"""Nupen tools - the per-package scratchpad: state/creator/scratch/<package>/ . Its own package may read and write it (files tools and
shell, through the policy); no other package can. It is removed when the package's lesson is recorded, unless kept."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

KEEP = ".keep"


def scratch_dir(state: Path, package: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", package or "anon")[:80] or "anon"
    return Path(state) / "scratch" / safe


def ensure(state: Path, package: str) -> Path:
    d = scratch_dir(state, package)
    d.mkdir(parents=True, exist_ok=True)
    return d


def keep(state: Path, package: str) -> None:
    ensure(state, package).joinpath(KEEP).write_text("kept\n", encoding="utf-8")


def cleanup(state: Path, package: str, force: bool = False) -> bool:
    """Delete the package's scratchpad (called when its lesson is recorded). Kept pads survive unless `force`. Returns True if removed."""
    d = scratch_dir(state, package)
    root = (Path(state) / "scratch").resolve()
    if not d.is_dir() or root not in d.resolve().parents:
        return False
    if (d / KEEP).exists() and not force:
        return False
    shutil.rmtree(d, ignore_errors=True)
    return not d.exists()
