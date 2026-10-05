"""Pinned wishlist entries that the GPU package compiler (creator.gpucompile) merges on EVERY recompile.

The voice modules (p2.voice_tts, p2.voice_persona_data, p2.voice_persona_17b) used to be hand-edited into phase2_package/wishlist.json, and a
recompile erased them. They now live in creator/gpu_pinned_wishlist.json; two decorators in gpucompile.py apply them:
  @pinned          on phase2_wishlist: pinned entries the compiler can compile are added (an entry with the same id is replaced by the pinned one)
  @pinned_written  on write_package:   every pinned entry, including `"external": true` ones (not a compiled training job, e.g. the Piper voice,
                   which is prepared on the PC), is present in the wishlist.json that the package writes
Edit the JSON file, not the generated wishlist. The decorators are the only change inside gpucompile.py, so a merge with other compiler work is clean.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

PINNED_FILE = Path(__file__).with_name("gpu_pinned_wishlist.json")


def load(path: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    try:
        d = json.loads(Path(path or PINNED_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in d.items() if isinstance(v, dict) and v.get("id") == k}


def _compilable(entry: Mapping[str, Any], module: Any) -> bool:
    if entry.get("external"):
        return False
    return entry.get("kind") != "infer" or hasattr(module, "compile_infer_job")      # an older compiler cannot compile inference entries


def _strip(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if k != "external"}


def merge(wishlist: list[dict[str, Any]], pins: Mapping[str, Mapping[str, Any]], compilable: Callable[[Mapping[str, Any]], bool] = lambda e: True) -> list[dict[str, Any]]:
    """The wishlist with each compilable pinned entry added (same id: replaced in place, order kept)."""
    out = list(wishlist)
    index = {w["id"]: i for i, w in enumerate(out)}
    skipped = {pid for pid, e in pins.items() if not compilable(e)}
    grew = True
    while grew:                                       # an entry that needs a skipped pinned job (deps / source_job) cannot be compiled either
        grew = False
        for pid, e in pins.items():
            needs = set(e.get("deps") or []) | {x["source_job"] for x in (e.get("extras") or []) if x.get("source_job")}
            if pid not in skipped and needs & skipped:
                skipped.add(pid)
                grew = True
    for pid, e in pins.items():
        if pid in skipped:
            continue
        if pid in index:
            out[index[pid]] = _strip(e)
        else:
            out.append(_strip(e))
    return out


def pinned(fn: Callable[..., list[dict[str, Any]]]) -> Callable[..., list[dict[str, Any]]]:
    @functools.wraps(fn)
    def wrapper(*a: Any, **k: Any) -> list[dict[str, Any]]:
        mod = sys.modules.get(fn.__module__)
        return merge(fn(*a, **k), load(), lambda e: _compilable(e, mod))
    return wrapper


def ensure_in_file(path: Path, pins: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Adds/replaces the pinned entries in a written wishlist.json; returns the ids it set."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    set_ids = []
    for pid, e in pins.items():
        if d.get(pid) != _strip(e):
            d[pid] = _strip(e)
            set_ids.append(pid)
    if set_ids:
        Path(path).write_text(json.dumps(d, indent=1), encoding="utf-8")
    return set_ids


def pinned_written(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*a: Any, **k: Any) -> Any:
        res = fn(*a, **k)
        out = k.get("out", a[1] if len(a) > 1 else None)
        if out is not None:
            ensure_in_file(Path(out) / "wishlist.json", load())
        return res
    return wrapper
