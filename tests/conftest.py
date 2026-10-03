"""Shared test fixtures.

cached_provenance (3 Oct 2026, measured: Ledger.append -> current_provenance re-hashes creator/ and engine/ of the real checkout on EVERY append,
~0.3-0.46 s each; it was 97% of some tests and planner 299 s -> 27 s, oversight 158 s -> 11 s with it cached): per test module the real value is
computed ONCE and reused with a fresh seed, config hash and timestamp - exactly what the kernel/swarm/claims/goals tests already did locally.
No test edits the real checkout's creator/ or engine/. A module that must exercise the real computation sets REAL_PROVENANCE = True
(tests/test_creator_ledger.py does: it covers current_provenance itself)."""
from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any, Iterator, Mapping, Optional

import pytest


@pytest.fixture(scope="module", autouse=True)
def cached_provenance(request: pytest.FixtureRequest) -> Iterator[None]:
    if getattr(request.module, "REAL_PROVENANCE", False):
        yield
        return
    from creator import ledger as LG
    base = LG.current_provenance()

    def cached(seed: Optional[int] = None, config: Optional[Mapping[str, Any]] = None, *a: Any, **k: Any) -> Any:
        cfg_hash = LG.sha256_text(LG.canonical(dict(config)))[:16] if config is not None else None
        return dataclasses.replace(base, seed=seed, config_hash=cfg_hash,
                                   timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    mp = pytest.MonkeyPatch()
    mp.setattr(LG, "current_provenance", cached)
    try:
        yield
    finally:
        mp.undo()
