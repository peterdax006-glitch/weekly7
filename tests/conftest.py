"""Shared test fixtures.

cached_provenance (3 Oct 2026, measured: Ledger.append -> current_provenance re-hashes creator/ and engine/ of the real checkout on EVERY append,
~0.3-0.46 s each; it was 97% of some tests and planner 299 s -> 27 s, oversight 158 s -> 11 s with it cached): per test module the real value is
computed ONCE and reused with a fresh seed, config hash and timestamp - exactly what the kernel/swarm/claims/goals tests already did locally.
No test edits the real checkout's creator/ or engine/. A module that must exercise the real computation sets REAL_PROVENANCE = True
(tests/test_creator_ledger.py does: it covers current_provenance itself).

h38 (3 Oct 2026): the base value is a PINNED constant, no longer read from the real checkout - nothing outside creator/ledger.py and its own
test reads a record's tree hashes or git commit (planner/schedule read only the timestamp), and reading creator/ + engine/ + `git status`
made every ledger-using test file depend on the whole tree (h34's reach traces: TREE-WIDE, never reusable across trees). And every test
process gets its own test-process memory file (NUPEN_TEST_MB_FILE in the temp dir): nested test launches used to rewrite the LIVE
state/creator/test_proc_mb.json the running swarm sizes its test slots from."""
from __future__ import annotations

import dataclasses
import datetime as dt
import os
import tempfile
from typing import Any, Iterator, Mapping, Optional

import pytest

os.environ.setdefault("NUPEN_SIGNER", "off")      # tests never send a shortcut to a signing service (creator/shortcutsign)
os.environ.setdefault("NUPEN_TEST_MB_FILE", os.path.join(tempfile.gettempdir(), f"nupen_test_mb_{os.getpid()}.json"))

# 5 Oct 2026 (h86, test speed): under pytest-xdist every worker is its own process; lightgbm (n_jobs=-1), OpenMP and BLAS would each start one
# thread per logical CPU, so N workers x all CPUs spin-wait against each other (measured: a 10 s test took 646 s under 12 workers).
# Inside a worker the libraries are pinned to one thread. A plain serial run (no PYTEST_XDIST_WORKER) is untouched.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_name, "1")

PINNED_TREE = "pinned-in-tests"


@pytest.fixture(scope="module", autouse=True)
def cached_provenance(request: pytest.FixtureRequest) -> Iterator[None]:
    if getattr(request.module, "REAL_PROVENANCE", False):
        yield
        return
    from creator import ledger as LG
    from creator import model as M
    base = M.Provenance(engine_tree_hash=PINNED_TREE, creator_tree_hash=PINNED_TREE, git_commit=PINNED_TREE, config_hash=None, seed=None,
                        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))

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


@pytest.fixture(autouse=True)
def scripted_workers_are_supervised(request: pytest.FixtureRequest) -> Iterator[None]:
    """h70 (4 Oct 2026): creator.safety refuses changes from UNSUPERVISED workers unless the coding trust gate opened their class, and
    runs the real protected suite after every merge. Tests that drive kernel cycles with scripted workers test the kernel's OTHER
    mechanics: for them every worker counts as the teacher (exactly the behaviour before the safety loop) and the post-merge suite is
    off (a nested 3-minute run of the real suite per adoption). A module that exercises the safety loop sets REAL_SAFETY = True
    (tests/test_creator_safety.py does)."""
    if getattr(request.module, "REAL_SAFETY", False):
        yield
        return
    try:
        from creator import safety as SF
    except ImportError:
        yield
        return
    real = SF.policy
    mp = pytest.MonkeyPatch()
    mp.setattr(SF, "supervised", lambda pol, by: True)
    mp.setattr(SF, "policy", lambda state: dataclasses.replace(real(state), full_suite=False))
    try:
        yield
    finally:
        mp.undo()
