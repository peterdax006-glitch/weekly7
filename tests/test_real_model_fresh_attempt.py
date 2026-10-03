"""Opt-in (NUPEN_REAL_MODEL_TEST=1): the REAL local model, started from this tree, makes a FRESH attempt (no old patch to resume)
on a planted tiny task through ModelStudent and ActionStudent. Asserts an attempt was made and a verdict returned; prints timings
(run with -s). Whether the 1.5B model solved it is reported, not asserted - a fresh attempt existing is the property under test."""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from creator import action_student as AS
from creator import generator as G
from creator import model_student as MS

pytestmark = [
    pytest.mark.skipif(not (G.SERVER_EXE.is_file() and G.DEFAULT_MODEL.is_file()), reason="local model runtime not installed"),
    pytest.mark.skipif(os.environ.get("NUPEN_REAL_MODEL_TEST") != "1", reason="loads a model server: opt-in with NUPEN_REAL_MODEL_TEST=1"),
]

SRC = "import csv\nimport sys\n\n\ndef rows(path):\n    with open(path) as fh:\n        return list(csv.reader(fh))\n\n\ndef argv():\n    return sys.argv\n"
PLAN = SimpleNamespace(step="efficiency", requirement_key="x.shrink", component="util", package_id="P9")
PKG = SimpleNamespace(objective="move the csv import into rows", outputs=("app/u.py",))


@pytest.fixture()
def work(tmp_path: Path) -> Path:
    w = tmp_path / "w"
    (w / "app").mkdir(parents=True)
    (w / "app/u.py").write_text(SRC, encoding="utf-8")
    (w / ".creator_task.md").write_text("Move `import csv` into rows(); keep sys.", encoding="utf-8")
    return w


def test_model_student_makes_a_fresh_attempt_with_the_real_model(work: Path, tmp_path: Path) -> None:
    st = MS.ModelStudent(tmp_path / "none.jsonl")
    t0 = time.monotonic()
    res = st(PLAN, PKG, work)
    print(f"\nModelStudent: {time.monotonic() - t0:.1f}s calls={st.last_calls} claimed_done={res.claimed_done} notes={res.notes[:120]!r}")
    assert st.last_calls >= 1 and "model call failed" not in res.notes


def test_action_student_makes_a_fresh_attempt_with_the_real_model(work: Path, tmp_path: Path) -> None:
    st = AS.ActionStudent(tmp_path / "none.jsonl", state_dir=tmp_path / "st", use_chooser=False, prescreen=False)
    t0 = time.monotonic()
    res = st(PLAN, PKG, work)
    print(f"\nActionStudent: {time.monotonic() - t0:.1f}s calls={st.last_calls} claimed_done={res.claimed_done} notes={res.notes[:120]!r}")
    assert st.last_calls >= 1 and "model call failed" not in res.notes
