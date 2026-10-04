"""h43 (3 Oct 2026): anticipation results no longer depend on Python's per-process string hash: equal-weight words are ordered by the word
and scores are summed exactly (fsum), so two processes with different PYTHONHASHSEED give the same report."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from creator import anticipation as A
words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima", "mike", "november"]
dirs, arts = [], []
for i in range(14):
    ws = [words[(i + k) % len(words)] for k in range(6)]
    dirs.append(A.Doc(1000.0 + i * 100, "owner wants " + " ".join(ws), "owner", f"d{i}"))
    arts.append(A.Doc(950.0 + i * 100, "nupen blueprint " + " ".join(reversed(ws)), "nupen", f"a{i}"))
print(json.dumps({"anticipate": A.anticipate(dirs, arts), "drill": A.drill(dirs, arts)}, sort_keys=True))
"""


def _run(seed: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(root)], env=env, capture_output=True, text=True, check=True, cwd=root).stdout
    return json.loads(out)


def test_report_is_the_same_under_any_hash_seed() -> None:
    runs = [_run(s) for s in ("1", "2", "3", "12345")]
    assert runs[0]["anticipate"]["pairs"], "the synthetic data must produce matches (else the test proves nothing)"
    assert all(r == runs[0] for r in runs[1:])
