"""K19: the Creator improves its own worker by measured trials (owner, 1 Oct 2026). Fake local model; temporary devbench."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creator import autotune as AT
from creator import devbench as D
from creator import generator as G
from creator import model as M
from creator.ledger import Ledger


def put(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def bench(tmp_path: Path) -> tuple[list[D.Task], dict]:
    tasks, sealed = tmp_path / "tasks", tmp_path / "sealed"
    for tid, split in [(f"T{i}", "dev") for i in range(6)] + [(f"H{i}", "holdout") for i in range(3)]:
        repo = tasks / tid / "repo"
        put(repo / "app/__init__.py", "")
        put(repo / "app/core.py", "def double(x):\n    return x + 1\n")
        put(repo / "tests/__init__.py", "")
        put(repo / "tests/test_core.py", "from app.core import double\n\n\ndef test_two():\n    assert double(3) == 6\n")
        put(tasks / tid / "task.json", json.dumps({"id": tid, "category": "bugfix", "split": split, "objective": "fix double"}))
        put(sealed / tid / "hidden/__init__.py", "")
        put(sealed / tid / "hidden/test_h.py", "from app.core import double\n\n\ndef test_h():\n    assert double(7) == 14\n")
    D.seal(tasks, sealed, sealed / "M.json")
    return D.load_tasks(tasks, sealed), D.load_manifest(sealed / "M.json")


class TempSensitiveLLM:
    """Answers correctly only at temperature >= 0.4 - so raising it (the first move proposed from 0.2) is a measurable gain."""
    model = "fake"

    def chat(self, messages, temperature: float = 0.2, **kw) -> str:
        body = "def double(x):\n    return 2 * x\n" if temperature >= 0.4 else "def double(x):\n    return x\n"
        return f"FILE: app/core.py\n```python\n{body}```\n"


def test_proposals_rotate_fields_and_never_repeat(tmp_path: Path) -> None:
    hist = tmp_path / "h.jsonl"
    base = G.WorkerConfig()
    seen = set()
    for i in range(6):
        f, cand = AT.propose(base, hist)
        assert cand.digest() not in seen and cand != base
        seen.add(cand.digest())
        with hist.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"candidate": cand.digest()}) + "\n")
    fields = {f for f, c in AT.neighbours(base)}
    assert fields == set(AT.GRID)


def test_a_better_setting_is_found_measured_and_adopted(tmp_path: Path, bench) -> None:
    tasks, m = bench
    led = Ledger(tmp_path / "l.jsonl", evidence_root=tmp_path)
    paths = dict(active=tmp_path / "active.json", history=tmp_path / "h.jsonl", experience=tmp_path / "none.jsonl",
                 frozen=tmp_path / "frozen.json", out_dir=tmp_path / "evals")
    AT.save_active(G.WorkerConfig(search_budget=0, temperature=0.2), "start", paths["active"])
    outcomes = []
    for _ in range(6):                                                   # rotate until the temperature trial comes up
        t = AT.trial(led, TempSensitiveLLM(), replicates=2, tasks=tasks, manifest=m, **paths)
        assert t is not None
        outcomes.append((t.field, t.verdict, t.adopted))
        if t.adopted:
            break
    assert any(f == "temperature" and adopted for f, _, adopted in outcomes), outcomes
    assert AT.load_active(paths["active"]).temperature == 0.4
    assert all(not a for f, _, a in outcomes if f != "temperature")      # changes that do not help are never adopted
    adopt = [e for e in led.of_type("Decision") if getattr(e.record, "verdict") is M.DecisionVerdict.ADOPT]
    assert len(adopt) == 1 and led.verify()
