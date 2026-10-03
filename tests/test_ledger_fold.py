"""h38 speed (3 Oct 2026): opening the ledger folds only the lines after a byte-identical, already verified prefix - and every tampering
the full fold catches is still caught when the prefix fold is warm."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from creator import ledger as L
from creator import ledgerfold as LF
from creator import model as M

K = M.Role.KERNEL


def prov() -> M.Provenance:
    return M.Provenance(engine_tree_hash="e" * 16, creator_tree_hash="c" * 16, git_commit="abc", config_hash=None, seed=0,
                        timestamp="2026-09-30T00:00:00+00:00")


def _world(led: L.Ledger, n: int) -> str:
    o = led.append(M.Objective(created_by=M.Role.OWNER, statement="develop yourself", acceptance_criteria=("a gain",)), provenance=prov())
    for i in range(n):
        led.append(M.Requirement(created_by=K, parents=(o,), key=f"R{i}", description="d", priority=M.Priority.HIGH, acceptance_test="t",
                                 measurement_method="m", failure_condition="f", validation_method="v", evidence_location="e"),
                   provenance=prov())
    return o


class _CountingJson:
    def __init__(self) -> None:
        self.loads_calls = 0

    def loads(self, s: str) -> Any:
        self.loads_calls += 1
        return json.loads(s)

    def __getattr__(self, name: str) -> Any:
        return getattr(json, name)


def _state(v: L.View) -> tuple[Any, ...]:
    return (v.head, [e.hash for e in v.entries], v.status, v.status_history, v.unique, v.children, v.digest())


@pytest.fixture(autouse=True)
def _cold() -> None:
    LF.clear()


def test_reopen_folds_only_new_lines_and_equals_a_full_fold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "dev.jsonl"
    a = L.Ledger(p, evidence_root=tmp_path)
    o = _world(a, 6)
    L.Ledger(p, evidence_root=tmp_path)                                  # verified full fold: the prefix is now known
    spy = _CountingJson()
    monkeypatch.setattr(L, "json", spy)
    again = L.Ledger(p, evidence_root=tmp_path)
    assert spy.loads_calls == 0                                          # old code: one json.loads per line (7)
    a.transition(o, M.Status.IN_PROGRESS, "start", K)                    # one more line, written by another instance
    newer = L.Ledger(p, evidence_root=tmp_path)
    assert spy.loads_calls == 1
    monkeypatch.setattr(L, "json", json)
    LF.clear()
    full = L.Ledger(p, evidence_root=tmp_path)
    assert _state(newer.view) == _state(full.view) and _state(again.view) != _state(full.view)
    assert newer.verify() == 8


def test_catch_up_after_another_writer_is_incremental_and_correct(tmp_path: Path) -> None:
    p = tmp_path / "dev.jsonl"
    a = L.Ledger(p, evidence_root=tmp_path)
    o = _world(a, 3)
    b = L.Ledger(p, evidence_root=tmp_path)
    a.transition(o, M.Status.IN_PROGRESS, "start", K)
    b.append(M.Requirement(created_by=K, parents=(o,), key="RB", description="d", priority=M.Priority.HIGH, acceptance_test="t",
                           measurement_method="m", failure_condition="f", validation_method="v", evidence_location="e"),
             provenance=prov())                                          # b catches up (incrementally) under the lock
    LF.clear()
    assert _state(b.view) == _state(L.Ledger(p, evidence_root=tmp_path).view) and b.verify() == 6
    with pytest.raises(L.LedgerError):                                   # a duplicate key is still refused from the caught-up view
        b.append(M.Requirement(created_by=K, parents=(o,), key="R1", description="d", priority=M.Priority.HIGH, acceptance_test="t",
                               measurement_method="m", failure_condition="f", validation_method="v", evidence_location="e"),
                 provenance=prov())


@pytest.mark.parametrize("attack", ["edit", "delete", "reorder", "insert", "truncate_middle", "rehash"])
def test_tampering_is_detected_with_a_warm_fold(tmp_path: Path, attack: str) -> None:
    p = tmp_path / "dev.jsonl"
    led = L.Ledger(p, evidence_root=tmp_path)
    _world(led, 3)
    L.Ledger(p, evidence_root=tmp_path)                                  # warm: the untampered prefix is cached
    lines = p.read_text(encoding="utf-8").splitlines()
    if attack == "edit":
        env = json.loads(lines[1])
        env["data"]["description"] = "rewritten"
        lines[1] = json.dumps(env)
    elif attack == "delete":
        del lines[1]
    elif attack == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    elif attack == "insert":
        lines.insert(1, lines[1])
    elif attack == "rehash":
        env = json.loads(lines[0])
        env.pop("hash")
        env["data"]["statement"] = "a different objective"
        env["hash"] = L._line_hash(env)
        lines[0] = L.canonical(env)
    else:
        lines = lines[:1] + lines[2:]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(L.LedgerError):
        L.Ledger(p, evidence_root=tmp_path)


def test_bad_appended_line_is_refused_and_not_cached(tmp_path: Path) -> None:
    p = tmp_path / "dev.jsonl"
    led = L.Ledger(p, evidence_root=tmp_path)
    _world(led, 2)
    L.Ledger(p, evidence_root=tmp_path)
    good = p.read_bytes()
    lines = good.decode("utf-8").splitlines()
    env = json.loads(lines[-1])
    env["seq"] = len(lines)
    env["prev"] = env["hash"]                                            # a well-formed line with a wrong id/hash
    p.write_bytes(good + (json.dumps(env) + "\n").encode("utf-8"))
    for _ in range(2):
        with pytest.raises(L.LedgerError):
            L.Ledger(p, evidence_root=tmp_path)


def test_a_replaced_checker_never_reuses_an_older_fold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "dev.jsonl"
    led = L.Ledger(p, evidence_root=tmp_path)
    _world(led, 2)
    L.Ledger(p, evidence_root=tmp_path)

    def refuse(*a: Any, **k: Any) -> M.Record:
        raise M.ModelError("rule changed")
    monkeypatch.setattr(M, "record_from_dict", refuse)
    with pytest.raises(L.LedgerError, match="no longer validates"):
        L.Ledger(p, evidence_root=tmp_path)
