"""Independent-validator regressions (C77 sec 48; CR203/CR206): bugs found by reading the Creator against its own contracts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from creator import agents as A
from creator import objective as O
from creator import selfmodel as SM
from creator.ledger import Ledger


def test_run_agent_survives_non_object_json_output(tmp_path: Path) -> None:
    """A CLI that prints valid JSON that is not an object (a bare string/number) is a CLI_ERROR, never a crash."""
    work = tmp_path / "job"
    work.mkdir()
    for payload in ('"hello"', "123", "null", "[1, 2]"):
        def runner(cmd: Sequence[str], cwd: Path, timeout: int, stdin_text: str, p: str = payload) -> tuple[int, str, str]:
            return 0, p, ""
        b = A.Budget(tmp_path / "b.json", A.BudgetPolicy(enabled=True, daily_calls=50, daily_usd=50.0, per_call_usd=0.25,
                                                        per_job_calls=50))
        run = A.run_agent(A.DEVELOPER, "j", "do it", work, b, runner, tmp_path / "runs")
        assert run.outcome == "CLI_ERROR" and not run.claimed_done


def test_requirement_dependency_on_later_numbered_component_is_recorded(tmp_path: Path) -> None:
    """K10 is built on K15 (COMPONENT_DEPENDS): compile order must be dependency order, not id order."""
    led = Ledger(tmp_path / "dev.jsonl", evidence_root=tmp_path)
    specs = [SM.CapabilitySpec(k, k, (f"pkg/{k}.py",), (), 0) for k in ("K01", "K05", "K06", "K10", "K15")]
    c = O.compile_capabilities(led, O.self_objective(led), specs)
    deps = set(getattr(led.get(c.requirement_ids["K10.exists"]), "depends_on"))
    assert c.requirement_ids["K15.tested"] in deps
    assert c.requirement_ids["K06.tested"] in deps
