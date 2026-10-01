"""CR04: the agent runtime (C77 secs 14-16, 44-46, 68). A FAKE runner stands in for the CLI: these tests spend no credits."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Sequence

import pytest

from creator import agents as A
from creator import devbench as D


def policy(**kw) -> A.BudgetPolicy:
    base = dict(enabled=True, daily_calls=3, daily_usd=1.0, per_call_usd=0.25, per_job_calls=2)
    base.update(kw)
    return A.BudgetPolicy(**base)


def fake(result: str, usd: float = 0.1, rc: int = 0, write: dict | None = None, is_error: bool = False):
    seen: dict = {}

    def runner(cmd: Sequence[str], cwd: Path, timeout: int, stdin_text: str) -> tuple[int, str, str]:
        seen.update(cmd=list(cmd), cwd=cwd, stdin=stdin_text)
        for rel, text in (write or {}).items():
            (cwd / rel).parent.mkdir(parents=True, exist_ok=True)
            (cwd / rel).write_text(text, encoding="utf-8")
        return rc, json.dumps({"type": "result", "result": result, "total_cost_usd": usd, "num_turns": 3,
                               "is_error": is_error}), ""
    runner.seen = seen                                                  # type: ignore[attr-defined]
    return runner


@pytest.fixture()
def work(tmp_path: Path) -> Path:
    w = tmp_path / "job"
    w.mkdir()
    return w


def test_disabled_by_default_and_refused_before_spending(tmp_path: Path, work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CREATOR_AGENT_CALLS", raising=False)
    b = A.Budget(tmp_path / "b.json")
    assert not b.policy.enabled
    r = fake("STATUS: DONE")
    with pytest.raises(A.BudgetError, match="disabled"):
        A.run_agent(A.DEVELOPER, "j", "do it", work, b, r, tmp_path / "runs")
    assert not r.seen and not (tmp_path / "b.json").exists()          # nothing ran, nothing recorded


def test_env_caps_cannot_exceed_hard_ceilings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREATOR_AGENT_CALLS", "1")
    monkeypatch.setenv("CREATOR_AGENT_DAILY_CALLS", "100000")
    monkeypatch.setenv("CREATOR_AGENT_DAILY_USD", "9999")
    p = A.BudgetPolicy.from_env()
    assert p.enabled and p.daily_calls == 50 and p.daily_usd == 20.0


def test_daily_and_per_job_caps(tmp_path: Path, work: Path) -> None:
    b = A.Budget(tmp_path / "b.json", policy(daily_calls=3, per_job_calls=2))
    A.run_agent(A.DEVELOPER, "job1", "x", work, b, fake("STATUS: DONE"), tmp_path / "runs")
    A.run_agent(A.DEVELOPER, "job1", "x", work, b, fake("STATUS: DONE"), tmp_path / "runs")
    with pytest.raises(A.BudgetError, match="per-job"):
        A.run_agent(A.DEVELOPER, "job1", "x", work, b, fake("STATUS: DONE"), tmp_path / "runs")
    A.run_agent(A.DEVELOPER, "job2", "x", work, b, fake("STATUS: DONE"), tmp_path / "runs")
    with pytest.raises(A.BudgetError, match="daily call cap"):
        A.run_agent(A.DEVELOPER, "job3", "x", work, b, fake("STATUS: DONE"), tmp_path / "runs")
    assert b.today()[0] == 3
    tomorrow = A.Budget(tmp_path / "b.json", policy(), clock=lambda: dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1))
    assert tomorrow.today() == (0, 0.0)


def test_spend_cap(tmp_path: Path, work: Path) -> None:
    b = A.Budget(tmp_path / "b.json", policy(daily_calls=10, daily_usd=0.4, per_call_usd=0.25, per_job_calls=9))
    A.run_agent(A.DEVELOPER, "j", "x", work, b, fake("ok", usd=0.2), tmp_path / "runs")
    with pytest.raises(A.BudgetError, match="spend cap"):
        A.run_agent(A.DEVELOPER, "j", "x", work, b, fake("ok", usd=0.2), tmp_path / "runs")


def test_command_is_confined_and_the_run_recorded(tmp_path: Path, work: Path) -> None:
    b = A.Budget(tmp_path / "b.json", policy())
    r = fake("changed the code\nSTATUS: DONE", usd=0.12)
    run = A.run_agent(A.DEVELOPER, "j", "fix it", work, b, r, tmp_path / "runs")
    cmd = r.seen["cmd"]
    assert r.seen["cwd"] == work.resolve()
    assert cmd[:2] == ["claude", "-p"] and "--permission-mode" in cmd and cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert "WebFetch" in cmd[cmd.index("--disallowedTools") + 1] and "--strict-mcp-config" in cmd and cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--setting-sources") + 1] == "project" and "--max-budget-usd" in cmd
    assert "fix it" in r.seen["stdin"] and not any("\n" in a for a in cmd)          # prompt on stdin, no multi-line argv
    assert run.outcome == "OK" and run.claimed_done and run.usd == 0.12 and run.turns == 3
    rd = Path(run.run_dir)
    assert {"prompt.txt", "stdout.json", "stderr.txt", "run.json", "command.json"} <= {p.name for p in rd.iterdir()}
    assert json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["calls"][0]["usd"] == 0.12


def test_claims_are_parsed_strictly(tmp_path: Path, work: Path) -> None:
    b = A.Budget(tmp_path / "b.json", policy(daily_calls=9, per_job_calls=9))
    for text, want in (("all good, STATUS: DONE inline", False), ("STATUS: NOT_DONE", False), ("x\nSTATUS: DONE\n", True),
                       ("STATUS: DONE\nSTATUS: NOT_DONE", False)):
        assert A.run_agent(A.DEVELOPER, "j", "x", work, b, fake(text), tmp_path / "runs").claimed_done is want, text
    err = A.run_agent(A.DEVELOPER, "j", "x", work, b, fake("STATUS: DONE", is_error=True), tmp_path / "runs")
    assert err.outcome == "CLI_ERROR" and not err.claimed_done


def test_contamination_voids_the_run(tmp_path: Path, work: Path) -> None:
    b = A.Budget(tmp_path / "b.json", policy())
    r = fake("STATUS: DONE", write={"tests/test_peek.py": "open('../../creator/devbench/sealed/D01/hidden/test_hidden.py')\n"})
    run = A.run_agent(A.DEVELOPER, "j", "x", work, b, r, tmp_path / "runs")
    assert run.outcome == "CONTAMINATED" and not run.claimed_done and run.contamination


def test_refuses_to_run_inside_the_repository(tmp_path: Path) -> None:
    b = A.Budget(tmp_path / "b.json", policy())
    with pytest.raises(A.BudgetError, match="inside the repository"):
        A.run_agent(A.DEVELOPER, "j", "x", A.REPO_ROOT / "creator", b, fake("STATUS: DONE"), tmp_path / "runs")


def _suite(tmp_path: Path) -> tuple[D.Task, dict]:
    tasks, sealed = tmp_path / "tasks", tmp_path / "sealed"
    repo = tasks / "T1" / "repo"
    for rel, text in {"app/__init__.py": "", "app/core.py": "def double(x):\n    return x + 1\n", "tests/__init__.py": "",
                      "tests/test_core.py": "from app.core import double\n\n\ndef test_one():\n    assert double(1) == 2\n"}.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    (tasks / "T1/task.json").write_text(json.dumps({"id": "T1", "category": "bugfix", "split": "dev", "objective": "double"}),
                                        encoding="utf-8")
    (sealed / "T1/hidden").mkdir(parents=True)
    (sealed / "T1/hidden/__init__.py").write_text("", encoding="utf-8")
    (sealed / "T1/hidden/test_h.py").write_text("from app.core import double\n\n\ndef test_h():\n    assert double(4) == 8\n",
                                                encoding="utf-8")
    D.seal(tasks, sealed, sealed / "M.json")
    return D.load_tasks(tasks, sealed)[0], D.load_manifest(sealed / "M.json")


def test_agent_solver_is_scored_by_the_evaluator_not_by_its_claim(tmp_path: Path) -> None:
    task, m = _suite(tmp_path)
    b = A.Budget(tmp_path / "b.json", policy(daily_calls=9, per_job_calls=9))
    good = A.AgentSolver(b, runner=fake("fixed\nSTATUS: DONE", write={"app/core.py": "def double(x):\n    return 2 * x\n"}),
                         runs_dir=tmp_path / "runs", name="good")
    bluff = A.AgentSolver(b, runner=fake("fixed\nSTATUS: DONE"), runs_dir=tmp_path / "runs", name="bluff")
    peek = A.AgentSolver(b, runner=fake("STATUS: DONE", write={"x.py": "# devbench/sealed\n"}), runs_dir=tmp_path / "runs",
                         name="peek")
    assert D.run_task(task, good, m).outcome == "SOLVED"
    assert D.run_task(task, bluff, m).outcome == "FALSE_COMPLETION"
    assert D.run_task(task, peek, m).outcome == "ERROR"
    off = A.AgentSolver(A.Budget(tmp_path / "b2.json", policy(enabled=False)), runner=fake("STATUS: DONE"), name="off")
    s = D.run_task(task, off, m)
    assert s.outcome == "UNSOLVED" and "budget refused" in s.notes and s.calls == 0


def test_a_call_that_reports_no_cost_is_charged_the_cap(tmp_path: Path, work: Path) -> None:
    """Regression (30 Sep): a real call printed plain text (flags lost to cmd.exe) and was booked as free."""
    b = A.Budget(tmp_path / "b.json", policy(per_call_usd=0.25))

    def text_only(cmd, cwd, timeout, stdin_text):
        return 0, "I could not write files.", ""

    run = A.run_agent(A.DEVELOPER, "j", "x", work, b, text_only, tmp_path / "runs")
    assert run.outcome == "CLI_ERROR" and b.today() == (1, 0.25)

    def no_cli(cmd, cwd, timeout, stdin_text):
        raise FileNotFoundError("claude")

    run2 = A.run_agent(A.DEVELOPER, "j2", "x", work, b, no_cli, tmp_path / "runs")
    assert run2.outcome == "CLI_ERROR" and b.today() == (2, 0.25)            # never started: nothing spent


def test_no_multiline_argument_reaches_the_command_line(tmp_path: Path) -> None:
    spec = A.AgentSpec(role="r", instructions="line one\nline two", model="sonnet\nx")
    with pytest.raises(ValueError, match="newline"):
        A.build_command(spec, tmp_path / "i.txt", 0.5)


def test_the_worker_runs_the_projects_python_with_pytest(tmp_path: Path) -> None:
    """Regression (30 Sep): the worker's `python` lacked pytest. Checked with the real environment, no LLM call."""
    import shutil
    import subprocess
    env = A.worker_env()
    exe = shutil.which("python", path=env["PATH"])
    assert exe is not None
    p = subprocess.run([exe, "-m", "pytest", "--version"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert p.returncode == 0 and "pytest" in (p.stdout + p.stderr)


def test_contamination_scan_can_be_limited_to_changed_files(tmp_path: Path) -> None:
    """Regression (1 Oct): in a full-repo sandbox, pre-existing files legitimately mention protected paths."""
    (tmp_path / "old.py").write_text("PATH = 'state/livesim'\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert A.scan_contamination([], tmp_path)
    assert A.scan_contamination([], tmp_path, files=[tmp_path / "new.py"]) == ()
    (tmp_path / "new.py").write_text("open('creator/devbench/sealed/D01')\n", encoding="utf-8")
    assert A.scan_contamination([], tmp_path, files=[tmp_path / "new.py"])
