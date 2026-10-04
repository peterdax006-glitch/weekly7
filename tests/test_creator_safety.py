"""h70: the checker / safety loop (creator.safety) - trust gate in DECIDE, adoption rate limit + cool-down, the full protected suite
after every merge, the event log and the owner's daily digest (scripts/nupen_digest.py). Kernel paths run on a TEMPORARY repo with
scripted workers; the real protected suite is replaced by stub runners except in the one runner test."""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest

from creator import codetrust as CT
from creator import kernel as K
from creator import safety as SF
from tests.test_creator_kernel import USER, USER_TEST, Scripted, cached_provenance, cfg, head, put, sh  # noqa: F401 - fixtures + helpers

REAL_SAFETY = True                                  # tests/conftest.py: this module runs the real safety loop
ROOT = Path(__file__).resolve().parents[1]
GOOD = {"pkg/user.py": USER, "tests/test_user.py": USER_TEST}


def _records(path: Path, classes: Sequence[str] = ("tests_only", "docs", "refactor", "bugfix", "feature"), n: int = 30) -> Path:
    """Attempt records of a setup that passes every task of `classes` with calibrated confidence (opens them in codetrust)."""
    rows = [{"cls": c, "passed": True, "confidence": 0.97, "task": f"{c}{i}"} for c in classes for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _policy(cfg: K.KernelConfig, **kw: Any) -> None:
    cfg.state.mkdir(parents=True, exist_ok=True)
    (cfg.state / SF.POLICY_FILE).write_text(json.dumps({"full_suite": False, **kw}), encoding="utf-8")


def _events(cfg: K.KernelConfig, *rows: dict) -> None:
    for r in rows:
        SF.record(cfg.state, r)


# ------------------------------------------------------------------------------------------------ policy and the pure gates
def test_the_policy_defaults_are_strict_and_the_file_only_overrides_what_it_names(tmp_path: Path) -> None:
    p = SF.policy(tmp_path)
    assert (p.max_adoptions_per_hour, p.cooldown_hours, p.supervised, dict(p.trust_records), p.full_suite) == \
        (2, 6.0, ("claude-session",), {}, True)
    (tmp_path / SF.POLICY_FILE).write_text('{"max_adoptions_per_hour": 1, "trust_records": {"w": "r.jsonl"}}', encoding="utf-8")
    p = SF.policy(tmp_path)
    assert p.max_adoptions_per_hour == 1 and p.trust_records == {"w": "r.jsonl"} and p.cooldown_hours == 6.0 and p.full_suite
    (tmp_path / SF.POLICY_FILE).write_text("not json", encoding="utf-8")
    assert SF.policy(tmp_path) == SF.Policy()


def test_the_policy_and_the_safety_records_are_protected_and_safety_is_measuring_code() -> None:
    from creator import sandbox as S
    for p in ("creator/safety.py", "state/creator/safety.json", "state/creator/safety/events.jsonl", "state/creator/safety/suite_cache.json"):
        assert S.is_protected(p), p
    assert CT.class_of_paths(["creator/safety.py"]) == "measuring" and "tests/test_creator_safety.py" in CT.PROTECTED_SUITE


def test_trust_refuses_without_a_measured_setup_and_opens_only_measured_classes(tmp_path: Path) -> None:
    pol = SF.Policy()
    ok, why = SF.trust(pol, "nupen-coder", ["pkg/user.py"], "add a user helper")
    assert not ok and "class feature" in why and "no measured setup for worker 'nupen-coder'" in why
    pol = SF.Policy(trust_records={"nupen-coder": str(_records(tmp_path / "r.jsonl", ("docs",)))})
    assert SF.trust(pol, "nupen-coder", ["docs/a.md"], "document")[0]
    ok, why = SF.trust(pol, "nupen-coder", ["pkg/user.py"], "add a user helper")
    assert not ok and "only 0 scored tasks" in why                     # feature was never measured: closed


def test_measuring_changes_are_never_adopted_unsupervised_even_by_a_perfect_setup(tmp_path: Path) -> None:
    recs = _records(tmp_path / "r.jsonl", CT.CLASSES)
    pol = SF.Policy(trust_records={"w": str(recs)})
    for paths in (["creator/kernel.py"], ["creator/safety.py"], ["tests/conftest.py"], ["creator/codetrust.py", "docs/x.md"]):
        ok, why = SF.trust(pol, "w", paths, "tidy")
        assert not ok and "class measuring" in why and "never" in why, (paths, why)


def test_the_rate_limit_counts_unsupervised_adoptions_in_the_last_hour(tmp_path: Path) -> None:
    pol, now = SF.Policy(max_adoptions_per_hour=2), 1_000_000.0
    assert SF.rate(tmp_path, pol, now) == ""
    for at, sup in ((now - 4000, False), (now - 100, True), (now - 50, True), (now - 30, False)):
        SF.record(tmp_path, {"at": at, "outcome": "ADOPTED", "supervised": sup})
    assert SF.rate(tmp_path, pol, now) == ""                           # 1 unsupervised in the hour; the teacher's do not count
    SF.record(tmp_path, {"at": now - 10, "outcome": "ADOPTED", "supervised": False})
    assert "2 unsupervised adoptions in the last hour (limit 2)" in SF.rate(tmp_path, pol, now)
    assert SF.rate(tmp_path, pol, now + 3600) == ""
    assert "limit 0" in SF.rate(tmp_path, SF.Policy(max_adoptions_per_hour=0), now + 9999)


def test_any_rollback_starts_a_cool_down(tmp_path: Path) -> None:
    pol, now = SF.Policy(cooldown_hours=6), 2_000_000.0
    SF.record(tmp_path, {"at": now - 3600, "outcome": "ROLLED_BACK", "supervised": True})
    assert "cool-down after a rollback: 5.0 h left of 6 h" in SF.rate(tmp_path, pol, now)
    assert SF.rate(tmp_path, pol, now + 5 * 3600 + 1) == ""


class _Sb:
    id = "sbx1"

    def diff(self) -> str:
        return "diff --git a/x b/x\n+1\n"


def _gate(cfg: K.KernelConfig, by: str, paths: Sequence[str], at_decide: bool = False, now: Optional[float] = None
          ) -> tuple[Optional[Exception], K.CycleReport]:
    from creator import planner as P
    from creator import model as M
    plan = P.Plan("g", "K02.exists", "K02", "exists", M.Role.IMPLEMENTER, "wp", "PKG1", "cp", "ex", 1)
    rep = K.CycleReport(1, "ERROR", "PKG1", "K02.exists")
    wp = type("WP", (), {"objective": "add a user helper"})()
    return SF.gate(cfg, plan, wp, _Sb(), rep, by, paths, at_decide=at_decide, now=now), rep


def test_the_gate_lets_the_teacher_through_and_refuses_unsupervised_work_with_its_diff_saved(cfg: K.KernelConfig) -> None:
    stop, rep = _gate(cfg, "claude-session", ["pkg/user.py"])
    assert stop is None and rep.details["safety"]["supervised"] and rep.details["safety"]["cls"] == "feature"
    stop, rep = _gate(cfg, "self-search-v1", ["pkg/user.py"])
    assert isinstance(stop, K._Reject) and str(stop).startswith(SF.NEEDS) and rep.details["safety"]["refused"] == "trust"
    saved = Path(rep.details["saved_diff"])
    assert saved.parent == cfg.state / "pending" and "+1" in saved.read_text(encoding="utf-8")


def test_the_gate_defers_an_open_class_over_the_rate_limit_and_never_counts_it_as_an_attempt(cfg: K.KernelConfig, tmp_path: Path) -> None:
    from creator import planner as P
    _policy(cfg, trust_records={"w": str(_records(tmp_path / "r.jsonl"))})
    now = time.time()
    stop, rep = _gate(cfg, "w", ["pkg/user.py"], now=now)
    assert stop is None and rep.details["safety"]["gate"].startswith("open: class feature: open")
    _events(cfg, {"at": now - 60, "outcome": "ADOPTED", "supervised": False}, {"at": now - 30, "outcome": "ADOPTED", "supervised": False})
    for at_decide in (False, True):
        stop, rep = _gate(cfg, "w", ["pkg/user.py"], at_decide=at_decide, now=now)
        assert isinstance(stop, K.Cancelled) and str(stop).startswith(P.DEFERRED_PREFIX) and "rate limit" in str(stop)
        assert any(str(stop).startswith(x) for x in P.NOT_AN_ATTEMPT)


# ------------------------------------------------------------------------------------------------ the gate in real kernel cycles
def test_an_unsupervised_change_is_refused_in_decide_before_any_test_runs(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    real = K.S.Sandbox.evaluate
    monkeypatch.setattr(K.S.Sandbox, "evaluate", lambda self, *a, **k: ran.append("evaluate") or real(self, *a, **k))
    before = head(cfg)
    rep = K.cycle(cfg, Scripted("self-search-v1", GOOD))
    assert rep.outcome == "REJECTED" and rep.reason.startswith(SF.NEEDS) and "class " in rep.reason, rep.reason
    assert head(cfg) == before and not (cfg.repo / "pkg" / "user.py").exists() and ran == []
    assert list((cfg.state / "pending").glob("*.patch"))                # the work waits for the teacher
    ev = SF.events(cfg.state)
    assert ev[-1]["outcome"] == "REJECTED" and ev[-1]["refused"] == "trust" and ev[-1]["by"] == "self-search-v1"
    assert ev[-1]["paths"] == ["pkg/user.py", "tests/test_user.py"] and not ev[-1]["supervised"]


def test_the_teacher_adopts_as_before_and_the_adoption_is_recorded(cfg: K.KernelConfig) -> None:
    _policy(cfg, supervised=["good"])
    rep = K.cycle(cfg, Scripted("good", GOOD))
    assert rep.outcome == "ADOPTED", rep.reason
    ev = SF.events(cfg.state)[-1]
    assert ev["outcome"] == "ADOPTED" and ev["supervised"] and ev["merge"] == rep.merge_commit and ev["cls"] == "feature"


def test_an_open_class_is_adopted_unsupervised_until_the_rate_limit(cfg: K.KernelConfig, tmp_path: Path) -> None:
    _policy(cfg, trust_records={"coder": str(_records(tmp_path / "r.jsonl"))}, max_adoptions_per_hour=1)
    rep = K.cycle(cfg, Scripted("coder", GOOD))
    assert rep.outcome == "ADOPTED", rep.reason
    assert not SF.events(cfg.state)[-1]["supervised"] and "open" in SF.events(cfg.state)[-1]["gate"]
    _events(cfg, {"at": time.time(), "outcome": "ADOPTED", "supervised": False})
    assert "limit 1" in SF.rate(cfg.state, SF.policy(cfg.state))


def test_a_rate_limited_change_is_deferred_with_its_work_kept(cfg: K.KernelConfig, tmp_path: Path) -> None:
    _policy(cfg, trust_records={"coder": str(_records(tmp_path / "r.jsonl"))})
    _events(cfg, {"at": time.time() - 100, "outcome": "ROLLED_BACK", "supervised": True})
    before = head(cfg)
    rep = K.cycle(cfg, Scripted("coder", GOOD))
    assert rep.outcome == "CANCELLED" and "deferred: safety rate limit: cool-down" in rep.reason, rep.reason
    assert head(cfg) == before and rep.details.get("saved_diff")
    from creator import planner as P
    from creator.ledger import Ledger
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    gap = [e.id for e in led.of_type("Gap") if "K02.exists" in getattr(e.record, "description")][0]
    assert P.attempts_for(led, gap) == []                               # not an attempt: re-planned after the cool-down


# ------------------------------------------------------------------------------------------------ the full protected suite after the merge
def _runner(fail_on: dict[str, list[str]], calls: list[tuple[str, tuple[str, ...]]], repo: Path) -> Any:
    """A stub suite runner: the failing case ids per tree ('main' = the repository, 'sandbox' = any other root)."""
    def run(root: Path, files: Sequence[str], junit: Path, timeout: float, python: str = "", workers: int = 0) -> dict[str, Any]:
        where = "main" if Path(root).resolve() == repo.resolve() else "sandbox"
        calls.append((where, tuple(files)))
        failed = [c for c in fail_on.get(where, []) if c.split("::", 1)[0] in files]
        return {"status": "FAILED" if failed else "PASSED", "failed": failed, "files": list(files), "seconds": 0.1, "workers": workers or 4,
                "cases": 10}
    return run


def _merged(cfg: K.KernelConfig) -> str:
    n = len(list((cfg.repo / "pkg").glob("extra*.py")))
    put(cfg.repo, f"pkg/extra{n}.py", "X = 1\n")
    sh(cfg.repo, "add", "-A")
    sh(cfg.repo, "commit", "-q", "-m", "merge stand-in")
    return head(cfg)


def test_a_new_protected_suite_failure_after_the_merge_is_a_revert_reason(cfg: K.KernelConfig) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    f = "tests/test_creator_kernel.py::test_x"
    rep = K.CycleReport(1, "ERROR", "PKG1")
    why = SF.suite_failure(cfg, _merged(cfg), rep, runner=_runner({"main": [f]}, calls, cfg.repo))
    assert why == f"protected suite failed after the merge: ['{f}']"
    assert calls[0] == ("main", SF.suite_files())                        # the FULL suite on main
    assert calls[1] == ("main", ("tests/test_creator_kernel.py",))       # the failure re-run serially (it must reproduce)
    assert calls[2] == ("sandbox", ("tests/test_creator_kernel.py",))    # the parent, unknown: only the failing file
    assert rep.details["safety"]["suite"]["failed"] == [f]
    assert not list(cfg.scratch.glob("*suitebase*")) if cfg.scratch.is_dir() else True


def test_a_flaky_or_pre_existing_failure_is_not_a_revert_reason(cfg: K.KernelConfig) -> None:
    f = "tests/test_creator_kernel.py::test_x"
    calls: list[tuple[str, tuple[str, ...]]] = []
    rep = K.CycleReport(1, "ERROR", "P1")                               # the parent (unknown) fails the same case: not caused by the merge
    assert SF.suite_failure(cfg, _merged(cfg), rep, runner=_runner({"main": [f], "sandbox": [f]}, calls, cfg.repo)) == ""
    assert rep.details["safety"]["suite"]["pre_existing"] == [f]
    flaky = _runner({"main": [f]}, calls, cfg.repo)
    n = [0]

    def once(*a: Any, **k: Any) -> dict[str, Any]:
        n[0] += 1
        r = flaky(*a, **k)
        return r if n[0] == 1 else dict(r, failed=[], status="PASSED")
    rep = K.CycleReport(1, "ERROR", "P2")                               # fails once, passes on the serial re-run: a flake
    assert SF.suite_failure(cfg, _merged(cfg), rep, runner=once) == "" and rep.details["safety"]["suite"]["failed"] == []


def test_the_suite_is_cached_by_tree_hash_and_a_missing_result_is_a_failure(cfg: K.KernelConfig) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    m = _merged(cfg)
    assert SF.suite_failure(cfg, m, K.CycleReport(1, "ERROR", "P"), runner=_runner({}, calls, cfg.repo)) == ""
    rep = K.CycleReport(1, "ERROR", "P")
    assert SF.suite_failure(cfg, m, rep, runner=_runner({}, calls, cfg.repo)) == "" and len(calls) == 1 and rep.details["safety"]["suite"]["cached"]

    def dead(*a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "TIMEOUT", "failed": [], "files": [], "seconds": 1.0, "workers": 1, "problems": ["pytest exceeded"]}
    why = SF.suite_failure(cfg, _merged(cfg), K.CycleReport(1, "ERROR", "P"), runner=dead)
    assert why.startswith("protected suite gave no result after the merge (TIMEOUT")

    def boom(*a: Any, **k: Any) -> dict[str, Any]:
        raise OSError("disk gone")
    assert "check crashed" in SF.suite_failure(cfg, _merged(cfg), K.CycleReport(1, "ERROR", "P"), runner=boom)


def test_a_merge_that_breaks_the_protected_suite_is_reverted_and_starts_the_cool_down(cfg: K.KernelConfig,
                                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    _policy(cfg, supervised=["good"], full_suite=True)
    f = "tests/test_creator_kernel.py::test_x"
    monkeypatch.setattr(SF, "run_suite", _runner({"main": [f]}, [], cfg.repo))
    before = head(cfg)
    rep = K.cycle(cfg, Scripted("good", GOOD))
    assert rep.outcome == "ROLLED_BACK" and "protected suite failed after the merge" in rep.reason, (rep.outcome, rep.reason)
    assert not (cfg.repo / "pkg" / "user.py").exists() and head(cfg) != before          # reverted by a new commit
    from creator import model as M
    from creator.ledger import Ledger
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    assert any(getattr(e.record, "verdict") is M.DecisionVerdict.ROLLBACK for e in led.of_type("Decision"))
    assert led.of_type("Repair") and SF.events(cfg.state)[-1]["outcome"] == "ROLLED_BACK"
    assert SF.rate(cfg.state, SF.policy(cfg.state)).startswith("cool-down after a rollback")


def test_the_real_runner_reports_failing_cases_at_below_normal_priority(tmp_path: Path) -> None:
    r = tmp_path / "t"
    put(r, "tests/test_a.py", "def test_ok():\n    assert True\n")
    put(r, "tests/test_b.py", "def test_bad():\n    assert 1 == 2\n\n\ndef test_good():\n    assert 1\n")
    out = SF.run_suite(r, ["tests/test_a.py", "tests/test_b.py", "tests/test_missing.py"], tmp_path / "j" / "s.xml", 300, sys.executable, 1)
    assert out["status"] == "FAILED" and out["failed"] == ["tests/test_b.py::test_bad"] and out["cases"] == 3 and out["workers"] == 1
    if (CT.PYLIB / "xdist").is_dir():
        par = SF.run_suite(r, ["tests/test_a.py", "tests/test_b.py"], tmp_path / "j" / "p.xml", 300, sys.executable, 2)
        assert par["workers"] == 2 and par["failed"] == out["failed"]
