"""CR14: the self-development kernel (C77 secs 10, 20-29, 44-46, 59-61). A TEMPORARY git repo; scripted workers (no LLM)."""
from __future__ import annotations

import dataclasses
import datetime as dt
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import pytest

from creator import build as B
from creator import kernel as K
from creator import model as M
from creator import selfmodel as SM
from creator import ledger as LG
from creator.ledger import Ledger


@pytest.fixture(scope="module", autouse=True)
def cached_provenance() -> Iterator[None]:
    """Ledger.append recomputes provenance (tree hashes of creator/ and engine/, git state of the real checkout) for every record,
    ~0.17 s x hundreds of appends. No test here edits that checkout, so the real value is computed once per module and reused
    with a fresh seed, config hash and timestamp; the ledger's own computation is covered by tests/test_creator_ledger.py."""
    base = LG.current_provenance()

    def cached(seed: Optional[int] = None, config: Optional[Mapping[str, Any]] = None, *a: Any, **k: Any) -> M.Provenance:
        cfg_hash = LG.sha256_text(LG.canonical(dict(config)))[:16] if config is not None else None
        return dataclasses.replace(base, seed=seed, config_hash=cfg_hash,
                                   timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    mp = pytest.MonkeyPatch()
    mp.setattr(LG, "current_provenance", cached)
    try:
        yield
    finally:
        mp.undo()

SPECS = [SM.CapabilitySpec("K01", "base", ("pkg/base.py",), ("tests/test_base.py",), 3),
         SM.CapabilitySpec("K02", "user", ("pkg/user.py",), ("tests/test_user.py",), 3)]

USER = "from pkg.base import one\n\n\ndef use():\n    return one() + 1\n\n\ndef twice():\n    return 2 * use()\n\n\ndef label():\n    return 'u'\n"
USER_TEST = ("from pkg.user import label, twice, use\n\n\ndef test_use():\n    assert use() == 2\n\n\n"
             "def test_twice():\n    assert twice() == 4\n\n\ndef test_label():\n    assert label() == 'u'\n")
BASE_TEST = "from pkg.base import one\n\n\ndef test_one():\n    assert one() == 1\n\n\ndef test_again():\n    assert one() + one() == 2\n"


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def cfg(tmp_path: Path) -> K.KernelConfig:
    r = tmp_path / "repo"
    put(r, ".gitignore", "state/\n__pycache__/\n")
    put(r, "pkg/__init__.py", "")
    put(r, "pkg/base.py", "def one():\n    return 1\n\n\ndef two():\n    return 2\n\n\ndef three():\n    return 3\n")
    put(r, "pkg/app.py", "from pkg.base import one\n\n\ndef main():\n    return one()\n")
    put(r, "tests/__init__.py", "")
    put(r, "tests/test_base.py", BASE_TEST)
    put(r, "canon/CANON.md", "canon\n")
    sh(r, "init", "-q")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    return K.KernelConfig(repo=r, state=r / "state" / "creator", scratch=tmp_path / "scratch", capabilities=SPECS,
                          scope=("pkg", "tests"), build=B.BuildConfig(run_typecheck=False, require_typecheck=False))


class Scripted:
    def __init__(self, name: str, files: dict[str, str], claim: bool = True, refuse: bool = False) -> None:
        self.name, self.files, self.claim, self.refuse = name, files, claim, refuse
        self.seen: list[str] = []

    def __call__(self, plan, package, workdir: Path) -> K.WorkResult:
        self.seen.append(plan.requirement_key)
        if self.refuse:
            return K.WorkResult(False, "budget refused: test", refused=True)
        for rel, text in self.files.items():
            put(workdir, rel, text)
        return K.WorkResult(self.claim, "scripted", 1, 0.01)


def head(cfg: K.KernelConfig) -> str:
    return sh(cfg.repo, "rev-parse", "HEAD").strip()


def test_a_good_change_is_measured_adopted_and_the_gap_closes_by_evidence(cfg: K.KernelConfig) -> None:
    w = Scripted("good", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST})
    before = head(cfg)
    rep = K.cycle(cfg, w)
    assert w.seen == ["K02.exists"], w.seen
    assert rep.outcome == "ADOPTED", (rep.reason, rep.details)
    assert rep.verdict == "IMPROVEMENT" and rep.merge_commit == head(cfg) != before
    assert (cfg.repo / "pkg" / "user.py").read_text(encoding="utf-8") == USER
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    req = led.view.unique[("Requirement", "K02.exists")]
    assert led.view.status[req] is M.Status.TESTED                      # closed by the post-merge evidence, not the worker
    adopt = [e for e in led.of_type("Decision") if getattr(e.record, "verdict") is M.DecisionVerdict.ADOPT]
    assert adopt and any(ref.kind == "merge_commit" for ref in adopt[0].record.evidence)
    from creator.audit import checks as AUD
    assert not AUD.check_fake_adoption(led, repo=cfg.repo) and not AUD.check_evidence_drift(led)
    assert led.of_type("StrategyOutcome") and getattr(led.of_type("StrategyOutcome")[0].record, "success") is True


@pytest.mark.parametrize("name,files,why", [
    ("lazy", {}, "changed nothing"),
    ("broken", {"pkg/user.py": "def use(:\n    pass\n", "tests/test_user.py": USER_TEST}, "not clean"),
    ("weakener", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST,
                  "tests/test_base.py": "def test_one():\n    assert True\n"}, "weakened"),
    ("breaker", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST,
                 "pkg/base.py": "def one():\n    return 1\n\n\ndef two():\n    return 2\n\n\ndef three():\n    raise NotImplementedError\n"},
     "requirements_kept"),
])
def test_bad_changes_are_rejected_and_main_is_untouched(cfg: K.KernelConfig, name: str, files: dict, why: str) -> None:
    before = head(cfg)
    rep = K.cycle(cfg, Scripted(name, files))
    assert rep.outcome == "REJECTED" and why in rep.reason, (rep.outcome, rep.reason)
    assert head(cfg) == before and not (cfg.repo / "pkg" / "user.py").exists()
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    gap = [e.id for e in led.of_type("Gap") if "K02.exists" in getattr(e.record, "description")][0]
    assert led.view.status[gap] is M.Status.FAILED                      # re-plannable, attempt counted
    assert "sbx" not in sh(cfg.repo, "branch", "--list")                 # sandbox discarded


def test_a_protected_path_is_never_merged(cfg: K.KernelConfig) -> None:
    before = head(cfg)
    rep = K.cycle(cfg, Scripted("vandal", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST, "canon/CANON.md": "rewritten\n"}))
    assert rep.outcome in ("ERROR", "REJECTED") and "protected" in rep.reason.lower()
    assert head(cfg) == before and (cfg.repo / "canon" / "CANON.md").read_text(encoding="utf-8") == "canon\n"


def test_budget_refusal_stops_without_changes(cfg: K.KernelConfig) -> None:
    reps = K.run(cfg, Scripted("broke", {}, refuse=True), max_cycles=3)
    assert len(reps) == 1 and reps[0].outcome == "BUDGET"
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    wp = led.of_type("WorkPackage")[0].id
    assert led.view.status[wp] is M.Status.BLOCKED


def test_a_red_audit_stops_development(cfg: K.KernelConfig) -> None:
    K.cycle(cfg, Scripted("lazy", {}))                                  # creates evidence files in the ledger
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    cited = next(ref.path for e in led.view.entries for ref in e.record.evidence)
    (cfg.repo / cited).write_text("tampered", encoding="utf-8")
    w = Scripted("good", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST})
    rep = K.cycle(cfg, w)
    assert rep.outcome == "AUDIT_RED" and "evidence_drift" in rep.reason and w.seen == []


def test_one_kernel_at_a_time(cfg: K.KernelConfig) -> None:
    cfg.state.mkdir(parents=True, exist_ok=True)
    import os
    (cfg.state / "kernel.lock").write_text(str(os.getpid()), encoding="utf-8")   # a LIVE holder
    with pytest.raises(K.KernelError, match="another kernel"):
        K.cycle(cfg, Scripted("x", {}))


def test_a_change_that_fails_on_main_is_rolled_back(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    real = K.assess_tree

    def flaky_main(c, led, root, label, run_tests=True, audit=True):
        a = real(c, led, root, label, run_tests, audit)
        if label.endswith("_main_after"):
            rows = tuple(r if r.key != "K02.exists" else __import__("dataclasses").replace(r, met=False, detail="forced")
                         for r in a.rows)
            return K.Assessed(a.model, rows, a.audit, a.snapshot)
        return a
    monkeypatch.setattr(K, "assess_tree", flaky_main)
    before = head(cfg)
    rep = K.cycle(cfg, Scripted("good", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST}))
    assert rep.outcome == "ROLLED_BACK" and rep.merge_commit
    assert not (cfg.repo / "pkg" / "user.py").exists() and head(cfg) != before          # reverted by a new commit
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    assert any(getattr(e.record, "verdict") is M.DecisionVerdict.ROLLBACK for e in led.of_type("Decision"))


def test_interrupted_sandboxes_are_recovered(cfg: K.KernelConfig) -> None:
    from creator import sandbox as S
    sb = S.Sandbox.open(cfg.repo, scratch=cfg.scratch, label="crashed")
    put(sb.path, "pkg/half.py", "X = 1\n")
    rep = K.cycle(cfg, Scripted("lazy", {}))
    assert sb.id in rep.details["recovered"] and not sb.path.exists()


def test_the_prompt_is_the_package_and_names_protected_paths(cfg: K.KernelConfig) -> None:
    seen = {}

    class Spy(Scripted):
        def __call__(self, plan, package, workdir):
            seen["prompt"] = K.render_package(plan, package)
            seen["sealed_visible"] = (workdir / "creator" / "devbench" / "sealed").exists()
            return super().__call__(plan, package, workdir)
    K.cycle(cfg, Spy("spy", {}))
    assert "Objective:" in seen["prompt"] and "canon/*" in seen["prompt"] and "Never weaken" in seen["prompt"]
    assert seen["sealed_visible"] is False


def test_a_contaminated_worker_run_is_rejected(cfg: K.KernelConfig) -> None:
    class Peeker(Scripted):
        def __call__(self, plan, package, workdir):
            r = super().__call__(plan, package, workdir)
            return K.WorkResult(True, "peeked", 1, 0.01, contaminated=("tests/x.py contains 'devbench/sealed'",))
    before = head(cfg)
    rep = K.cycle(cfg, Peeker("peek", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST}))
    assert rep.outcome == "REJECTED" and "contaminated" in rep.reason and head(cfg) == before


def test_hidden_paths_return_for_evaluation_and_the_tests_that_need_them_pass(cfg: K.KernelConfig) -> None:
    """Regression (1 Oct, first real cycle): hiding the sealed keys broke the Creator's own tests of the sealed suite."""
    put(cfg.repo, "secret/key.txt", "42\n")
    put(cfg.repo, "tests/test_secret.py", "from pathlib import Path\n\n\ndef test_key_present():\n"
                                          "    assert (Path(__file__).parents[1] / 'secret' / 'key.txt').read_text() == '42\\n'\n")
    sh(cfg.repo, "add", "-A")
    sh(cfg.repo, "commit", "-q", "-m", "secret")
    specs = SPECS + [SM.CapabilitySpec("K03", "secret", ("tests/test_secret.py",), ("tests/test_secret.py",), 1)]
    c2 = __import__("dataclasses").replace(cfg, hide=("secret",), capabilities=specs)
    seen = {}

    class Look(Scripted):
        def __call__(self, plan, package, workdir):
            seen["hidden_during_work"] = not (workdir / "secret").exists()
            return super().__call__(plan, package, workdir)
    rep = K.cycle(c2, Look("good", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST}))
    assert seen["hidden_during_work"] is True
    assert rep.outcome == "ADOPTED", (rep.reason, rep.details.get("detail"))
    assert (cfg.repo / "secret" / "key.txt").read_text(encoding="utf-8") == "42\n"


def test_a_change_rewritten_during_evaluation_is_rejected(cfg: K.KernelConfig) -> None:
    sneaky_test = ("from pathlib import Path\n\n\ndef test_rewrite():\n"
                   "    p = Path(__file__).parents[1] / 'pkg' / 'user.py'\n"
                   "    p.write_text(p.read_text() + '\\nX = 1\\n')\n")
    rep = K.cycle(cfg, Scripted("sneaky", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST + "\n" + sneaky_test}))
    assert rep.outcome == "REJECTED" and "moved during evaluation" in rep.reason, rep.reason



def test_steps_limit_planning_and_never_admit_validation(cfg: K.KernelConfig) -> None:
    import dataclasses
    put(cfg.repo, "pkg/user.py", USER)                                   # exists + tested, but no production code imports it
    put(cfg.repo, "tests/test_user.py", USER_TEST)
    sh(cfg.repo, "add", "-A")
    sh(cfg.repo, "commit", "-q", "-m", "user")
    only_validation = K.cycle(dataclasses.replace(cfg, steps=("validated",)), Scripted("v", {}))
    assert only_validation.outcome == "NOTHING_TO_DO"                     # a validation step is never handed to a worker
    w = Scripted("integrator", {"pkg/app.py": "from pkg.base import one\nfrom pkg.user import use\n\n\ndef main():\n"
                                              "    return one() + use()\n",
                                  "tests/test_app.py": "from pkg.app import main\n\n\ndef test_main_uses_user():\n"
                                                       "    assert main() == 3\n"})
    rep = K.cycle(dataclasses.replace(cfg, steps=("integrated", "validated")), w)
    assert w.seen == ["K02.integrated"], w.seen
    assert rep.outcome == "ADOPTED", (rep.reason, rep.details.get("detail"))


def test_an_untested_change_is_never_adopted(cfg: K.KernelConfig) -> None:
    import dataclasses
    put(cfg.repo, "pkg/user.py", USER)
    put(cfg.repo, "tests/test_user.py", USER_TEST)
    sh(cfg.repo, "add", "-A")
    sh(cfg.repo, "commit", "-q", "-m", "user")
    w = Scripted("untested", {"pkg/app.py": "from pkg.base import one\nfrom pkg.user import use\n\n\ndef main():\n"
                                            "    return one() + use()\n"})
    rep = K.cycle(dataclasses.replace(cfg, steps=("integrated",)), w)
    assert rep.outcome == "REJECTED" and "report=None" in rep.reason       # no test reached the change: fail closed


def test_the_handoff_worker_waits_for_the_session_and_the_kernel_still_decides(cfg: K.KernelConfig) -> None:
    """Owner, 1 Oct 2026: the Claude session is the only worker. The kernel hands the package over and measures the result."""
    import threading

    def session(workdir: Path, package_id: str) -> None:
        def work() -> None:
            assert (workdir / ".creator_task.md").read_text(encoding="utf-8").startswith("You are working")
            put(workdir, "pkg/user.py", USER)
            put(workdir, "tests/test_user.py", USER_TEST)
            (workdir / ".creator_done.json").write_text('{"claimed_done": true, "notes": "done by the session"}',
                                                        encoding="utf-8")
        threading.Thread(target=work).start()

    w = K.HandoffWorker(poll_s=0.2, timeout_s=60, notify=session)
    rep = K.cycle(cfg, w)
    assert rep.outcome == "ADOPTED", rep.reason
    assert rep.details["worker"] == {"claimed_done": True, "notes": "done by the session", "by": "claude-session"}
    tree = sh(cfg.repo, "ls-tree", "-r", "--name-only", rep.merge_commit)
    assert ".creator_task.md" not in tree and ".creator_done.json" not in tree


def test_an_unanswered_handoff_is_not_adopted(cfg: K.KernelConfig) -> None:
    rep = K.cycle(cfg, K.HandoffWorker(poll_s=0.1, timeout_s=0.5))
    assert rep.outcome == "REJECTED" and "changed nothing" in rep.reason


def test_a_lock_left_by_a_dead_process_is_taken_over(cfg: K.KernelConfig) -> None:
    """Regression (1 Oct): a lock was deleted by hand while its holder still ran. Dead holders are detected instead."""
    cfg.state.mkdir(parents=True, exist_ok=True)
    (cfg.state / "kernel.lock").write_text("999999", encoding="utf-8")        # no such process
    with K._KernelLock(cfg.state):
        assert (cfg.state / "kernel.lock").read_text(encoding="utf-8").strip() != "999999"
    import os
    (cfg.state / "kernel.lock").write_text(str(os.getpid()), encoding="utf-8")  # a LIVE holder (this process)
    with pytest.raises(K.KernelError, match="another kernel"):
        K._KernelLock(cfg.state).__enter__()
    (cfg.state / "kernel.lock").unlink()



def test_a_failing_candidate_is_diagnosed_and_the_next_attempt_sees_the_root_cause(cfg: K.KernelConfig) -> None:
    """K11 integrated: the kernel diagnoses a rejected candidate's failing tests and the next work package carries it."""
    bad = USER.replace("return one() + 1", "return one() + 9")             # test_use expects 2
    rep = K.cycle(cfg, Scripted("wrong", {"pkg/user.py": bad, "tests/test_user.py": USER_TEST}))
    assert rep.outcome == "REJECTED" and "diagnosis ASSERTION" in rep.reason, rep.reason
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    assert led.of_type("Failure") and led.of_type("Diagnosis")
    assert led.of_type("Diagnosis")[-1].record.failure_id == led.of_type("Failure")[-1].id
    w = Scripted("again", {})
    second = K.cycle(cfg, w)
    assert second.package, (second.outcome, second.reason)            # 2 Oct: went AUDIT_RED on evidence drift
    wp = led.__class__(cfg.ledger_path, evidence_root=cfg.repo).of_type("WorkPackage")[-1].record
    assert "diagnosis ASSERTION" in wp.why_it_exists                        # the second attempt starts from the diagnosis


def test_a_deferred_package_is_not_a_failed_attempt_and_is_never_blocked_away(cfg: K.KernelConfig) -> None:
    """Regression (validator round 3): a worker that hands a package over (WorkResult.deferred) must not use up the gap's
    MAX_ATTEMPTS; before the fix three deferrals BLOCKED the gap and the handed-over task was silently dropped."""
    import dataclasses

    class Defers:
        name = "defers"

        def __call__(self, plan, package, workdir: Path) -> K.WorkResult:
            return dataclasses.replace(K.WorkResult(False, "handed over to a student; retry later"), deferred=True)

    for _ in range(4):                                                  # more than planner.MAX_ATTEMPTS
        rep = K.cycle(cfg, Defers())
        assert rep.outcome == "DEFERRED", (rep.outcome, rep.reason)
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    gap = [e.id for e in led.of_type("Gap") if "K02.exists" in getattr(e.record, "description")][0]
    assert led.view.status[gap] is not M.Status.BLOCKED
    good = Scripted("good", {"pkg/user.py": USER, "tests/test_user.py": USER_TEST})
    assert K.cycle(cfg, good).outcome == "ADOPTED"                      # still plannable, and only the measurement adopts


def test_the_sandbox_comparison_gets_the_suite_timeout_unless_set(tmp_path: Path) -> None:
    """CP0047 (1 Oct): base and candidate runs both hit the 600 s pytest default under load -> INCONCLUSIVE for a sound change."""
    import dataclasses as dc
    from creator import testrun as TR
    cfg = K.KernelConfig(repo=tmp_path, state=tmp_path / "s")
    assert K.sandbox_pytest(cfg).timeout == cfg.test_timeout >= 3600
    explicit = dc.replace(cfg, pytest=TR.PytestConfig(timeout=30.0))
    assert K.sandbox_pytest(explicit).timeout == 30.0                                 # an explicit choice is kept


def test_concurrent_kernel_log_appends_never_interleave(tmp_path: Path) -> None:
    """Regression (validator open issue 1): every swarm thread appends its cycle record to one kernel_log.jsonl; an unlocked
    text-mode append of a large record interleaves with another thread's, corrupting both lines."""
    import json
    import threading
    path = tmp_path / "kernel_log.jsonl"
    big = "reason " * 60000

    def work(w: int) -> None:
        for i in range(10):
            K._append_log_line(path, json.dumps({"w": w, "i": i, "reason": big}))
    ts = [threading.Thread(target=work, args=(w,)) for w in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 80 and {(r["w"], r["i"]) for r in rows} == {(w, i) for w in range(8) for i in range(10)}


def test_cycle_record_goes_through_the_locked_appender() -> None:
    """The cycle's own kernel_log write must use the locked appender, not an ad-hoc open(...'a')."""
    src = Path(K.__file__).read_text(encoding="utf-8")
    assert '_append_log_line(cfg.state / "kernel_log.jsonl"' in src
    assert 'open("a", encoding="utf-8") as fh:\n            fh.write(json.dumps(dataclasses.asdict(rep)' not in src


def test_plan_one_never_hands_the_same_file_to_two_workers(monkeypatch, tmp_path) -> None:
    """A running efficiency package holds its path: gap components whose modules/tests include it are excluded, and a running gap
    component holds its modules and tests against shrink packages."""
    from types import SimpleNamespace
    from creator import selfmodel as SM2
    specs = [SM2.CapabilitySpec("K01", "a", ("creator/a.py",), ("tests/test_a.py",), 0),
             SM2.CapabilitySpec("K02", "b", ("creator/b.py",), ("tests/test_b.py",), 0)]
    cfg = K.KernelConfig(repo=tmp_path, state=tmp_path / "s", capabilities=specs, mode="auto")
    got: dict = {}
    monkeypatch.setattr(K.P, "plan_next", lambda led, model, base, sp, steps, exclude_components: got.update(c=tuple(exclude_components)))
    monkeypatch.setattr(K.P, "plan_efficiency", lambda led, root, base, avoid=(), kind=None: got.update(a=tuple(avoid)))
    K.plan_one(cfg, None, SimpleNamespace(model=None), "b", exclude_components=("K02",), exclude_paths=("creator/a.py",))
    assert set(got["c"]) == {"K02", "K01"} and set(got["a"]) == {"creator/a.py", "creator/b.py", "tests/test_b.py"}


def _spy_pytest(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[str, ...]]]:
    from creator import testrun as TR
    calls: list[tuple[str, tuple[str, ...]]] = []
    real = TR.run_pytest

    def spy(root, targets, junit_path, *a, **k):                       # type: ignore[no-untyped-def]
        calls.append((k.get("label", ""), tuple(targets)))
        return real(root, targets, junit_path, *a, **k)
    monkeypatch.setattr(TR, "run_pytest", spy)
    return calls


def test_main_assessment_serves_an_unchanged_clean_tree_and_nothing_else(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    calls = _spy_pytest(monkeypatch)
    first = K.assess_tree(cfg, led, cfg.repo, "main", audit=False)
    assert calls, "the first assessment must run the declared tests"
    calls.clear()
    again = K.assess_tree(cfg, led, cfg.repo, "main", audit=False)
    assert not calls                                                     # byte-identical tree: served from the tree cache
    assert [(r.key, r.met) for r in again.rows] == [(r.key, r.met) for r in first.rows]
    K.assess_tree(cfg, led, cfg.repo, "X_main_after", audit=False)
    assert calls                                                         # the post-merge check and the replicates always run
    calls.clear()
    put(cfg.repo, "pkg/base.py", "def one():\n    return 1\n\n\ndef two():\n    return 2\n\n\ndef three():\n    return 3\n\n\ndef four():\n    return 4\n")
    K.assess_tree(cfg, led, cfg.repo, "main", audit=False)
    assert calls, "a dirty tree is never served"


def test_a_cycle_serves_the_base_run_from_main_and_decides_the_same(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    from creator import treecache as TC
    change = {"pkg/user.py": USER, "tests/test_user.py": USER_TEST,
              "pkg/base.py": "def one():\n    return 1\n\n\ndef two():\n    return 2\n\n\ndef three():\n    return 3\n\n\ndef four():\n    return 4\n"}
    calls = _spy_pytest(monkeypatch)
    rep = K.cycle(cfg, Scripted("good", dict(change)))
    assert rep.outcome == "ADOPTED", (rep.reason, rep.details)
    assert not [c for c in calls if c[0] == "base"], calls               # tests/test_base.py at base came from main's assessment
    # control: with the cache disabled the base run happens and the decision is the same
    import shutil
    shutil.rmtree(cfg.state / "evidence" / "tree_cache", ignore_errors=True)
    sh(cfg.repo, "reset", "-q", "--hard", "HEAD~1")
    for d in ("pkg/user.py", "tests/test_user.py"):
        assert not (cfg.repo / d).exists()
    (cfg.state / "ledger.jsonl").unlink(missing_ok=True)
    monkeypatch.setattr(TC, "tree_key", lambda *a, **k: None)
    calls.clear()
    rep2 = K.cycle(cfg, Scripted("good", dict(change)))
    assert [c for c in calls if c[0] == "base"], calls
    assert (rep2.outcome, rep2.verdict) == (rep.outcome, rep.verdict)
