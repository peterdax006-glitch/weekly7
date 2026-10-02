"""CR098: integration tests - SEVERAL real Creator components driven together on a temporary git repo. No fakes for the components
under test: objective compile -> gap sync -> schedule -> planner -> kernel.execute -> sandbox -> evaluation -> improvement verdict ->
merge -> post-merge assessment/audit -> curriculum lesson. The only scripted part is the WORKER (the thing that writes code), as in
the kernel tests; every judgement downstream of it is the real code."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from creator import build as B
from creator import curriculum as CUR
from creator import gaps as G
from creator import kernel as K
from creator import model as M
from creator import reproduce as RP
from creator import sandbox as S
from creator import schedule as SCH
from creator import selfmodel as SM
from creator.audit import checks as AUD
from creator.ledger import Ledger

SPECS = [SM.CapabilitySpec("K01", "base", ("pkg/base.py",), ("tests/test_base.py",), 3),
         SM.CapabilitySpec("K02", "user", ("pkg/user.py",), ("tests/test_user.py",), 3)]
USER = "from pkg.base import one\n\n\ndef use():\n    return one() + 1\n\n\ndef twice():\n    return 2 * use()\n\n\ndef label():\n    return 'u'\n"
USER_TEST = ("from pkg.user import label, twice, use\n\n\ndef test_use():\n    assert use() == 2\n\n\n"
             "def test_twice():\n    assert twice() == 4\n\n\ndef test_label():\n    assert label() == 'u'\n")
BASE_TEST = "from pkg.base import one\n\n\ndef test_one():\n    assert one() == 1\n\n\ndef test_again():\n    assert one() + one() == 2\n"
BASE = "def one():\n    return 1\n\n\ndef two():\n    return 2\n\n\ndef three():\n    return 3\n"
BREAKER = "def one():\n    return 1\n\n\ndef two():\n    return 2\n\n\ndef three():\n    raise NotImplementedError\n"


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
    put(r, "pkg/base.py", BASE)
    put(r, "tests/__init__.py", "")
    put(r, "tests/test_base.py", BASE_TEST)
    put(r, "canon/CANON.md", "canon\n")
    sh(r, "init", "-q")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    return K.KernelConfig(repo=r, state=r / "state" / "creator", scratch=tmp_path / "scratch", capabilities=SPECS,
                          scope=("pkg", "tests"), build=B.BuildConfig(run_typecheck=False, require_typecheck=False))


class Scripted:
    """The only fake: the worker that writes the change."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files, self.name, self.plans = files, "scripted", []

    def __call__(self, plan, package, workdir: Path) -> K.WorkResult:
        self.plans.append((plan.requirement_key, plan.package_id))
        for rel, text in self.files.items():
            put(workdir, rel, text)
        return K.WorkResult(True, "scripted", 1, 0.01, reasoning="wrote the module and its tests")


def decisions(led: Ledger, verdict: M.DecisionVerdict) -> list[str]:
    """Decisions about CHANGES (subject = the Experiment). Since K08 design was integrated, the planner also records a REJECT for
    every design option it did not choose (subject = a DesignOption); those are not decisions about the change."""
    return [e.id for e in led.of_type("Decision") if getattr(e.record, "verdict") is verdict
            and str(getattr(e.record, "subject_id")).startswith(M.Experiment.PREFIX)]


def test_the_whole_pipeline_from_objective_to_resolved_lesson(cfg: K.KernelConfig, tmp_path: Path) -> None:
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    # 1. objective compile -> assessment -> gaps (real prepare: recover, compile, assess, gap sync, audit)
    main, _recovered, stop = K.prepare(cfg, led)
    assert stop is None and main is not None
    assert len(led.of_type("Capability")) == 2 and led.of_type("Requirement")
    open_gaps = [e for e in led.of_type("Gap") if led.view.status[e.id] in M.OPEN_STATES]
    assert any("K02.exists" in getattr(g.record, "description") for g in open_gaps)
    # 2. schedule: the batch the swarm would take names the same component the planner then plans
    batch = SCH.next_batch(led, 1, cfg.specs(), steps=("exists",))
    assert [(n.component, n.step) for n in batch.picks] == [("K02", "exists")]
    # 3. planner: one package for that gap, based on the real HEAD
    base_sha = S.head(cfg.repo)
    plan = K.plan_one(cfg, led, main, base_sha)
    assert plan is not None and plan.requirement_key == "K02.exists" and plan.component == "K02"
    # 4. kernel.execute with the curriculum recording the lesson (claude_step wraps the scripted worker as the session)
    cur = CUR.Curriculum(tmp_path / "lessons.jsonl")
    worker = cur.claude_step(Scripted({"pkg/user.py": USER, "tests/test_user.py": USER_TEST}))
    rep = K.execute(cfg, worker, plan, main, base_sha, 1, led, [])
    cur.resolve(rep)
    # 5. sandbox + evaluation + verdict + merge
    assert rep.outcome == "ADOPTED", (rep.reason, rep.details)
    assert rep.verdict == "IMPROVEMENT" and rep.merge_commit == sh(cfg.repo, "rev-parse", "HEAD").strip() != base_sha
    assert (cfg.repo / "pkg" / "user.py").read_text(encoding="utf-8") == USER
    assert base_sha in sh(cfg.repo, "rev-list", rep.merge_commit).split()                     # history is extended, never rewritten
    assert "pkg/user.py" in sh(cfg.repo, "ls-tree", "-r", "--name-only", rep.merge_commit)
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    claim = [e for e in led.of_type("ImprovementClaim")][-1]
    assert getattr(claim.record, "verdict") is M.Verdict.IMPROVEMENT
    adopt = decisions(led, M.DecisionVerdict.ADOPT)
    assert len(adopt) == 1 and getattr(led.get(adopt[0]), "claim_id") == claim.id
    # 6. post-merge: the requirement is closed by the post-merge evidence and the audit of the merged tree is clean
    req = led.view.unique[("Requirement", "K02.exists")]
    assert led.view.status[req] is M.Status.TESTED
    audit = AUD.audit(led, repo=cfg.repo, only=("ledger_integrity", "evidence_drift", "fake_adoption", "claim_recompute",
                                                 "strategy_attribution"))
    assert audit.clean, audit.to_dict()
    led.verify()
    # 7. curriculum: the lesson was recorded with the change and resolved from the cycle outcome
    (lesson,) = cur.log.lessons()
    assert lesson.adopted is True and lesson.verdict.startswith("ADOPTED") and lesson.package_id == rep.package
    assert set(lesson.files_after) == {"pkg/user.py", "tests/test_user.py"} and lesson.task_kind == "gap"
    # 8. an independent reproduction of the whole run agrees
    rr = RP.reproduce(cfg.repo, cfg.ledger_path)
    assert rr["chain"]["ok"] and rr["overall"] == RP.REPRODUCED, rr["claims"]


def test_a_regressing_change_is_rejected_nothing_merges_and_the_ledger_verifies(cfg: K.KernelConfig, tmp_path: Path) -> None:
    head = sh(cfg.repo, "rev-parse", "HEAD").strip()
    cur = CUR.Curriculum(tmp_path / "lessons.jsonl")
    bad = cur.claude_step(Scripted({"pkg/user.py": USER, "tests/test_user.py": USER_TEST, "pkg/base.py": BREAKER}))
    reports = K.run(cfg, bad, max_cycles=1, on_cycle=cur.resolve)
    (rep,) = reports
    assert rep.outcome == "REJECTED" and "requirements_kept" in rep.reason, (rep.outcome, rep.reason)
    assert sh(cfg.repo, "rev-parse", "HEAD").strip() == head and not (cfg.repo / "pkg" / "user.py").exists()
    assert (cfg.repo / "pkg" / "base.py").read_text(encoding="utf-8") == BASE
    assert sh(cfg.repo, "status", "--porcelain").strip() == ""
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    led.verify()
    assert decisions(led, M.DecisionVerdict.ADOPT) == [] and len(decisions(led, M.DecisionVerdict.REJECT)) == 1
    assert not any(ref.kind == "merge_commit" for e in led.view.entries for ref in e.record.evidence)
    gap = [e.id for e in led.of_type("Gap") if "K02.exists" in getattr(e.record, "description")][0]
    assert led.view.status[gap] is M.Status.FAILED                         # re-plannable, the attempt is counted
    req = led.view.unique[("Requirement", "K02.exists")]
    assert led.view.status[req] is not M.Status.TESTED
    assert AUD.audit(led, repo=cfg.repo, only=("ledger_integrity", "evidence_drift", "fake_adoption")).clean
    (lesson,) = cur.log.lessons()
    assert lesson.adopted is False and lesson.verdict.startswith("REJECTED")
    # the rejection did not poison the system: the next, sound change on the same ledger is planned again and adopted
    good = cur.claude_step(Scripted({"pkg/user.py": USER, "tests/test_user.py": USER_TEST}))
    ok = K.run(cfg, good, max_cycles=1, on_cycle=cur.resolve)[0]
    assert ok.outcome == "ADOPTED", (ok.reason, ok.details)
    assert [x.adopted for x in cur.log.lessons()] == [False, True]
    Ledger(cfg.ledger_path, evidence_root=cfg.repo).verify()


def test_a_tampered_ledger_stops_the_next_cycle_before_any_worker_runs(cfg: K.KernelConfig) -> None:
    from creator.ledger import LedgerError
    K.cycle(cfg, Scripted({"pkg/user.py": USER, "tests/test_user.py": USER_TEST}))
    head = sh(cfg.repo, "rev-parse", "HEAD").strip()
    raw = cfg.ledger_path.read_bytes().split(b"\n")
    raw[3] = raw[3].replace(b"a", b"b", 1)
    cfg.ledger_path.write_bytes(b"\n".join(raw))
    w = Scripted({"pkg/other.py": "X = 1\n"})
    with pytest.raises(LedgerError, match="hash mismatch"):
        K.cycle(cfg, w)
    assert w.plans == [] and sh(cfg.repo, "rev-parse", "HEAD").strip() == head


def test_a_post_merge_rollback_is_recorded_diagnosed_and_not_left_unresolved(cfg: K.KernelConfig,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Kernel + audit together (CR204): main fails after the merge -> reverted, and the rollback leaves a Failure, a Diagnosis and a
    Repair in the ledger, so the unresolved-failures audit stays green while a hand-built rollback with none of them would be red."""
    import dataclasses
    real = K.assess_tree

    def flaky_main(c, led, root, label, run_tests=True, audit=True, reuse=None):
        a = real(c, led, root, label, run_tests, audit, reuse)
        if label.endswith("_main_after"):
            rows = tuple(r if r.key != "K02.exists" else dataclasses.replace(r, met=False, detail="forced") for r in a.rows)
            return K.Assessed(a.model, rows, a.audit, a.snapshot)
        return a
    monkeypatch.setattr(K, "assess_tree", flaky_main)
    rep = K.cycle(cfg, Scripted({"pkg/user.py": USER, "tests/test_user.py": USER_TEST}))
    assert rep.outcome == "ROLLED_BACK" and rep.merge_commit
    assert not (cfg.repo / "pkg" / "user.py").exists()
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    assert len(decisions(led, M.DecisionVerdict.ROLLBACK)) == 1
    fl, dg, rp = led.of_type("Failure"), led.of_type("Diagnosis"), led.of_type("Repair")
    assert len(fl) == len(dg) == len(rp) == 1 and getattr(dg[0].record, "failure_id") == fl[0].id
    assert AUD.check_unresolved_failures(led) == [] and AUD.failure_report(led)["critical"] == 2     # the failure and the decision
    led.verify()
