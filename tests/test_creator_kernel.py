"""CR14: the self-development kernel (C77 secs 10, 20-29, 44-46, 59-61). A TEMPORARY git repo; scripted workers (no LLM)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from creator import build as B
from creator import kernel as K
from creator import model as M
from creator import selfmodel as SM
from creator.ledger import Ledger

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
    (cfg.state / "kernel.lock").write_text("999", encoding="utf-8")
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
