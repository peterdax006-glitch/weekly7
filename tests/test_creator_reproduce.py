"""CR207/CR208: independent reproduction on temporary git repos + ledgers (chain, evidence, verdict recompute, ADOPT commits, rerun)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from creator import ledger as L
from creator import model as M
from creator import reproduce as R

K = M.Role.KERNEL


def prov() -> M.Provenance:
    return M.Provenance(engine_tree_hash="e" * 16, creator_tree_hash="c" * 16, git_commit="abc", config_hash=None, seed=0,
                        timestamp="2026-09-30T00:00:00+00:00")


def comp(fn: str = "creator.model:improvement_verdict") -> M.ComputationRef:
    return M.ComputationRef(function=fn, code_hash="ab" * 8, inputs_sha256="cd" * 8, output_sha256="ef" * 8)


def git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def make_repo(tmp: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    git(repo, "config", "commit.gpgsign", "false")
    (repo / "README").write_text("x", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    for name, text in files.items():
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "feature")
    return repo, git(repo, "rev-parse", "HEAD")


def meas(led: L.Ledger, parent: str, value: float, split: M.Split = M.Split.DEV, metric: str = "solve_rate") -> str:
    return led.append(M.Measurement(created_by=K, parents=(parent,), metric=metric, value=value, stderr=0.01, n=50, population="d",
                                    conditions="same", higher_is_better=True, split=split,
                                    computation=comp("creator.evaluate:measure")), provenance=prov())


def adopted_ledger(repo: Path, commit: str, metric: str = "solve_rate", cand: float = 0.7,
                   evidence_kind: str = "merge_commit") -> tuple[L.Ledger, str, str]:
    """A ledger with one valid IMPROVEMENT claim and an ADOPT citing `commit`. Returns (ledger, claim_id, decision_id)."""
    led = L.Ledger(repo / "state" / "ledger.jsonl", evidence_root=repo)
    o = led.append(M.Objective(created_by=M.Role.OWNER, statement="develop", acceptance_criteria=("gain",)), provenance=prov())
    gap = led.append(M.Gap(created_by=K, parents=(o,), kind=list(M.GapKind)[0], description="d", importance=0.5), provenance=prov())
    ex = led.append(M.Experiment(created_by=K, parents=(gap,), hypothesis="h", design="d", metrics=(metric,), seed=1,
                                 baseline_ref="b", candidate_ref="c"), provenance=prov())
    base = [meas(led, ex, 0.50, metric=metric), meas(led, ex, 0.51, metric=metric)]
    cands = [meas(led, ex, cand, metric=metric), meas(led, ex, cand + 0.01, metric=metric)]
    reg = (meas(led, ex, 0.9, metric="regression_pass"), meas(led, ex, 0.9, metric="regression_pass"))
    hold = (meas(led, ex, 0.40, M.Split.HOLDOUT, metric=metric), meas(led, ex, 0.60, M.Split.HOLDOUT, metric=metric))
    cid = led.append(M.ImprovementClaim(created_by=M.Role.VALIDATOR, parents=(ex,), subject_id=ex, baseline_ids=tuple(base),
                                        candidate_ids=tuple(cands), verdict=M.Verdict.IMPROVEMENT, computation=comp(),
                                        regression_baseline_ids=(reg[0],), regression_candidate_ids=(reg[1],),
                                        holdout_baseline_id=hold[0], holdout_candidate_id=hold[1]), provenance=prov())
    ev_path = repo / "state" / "cycles" / commit
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    ev_path.write_text(commit, encoding="utf-8")
    ev = M.EvidenceRef.of(ev_path, repo, kind=evidence_kind)
    did = led.append(M.Decision(created_by=K, subject_id=ex, verdict=M.DecisionVerdict.ADOPT, reason="measured", claim_id=cid,
                                evidence=(ev,)), provenance=prov())
    return led, cid, did


GOOD = {"creator/feature.py": "def f() -> int:\n    return 1\n", "state/data.json": "{\"a\": 1}"}


def test_clean_ledger_reproduces(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path, GOOD)
    led, cid, _ = adopted_ledger(repo, sha)
    rep = R.reproduce(repo, led.path)
    assert rep["chain"]["ok"] and rep["chain"]["records"] == len(led.view.entries)
    assert rep["counts"]["claims"] == 1 and rep["counts"][R.REPRODUCED] == 1 and rep["overall"] == R.REPRODUCED
    assert rep["claims"][0]["claim"] == cid and rep["claims"][0]["recomputed_verdict"] == "IMPROVEMENT"
    assert rep["claims"][0]["adoptions"][0]["commit"] == sha and rep["evidence"]["changed"] == 0


def test_tampered_evidence_is_not_reproduced(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path, GOOD)
    led, _, _ = adopted_ledger(repo, sha)
    (repo / "state" / "cycles" / sha).write_text("something else", encoding="utf-8")
    rep = R.reproduce(repo, led.path)
    assert rep["claims"][0]["status"] == R.NOT_REPRODUCED and rep["evidence"]["changed"] == 1
    assert any("evidence changed" in r for r in rep["claims"][0]["reasons"])


def test_missing_evidence_is_unverifiable_not_reproduced(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path, GOOD)
    led, _, _ = adopted_ledger(repo, sha)
    (repo / "state" / "cycles" / sha).unlink()
    rep = R.reproduce(repo, led.path)
    assert rep["claims"][0]["status"] == R.UNVERIFIABLE and rep["evidence"]["missing"] == 1


def test_claim_that_no_longer_reproduces(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path, GOOD)
    led, _, _ = adopted_ledger(repo, sha)

    def stricter_rule(*a: object, **k: object) -> tuple[M.Verdict, dict[str, object]]:
        return M.Verdict.INSUFFICIENT_EVIDENCE, {"why": "rule changed"}

    rep = R.reproduce(repo, led.path, rule=stricter_rule)
    c = rep["claims"][0]
    assert c["status"] == R.NOT_REPRODUCED and c["recomputed_verdict"] == "INSUFFICIENT_EVIDENCE" and rep["counts"][R.NOT_REPRODUCED] == 1


def test_edited_ledger_line_breaks_the_chain(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path, GOOD)
    led, _, _ = adopted_ledger(repo, sha)
    lines = led.path.read_text(encoding="utf-8").split("\n")
    i = next(n for n, ln in enumerate(lines) if '"rtype":"ImprovementClaim"' in ln)
    lines[i] = lines[i].replace("IMPROVEMENT", "NO_EFFECT", 1)
    led.path.write_text("\n".join(lines), encoding="utf-8")
    rep = R.reproduce(repo, led.path)
    assert not rep["chain"]["ok"] and "hash mismatch" in rep["chain"]["error"] and rep["overall"] == R.UNVERIFIABLE
    assert rep["counts"]["claims"] == 0                               # the edited claim is never trusted or counted


def test_missing_merge_commit_is_not_reproduced(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path, GOOD)
    led, _, _ = adopted_ledger(repo, "1" * 40)
    rep = R.reproduce(repo, led.path)
    c = rep["claims"][0]
    assert c["status"] == R.NOT_REPRODUCED and "does not exist in git" in c["adoptions"][0]["reasons"][0]


def test_commit_with_unparseable_file_is_not_reproduced(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path, {"creator/broken.py": "def f(:\n", "state/x.json": "{nope"})
    led, _, _ = adopted_ledger(repo, sha)
    rep = R.reproduce(repo, led.path)
    reasons = rep["claims"][0]["adoptions"][0]["reasons"]
    assert rep["claims"][0]["status"] == R.NOT_REPRODUCED and len(reasons) == 2


def test_not_a_git_repo_is_unverifiable(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path, GOOD)
    led, _, _ = adopted_ledger(repo, sha)
    plain = tmp_path / "plain"
    (plain / "state" / "cycles").mkdir(parents=True)
    (plain / "state" / "cycles" / sha).write_text(sha, encoding="utf-8")
    rep = R.reproduce(plain, led.path)
    assert rep["claims"][0]["status"] == R.UNVERIFIABLE


def test_rerun_matches_recorded_pass_rate_and_removes_worktree(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path, {"tests/test_ok.py": "def test_a():\n    assert True\n"})
    led, _, _ = adopted_ledger(repo, sha, metric=R.RERUN_METRIC, cand=0.995)       # recorded pass rate ~1.0
    rep = R.reproduce(repo, led.path, rerun=True, rerun_timeout=60)
    rr = rep["claims"][0]["adoptions"][0]["rerun"]
    assert rr["status"] == "RAN" and rr["pass_rate"] == 1.0 and rr["compare"] == R.REPRODUCED
    assert rep["claims"][0]["status"] == R.REPRODUCED
    assert "repro_wt_" not in git(repo, "worktree", "list")


def test_rerun_that_disagrees_with_the_record(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path, {"tests/test_bad.py": "def test_a():\n    assert False\n"})
    led, _, _ = adopted_ledger(repo, sha, metric=R.RERUN_METRIC, cand=0.995)
    rep = R.reproduce(repo, led.path, rerun=True, rerun_timeout=60)
    rr = rep["claims"][0]["adoptions"][0]["rerun"]
    assert rr["pass_rate"] == 0.0 and rr["compare"] == R.NOT_REPRODUCED and rep["claims"][0]["status"] == R.NOT_REPRODUCED
    assert "repro_wt_" not in git(repo, "worktree", "list")


def test_script_is_a_fresh_process_and_never_writes_the_ledger(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path, GOOD)
    led, _, _ = adopted_ledger(repo, sha)
    before = led.path.read_bytes()
    out = tmp_path / "out.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "reproduce.py"
    p = subprocess.run([sys.executable, str(script), "--repo", str(repo), "--ledger", str(led.path), "--out", str(out)],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["counts"]["REPRODUCED"] == 1
    assert led.path.read_bytes() == before and not led.path.with_suffix(".jsonl.lock").exists()
