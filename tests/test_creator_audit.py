"""CR16: the auditor and the adversary (C77 secs 33-35, 47, 48, 50, 68)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creator import model as M
from creator.audit import checks as A
from creator.ledger import Ledger


@pytest.fixture()
def led(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "dev.jsonl", evidence_root=tmp_path)


def comp(fn: str = M.VERDICT_FUNCTION) -> M.ComputationRef:
    return M.ComputationRef(function=fn, code_hash="ab" * 8, inputs_sha256="cd" * 8, output_sha256="ef" * 8)


def test_every_attack_is_caught_for_the_right_reason() -> None:
    res = A.adversary()
    assert len(res) == len(A.ATTACKS) and all(a.caught for a in res), [a for a in res if not a.caught]


def test_an_attack_that_fails_for_another_reason_is_not_counted(tmp_path: Path) -> None:
    def broken(d: Path) -> str:
        raise KeyError("unrelated crash")
    [a] = A.adversary((("x", "rule", broken, r"TESTED needs"),))
    assert not a.caught and "another reason" in a.detail


def test_disabling_a_check_makes_its_attack_uncaught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation test: the adversary must notice when a defence is switched off."""
    monkeypatch.setattr(A, "check_test_weakening", lambda before, after: [])
    monkeypatch.setattr(A, "check_evidence_drift", lambda led, **_: [])
    res = {a.name: a.caught for a in A.adversary()}
    assert res["weaken_tests"] is False and res["evidence_swap"] is False and res["forge_tested"] is True


def test_clean_ledger_audits_clean_and_drift_is_found(led: Ledger, tmp_path: Path) -> None:
    (tmp_path / "log.txt").write_text("1 passed", encoding="utf-8")
    led.append(M.TestRun(created_by=M.Role.KERNEL, command="t", passed=1, failed=0, errors=0, skipped=0, duration_s=0.1,
                         evidence=(M.EvidenceRef.of(tmp_path / "log.txt", tmp_path),)))
    only = ("ledger_integrity", "evidence_drift", "fake_adoption", "claim_recompute")
    assert A.audit(led, only=only).clean
    (tmp_path / "log.txt").unlink()
    r = A.audit(led, only=only)
    assert r.count("CRITICAL") == 1 and r.findings[0].check == "evidence_drift" and "missing" in r.findings[0].detail


def _claim_world(led: Ledger) -> tuple[str, str]:
    o = led.append(M.Objective(created_by=M.Role.OWNER, statement="s", acceptance_criteria=("a",)))
    g = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(o,), kind=M.GapKind.TESTING, description="g", importance=0.5))
    ex = led.append(M.Experiment(created_by=M.Role.KERNEL, parents=(g,), hypothesis="h", design="d", metrics=("m",), seed=0,
                                 baseline_ref="b", candidate_ref="c"))
    root = led.evidence_root
    (root / "scores.json").write_text("[]", encoding="utf-8")
    ev = (M.EvidenceRef.of(root / "scores.json", root),)

    def ms(v: float, split: M.Split = M.Split.DEV, metric: str = "solve_rate") -> str:
        return led.append(M.Measurement(created_by=M.Role.VALIDATOR, parents=(ex,), metric=metric, value=v, stderr=0.01, n=50,
                                        population="p", conditions="c", higher_is_better=True, split=split,
                                        computation=comp("creator.evaluate:metrics_of"), evidence=ev))
    base, cand = [ms(0.5), ms(0.51)], [ms(0.7), ms(0.72)]
    reg = (ms(0.9, metric="no_regression"), ms(0.9, metric="no_regression"))
    hold = (ms(0.4, M.Split.HOLDOUT), ms(0.6, M.Split.HOLDOUT))
    cid = led.append(M.ImprovementClaim(created_by=M.Role.VALIDATOR, parents=(ex,), subject_id=ex, baseline_ids=tuple(base),
                                        candidate_ids=tuple(cand), verdict=M.Verdict.IMPROVEMENT, computation=comp(),
                                        regression_baseline_ids=(reg[0],), regression_candidate_ids=(reg[1],),
                                        holdout_baseline_id=hold[0], holdout_candidate_id=hold[1]))
    return ex, cid


def test_claims_are_recomputed_with_the_current_rule(led: Ledger, monkeypatch: pytest.MonkeyPatch) -> None:
    _claim_world(led)
    assert not A.check_claim_recompute(led)
    monkeypatch.setattr(M, "improvement_verdict", lambda *a, **k: (M.Verdict.NO_EFFECT, {}))
    [f] = A.check_claim_recompute(led)
    assert f.severity == "CRITICAL" and "current rule gives NO_EFFECT" in f.detail


def test_adoption_must_cite_a_real_merge(led: Ledger, tmp_path: Path) -> None:
    ex, cid = _claim_world(led)
    led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=ex, verdict=M.DecisionVerdict.ADOPT, reason="measured",
                          claim_id=cid))
    [f] = A.check_fake_adoption(led, repo=tmp_path)
    assert f.severity == "HIGH" and "merge commit" in f.detail
    (tmp_path / "merge").mkdir()
    (tmp_path / "merge" / "deadbeef00").write_text("x", encoding="utf-8")
    led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=ex, verdict=M.DecisionVerdict.ADOPT, reason="measured",
                          claim_id=cid, evidence=(M.EvidenceRef.of(tmp_path / "merge" / "deadbeef00", tmp_path, "merge_commit"),)))
    found = A.check_fake_adoption(led, repo=tmp_path)
    assert any(f.severity == "CRITICAL" and "not in git" in f.detail for f in found)


def test_test_shape_and_weakening() -> None:
    src = ("import pytest\n\ndef test_a():\n    assert 1 == 1\n    with pytest.raises(ValueError):\n        f()\n\n"
           "def helper():\n    pass\n")
    assert A.test_shape(src) == A.TestShape(1, 2, 0, 0)
    loose = "import pytest\n\n@pytest.mark.xfail\ndef test_a():\n    assert x == pytest.approx(1, rel=0.5)\n    assert True\n"
    assert A.test_shape(loose) == A.TestShape(1, 2, 1, 2)
    kinds = {f.detail.split(" ")[0] for f in A.check_test_weakening({"tests/test_a.py": src}, {"tests/test_a.py": loose})}
    assert "skip/xfail" in kinds and "looser" in kinds
    assert A.check_test_weakening({"tests/test_a.py": src}, {})[0].detail == "test file deleted"
    assert not A.check_test_weakening({"tests/test_a.py": src}, {"tests/test_a.py": src + "\ndef test_b():\n    assert f(2) == 4\n"})
    assert not A.check_test_weakening({"app/core.py": "x = 1"}, {})                # not a test file


def test_answer_literals_are_distinctive_only() -> None:
    assert A._distinctive(299993) and A._distinctive("already-sluggy") and A._distinctive("2023-07-04")
    for common in (2000, 10000, "no:cacheprovider", "[a-z]+", "pytest", "app.core", "--rootdir", True, "hi"):
        assert not A._distinctive(common), common


def test_the_real_creator_has_no_hardcoded_answers_and_the_suite_is_sealed() -> None:
    assert A.check_hardcoded_answers() == []
    assert A.check_sealed_suite() == []


def test_budget_anomalies(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"calls": [{"run": "r1", "usd": 0.0, "outcome": "OK"}, {"run": "r2", "usd": 0.1, "outcome": "OK"},
                                       {"run": "r3", "usd": 0.0, "outcome": "REFUSED"}]}), encoding="utf-8")
    [f] = A.check_budget_anomalies(budget_file=p)
    assert f.subject == "r1"


def test_a_crashing_check_is_reported_not_skipped(led: Ledger, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_: object) -> list:
        raise RuntimeError("check broke")
    monkeypatch.setitem(A.CHECKS, "ledger_integrity", boom)
    r = A.audit(led, only=("ledger_integrity",))
    assert not r.clean and "check broke" in r.errors["ledger_integrity"]


def test_the_sealed_suite_checked_is_the_audited_repositorys(tmp_path: Path) -> None:
    """Regression (1 Oct): the check always read the global suite, so kernel tests in a sealed-keys-hidden sandbox went red."""
    assert A.check_sealed_suite(repo=tmp_path) == []                                 # no devbench at all: nothing to check
    (tmp_path / "creator" / "devbench" / "tasks").mkdir(parents=True)
    [f] = A.check_sealed_suite(repo=tmp_path)                                         # half a devbench: caught
    assert f.severity == "CRITICAL" and f.subject == "MANIFEST"
    assert A.check_sealed_suite() == []                                               # the real repository stays clean


def test_learning_stores_must_not_contain_the_holdout(tmp_path: Path) -> None:
    """CR201: experience that mentions a holdout task or a sealed answer literal is memorization, not learning."""
    import shutil
    repo = tmp_path / "r"
    shutil.copytree(A.REPO_ROOT / "creator" / "devbench" / "sealed", repo / "creator" / "devbench" / "sealed")
    st = repo / "state" / "creator"
    st.mkdir(parents=True)
    (st / "dev_memory.jsonl").write_text('{"kind": "repair", "subject": "D01 bugfix", "detail": "fixed"}\n', encoding="utf-8")
    assert A.check_memorization(repo=repo) == []                                     # dev tasks may be learned from
    from creator import devbench as D
    holdout = D.load_manifest(repo / "creator" / "devbench" / "sealed" / "MANIFEST.json")["splits"]["holdout"][0]
    (st / "generator_memory_dev.jsonl").write_text('{"objective": "slugify", "files": {"a.py": "hello-world 299993"}}\n',
                                                   encoding="utf-8")
    assert A.check_memorization(repo=repo) == []                                     # dev answers are fair to learn from
    (st / "generator_memory_dev.jsonl").unlink()
    (st / "generator_memory_x.jsonl").write_text(f'{{"objective": "solve {holdout}", "files": {{"a.py": "x"}}}}\n',
                                                 encoding="utf-8")
    found = A.check_memorization(repo=repo)
    assert {f.subject for f in found} == {"generator_memory_x.jsonl"} and all(f.severity == "CRITICAL" for f in found)
    assert any("holdout" in f.detail for f in found)


def test_final_campaign_defences_are_load_bearing(monkeypatch: pytest.MonkeyPatch) -> None:
    """CR20: each attack the campaign found UNCAUGHT must become uncaught again when its defence is switched off, while the
    attacks that were already refused by the ledger/kernel stay caught."""
    monkeypatch.setattr(A, "check_test_weakening", lambda before, after: [])
    monkeypatch.setattr(A, "check_strategy_attribution", lambda led, **_: [])
    monkeypatch.setattr(A, "SEALED_PINS", frozenset({"x"}))
    monkeypatch.setattr(A, "check_sealed_suite", lambda **_: [])
    mine = ("conftest_skip", "or_true_assert", "forged_self_share", "launder_claude_share", "reseal_devbench", "delete_test_file",
            "steal_live_lock", "mixed_populations")                     # the rest are exercised by the full run above (speed)
    res = {a.name: a.caught for a in A.adversary(tuple(x for x in A.ATTACKS if x[0] in mine))}
    for n in mine[:6]:
        assert res[n] is False, n
    assert res["steal_live_lock"] is True and res["mixed_populations"] is True


def test_harness_files_cannot_silence_tests() -> None:
    hook = "import pytest\n\ndef pytest_collection_modifyitems(items):\n    pass\n"
    [f] = A.check_test_weakening({}, {"tests/conftest.py": hook})
    assert f.severity == "CRITICAL" and "harness" in f.detail
    assert not A.check_test_weakening({"conftest.py": hook}, {"conftest.py": hook})       # an existing hook is not new
    assert not A.check_test_weakening({}, {"conftest.py": "import pytest\n\n@pytest.fixture\ndef x():\n    return 1\n"})
    assert A.check_test_weakening({}, {"pytest.ini": "[pytest]\naddopts = --deselect tests/test_a.py\n"})
    from creator import kernel as K
    assert "conftest.py" in A.HARNESS_FILES and K.AUD.HARNESS_FILES is A.HARNESS_FILES


def test_attribution_check(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    ok = A._strategy_world(tmp_path / "a", [("claude-session", True), ("self-search", False), ("self-forged", False)])
    assert A.check_strategy_attribution(ok) == []                                           # failures are never credited
    (tmp_path / "b").mkdir()
    assert A.check_strategy_attribution(A._strategy_world(tmp_path / "b", [("self-search", True)])) == []
    (tmp_path / "c").mkdir()
    [f] = A.check_strategy_attribution(A._strategy_world(tmp_path / "c", [("self-search", True), ("claude-session", True)]))
    assert "several authors" in f.detail


def test_vacuous_or_assert_is_loose() -> None:
    assert A.test_shape("def test_a():\n    assert f() == 1 or True\n").approx_loose == 1
    assert A.test_shape("def test_a():\n    assert f() == 1 or g()\n").approx_loose == 0


def test_the_pinned_manifest_digest_matches_the_sealed_suite() -> None:
    from creator import devbench as D
    assert D.load_manifest()["digest"] in A.SEALED_PINS


def test_every_real_own_worker_is_a_registered_self_worker() -> None:
    """Regression (2 Oct): the registry missed the real workers, so a genuine self-built adoption would have gone red."""
    from creator import selfworkers as SW
    from creator import testgen as TG
    assert {SW.RuleWorker.name, SW.SearchWorker.name, TG.TestGenWorker.name} <= set(A.SELF_WORKERS)


def test_superseded_evidence_preserved_in_git_history_is_not_drift_but_uncommitted_change_is(tmp_path: Path) -> None:
    """2 Oct: validation round 4 rewrote VALIDATION_REPORT.json and 10 earlier TestRuns citing the old bytes went CRITICAL although
    git held those exact bytes. Superseded-but-committed evidence is verifiable; bytes that were never committed are not."""
    import subprocess

    def g(*a: str) -> None:
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path, check=True, capture_output=True)
    g("init", "-q")
    g("config", "core.autocrlf", "false")                                  # as in the real repo: blobs are the disk bytes
    led = Ledger(tmp_path / "dev.jsonl", evidence_root=tmp_path)
    rep = tmp_path / "report.json"
    rep.write_bytes(b'{"round": 1}\n')
    g("add", "report.json")
    g("commit", "-q", "-m", "round 1")
    led.append(M.TestRun(created_by=M.Role.VALIDATOR, command="t", passed=1, failed=0, errors=0, skipped=0, duration_s=0.1,
                         evidence=(M.EvidenceRef.of(rep, tmp_path),)))
    rep.write_bytes(b'{"round": 2}\n')                   # superseded in place, old bytes committed
    g("commit", "-q", "-am", "round 2")
    assert A.check_evidence_drift(led) == []
    rep.write_bytes(b'{"round": 3, "never": "committed"}\n')
    led.append(M.TestRun(created_by=M.Role.VALIDATOR, command="t", passed=1, failed=0, errors=0, skipped=0, duration_s=0.1,
                         evidence=(M.EvidenceRef.of(rep, tmp_path),)))
    rep.write_bytes(b'{"round": 4}\n')                   # round-3 bytes exist nowhere now
    found = A.check_evidence_drift(led)
    assert len(found) == 1 and found[0].severity == "CRITICAL" and "changed" in found[0].detail
