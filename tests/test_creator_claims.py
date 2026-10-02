"""CR208: the claims register. Every major claim has evidence refs and a recomputation; a claim without recomputable evidence, or a
fabricated one, is UNSUPPORTED. Temporary repos and ledgers."""
from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest

from creator import claims as C
from creator import model as M
from creator.ledger import Ledger


@pytest.fixture(autouse=True)
def fast_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    prov = M.Provenance(engine_tree_hash="e", creator_tree_hash="c", git_commit="test", config_hash=None, seed=None,
                        timestamp="2026-10-02T00:00:00+00:00")
    monkeypatch.setattr("creator.ledger.current_provenance", lambda *a, **k: prov)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    (r / "a.py").write_text("X = 1\n", encoding="utf-8")
    git(r, "init", "-q")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "m")
    return r


def build(repo: Path, merge_commit: str | None = None) -> tuple[Path, str, str]:
    """A ledger with one IMPROVEMENT claim (measurements cite an evidence file) and one ADOPT decision citing `merge_commit`."""
    led = Ledger(repo / "state" / "ledger.jsonl", evidence_root=repo)
    o = led.append(M.Objective(created_by=M.Role.OWNER, statement="o", acceptance_criteria=("x",)))
    g = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(o,), kind=M.GapKind.TESTING, description="g", importance=0.5))
    ex = led.append(M.Experiment(created_by=M.Role.KERNEL, parents=(g,), hypothesis="h", design="d", metrics=("m",), seed=0,
                                 baseline_ref="b", candidate_ref="c"))
    (repo / "ev.json").write_text("[]", encoding="utf-8")
    ev = (M.EvidenceRef.of(repo / "ev.json", repo),)
    comp = M.ComputationRef(function="creator.evaluate:metrics_of", code_hash="ab" * 8, inputs_sha256="cd" * 8, output_sha256="ef" * 8)

    def ms(v: float, metric: str = "m", split: M.Split = M.Split.DEV) -> str:
        return led.append(M.Measurement(created_by=M.Role.VALIDATOR, parents=(ex,), metric=metric, value=v, stderr=0.01, n=50,
                                        population="p", conditions="c", higher_is_better=True, split=split, computation=comp,
                                        evidence=ev))
    base, cand = [ms(0.5), ms(0.51)], [ms(0.7), ms(0.72)]
    rb, rc, hb, hc = ms(0.9, "g"), ms(0.9, "g"), ms(0.4, "h", M.Split.HOLDOUT), ms(0.6, "h", M.Split.HOLDOUT)
    cid = led.append(M.ImprovementClaim(created_by=M.Role.VALIDATOR, parents=(ex,), subject_id=ex, baseline_ids=tuple(base),
                                        candidate_ids=tuple(cand), verdict=M.Verdict.IMPROVEMENT,
                                        computation=dataclasses.replace(comp, function=M.VERDICT_FUNCTION),
                                        regression_baseline_ids=(rb,), regression_candidate_ids=(rc,),
                                        holdout_baseline_id=hb, holdout_candidate_id=hc))
    sha = merge_commit or git(repo, "rev-parse", "HEAD")
    (repo / "m").mkdir(exist_ok=True)
    (repo / "m" / sha).write_text(sha + "\n", encoding="utf-8")
    did = led.append(M.Decision(created_by=M.Role.VALIDATOR, subject_id=ex, verdict=M.DecisionVerdict.ADOPT, reason="measured",
                                claim_id=cid, evidence=(M.EvidenceRef.of(repo / "m" / sha, repo, "merge_commit"),)))
    return led.path, cid, did


def by_id(reg: dict) -> dict:
    return {c["id"]: c for c in reg["claims"]}


def test_a_sound_claim_and_adoption_are_supported_with_evidence_and_recomputation(repo: Path) -> None:
    path, cid, did = build(repo)
    reg = C.build_register(repo, path)
    claims = by_id(reg)
    c, a = claims[f"claim:{cid}"], claims[f"adoption:{did}"]
    assert c["status"] == a["status"] == C.SUPPORTED, (c["why"], a["why"])
    assert c["recomputation"]["recomputed_verdict"] == "IMPROVEMENT" and any(e.get("path") == "ev.json" and e["ok"] for e in c["evidence"])
    assert a["recomputation"]["merge_commit_checked"] is True


def test_changed_evidence_makes_the_claim_unsupported(repo: Path) -> None:
    path, cid, _did = build(repo)
    (repo / "ev.json").write_text("[1]", encoding="utf-8")                      # the cited file was edited afterwards
    c = by_id(C.build_register(repo, path))[f"claim:{cid}"]
    assert c["status"] == C.UNSUPPORTED and any(not e.get("ok", True) for e in c["evidence"])


def test_an_adoption_citing_a_commit_that_is_not_in_git_is_unsupported(repo: Path) -> None:
    path, cid, did = build(repo, merge_commit="de" * 20)
    claims = by_id(C.build_register(repo, path))
    assert claims[f"adoption:{did}"]["status"] == C.UNSUPPORTED
    assert claims[f"claim:{cid}"]["status"] == C.UNSUPPORTED                     # reproduction ties a claim to its adoption's commit
    assert "does not exist in git" in claims[f"claim:{cid}"]["why"]


def test_a_fabricated_claim_is_flagged_and_an_honest_one_is_not(repo: Path) -> None:
    path, _cid, _did = build(repo)
    (repo / "numbers.json").write_text(json.dumps({"tests": {"passed": 120}}), encoding="utf-8")
    sha = C._sha(repo / "numbers.json")
    extra = [
        {"id": "honest", "statement": "120 tests passed", "evidence": [{"path": "numbers.json", "sha256": sha}],
         "recompute": {"kind": "json", "path": "numbers.json", "pointer": "tests/passed", "expected": 120}},
        {"id": "inflated", "statement": "150 tests passed", "evidence": [{"path": "numbers.json", "sha256": sha}],
         "recompute": {"kind": "json", "path": "numbers.json", "pointer": "tests/passed", "expected": 150}},
        {"id": "no_evidence", "statement": "the system is 10x faster", "evidence": [],
         "recompute": {"kind": "json", "path": "numbers.json", "pointer": "tests/passed", "expected": 120}},
        {"id": "no_recompute", "statement": "everything works", "evidence": [{"path": "numbers.json"}]},
        {"id": "ghost_file", "statement": "see report", "evidence": [{"path": "report_that_does_not_exist.json"}],
         "recompute": {"kind": "json", "path": "report_that_does_not_exist.json", "pointer": "x", "expected": 1}},
        {"id": "stale_hash", "statement": "120 tests passed", "evidence": [{"path": "numbers.json", "sha256": "0" * 64}],
         "recompute": {"kind": "json", "path": "numbers.json", "pointer": "tests/passed", "expected": 120}},
    ]
    reg = C.build_register(repo, path, extra)
    got = {k: v["status"] for k, v in by_id(reg).items() if k.startswith("extra:")}
    assert got == {"extra:honest": C.SUPPORTED, "extra:inflated": C.UNSUPPORTED, "extra:no_evidence": C.UNSUPPORTED,
                   "extra:no_recompute": C.UNSUPPORTED, "extra:ghost_file": C.UNSUPPORTED, "extra:stale_hash": C.UNSUPPORTED}
    assert {"extra:inflated", "extra:no_evidence"} <= set(reg["unsupported"])
    assert reg["counts"]["UNSUPPORTED"] == len(reg["unsupported"])


def test_headline_numbers_are_recomputed_from_other_sources(repo: Path) -> None:
    path, _cid, _did = build(repo)
    (repo / "creator").mkdir()
    (repo / "creator" / "capabilities.json").write_text(json.dumps({"capabilities": [{"id": "K01"}, {"id": "K02"}]}), encoding="utf-8")
    st = repo / "state" / "creator"
    st.mkdir(parents=True)
    attacks = [{"name": "a", "caught": True}, {"name": "b", "caught": True}, {"name": "c", "caught": False}]
    (st / "AUDIT.json").write_text(json.dumps({"audit": {"findings": [{"severity": "HIGH"}]}, "adversary": attacks}), encoding="utf-8")

    def status(adv_caught: int, ids=("K01", "K02")) -> None:
        (st / "STATUS.json").write_text(json.dumps({"capabilities": [{"id": i} for i in ids], "ledger_records": 3,
                                                    "adversary": {"attacks": 3, "caught": adv_caught},
                                                    "audit": {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 0, "LOW": 0}}), encoding="utf-8")
    status(2)
    ok = by_id(C.build_register(repo, path))
    assert {ok[k]["status"] for k in ("headline:capabilities", "headline:adversary", "headline:audit", "headline:ledger_records")} == {C.SUPPORTED}
    status(3, ids=("K01",))                                                    # 'the adversary caught everything', a capability dropped
    bad = by_id(C.build_register(repo, path))
    assert bad["headline:adversary"]["status"] == C.UNSUPPORTED and "AUDIT.json lists" in bad["headline:adversary"]["why"]
    assert bad["headline:capabilities"]["status"] == C.UNSUPPORTED and "missing ['K02']" in bad["headline:capabilities"]["why"]


def test_a_claim_of_more_ledger_records_than_the_chain_holds_is_unsupported(repo: Path) -> None:
    path, _cid, _did = build(repo)
    st = repo / "state" / "creator"
    st.mkdir(parents=True)
    (st / "STATUS.json").write_text(json.dumps({"capabilities": [], "ledger_records": 99999}), encoding="utf-8")
    assert by_id(C.build_register(repo, path))["headline:ledger_records"]["status"] == C.UNSUPPORTED


def test_the_script_writes_the_register(repo: Path, tmp_path: Path) -> None:
    path, _cid, _did = build(repo)
    import importlib.util
    spec = importlib.util.spec_from_file_location("claims_register", Path(C.__file__).resolve().parents[1] / "scripts" / "claims_register.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "CLAIMS.json"
    assert mod.main(["--repo", str(repo), "--ledger", str(path), "--out", str(out)]) == 0
    reg = json.loads(out.read_text(encoding="utf-8"))
    assert reg["counts"]["claims"] == len(reg["claims"]) >= 2 and all("evidence" in c and "recomputation" in c for c in reg["claims"])
    assert mod.main(["--repo", str(repo), "--ledger", str(path), "--out", str(out), "--strict"]) == 1     # headline files are absent here
