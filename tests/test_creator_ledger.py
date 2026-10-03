"""CR01: the Creator's development ledger (C77 secs 7, 30, 35, 59-61). Every refusal is tested on the case it exists for."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creator import ledger as L
from creator import model as M

K = M.Role.KERNEL
REAL_PROVENANCE = True        # tests/conftest.py: this file covers current_provenance itself - never the cached one


def prov(seed: int = 0) -> M.Provenance:
    return M.Provenance(engine_tree_hash="e" * 16, creator_tree_hash="c" * 16, git_commit="abc", config_hash=None, seed=seed,
                        timestamp="2026-09-30T00:00:00+00:00")


@pytest.fixture()
def led(tmp_path: Path) -> L.Ledger:
    return L.Ledger(tmp_path / "dev.jsonl", evidence_root=tmp_path)


def objective(led: L.Ledger) -> str:
    return led.append(M.Objective(created_by=M.Role.OWNER, statement="develop yourself", acceptance_criteria=("a measured gain",)),
                      provenance=prov())


def requirement(led: L.Ledger, parent: str, key: str = "R1") -> str:
    return led.append(M.Requirement(created_by=K, parents=(parent,), key=key, description="d", priority=M.Priority.HIGH,
                                    acceptance_test="t", measurement_method="m", failure_condition="f", validation_method="v",
                                    evidence_location="e"), provenance=prov())


def evidence(tmp_path: Path, name: str = "out.txt", text: str = "1 passed") -> M.EvidenceRef:
    (tmp_path / name).write_text(text, encoding="utf-8")
    return M.EvidenceRef.of(tmp_path / name, tmp_path, kind="log")


def comp(fn: str = "creator.model:improvement_verdict") -> M.ComputationRef:
    return M.ComputationRef(function=fn, code_hash="ab" * 8, inputs_sha256="cd" * 8, output_sha256="ef" * 8)


def measurement(led: L.Ledger, parent: str, value: float, split: M.Split = M.Split.DEV, metric: str = "solve_rate",
                stderr: float = 0.01) -> str:
    return led.append(M.Measurement(created_by=K, parents=(parent,), metric=metric, value=value, stderr=stderr, n=50,
                                    population="devbench", conditions="same", higher_is_better=True, split=split,
                                    computation=comp("creator.evaluate:measure")), provenance=prov())


# ------------------------------------------------------------------------------------------------ append / reload / chain

def test_empty_ledger(led: L.Ledger) -> None:
    assert led.verify() == 0 and led.view.head == L.GENESIS and led.snapshot()["records"] == 0


def test_append_reload_and_provenance(tmp_path: Path, led: L.Ledger) -> None:
    o = objective(led)
    r = requirement(led, o)
    again = L.Ledger(tmp_path / "dev.jsonl", evidence_root=tmp_path)
    assert again.verify() == 2 and again.view.head == led.view.head
    assert again.view.status[o] is M.Status.NOT_STARTED and again.get(r).parents == (o,)
    for e in again.view.entries:
        assert e.provenance.engine_tree_hash and e.provenance.creator_tree_hash and e.provenance.timestamp
    assert again.lineage(r) == [o] and again.descendants(o) == [r]


def test_computed_provenance_when_none_given(led: L.Ledger) -> None:
    rid = led.append(M.Objective(created_by=M.Role.OWNER, statement="s", acceptance_criteria=("a",)), seed=7, config={"k": 1})
    p = led.view.by_id[rid].provenance
    assert p.seed == 7 and p.config_hash and len(p.creator_tree_hash) == 16 and p.git_commit


@pytest.mark.parametrize("attack", ["edit", "delete", "reorder", "insert", "truncate_middle"])
def test_tampering_is_detected(tmp_path: Path, led: L.Ledger, attack: str) -> None:
    o = objective(led)
    requirement(led, o, "R1")
    requirement(led, o, "R2")
    p = tmp_path / "dev.jsonl"
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
    else:
        lines = lines[:1] + lines[2:]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(L.LedgerError):
        L.Ledger(p, evidence_root=tmp_path)


def test_rehashed_edit_is_still_caught_by_the_chain(tmp_path: Path, led: L.Ledger) -> None:
    """An attacker who recomputes the edited line's own hash still breaks the next line's prev link."""
    o = objective(led)
    requirement(led, o, "R1")
    p = tmp_path / "dev.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    env = json.loads(lines[0])
    env.pop("hash")
    env["data"]["statement"] = "a different objective"
    env["hash"] = L._line_hash(env)
    lines[0] = L.canonical(env)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(L.LedgerError):
        L.Ledger(p, evidence_root=tmp_path)


def test_torn_final_line_is_refused_then_repaired_and_kept_as_evidence(tmp_path: Path, led: L.Ledger) -> None:
    objective(led)
    p = tmp_path / "dev.jsonl"
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 1, "id": "OBJ-')                       # an append interrupted mid-line
    with pytest.raises(L.LedgerError, match="torn"):
        L.Ledger(p, evidence_root=tmp_path)
    fixed = L.Ledger(p, evidence_root=tmp_path, repair_torn=True)
    assert fixed.verify() == 1 and (tmp_path / "dev.jsonl.torn").read_text(encoding="utf-8").count("OBJ-") == 1


def test_two_writers_never_fork_the_chain(tmp_path: Path) -> None:
    a = L.Ledger(tmp_path / "dev.jsonl", evidence_root=tmp_path)
    b = L.Ledger(tmp_path / "dev.jsonl", evidence_root=tmp_path)
    o = objective(a)
    requirement(b, o, "R1")                                      # b catches up under the lock before validating
    objective(a)
    assert L.Ledger(tmp_path / "dev.jsonl", evidence_root=tmp_path).verify() == 3


# ------------------------------------------------------------------------------------------------ refusals

def test_missing_parent_and_wrong_parent_type_are_refused(led: L.Ledger) -> None:
    with pytest.raises(L.LedgerError, match="does not exist"):
        requirement(led, "OBJ-" + "0" * 20)
    o = objective(led)
    m = measurement(led, o, 0.5)
    with pytest.raises(L.LedgerError, match="parent of type"):
        requirement(led, m)


def test_unique_keys(led: L.Ledger) -> None:
    o = objective(led)
    requirement(led, o, "R1")
    with pytest.raises(L.LedgerError, match="already exists"):
        requirement(led, o, "R1")


def test_status_machine_from_the_actual_state(tmp_path: Path, led: L.Ledger) -> None:
    o = objective(led)
    r = requirement(led, o)
    with pytest.raises(L.LedgerError, match="illegal transition NOT_STARTED -> VALIDATED"):
        led.transition(r, M.Status.VALIDATED, "skip", M.Role.AUDITOR)
    led.transition(r, M.Status.IN_PROGRESS, "start", K)
    led.transition(r, M.Status.IMPLEMENTED, "code written", K)
    with pytest.raises(L.LedgerError, match="TestRun"):
        led.transition(r, M.Status.TESTED, "trust me", K)               # code exists is not tested
    forged = M.Transition(created_by=K, subject_id=r, from_state=M.Status.NOT_STARTED, to_state=M.Status.IN_PROGRESS, reason="x")
    with pytest.raises(L.LedgerError, match="stale or forged"):
        led.append(forged, provenance=prov())
    assert led.view.status[r] is M.Status.IMPLEMENTED and len(led.view.status_history[r]) == 2


def test_tested_needs_a_passing_test_run(tmp_path: Path, led: L.Ledger) -> None:
    o = objective(led)
    r = requirement(led, o)
    led.transition(r, M.Status.IN_PROGRESS, "start", K)
    led.transition(r, M.Status.IMPLEMENTED, "code", K)
    bad = led.append(M.TestRun(created_by=M.Role.TESTER, parents=(r,), command="pytest", passed=3, failed=1, errors=0, skipped=0,
                               duration_s=1.0, evidence=(evidence(tmp_path, "bad.txt", "3 passed 1 failed"),)), provenance=prov())
    with pytest.raises(L.LedgerError, match="not PASS"):
        led.transition(r, M.Status.TESTED, "tests", K, justification_ids=[bad])
    good = led.append(M.TestRun(created_by=M.Role.TESTER, parents=(r,), command="pytest", passed=4, failed=0, errors=0, skipped=0,
                                duration_s=1.0, evidence=(evidence(tmp_path),)), provenance=prov())
    led.transition(r, M.Status.TESTED, "tests pass", K, justification_ids=[good])
    assert led.view.status[r] is M.Status.TESTED


def test_validated_needs_an_independent_role_and_evidence(tmp_path: Path, led: L.Ledger) -> None:
    o = objective(led)
    r = requirement(led, o)
    tr = None
    for to in (M.Status.IN_PROGRESS, M.Status.IMPLEMENTED):
        led.transition(r, to, "step", K)
    tr = led.append(M.TestRun(created_by=M.Role.TESTER, parents=(r,), command="pytest", passed=4, failed=0, errors=0, skipped=0,
                              duration_s=1.0, evidence=(evidence(tmp_path),)), provenance=prov())
    led.transition(r, M.Status.TESTED, "tests", K, justification_ids=[tr])
    led.transition(r, M.Status.INTENDED_BEHAVIOR_VERIFIED, "behaviour", K, justification_ids=[tr])
    with pytest.raises(M.ModelError, match="may not validate"):
        led.transition(r, M.Status.VALIDATED, "self-approved", K, justification_ids=[tr], evidence=[evidence(tmp_path)])
    with pytest.raises(L.LedgerError, match="evidence on the transition"):
        led.transition(r, M.Status.VALIDATED, "no evidence", M.Role.AUDITOR, justification_ids=[tr])
    led.transition(r, M.Status.VALIDATED, "audited", M.Role.AUDITOR, justification_ids=[tr], evidence=[evidence(tmp_path)])
    assert led.view.status[r] is M.Status.VALIDATED


def test_evidence_must_hash_to_what_is_cited(tmp_path: Path, led: L.Ledger) -> None:
    o = objective(led)
    ev = evidence(tmp_path, "log.txt", "original")
    (tmp_path / "log.txt").write_text("changed after citing", encoding="utf-8")
    with pytest.raises(L.LedgerError, match="evidence changed"):
        led.append(M.TestRun(created_by=M.Role.TESTER, parents=(o,), command="pytest", passed=1, failed=0, errors=0, skipped=0,
                             duration_s=0.1, evidence=(ev,)), provenance=prov())


# ------------------------------------------------------------------------------------------------ verdicts are computed

def _claim_world(led: L.Ledger) -> tuple[str, str, list[str], list[str], tuple[str, str], tuple[str, str]]:
    o = objective(led)
    gap = led.append(M.Gap(created_by=K, parents=(o,), kind=list(M.GapKind)[0], description="d", importance=0.5), provenance=prov())
    ex = led.append(M.Experiment(created_by=K, parents=(gap,), hypothesis="h", design="d", metrics=("solve_rate",), seed=1,
                                 baseline_ref="base", candidate_ref="cand"), provenance=prov())
    base = [measurement(led, ex, 0.50), measurement(led, ex, 0.51)]
    cand = [measurement(led, ex, 0.70), measurement(led, ex, 0.72)]
    reg = (measurement(led, ex, 0.9, metric="regression_pass"), measurement(led, ex, 0.9, metric="regression_pass"))
    hold = (measurement(led, ex, 0.40, M.Split.HOLDOUT), measurement(led, ex, 0.60, M.Split.HOLDOUT))
    return o, ex, base, cand, reg, hold


def _claim(cp: str, base: list[str], cand: list[str], reg: tuple[str, str], hold: tuple[str, str], verdict: M.Verdict) -> M.ImprovementClaim:
    return M.ImprovementClaim(created_by=M.Role.VALIDATOR, parents=(cp,), subject_id=cp, baseline_ids=tuple(base),
                              candidate_ids=tuple(cand), verdict=verdict, computation=comp(),
                              regression_baseline_ids=(reg[0],), regression_candidate_ids=(reg[1],),
                              holdout_baseline_id=hold[0], holdout_candidate_id=hold[1])


def test_improvement_claim_verdict_is_recomputed(led: L.Ledger) -> None:
    o, ex, base, cand, reg, hold = _claim_world(led)
    with pytest.raises(L.LedgerError, match="rule computes"):
        led.append(_claim(ex, base, cand, reg, hold, M.Verdict.NO_EFFECT), provenance=prov())
    cid = led.append(_claim(ex, base, cand, reg, hold, M.Verdict.IMPROVEMENT), provenance=prov())
    gap2 = led.append(M.Gap(created_by=K, parents=(o,), kind=list(M.GapKind)[0], description="e", importance=0.1), provenance=prov())
    other = led.append(M.Experiment(created_by=K, parents=(gap2,), hypothesis="h2", design="d", metrics=("m",), seed=2,
                                    baseline_ref="b", candidate_ref="c"), provenance=prov())
    with pytest.raises(L.LedgerError, match="different subject"):
        led.append(M.Decision(created_by=K, subject_id=other, verdict=M.DecisionVerdict.ADOPT, reason="r", claim_id=cid),
                   provenance=prov())
    led.append(M.Decision(created_by=K, subject_id=ex, verdict=M.DecisionVerdict.ADOPT, reason="measured", claim_id=cid),
               provenance=prov())


def test_adopt_without_an_improvement_is_refused(led: L.Ledger) -> None:
    _, cp, base, cand, reg, hold = _claim_world(led)
    flat = [measurement(led, cp, 0.50), measurement(led, cp, 0.505)]
    v, _ = M.improvement_verdict([led.get(i) for i in base], [led.get(i) for i in flat],         # type: ignore[misc]
                                 [(led.get(reg[0]), led.get(reg[1]))], (led.get(hold[0]), led.get(hold[1])))  # type: ignore[arg-type]
    cid = led.append(_claim(cp, base, flat, reg, hold, v), provenance=prov())
    assert v is not M.Verdict.IMPROVEMENT
    with pytest.raises(L.LedgerError, match="not IMPROVEMENT"):
        led.append(M.Decision(created_by=K, subject_id=cp, verdict=M.DecisionVerdict.ADOPT, reason="looks better", claim_id=cid),
                   provenance=prov())


# ------------------------------------------------------------------------------------------------ checkpoints and views

def test_checkpoint_pins_the_head_and_the_folded_state(led: L.Ledger) -> None:
    o = objective(led)
    led.transition(o, M.Status.IN_PROGRESS, "go", K)
    cid = led.checkpoint("after start")
    ck = led.get(cid)
    assert isinstance(ck, M.CheckpointMarker) and ck.chain_records == 2
    stale = M.CheckpointMarker(created_by=K, label="stale", chain_head=ck.chain_head, chain_records=2, view_digest=ck.view_digest)
    with pytest.raises(L.LedgerError, match="chain_head"):
        led.append(stale, provenance=prov())


def test_snapshot_and_queries(led: L.Ledger) -> None:
    o = objective(led)
    r1, r2 = requirement(led, o, "R1"), requirement(led, o, "R2")
    led.transition(r1, M.Status.IN_PROGRESS, "go", K)
    snap = led.snapshot()
    assert snap["by_type"] == {"Objective": 1, "Requirement": 2, "Transition": 1}
    assert snap["by_status"]["IN_PROGRESS"] == 1 and set(snap["open"]) == {o, r1, r2}
    assert [e.id for e in led.with_status(M.Status.IN_PROGRESS)] == [r1] and {e.id for e in led.about(o)} == {r1, r2}


# ------------------------------------------------------------------------------------------------ provenance speed-ups: same values, edits still seen

def _ref_tree_hash(root: Path, pattern: str = "*.py") -> str:
    """The original, uncached tree_hash, kept here as the reference the cached one must equal byte for byte."""
    import hashlib
    h = hashlib.sha256()
    for p in sorted(root.rglob(pattern)):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def test_tree_hash_cache_equals_uncached_on_the_real_tree() -> None:
    assert L.tree_hash(L.CREATOR_DIR) == _ref_tree_hash(L.CREATOR_DIR)
    assert L.tree_hash(L.CREATOR_DIR) == _ref_tree_hash(L.CREATOR_DIR)                 # a cache hit gives the same value


def test_tree_hash_cache_sees_edit_add_remove_rename_and_crlf(tmp_path: Path) -> None:
    (tmp_path / "Sub").mkdir()
    (tmp_path / "__pycache__").mkdir()
    a, b = tmp_path / "a.py", tmp_path / "Sub" / "B.py"
    a.write_bytes(b"x = 1\r\n")
    b.write_bytes(b"y = 2\n")
    (tmp_path / "__pycache__" / "c.py").write_bytes(b"ignored")
    seen = [L.tree_hash(tmp_path)]
    assert seen[0] == _ref_tree_hash(tmp_path)
    assert L.tree_hash(tmp_path) == seen[0]
    a.write_bytes(b"x = 1\n")                                                           # CRLF folded: same hash
    assert L.tree_hash(tmp_path) == seen[0]
    a.write_bytes(b"x = 9\n")                                                           # same size, new content
    seen.append(L.tree_hash(tmp_path))
    b.write_bytes(b"y = 22\n")                                                          # new size
    seen.append(L.tree_hash(tmp_path))
    (tmp_path / "z.py").write_bytes(b"")                                                # added
    seen.append(L.tree_hash(tmp_path))
    (tmp_path / "z.py").rename(tmp_path / "Z2.py")                                      # renamed
    seen.append(L.tree_hash(tmp_path))
    (tmp_path / "Z2.py").unlink()                                                       # removed
    seen.append(L.tree_hash(tmp_path))
    assert seen[1] != seen[0] and seen[2] != seen[1] and seen[3] != seen[2] and seen[4] != seen[3] and seen[5] != seen[4]
    assert L.tree_hash(tmp_path) == _ref_tree_hash(tmp_path)


def _git(repo: Path, *args: str) -> None:
    import subprocess
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True, capture_output=True)


def test_git_commit_cache_equals_uncached_and_sees_edit_and_new_commit(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    (repo / "creator").mkdir(parents=True)
    (repo / "engine").mkdir()
    (repo / "creator" / "m.py").write_text("a = 1\n")
    (repo / "engine" / "e.py").write_text("b = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "one")
    c1 = L._git_commit(repo)
    assert c1 == L._git_commit_uncached(repo) and not c1.endswith("+dirty")
    assert L._git_commit(repo) == c1
    (repo / "creator" / "m.py").write_text("a = 2\n")                                   # edit -> dirty, at once
    c2 = L._git_commit(repo)
    assert c2 == c1 + "+dirty" == L._git_commit_uncached(repo)
    (repo / "engine" / "new.txt").write_text("untracked")                               # untracked non-py file still dirty
    assert L._git_commit(repo) == c2
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "two")                                             # new commit -> new HEAD, clean
    c3 = L._git_commit(repo)
    assert c3 != c1 and not c3.endswith("+dirty") and c3 == L._git_commit_uncached(repo)
    (repo / "creator" / "m.py").write_text("a = 1\n")
    assert L._git_commit(repo) == c3 + "+dirty"
    _git(repo, "checkout", "-q", "--", "creator/m.py")                                  # reverted -> clean again
    assert L._git_commit(repo) == c3
    _git(repo, "commit", "-q", "--allow-empty", "-m", "three")                          # empty commit: only the ref moves
    assert L._git_commit(repo) not in (c1, c3) and L._git_commit(repo) == L._git_commit_uncached(repo)


def test_current_provenance_equals_uncached_composition() -> None:
    p = L.current_provenance(seed=3, config={"k": 1})
    assert p.creator_tree_hash == _ref_tree_hash(L.CREATOR_DIR)
    assert p.git_commit == L._git_commit_uncached(L.REPO_ROOT)


def test_tree_hash_sees_a_same_size_rewrite_that_keeps_its_timestamp(tmp_path: Path) -> None:
    """Racily clean (1 Oct): a rewrite inside one timestamp tick keeps (mtime_ns, size); the cache must not trust such a key."""
    import os
    f = tmp_path / "a.py"
    f.write_bytes(b"x = 1\n")
    st = f.stat()
    first = L.tree_hash(tmp_path)
    f.write_bytes(b"x = 9\n")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))                                  # same size, same mtime: stat sees nothing
    assert L.tree_hash(tmp_path) != first and L.tree_hash(tmp_path) == _ref_tree_hash(tmp_path)


def test_git_state_is_not_cached_while_files_are_racy(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "creator").mkdir()
    (tmp_path / "creator" / "m.py").write_text("x = 1\n", encoding="utf-8")
    assert L._git_state_key(tmp_path) is None                                        # just written: racy, so git is asked
