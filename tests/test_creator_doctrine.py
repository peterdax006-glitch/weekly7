"""creator/doctrine.py: per-rule verdicts for the process/doctrine sections of CREATOR_MASTER_PROMPT.md.

Every computed check is exercised on a violating tree AND a compliant tree (a check that cannot fail proves nothing), the rule
table is held to its own contract, and the roll-up never reaches VALIDATED."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creator import doctrine as D


def _cycle(root: Path, name: str, outcome: str = "ADOPTED", files: tuple[str, ...] = ("creator/a.py",), lines: int = 5,
           verdict: str = "IMPROVEMENT", lo: float = 1.0, requirement: str = "K07.exists") -> None:
    d = root / "state" / "creator" / "cycles" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "cycle.json").write_text(json.dumps({"package": name, "outcome": outcome, "requirement": requirement, "verdict": verdict,
                                              "details": {"detail": {"lo": lo, "min_effect": 0.0}}}), encoding="utf-8")
    (d / "evaluation.json").write_text(json.dumps({"changed": {f: "M" for f in files}}), encoding="utf-8")
    body = "".join(f"+line {i}\n" for i in range(lines))
    (d / "diff.patch").write_text("--- a/x\n+++ b/x\n" + body, encoding="utf-8")


def _status(root: Path, states: dict[str, str], share: float | None = None) -> None:
    out = root / "state" / "creator"
    out.mkdir(parents=True, exist_ok=True)
    (out / "STATUS.json").write_text(json.dumps({"capabilities": [{"id": k, "state": v} for k, v in states.items()],
                                                 "curriculum": {"teacher_share": share, "adopted_by": {}}}), encoding="utf-8")


def _junit(root: Path, test_file: str, cases: dict[str, str], bind: bool = True) -> None:
    d = root / D.JUNIT
    d.mkdir(parents=True, exist_ok=True)
    xml = "".join(f'<testcase name="{n}">' + ("<failure/>" if r == "fail" else "<skipped/>" if r == "skip" else "") + "</testcase>"
                  for n, r in cases.items())
    (d / (test_file.replace("/", "__") + ".xml")).write_text(f"<testsuites><testsuite>{xml}</testsuite></testsuites>", encoding="utf-8")
    if bind:                                                  # what collect_test_evidence records when it runs the file
        from creator import selfmodel as SM
        from creator import testrun as TR
        store = root / D.EVIDENCE_STORE
        known = json.loads(store.read_text(encoding="utf-8")) if store.is_file() else {}
        known[test_file] = {"test_file": test_file, "outcome": "PASS", "where": "",
                            "source_digest": SM.reach_digest(TR.ImportGraph.build(root), test_file, {})}
        store.write_text(json.dumps(known), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------- the checks
def test_status_states() -> None:
    assert D.check_status_states(D.ROOT)[0] == D.PASS


def test_execution_order() -> None:
    st, ev = D.check_execution_order(D.ROOT)
    assert st == D.PASS and "INTENDED_BEHAVIOR_VERIFIED" in ev


def test_limitation_recordable() -> None:
    assert D.check_limitation_recordable(D.ROOT)[0] == D.PASS


def test_package_fields() -> None:
    assert D.check_package_fields(D.ROOT)[0] == D.PASS


def test_package_size(tmp_path: Path) -> None:
    assert D.check_package_size(tmp_path)[0] == D.PENDING
    _cycle(tmp_path, "CP1", lines=10)
    assert D.check_package_size(tmp_path)[0] == D.PASS
    _cycle(tmp_path, "CP2", lines=D.MAX_PACKAGE_DIFF_LINES + 1)
    st, ev = D.check_package_size(tmp_path)
    assert st == D.FAIL and "CP2" in ev
    _cycle(tmp_path, "CP2", files=tuple(f"creator/f{i}.py" for i in range(D.MAX_PACKAGE_FILES + 1)))
    assert D.check_package_size(tmp_path)[0] == D.FAIL


def test_self_target(tmp_path: Path) -> None:
    assert D.check_self_target(tmp_path)[0] == D.PENDING
    _cycle(tmp_path, "CP1", files=("creator/a.py", "tests/test_a.py"))
    assert D.check_self_target(tmp_path)[0] == D.PASS
    _cycle(tmp_path, "CP2", files=("engine/learning/x.py",))
    st, ev = D.check_self_target(tmp_path)
    assert st == D.FAIL and "engine/learning/x.py" in ev


def test_work_traces_to_requirements(tmp_path: Path) -> None:
    _cycle(tmp_path, "CP1", requirement="K07.exists")
    _cycle(tmp_path, "CP2", requirement="EFF.size")
    assert D.check_work_traces_to_requirements(tmp_path)[0] == D.PASS
    _cycle(tmp_path, "CP3", requirement="K07.make_it_pretty")
    assert D.check_work_traces_to_requirements(tmp_path)[0] == D.FAIL
    (tmp_path / "state" / "creator" / "cycles" / "CP3" / "cycle.json").write_text(json.dumps({"package": "CP3", "requirement": "weekly7.finish"}))
    assert D.check_work_traces_to_requirements(tmp_path)[0] == D.FAIL


def test_complexity_vs_capability(tmp_path: Path) -> None:
    assert D.check_complexity_vs_capability(tmp_path)[0] == D.PENDING
    _cycle(tmp_path, "CP1", lines=300, lo=1.0)
    st, ev = D.check_complexity_vs_capability(tmp_path)
    assert st == D.PASS and "+300" in ev
    _cycle(tmp_path, "CP2", lines=300, lo=0.0)                           # big, but no measured gain
    assert D.check_complexity_vs_capability(tmp_path)[0] == D.FAIL
    _cycle(tmp_path, "CP2", verdict="INSUFFICIENT_EVIDENCE", lo=2.0)
    assert D.check_complexity_vs_capability(tmp_path)[0] == D.FAIL
    _cycle(tmp_path, "CP2", outcome="REJECTED", lines=900, lo=-3.0)       # rejected cycles do not count as adopted
    assert D.check_complexity_vs_capability(tmp_path)[0] == D.PASS


def test_depth_is_behaviour(tmp_path: Path) -> None:
    assert D.check_depth_is_behaviour(D.ROOT)[0] == D.PASS
    (tmp_path / "creator").mkdir()
    caps = tmp_path / "creator" / "capabilities.json"
    caps.write_text(json.dumps({"capabilities": [{"id": "K01", "tests": ["t.py"], "floor": 0}]}), encoding="utf-8")
    assert D.check_depth_is_behaviour(tmp_path)[0] == D.PASS
    for bad in ({"id": "K01", "tests": ["t.py"], "floor": 500}, {"id": "K01", "tests": [], "floor": 0},
                {"id": "K01", "tests": ["t.py"], "floor": 0, "state": "VALIDATED"}):
        caps.write_text(json.dumps({"capabilities": [bad]}), encoding="utf-8")
        assert D.check_depth_is_behaviour(tmp_path)[0] == D.FAIL
    assert D.check_depth_is_behaviour(tmp_path / "nowhere")[0] == D.FAIL


def test_meaningful_ignores_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    assert D.check_meaningful_ignores_padding(D.ROOT)[0] == D.PASS
    from creator import efficiency as E
    monkeypatch.setattr(E, "ast_size", lambda src: len(src))               # a ruler that counts raw characters is padded easily
    st, ev = D.check_meaningful_ignores_padding(D.ROOT)
    assert st == D.FAIL and "ast_size moved" in ev


def test_capability_tested(tmp_path: Path) -> None:
    assert D.check_capability_tested(tmp_path, "K01")[0] == D.UNPROVEN
    _status(tmp_path, {"K01": "TESTED", "K02": "IMPLEMENTED"})
    assert D.check_capability_tested(tmp_path, "K01")[0] == D.PASS
    st, ev = D.check_capability_tested(tmp_path, "K01", "K02", "K99")
    assert st == D.UNPROVEN and "K02" in ev and "K99" in ev


def test_claude_share(tmp_path: Path) -> None:
    _status(tmp_path, {}, None)
    assert D.check_claude_share(tmp_path)[0] == D.PENDING
    _status(tmp_path, {}, 0.83)
    st, ev = D.check_claude_share(tmp_path)
    assert st == D.PENDING and "0.83" in ev
    _status(tmp_path, {}, 0.2)
    assert D.check_claude_share(tmp_path)[0] == D.PASS


def test_recursion_demonstrated(tmp_path: Path) -> None:
    assert D.check_recursion_demonstrated(tmp_path)[0] == D.PENDING
    led = tmp_path / "state" / "creator" / "ledger.jsonl"
    led.parent.mkdir(parents=True)
    led.write_text('{"data": {"statement": "other"}}\n', encoding="utf-8")
    assert D.check_recursion_demonstrated(tmp_path)[0] == D.PENDING
    led.write_text('{"rtype": "Finding", "data": {"statement": "RECURSION_STEP {}"}}\n', encoding="utf-8")
    assert D.check_recursion_demonstrated(tmp_path)[0] == D.PASS


def test_goals_pipeline(tmp_path: Path) -> None:
    assert D.check_goals_pipeline(tmp_path)[0] == D.PENDING
    (tmp_path / "creator").mkdir()
    (tmp_path / "creator" / "goals.py").write_text("X = 1\n", encoding="utf-8")
    assert D.check_goals_pipeline(tmp_path)[0] == D.PASS


def test_authoritative_documents(tmp_path: Path) -> None:
    st, ev = D.check_authoritative_documents(tmp_path)
    assert st == D.FAIL and "BIBLE.md" in ev
    assert D.check_authoritative_documents(D.ROOT)[0] == D.PASS


def test_subsections(tmp_path: Path) -> None:
    st, ev = D.check_subsections(D.ROOT, "CR300")
    assert st in (D.PASS, D.UNPROVEN, D.FAIL) and "CR300" in ev
    st, ev = D.check_subsections(tmp_path, "CR300")                       # an empty tree: no capability is TESTED
    assert st != D.PASS and "CR300.1" in ev


def test_loop_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    assert D.check_loop_stages(D.ROOT)[0] == D.PASS
    monkeypatch.setattr(D, "LOOP_SYMBOLS", (("ADOPT", "creator/kernel.py::no_such_function"),))
    st, ev = D.check_loop_stages(D.ROOT)
    assert st == D.FAIL and "no_such_function" in ev


# ---------------------------------------------------------------------------------------------------- rule evaluation
def test_a_cited_test_is_proven_only_by_a_passing_junit_result(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_a():\n    pass\n\ndef test_b():\n    pass\n", encoding="utf-8")
    rule = D.Rule("S", "S.1", "t", D.MAPPED, ("tests/test_x.py::test_a",))
    assert D.evaluate_rule(tmp_path, rule).status == D.UNPROVEN             # no junit: not assumed to pass
    _junit(tmp_path, "tests/test_x.py", {"test_b": "ok"})
    assert D.evaluate_rule(tmp_path, rule).status == D.UNPROVEN             # junit without that case
    _junit(tmp_path, "tests/test_x.py", {"test_a": "fail"})
    assert D.evaluate_rule(tmp_path, rule).status == D.FAIL
    _junit(tmp_path, "tests/test_x.py", {"test_a": "skip"})
    assert D.evaluate_rule(tmp_path, rule).status == D.FAIL                 # a skipped test proves nothing
    _junit(tmp_path, "tests/test_x.py", {"test_a[1]": "ok", "test_a[2]": "fail"})
    assert D.evaluate_rule(tmp_path, rule).status == D.FAIL                 # one failing parameter fails the case
    _junit(tmp_path, "tests/test_x.py", {"test_a[1]": "ok", "test_a[2]": "ok"})
    assert D.evaluate_rule(tmp_path, rule).status == D.PASS
    gone = D.Rule("S", "S.2", "t", D.MAPPED, ("tests/test_x.py::test_gone",))
    assert D.evaluate_rule(tmp_path, gone).status == D.FAIL                 # a reference to nothing is a failure
    assert D.evaluate_rule(tmp_path, D.Rule("S", "S.3", "t", D.MAPPED, ("creator/nothing.py::f",))).status == D.FAIL


def test_junit_evidence_is_bound_to_the_code_it_ran_against(tmp_path: Path) -> None:
    (tmp_path / "creator").mkdir()
    (tmp_path / "creator" / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("from creator import m\n\ndef test_a():\n    assert m.f() == 1\n", encoding="utf-8")
    rule = D.Rule("S", "S.1", "t", D.MAPPED, ("tests/test_x.py::test_a",))
    _junit(tmp_path, "tests/test_x.py", {"test_a": "ok"})
    assert D.evaluate_rule(tmp_path, rule).status == D.PASS
    (tmp_path / "creator" / "m.py").write_text("def f():\n    return 2\n", encoding="utf-8")      # a REACHED module changes
    r = D.evaluate_rule(tmp_path, rule)
    assert r.status == D.UNPROVEN and "stale" in r.evidence
    (tmp_path / "creator" / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    assert D.evaluate_rule(tmp_path, rule).status == D.PASS
    _junit(tmp_path, "tests/test_x.py", {"test_a": "ok"}, bind=False)
    (tmp_path / "state" / "creator" / "test_evidence.json").unlink()                                 # junit with no record
    assert D.evaluate_rule(tmp_path, rule).status == D.UNPROVEN


def test_a_check_rule_needs_its_check_and_its_test_and_a_crash_is_a_failure(tmp_path: Path) -> None:
    assert D.evaluate_rule(tmp_path, D.Rule("S", "S.1", "t", D.CHECK, check="nope")).status == D.FAIL
    D.CHECKS["boom"] = lambda root: 1 / 0                                    # type: ignore[assignment,return-value]
    try:
        r = D.evaluate_rule(tmp_path, D.Rule("S", "S.2", "t", D.CHECK, check="boom"))
    finally:
        del D.CHECKS["boom"]
    assert r.status == D.FAIL and "crashed" in r.evidence


def test_the_roll_up_order_and_it_never_validates() -> None:
    def res(*statuses: str) -> list[D.Result]:
        return [D.Result(D.Rule("S", f"S.{i}", "t", D.INTENT), s, "") for i, s in enumerate(statuses)]
    assert D.roll_up("S", res(D.PASS, D.PASS))[0] == "TESTING"
    assert D.roll_up("S", res(D.PASS, D.NOT_CHECKABLE))[0] == "IMPLEMENTED"
    assert D.roll_up("S", res(D.PASS, D.PENDING))[0] == "IMPLEMENTED"
    assert D.roll_up("S", res(D.PENDING, D.UNPROVEN))[0] == "IN_PROGRESS"
    assert D.roll_up("S", res(D.UNPROVEN, D.FAIL, D.PASS))[0] == "FAILED"
    assert D.roll_up("OTHER", res(D.PASS))[0] == "NOT_STARTED"


# ---------------------------------------------------------------------------------------------------- the rule table
def test_every_listed_section_has_rules_and_nothing_else_does() -> None:
    assert set(D.RULES_BY_SECTION) == set(D.TITLES) and len(D.SECTIONS) == 28
    assert all(D.RULES_BY_SECTION[s] for s in D.SECTIONS)
    assert {r.section for r in D.RULES} == set(D.SECTIONS)


def test_rule_ids_are_unique_and_kinds_are_complete() -> None:
    ids = [r.rule_id for r in D.RULES]
    assert len(ids) == len(set(ids))
    for r in D.RULES:
        assert r.kind in (D.MAPPED, D.CHECK, D.INTENT)
        if r.kind == D.INTENT:
            assert r.note and not r.refs and not r.check, r.rule_id          # an intent rule says WHY it is not checkable
        if r.kind == D.MAPPED:
            assert r.refs and not r.check, r.rule_id
        if r.kind == D.CHECK:
            assert r.check in D.CHECKS, r.rule_id


def test_every_computed_rule_cites_at_least_one_test_and_every_check_has_its_own_test() -> None:
    for r in D.RULES:
        if r.kind != D.INTENT:
            assert any(ref.startswith("tests/") for ref in r.refs), f"{r.rule_id} cites no test"
    me = Path(__file__).read_text(encoding="utf-8")
    for name in D.CHECKS:
        assert f"def test_{name}(" in me, f"check {name} has no test named test_{name}"


def test_every_reference_in_the_table_exists_in_this_tree() -> None:
    gone = [f"{r.rule_id} {ref}" for r in D.RULES for ref in r.refs if not D._ref_exists(D.ROOT, ref)]
    assert not gone, gone


def test_intent_rules_are_a_minority_and_name_their_reason() -> None:
    intent = [r for r in D.RULES if r.kind == D.INTENT]
    assert 0 < len(intent) < len(D.RULES) / 4
    assert all(len(r.note) > 20 for r in intent)


def test_the_map_lists_every_rule_once_and_every_section_status() -> None:
    results = D.evaluate(D.ROOT)
    text = D.render_map(results, D.TITLES)
    assert len(results) == len(D.RULES)
    for r in D.RULES:
        assert text.count(f"| {r.rule_id} |") == 1
    for s in D.SECTIONS:
        assert f"## {s} - " in text
    assert "VALIDATED" not in {D.roll_up(s, results)[0] for s in D.SECTIONS}
