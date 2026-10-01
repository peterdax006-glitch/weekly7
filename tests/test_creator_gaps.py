"""CR08: gap analysis from evidence (C77 secs 14, 41, 42). A TEMPORARY project tree; test results are real pytest runs."""
from __future__ import annotations

from pathlib import Path

import pytest

from creator import gaps as G
from creator import model as M
from creator import objective as O
from creator import selfmodel as SM
from creator.ledger import Ledger


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


SPECS = [SM.CapabilitySpec("K01", "base", ("pkg/base.py",), ("tests/test_base.py",), 3),
         SM.CapabilitySpec("K02", "user", ("pkg/user.py",), ("tests/test_user.py",), 50)]


@pytest.fixture()
def proj(tmp_path: Path) -> Path:
    r = tmp_path / "proj"
    put(r, "pkg/__init__.py", "")
    put(r, "pkg/base.py", "def one():\n    return 1\n\n\ndef two():\n    return 2\n\n\ndef three():\n    return 3\n")
    put(r, "pkg/user.py", "from pkg.base import one\n\n\ndef use():\n    return one() + 1\n\n\ndef later():\n    pass\n")
    put(r, "tests/__init__.py", "")
    put(r, "tests/test_base.py", "from pkg.base import one\n\n\ndef test_one():\n    assert one() == 1\n")
    put(r, "tests/test_user.py", "from pkg.user import use\n\n\ndef test_use():\n    assert use() == 3\n")     # fails
    return r


def world(proj: Path) -> tuple[Ledger, SM.SelfModel, O.Compiled]:
    led = Ledger(proj / "dev.jsonl", evidence_root=proj)
    oid = O.self_objective(led)
    comp = O.compile_capabilities(led, oid, SPECS)
    ev = SM.collect_test_evidence(proj, ["tests/test_base.py", "tests/test_user.py"], proj / "ev.json")
    model = SM.build(proj, scope=("pkg", "tests"), capabilities=SPECS, test_evidence=ev, include_versions=False)
    return led, model, comp


def test_assessment_is_computed_from_evidence(proj: Path) -> None:
    led, model, _ = world(proj)
    a = {r.key: r for r in G.assess(led, model)}
    assert a["K01.exists"].met and a["K01.tested"].met and a["K01.depth"].met and a["K01.no_stubs"].met
    assert a["K01.integrated"].met                                      # pkg/user.py imports it
    assert not a["K02.tested"].met and "FAIL" in a["K02.tested"].detail
    assert not a["K02.no_stubs"].met and "later" in a["K02.no_stubs"].detail
    assert not a["K02.depth"].met and not a["K02.integrated"].met
    assert not a["K01.validated"].met                                   # only an independent role can make it so


def test_sync_opens_gaps_and_closes_met_requirements_with_a_computed_testrun(proj: Path) -> None:
    led, model, comp = world(proj)
    rep = G.sync(led, model)
    req = comp.requirement_ids
    assert led.view.status[req["K01.tested"]] is M.Status.TESTED
    just = [e for e in led.of_type("TestRun") if req["K01.tested"] in e.record.subject_ids]
    assert just and just[0].record.evidence and not just[0].record.evidence[0].problem(proj)
    assert G.open_gaps_for(led, req["K02.tested"]) and G.open_gaps_for(led, req["K01.validated"])
    assert not G.open_gaps_for(led, req["K01.tested"])
    again = G.sync(led, model)
    assert again.opened == () and again.closed == ()                    # idempotent
    assert rep.met == sum(1 for r in G.assess(led, model) if r.met)


def test_fixing_closes_the_gap_and_breaking_marks_a_regression(proj: Path) -> None:
    led, model, comp = world(proj)
    G.sync(led, model)
    req = comp.requirement_ids
    gap = G.open_gaps_for(led, req["K02.tested"])[0]
    put(proj, "pkg/user.py", "from pkg.base import one\n\n\ndef use():\n    return one() + 2\n\n\ndef later():\n    pass\n")
    ev = SM.collect_test_evidence(proj, ["tests/test_user.py"], proj / "ev.json")
    fixed = SM.build(proj, scope=("pkg", "tests"), capabilities=SPECS, test_evidence=ev, include_versions=False)
    rep = G.sync(led, fixed)
    assert gap in rep.closed and led.view.status[gap] is M.Status.TESTED and led.view.status[req["K02.tested"]] is M.Status.TESTED
    put(proj, "pkg/base.py", "def one():\n    return 0\n\n\ndef two():\n    return 2\n\n\ndef three():\n    return 3\n")
    ev = SM.collect_test_evidence(proj, ["tests/test_base.py", "tests/test_user.py"], proj / "ev.json")
    broken = SM.build(proj, scope=("pkg", "tests"), capabilities=SPECS, test_evidence=ev, include_versions=False)
    rep2 = G.sync(led, broken)
    assert "K01.tested" in rep2.regressed and led.view.status[req["K01.tested"]] is M.Status.FAILED
    assert G.open_gaps_for(led, req["K01.tested"])


def test_stale_evidence_does_not_close_a_gap(proj: Path) -> None:
    led, model, comp = world(proj)
    G.sync(led, model)
    put(proj, "pkg/user.py", "from pkg.base import one\n\n\ndef use():\n    return one() + 2\n\n\ndef later():\n    pass\n")
    stale = SM.build(proj, scope=("pkg", "tests"), capabilities=SPECS,                      # old results, new source
                     test_evidence=SM.load_test_evidence(proj / "ev.json"), include_versions=False)
    G.sync(led, stale)
    assert G.open_gaps_for(led, comp.requirement_ids["K02.tested"])


def test_ranking_puts_unblocked_important_gaps_first(proj: Path) -> None:
    led, model, _ = world(proj)
    G.sync(led, model)
    r = G.ranked(led)
    assert r and not r[0].blocked_by
    first_blocked = next((i for i, g in enumerate(r) if g.blocked_by), len(r))
    assert all(g.blocked_by for g in r[first_blocked:])
    k01v = [g for g in r if g.requirement_key == "K01.validated"]
    assert k01v and k01v[0].blocked_by == ()                       # all of K01's prerequisites are met
    k02v = [g for g in r if g.requirement_key == "K02.validated"]
    assert k02v and k02v[0].blocked_by                              # K02 still has open prerequisite gaps
    s = G.summary(led)
    assert s["open_gaps"] == len(r) and s["requirements"] == 2 * len(O.LADDER)
