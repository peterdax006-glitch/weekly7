"""CR02: the Creator's evidence-derived self-model (C77 secs 11, 25, 41, 42)."""
from __future__ import annotations

from pathlib import Path

import pytest

from creator import selfmodel as SM


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    r = tmp_path / "proj"
    write(r, "pkg/__init__.py", "")
    write(r, "pkg/core.py", '"""Core maths."""\n\n\ndef add(a: int, b: int) -> int:\n    """Add."""\n    return a + b\n\n\n'
                            'class Box:\n    """A box."""\n\n    def size(self) -> int:\n        return 1\n\n'
                            '    def later(self):\n        raise NotImplementedError\n')
    write(r, "pkg/util.py", "from pkg.core import add\n\n\ndef twice(x):\n    return add(x, x)\n\n\ndef todo():\n    pass\n")
    write(r, "pkg/lonely.py", "def alone():\n    return 0\n")
    write(r, "pkg/broken.py", "def oops(:\n    pass\n")
    write(r, "tests/test_core.py", "from pkg.core import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    write(r, "tests/test_util.py", "from pkg.util import twice\n\n\ndef test_twice():\n    assert twice(2) == 5\n")
    return r


SPECS = [SM.CapabilitySpec("C1", "core", ("pkg/core.py",), ("tests/test_core.py",), floor=1),
         SM.CapabilitySpec("C2", "util", ("pkg/util.py",), ("tests/test_util.py",), floor=1000),
         SM.CapabilitySpec("C3", "lonely", ("pkg/lonely.py",), ("tests/test_lonely.py",)),
         SM.CapabilitySpec("C4", "future", ("pkg/future.py",), ("tests/test_future.py",))]


def model(tree: Path, evidence: dict | None = None) -> SM.SelfModel:
    return SM.build(tree, scope=("pkg", "tests"), capabilities=SPECS, test_evidence=evidence or {}, include_versions=False)


def test_components_interfaces_and_stubs(tree: Path) -> None:
    m = model(tree)
    core = m.components["pkg/core.py"]
    names = {i.name for i in core.interfaces}
    assert {"add", "Box", "Box.size", "Box.later"} <= names and core.doc == "Core maths."
    assert any("Box.later" in s and "NotImplementedError" in s for s in core.stubs)
    assert any(s.startswith("todo:") and "pass only" in s for s in m.components["pkg/util.py"].stubs)
    assert m.components["pkg/broken.py"].parse_error
    assert core.meaningful > 0 and len(core.sha256) == 64


def test_dependencies_and_tests_of(tree: Path) -> None:
    m = model(tree)
    assert "pkg/core.py" in m.dependencies["pkg/util.py"]
    assert "pkg/util.py" in m.dependents["pkg/core.py"]
    assert m.tests_of["pkg/core.py"] == ("tests/test_core.py",) and "tests/test_util.py" in m.tests_of["pkg/util.py"]
    kinds = {(x.kind, x.subject) for x in m.limitations}
    assert ("UNTESTED_MODULE", "pkg/lonely.py") in kinds and ("UNREACHED", "pkg/lonely.py") in kinds
    assert ("UNPARSABLE", "pkg/broken.py") in kinds


def test_capability_states_are_computed_from_evidence(tree: Path, tmp_path: Path) -> None:
    store = tmp_path / "ev.json"
    ev = SM.collect_test_evidence(tree, ["tests/test_core.py", "tests/test_util.py"], store)
    assert ev["tests/test_core.py"].outcome == "PASS" and ev["tests/test_util.py"].outcome == "FAIL"
    m = model(tree, ev)
    st = {c.id: c for c in m.capabilities}
    assert st["C1"].state == "TESTED" and st["C2"].state == "FAILED" and st["C2"].below_floor
    assert st["C3"].state == "IMPLEMENTED" and st["C3"].uncertainty == "UNTESTED" and st["C4"].state == "NOT_STARTED"
    assert any(x.kind == "FAILING_TESTS" and x.subject == "C2" for x in m.limitations)
    assert any(x.kind == "BELOW_FLOOR" and x.subject == "C2" for x in m.limitations)
    assert SM.load_test_evidence(store)["tests/test_core.py"].outcome == "PASS"


def test_an_edit_makes_old_test_evidence_stale(tree: Path, tmp_path: Path) -> None:
    ev = SM.collect_test_evidence(tree, ["tests/test_core.py"], tmp_path / "ev.json")
    assert model(tree, ev).capability("C1").state == "TESTED"
    write(tree, "pkg/core.py", (tree / "pkg/core.py").read_text(encoding="utf-8") + "\n\ndef extra():\n    return 1\n")
    c1 = model(tree, ev).capability("C1")
    assert c1.state == "IMPLEMENTED" and c1.last_results["tests/test_core.py"] == "STALE"


def test_diagnose_prefers_evidence_over_claims(tree: Path, tmp_path: Path) -> None:
    ev = SM.collect_test_evidence(tree, ["tests/test_core.py", "tests/test_util.py"], tmp_path / "ev.json")
    m = model(tree, ev)
    found = {c.capability: c for c in SM.diagnose(m, {"C1": ("TESTED", "ledger"), "C2": ("TESTED", "checklist"),
                                                      "C3": ("VALIDATED", "checklist"), "C4": ("IMPLEMENTED", "notes"),
                                                      "C9": ("TESTED", "memo"), })}
    assert "C1" not in found                                            # claim matches evidence
    assert found["C2"].evidenced == "FAILED" and "fail" in found["C2"].why
    assert found["C3"].claimed == "VALIDATED" and "not even TESTED" in found["C3"].why
    assert found["C4"].evidenced == "NOT_STARTED" and found["C9"].evidenced == "UNKNOWN"
    behind = SM.diagnose(m, {"C1": ("IMPLEMENTED", "stale checklist")})
    assert behind and "behind" in behind[0].why


def test_the_model_is_deterministic_and_saved(tree: Path, tmp_path: Path) -> None:
    a, b = model(tree), model(tree)
    assert a.digest() == b.digest()
    write(tree, "pkg/new.py", "X = 1\n")
    assert model(tree).digest() != a.digest()
    out = tmp_path / "sm.json"
    assert SM.save(a, out) == a.digest() and out.read_text(encoding="utf-8").count('"digest"') == 1


def test_empty_scope(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    m = SM.build(tmp_path / "empty", scope=("nothing",), capabilities=[], include_versions=False)
    assert not m.components and not m.capabilities and not m.limitations


def test_scans_the_real_creator_package() -> None:
    m = SM.build(scope=("creator",), include_versions=False)
    assert "creator/ledger.py" in m.components and m.capability("K01").present_modules
    assert m.capability("K03").state == "NOT_STARTED" or m.capability("K03").present_modules



def test_parallel_evidence_matches_serial(tree: Path, tmp_path: Path) -> None:
    serial = SM.collect_test_evidence(tree, ["tests/test_core.py", "tests/test_util.py"], tmp_path / "a.json")
    par = SM.collect_test_evidence(tree, ["tests/test_core.py", "tests/test_util.py"], tmp_path / "b.json", parallel=2)
    assert {k: (v.outcome, v.source_digest) for k, v in serial.items()} == {k: (v.outcome, v.source_digest) for k, v in par.items()}
