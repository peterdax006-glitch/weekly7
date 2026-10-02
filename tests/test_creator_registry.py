"""Sparse activation (owner, 2 Oct 2026): optional capabilities load on demand through creator.registry, and the kernel REJECTS an adopted
change that grows what the kernel / swarm load at start (unless an eager dependency is the package's objective, justified on record)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from creator import build as B
from creator import efficiency as E
from creator import kernel as K
from creator import registry as REG
from creator import selfmodel as SM
from tests.test_creator_kernel import Scripted, cached_provenance, head, put, sh  # noqa: F401 - fixture + helpers

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ the registry

def test_get_imports_on_first_use_and_caches() -> None:
    code = ("import sys\nfrom creator import registry as R\n"
            "assert 'creator.doctrine' not in sys.modules\n"
            "m = R.get('doctrine')\nassert 'creator.doctrine' in sys.modules and R.get('doctrine') is m\n"
            "assert 'doctrine' in R.loaded() and 'goals' not in R.loaded()\n")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_attribute_targets_unknown_names_and_missing_modules() -> None:
    assert REG.get("lesson_student").__name__ == "LessonStudent"
    with pytest.raises(REG.CapabilityError):
        REG.get("no_such_capability")
    assert REG.optional("no_such_capability") is None
    REG.register("ghost", "creator.does_not_exist_anywhere")
    try:
        assert REG.optional("ghost") is None
        with pytest.raises(ImportError):
            REG.get("ghost")
    finally:
        REG.CAPABILITIES.pop("ghost", None)
    REG.register("ghost2", "creator.registry:no_such_attr")
    try:
        assert REG.optional("ghost2") is None
    finally:
        REG.CAPABILITIES.pop("ghost2", None)


def test_every_registered_capability_resolves() -> None:
    missing = [n for n in REG.names() if REG.optional(n) is None]
    assert not missing, missing


def test_the_rule_text_names_the_registry_and_reaches_the_worker_prompt() -> None:
    t = REG.sparse_rule_text()
    assert "creator.registry" in t and "module-level imports only for the core" in t
    from creator import model as M
    from creator import planner as P
    plan = P.Plan("g", "K.x", "c", "exists", M.Role.IMPLEMENTER, "w", "p", "cp", "e", 1)
    wp = M.WorkPackage(created_by=M.Role.IMPLEMENTER, package_id="WP1", objective="o", why_it_exists="w", prerequisites=("none",),
                       inputs=(), outputs=(), implementation_requirements=("do",), interfaces=("i",), data_flow="d", dependencies=(),
                       test_requirements=("t",), validation_requirements=("v",), expected_failure_modes=("e",), evidence_requirements=("ev",),
                       failure_conditions=("f",), rollback_requirements=("r",), completion_criteria=("c",), anti_premature_completion=("a",), meaningful_code_depth=1)
    assert "creator.registry" in K.render_package(plan, wp)


def test_the_entry_points_do_not_import_optional_capabilities() -> None:
    """The swarm entry point starts without the optional capabilities loaded (they are fetched through the registry)."""
    code = ("import sys\nsys.path[:0] = [sys.argv[1], sys.argv[1] + '/scripts']\nimport creator_swarm\n"
            "bad = [m for m in ('creator.goals', 'creator.constraints', 'creator.curriculum', 'creator.process_levers',\n"
            "                   'creator.selfworkers', 'creator.student', 'creator.model_student', 'creator.recursion') if m in sys.modules]\n"
            "assert not bad, bad\n")
    p = subprocess.run([sys.executable, "-c", code, str(ROOT)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_the_real_start_load_stays_at_the_lazy_level() -> None:
    """Regression: the start-time load of the swarm entry point stays at the lazy level reached on 2 Oct (generous headroom)."""
    load = E.start_load(ROOT)
    assert set(load) == set(E.START_ENTRIES) and load["scripts/creator_swarm.py"] < 85_000, load


# ------------------------------------------------------------------------------------------------ the guard (pure)

HEAVY = "".join(f"def f{i}():\n    return {i} + 1\n\n\n" for i in range(80))


def _tree(root: Path) -> Path:
    put(root, "creator/__init__.py", "")
    put(root, "creator/kernel.py", "from creator import base\n\n\ndef run():\n    return base.one()\n")
    put(root, "creator/base.py", "def one():\n    return 1\n\n\n" + HEAVY * 20)
    put(root, "creator/heavy.py", HEAVY)
    put(root, "scripts/creator_swarm.py", "from creator import kernel\n\n\ndef main():\n    return kernel.run()\n")
    return root


def _copy(src: Path, dst: Path) -> Path:
    for f in src.rglob("*.py"):
        put(dst, f.relative_to(src).as_posix(), f.read_text(encoding="utf-8"))
    return dst


def test_a_module_level_import_of_a_heavy_module_is_a_regression(tmp_path: Path) -> None:
    base = _tree(tmp_path / "base")
    cand = _copy(base, tmp_path / "cand")
    put(cand, "creator/kernel.py", "from creator import base\nfrom creator import heavy\n\n\ndef run():\n    return base.one() + heavy.f1()\n")
    why = E.start_load_regression(base, cand, "K.x")
    assert len(why) == 2 and "creator.registry" in why[0], why                 # kernel and swarm entry both load more


def test_a_guarded_module_level_import_still_counts(tmp_path: Path) -> None:
    base = _tree(tmp_path / "base")
    cand = _copy(base, tmp_path / "cand")
    put(cand, "creator/kernel.py", "from creator import base\ntry:\n    from creator import heavy\nexcept ImportError:\n    heavy = None\n")
    assert E.start_load_regression(base, cand, "K.x")


def test_the_same_capability_behind_the_registry_is_not_a_regression(tmp_path: Path) -> None:
    base = _tree(tmp_path / "base")
    cand = _copy(base, tmp_path / "cand")
    put(cand, "creator/kernel.py", "from creator import base\nfrom creator import registry\n\n\ndef run():\n"
                                   "    return base.one() + registry.get('heavy').f1()\n")
    put(cand, "creator/registry.py", "import importlib\n\n\ndef get(n):\n    return importlib.import_module('creator.' + n)\n")
    assert E.start_load_regression(base, cand, "K.x") == []
    put(cand, "creator/kernel.py", "from creator import base\n\n\ndef run():\n    from creator import heavy\n    return heavy.f1()\n")
    assert E.start_load_regression(base, cand, "K.x") == []                    # a function-level import is lazy too


def test_an_eager_dependency_needs_its_recorded_justification(tmp_path: Path) -> None:
    base = _tree(tmp_path / "base")
    cand = _copy(base, tmp_path / "cand")
    put(cand, "creator/kernel.py", "from creator import base\nfrom creator import heavy\n")
    assert E.start_load_regression(base, cand, "EAGER.heavy")
    put(cand, "creator/EAGER.md", "EAGER.heavy: x\n")                          # too short to be a reason
    assert E.start_load_regression(base, cand, "EAGER.heavy")
    put(cand, "creator/EAGER.md", "EAGER.other: the kernel cannot decide anything without it\n")   # for another package
    assert E.start_load_regression(base, cand, "EAGER.heavy")
    put(cand, "creator/EAGER.md", "- EAGER.heavy: the kernel cannot decide anything without it\n")
    assert E.start_load_regression(base, cand, "EAGER.heavy") == []


def test_a_small_addition_within_tolerance_passes(tmp_path: Path) -> None:
    base = _tree(tmp_path / "base")
    cand = _copy(base, tmp_path / "cand")
    put(cand, "creator/base.py", (cand / "creator/base.py").read_text(encoding="utf-8") + "\n\ndef two():\n    return 2\n")
    assert E.start_load_regression(base, cand, "K.x") == []


# ------------------------------------------------------------------------------------------------ the guard in the kernel cycle

SPECS = [SM.CapabilitySpec("K01", "base", ("creator/base.py",), ("tests/test_base.py",), 3),
         SM.CapabilitySpec("K02", "user", ("creator/user.py",), ("tests/test_user.py",), 3)]
USER = "from creator.heavy import f1\n\n\ndef use():\n    return f1() - 1\n\n\ndef twice():\n    return 2 * use()\n"
USER_TEST = ("from creator.user import twice, use\n\n\ndef test_use():\n    assert use() == 1\n\n\ndef test_twice():\n    assert twice() == 2\n")
BASE_TEST = "from creator.base import one\n\n\ndef test_one():\n    assert one() == 1\n\n\ndef test_again():\n    assert one() + one() == 2\n"


@pytest.fixture()
def kcfg(tmp_path: Path) -> K.KernelConfig:
    r = tmp_path / "repo"
    put(r, ".gitignore", "state/\n__pycache__/\n")
    _tree(r)
    put(r, "tests/__init__.py", "")
    put(r, "tests/test_base.py", BASE_TEST)
    put(r, "canon/CANON.md", "canon\n")
    sh(r, "init", "-q")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    return K.KernelConfig(repo=r, state=r / "state" / "creator", scratch=tmp_path / "scratch", capabilities=SPECS,
                          scope=("creator", "tests"), build=B.BuildConfig(run_typecheck=False, require_typecheck=False))


def test_the_kernel_rejects_a_candidate_that_adds_an_eager_heavy_import(kcfg: K.KernelConfig) -> None:
    before = head(kcfg)
    eager = "from creator import base\nfrom creator import user\n\n\ndef run():\n    return base.one() + user.use()\n"
    rep = K.cycle(kcfg, Scripted("eager", {"creator/user.py": USER, "tests/test_user.py": USER_TEST, "creator/kernel.py": eager}))
    assert rep.outcome == "REJECTED" and "start-time load grew" in rep.reason, (rep.outcome, rep.reason)
    assert rep.details["start_load"] and head(kcfg) == before


def test_the_kernel_adopts_the_same_capability_behind_the_registry(kcfg: K.KernelConfig) -> None:
    lazy = ("from creator import base\n\n\ndef run():\n    import importlib\n"
            "    return base.one() + importlib.import_module('creator.user').use()\n")
    before = head(kcfg)
    rep = K.cycle(kcfg, Scripted("lazy", {"creator/user.py": USER, "tests/test_user.py": USER_TEST, "creator/kernel.py": lazy}))
    assert rep.outcome == "ADOPTED", (rep.reason, rep.details.get("start_load"))
    assert head(kcfg) != before
