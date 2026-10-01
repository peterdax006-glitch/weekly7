"""K17: the Creator shrinks its own code and memory (owner, 1 Oct 2026; ruling 'capability, not lines')."""
from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

from creator import build as B
from creator import efficiency as E
from creator import kernel as K
from creator import model as M
from creator import selfmodel as SM
from creator.ledger import Ledger


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


# ------------------------------------------------------------------------------------------------ measurement

def test_size_ignores_formatting_comments_and_docstrings() -> None:
    a = "def f(x):\n    y = x + 1\n    return y\n"
    b = 'def f(x):\n    """Docs."""\n    # comment\n    y = x + 1; return y\n'
    assert E.ast_size(a) == E.ast_size(b) > 0
    assert E.ast_size("def f(x):\n    return x + 1\n") < E.ast_size(a)
    assert E.ast_size("def f(:\n") == -1


def test_targets_are_editable_and_skip_recent_failures(tmp_path: Path) -> None:
    put(tmp_path, "creator/model.py", "x = 1\n" * 50)                   # protected measuring stick
    put(tmp_path, "creator/big.py", "x = 1\n" * 30)
    put(tmp_path, "creator/small.py", "x = 1\n")
    put(tmp_path, "creator/devbench/tasks/T/repo/app.py", "x = 1\n" * 99)  # benchmark material, not code
    assert E.pick_target(tmp_path) == ("creator/big.py", E.ast_size("x = 1\n" * 30))
    assert E.pick_target(tmp_path, avoid=("creator/big.py",))[0] == "creator/small.py"
    assert E.package_size(tmp_path) == sum(E.sizes(tmp_path).values())
    put(tmp_path, "creator/broken.py", "def f(:\n")
    with pytest.raises(ValueError, match="unparsable"):
        E.package_size(tmp_path)


def test_test_count(tmp_path: Path) -> None:
    put(tmp_path, "tests/test_a.py", "def test_x():\n    assert 1\n\n\ndef helper():\n    pass\n\n\ndef test_y():\n    assert 2\n")
    assert E.test_count(tmp_path) == 2


def test_peak_memory_of_the_real_creator_is_measured() -> None:
    xs = E.peak_memory_mb(E.Path(__file__).resolve().parents[1], replicates=1)
    assert len(xs) == 1 and 10 < xs[0] < 4000


# ------------------------------------------------------------------------------------------------ the shrink cycle

BIG = ("def add(a, b):\n    total = 0\n    total = total + a\n    total = total + b\n    return total\n\n\n"
       "def add3(a, b, c):\n    total = 0\n    total = total + a\n    total = total + b\n    total = total + c\n    return total\n\n\n"
       "def double(x):\n    result = add(x, x)\n    return result\n")
SMALL = "def add(a, b):\n    return a + b\n\n\ndef add3(a, b, c):\n    return a + b + c\n\n\ndef double(x):\n    return add(x, x)\n"
TESTS = ("from creator.big import add, add3, double\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\n"
         "def test_add3():\n    assert add3(1, 2, 3) == 6\n\n\ndef test_double():\n    assert double(4) == 8\n")
SPECS = [SM.CapabilitySpec("K01", "big", ("creator/big.py",), ("tests/test_big.py",), 0)]


@pytest.fixture()
def cfg(tmp_path: Path) -> K.KernelConfig:
    r = tmp_path / "repo"
    put(r, ".gitignore", "state/\n__pycache__/\n")
    put(r, "creator/__init__.py", "")
    put(r, "creator/big.py", BIG)
    put(r, "creator/app.py", "from creator.big import double\n\n\ndef main():\n    return double(2)\n")
    put(r, "tests/__init__.py", "")
    put(r, "tests/test_big.py", TESTS)
    put(r, "tests/test_app.py", "from creator.app import main\n\n\ndef test_main():\n    assert main() == 4\n")
    sh(r, "init", "-q")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    return K.KernelConfig(repo=r, state=r / "state" / "creator", scratch=tmp_path / "scratch", capabilities=SPECS,
                          scope=("creator", "tests"), build=B.BuildConfig(run_typecheck=False, require_typecheck=False),
                          mode="efficiency", measure_memory=False)


class Shrinker:
    name = "scripted-shrinker"

    def __init__(self, files: dict[str, str], delete: tuple[str, ...] = ()) -> None:
        self.files, self.delete = files, delete
        self.seen: list[str] = []

    def __call__(self, plan, package, workdir: Path) -> K.WorkResult:
        self.seen.append(plan.component)
        for rel, text in self.files.items():
            put(workdir, rel, text)
        for rel in self.delete:
            (workdir / rel).unlink()
        return K.WorkResult(True, "scripted")


def test_a_real_shrink_is_measured_and_adopted(cfg: K.KernelConfig) -> None:
    before = E.package_size(cfg.repo)
    w = Shrinker({"creator/big.py": SMALL})
    rep = K.cycle(cfg, w)
    assert w.seen == ["creator/big.py"]
    assert rep.outcome == "ADOPTED", (rep.reason, rep.details.get("detail"))
    assert E.package_size(cfg.repo) < before and (cfg.repo / "creator/big.py").read_text(encoding="utf-8") == SMALL
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    gap = [e.id for e in led.of_type("Gap") if getattr(e.record, "description").startswith("shrink creator/big.py")][0]
    assert led.view.status[gap] is M.Status.TESTED
    assert rep.details["detail"]["size"][1] < rep.details["detail"]["size"][0]


@pytest.mark.parametrize("name,files,delete,why", [
    ("mover", {"creator/big.py": "from creator.helper import add, add3, double  # noqa: F401\n", "creator/helper.py": BIG}, (),
     "holdout"),
    ("reformatter", {"creator/big.py": BIG.replace("    total = total + a\n    total = total + b\n",
                                                   "    total = total + a; total = total + b\n")}, (), "claim"),
    ("amputator", {"creator/big.py": "def add(a, b):\n    return a + b\n\n\ndef double(x):\n    return add(x, x)\n"}, (),
     "not clean"),
    ("test-deleter", {"creator/big.py": SMALL, "tests/test_big.py": TESTS.split("\n\n\ndef test_double")[0] + "\n"}, (), "test"),
])
def test_gamed_shrinks_are_rejected(cfg: K.KernelConfig, name: str, files: dict, delete: tuple, why: str) -> None:
    head = sh(cfg.repo, "rev-parse", "HEAD")
    rep = K.cycle(cfg, Shrinker(files, delete))
    assert rep.outcome == "REJECTED", (name, rep.outcome, rep.reason)
    assert why in rep.reason, (name, rep.reason)
    assert sh(cfg.repo, "rev-parse", "HEAD") == head and (cfg.repo / "creator/big.py").read_text(encoding="utf-8") == BIG


def test_gap_work_comes_first_in_auto_mode(cfg: K.KernelConfig) -> None:
    specs = SPECS + [SM.CapabilitySpec("K02", "missing", ("creator/missing.py",), ("tests/test_missing.py",), 0)]
    w = Shrinker({})
    K.cycle(dataclasses.replace(cfg, mode="auto", capabilities=specs), w)
    assert w.seen == ["K02"]                                             # a capability gap outranks a shrink
