"""K21: the system's own workers - it works on itself with no AI (owner, 1 Oct 2026)."""
from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

from creator import build as B
from creator import efficiency as E
from creator import kernel as K
from creator import objective as O
from creator import planner as P
from creator import selfmodel as SM
from creator import selfworkers as SW
from creator.ledger import Ledger


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


# ------------------------------------------------------------------------------------------------ the transforms

def test_unused_imports_drops_only_what_nothing_uses() -> None:
    src = "import os\nimport json\nfrom typing import Any, Optional\n\n\ndef f(x: 'Optional[int]') -> str:\n    return json.dumps(x)\n"
    out, rounds = src, 0
    while (nxt := SW.unused_imports(out)) is not None:                 # one import statement per application
        out, rounds = nxt, rounds + 1
    assert rounds == 2 and "import os" not in out and "Any" not in out
    assert "import json" in out and "from typing import Optional" in out                  # a string annotation counts as use
    assert SW.unused_imports("from __future__ import annotations\nx = 1\n") is None


def test_lazy_imports_moves_a_component_into_its_only_users() -> None:
    src = ('"""Doc."""\nfrom creator import heavy\nfrom creator import used_at_top\n\nX = used_at_top.Y\n\n\n'
           'def a():\n    """Doc a."""\n    return heavy.go()\n\n\ndef b(n):\n    return heavy.go() + n\n\n\ndef c():\n    return 1\n')
    out = SW.lazy_imports(src)
    assert out is not None
    assert "from creator import used_at_top" in out.split("def a")[0]                     # module-level use: stays eager
    assert "from creator import heavy" not in out.split("def a")[0]
    assert out.count("from creator import heavy") == 2                                     # one per user, none in c()
    assert '"""Doc a."""\n    from creator import heavy\n    return heavy.go()' in out
    compile(out, "x.py", "exec")
    assert SW.lazy_imports("import json\n\n\ndef f():\n    return json\n") is None           # not a Creator component
    assert SW.lazy_imports("from creator import m\n\n\ndef f(x: m.T):\n    return x\n") is None  # annotation: refused


def test_dead_private_respects_references() -> None:
    src = "def _unused():\n    return 1\n\n\ndef _used():\n    return 2\n\n\ndef api():\n    return _used()\n"
    out = SW.dead_private(src, referenced=lambda name: False)
    assert out is not None and "_unused" not in out and "_used" in out
    assert SW.dead_private(src, referenced=lambda name: name == "_unused") is None


def test_self_first_hands_off_only_what_its_own_workers_cannot_do(tmp_path: Path) -> None:
    calls = []

    class Nope:
        name = "self-nope"

        def __call__(self, plan, package, workdir):
            calls.append("own")
            return K.WorkResult(False, "cannot")

    class Session:
        name = "claude-session"

        def __call__(self, plan, package, workdir):
            calls.append("session")
            return K.WorkResult(True, "done by the session")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    r = SW.SelfFirst([Nope()], Session())(None, None, tmp_path)
    assert calls == ["own", "session"] and r.by == "claude-session" and r.claimed_done


# ------------------------------------------------------------------------------------------------ real cycles

BIG = ("import os\nfrom creator import helper\n\n\ndef _never_called():\n    return 42\n\n\n"
       "def add(a, b):\n    return helper.plus(a, b)\n")
HELPER = "def plus(a, b):\n    return a + b\n\n\ndef unused_but_public():\n    return 0\n" + "\n".join(
    f"\n\ndef pad{i}(x):\n    return x * {i} + {i}" for i in range(30)) + "\n"


@pytest.fixture()
def cfg(tmp_path: Path) -> K.KernelConfig:
    r = tmp_path / "repo"
    put(r, ".gitignore", "state/\n__pycache__/\n")
    put(r, "creator/__init__.py", "")
    put(r, "creator/helper.py", HELPER)
    put(r, "creator/big.py", BIG)
    put(r, "creator/kernel.py", "from creator import big\n\n\ndef start():\n    return big.add(1, 2)\n")
    put(r, "tests/__init__.py", "")
    put(r, "tests/test_big.py", "from creator.big import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    put(r, "tests/test_kernel.py", "from creator.kernel import start\n\n\ndef test_start():\n    assert start() == 3\n")
    sh(r, "init", "-q")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    specs = [SM.CapabilitySpec("K01", "big", ("creator/big.py",), ("tests/test_big.py",), 0)]
    return K.KernelConfig(repo=r, state=r / "state" / "creator", scratch=tmp_path / "scratch", capabilities=specs,
                          scope=("creator", "tests"), build=B.BuildConfig(run_typecheck=False, require_typecheck=False),
                          mode="efficiency", measure_memory=False)


def own() -> SW.SelfFirst:
    return SW.SelfFirst([SW.RuleWorker(), SW.SearchWorker()], None)


def test_the_system_shrinks_itself_with_its_own_rules(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    real = P.plan_efficiency
    monkeypatch.setattr(P, "plan_efficiency",
                        lambda led, root, base, avoid=(), kind=None: real(led, root, base, ("creator/helper.py",), "size"))
    before = E.sizes(cfg.repo)["creator/big.py"]
    rep = K.cycle(cfg, own())
    assert rep.package and rep.outcome == "ADOPTED", (rep.reason, rep.details.get("detail"))
    after = (cfg.repo / "creator/big.py").read_text(encoding="utf-8")
    assert "import os" not in after and "_never_called" not in after and E.sizes(cfg.repo)["creator/big.py"] < before
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    dep = O.claude_dependence(led)
    assert dep["self_share"] == 1.0 and dep["adopted_by_worker"] == {"self-rules-v1": 1}


def test_the_system_loads_less_with_its_own_rules(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    real = P.plan_efficiency
    monkeypatch.setattr(P, "plan_efficiency", lambda led, root, base, avoid=(), kind=None: real(led, root, base, avoid, "activation"))
    before = E.activation(cfg.repo)
    rep = K.cycle(cfg, own())
    assert rep.outcome == "ADOPTED", (rep.reason, rep.details.get("detail"))
    after = E.activation(cfg.repo)
    assert after["active_nodes"] < before["active_nodes"]
    kernel_top = (cfg.repo / "creator/kernel.py").read_text(encoding="utf-8").split("def ")[0]
    assert "from creator import big" not in kernel_top                  # the kernel no longer loads big (or helper) at start
    assert O.claude_dependence(Ledger(cfg.ledger_path, evidence_root=cfg.repo))["self_share"] == 1.0


def test_search_is_not_used_on_code_whose_tests_already_pass(cfg: K.KernelConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (1 Oct): with no rule applicable, the search worker mutated untested code, which grew the module."""
    real = P.plan_efficiency
    monkeypatch.setattr(P, "plan_efficiency",
                        lambda led, root, base, avoid=(), kind=None: real(led, root, base, ("creator/big.py", "creator/kernel.py"), "size"))
    rep = K.cycle(cfg, own())
    assert rep.outcome == "REJECTED" and "changed nothing" in rep.reason            # helper.py: no rule applies, nothing done
