"""K20: the Creator's own worker inside the kernel - local model edits, tests decide (owner, 1 Oct 2026). Fake model."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from creator import build as B
from creator import generator as G
from creator import kernel as K
from creator import localworker as LW
from creator import selfmodel as SM


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


def test_edits_apply_exactly_or_by_lines_and_refuse_ambiguity(tmp_path: Path) -> None:
    put(tmp_path, "a.py", "def f(x):\n    if x:\n        return 1\n    return 2\n")
    reply = ("FILE: a.py\n<<<<<<< SEARCH\n  if x:\n      return 1\n=======\nif x > 0:\n    return 10\n>>>>>>> REPLACE\n"
             "FILE: ../escape.py\n<<<<<<< SEARCH\n\n=======\nx = 1\n>>>>>>> REPLACE\n")
    edits = G.parse_edits(reply)
    assert [e[0] for e in edits] == ["a.py"]                            # the path escape is dropped
    applied, refused = G.apply_edits(tmp_path, edits)
    assert applied == ["a.py"] and not refused
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "def f(x):\n    if x > 0:\n        return 10\n    return 2\n"
    put(tmp_path, "b.py", "x = 1\nx = 1\n")
    assert G.apply_edits(tmp_path, [("b.py", "x = 1", "x = 2")])[1]       # two matches: refused, not guessed
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "x = 1\nx = 1\n"
    assert G.apply_edits(tmp_path, [("new.py", "", "y = 2")])[0] == ["new.py"]


class FakeModel:
    def __init__(self, replies: list[str]) -> None:
        self.replies, self.calls, self.seen = list(replies), 0, []

    def __enter__(self) -> "FakeModel":
        return self

    def __exit__(self, *a) -> None:
        pass

    def chat(self, messages, **kw) -> str:
        self.calls += 1
        self.seen.append(messages[-1]["content"])
        return self.replies.pop(0) if self.replies else "nothing"


SPECS = [SM.CapabilitySpec("K01", "base", ("pkg/base.py",), ("tests/test_base.py",), 0),
         SM.CapabilitySpec("K02", "user", ("pkg/user.py",), ("tests/test_user.py",), 0)]
CREATE = ("FILE: pkg/user.py\n<<<<<<< SEARCH\n\n=======\nfrom pkg.base import one\n\n\ndef use():\n    return one() + 1\n"
          ">>>>>>> REPLACE\nFILE: tests/test_user.py\n<<<<<<< SEARCH\n\n=======\nfrom pkg.user import use\n\n\n"
          "def test_use():\n    assert use() == 2\n>>>>>>> REPLACE\n")
WRONG = CREATE.replace("one() + 1", "one() + 5")


@pytest.fixture()
def cfg(tmp_path: Path) -> K.KernelConfig:
    r = tmp_path / "repo"
    put(r, ".gitignore", "state/\n__pycache__/\n")
    put(r, "pkg/__init__.py", "")
    put(r, "pkg/base.py", "def one():\n    return 1\n")
    put(r, "pkg/app.py", "from pkg.base import one\n\n\ndef main():\n    return one()\n")
    put(r, "tests/__init__.py", "")
    put(r, "tests/test_base.py", "from pkg.base import one\n\n\ndef test_one():\n    assert one() == 1\n")
    sh(r, "init", "-q")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    return K.KernelConfig(repo=r, state=r / "state" / "creator", scratch=tmp_path / "scratch", capabilities=SPECS,
                          scope=("pkg", "tests"), build=B.BuildConfig(run_typecheck=False, require_typecheck=False), mode="gaps")


def test_the_local_worker_builds_a_missing_component_and_the_kernel_adopts_it(cfg: K.KernelConfig) -> None:
    fake = FakeModel([WRONG, CREATE])                                   # first attempt fails its test, second is right
    rep = K.cycle(cfg, LW.LocalWorker(llm_factory=lambda: fake))
    assert rep.outcome == "ADOPTED", rep.reason
    assert fake.calls == 2 and "The tests fail" in fake.seen[1]
    assert (cfg.repo / "pkg/user.py").is_file()


def test_a_local_worker_that_cannot_do_it_is_rejected(cfg: K.KernelConfig) -> None:
    rep = K.cycle(cfg, LW.LocalWorker(llm_factory=lambda: FakeModel(["I am not sure"] * 3)))
    assert rep.outcome == "REJECTED" and "changed nothing" in rep.reason
