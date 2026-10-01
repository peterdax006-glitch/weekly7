"""K22: several of the system's own workers in parallel on disjoint parts, sized by free RAM (owner, 1 Oct 2026)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from creator import build as B
from creator import kernel as K
from creator import objective as O
from creator import planner as P
from creator import selfmodel as SM
from creator import selfworkers as SW
from creator import swarm as W
from creator.ledger import Ledger


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


def module(i: int) -> str:
    pad = "".join(f"\n\ndef f{i}_{k}(x):\n    return x + {k}\n" for k in range(i + 3))
    return f"import os\nimport json\n\n\ndef _dead_{i}():\n    return {i}\n\n\ndef use_{i}(x):\n    return x * {i}\n{pad}"


@pytest.fixture()
def cfg(tmp_path: Path) -> K.KernelConfig:
    r = tmp_path / "repo"
    put(r, ".gitignore", "state/\n__pycache__/\n")
    put(r, "creator/__init__.py", "")
    put(r, "tests/__init__.py", "")
    for i in (1, 2, 3):
        put(r, f"creator/m{i}.py", module(i))
        put(r, f"tests/test_m{i}.py", f"from creator.m{i} import use_{i}\n\n\ndef test_use():\n    assert use_{i}(2) == {2 * i}\n")
    sh(r, "init", "-q")
    sh(r, "add", "-A")
    sh(r, "commit", "-q", "-m", "base")
    specs = [SM.CapabilitySpec(f"K0{i}", f"m{i}", (f"creator/m{i}.py",), (f"tests/test_m{i}.py",), 0) for i in (1, 2, 3)]
    return K.KernelConfig(repo=r, state=r / "state" / "creator", scratch=tmp_path / "scratch", capabilities=specs,
                          scope=("creator", "tests"), build=B.BuildConfig(run_typecheck=False, require_typecheck=False),
                          mode="efficiency", measure_memory=False)


@pytest.fixture(autouse=True)
def size_only(monkeypatch: pytest.MonkeyPatch) -> None:
    real = P.plan_efficiency
    monkeypatch.setattr(P, "plan_efficiency", lambda led, root, base, avoid=(), kind=None: real(led, root, base, avoid, "size"))


def own() -> SW.SelfFirst:
    return SW.SelfFirst([SW.RuleWorker(), SW.SearchWorker()], None)


def test_parallel_workers_on_disjoint_modules_all_get_adopted(cfg: K.KernelConfig) -> None:
    gov = W.Governor(start_gb=0, per_worker_gb=0, low_gb=-1, max_workers=3, free=lambda: 10.0)
    rnd = W.run_round(cfg, own, gov, max_packages=3, poll_s=0.2)
    assert rnd.peak_parallel == 3 and rnd.pulled_back == 0
    assert sorted(r.outcome for r in rnd.reports) == ["ADOPTED"] * 3, [(r.package, r.outcome, r.reason) for r in rnd.reports]
    targets = {Ledger(cfg.ledger_path, evidence_root=cfg.repo).get(
        next(e.id for e in Ledger(cfg.ledger_path, evidence_root=cfg.repo).of_type("WorkPackage")
             if e.record.package_id == r.package)).outputs[0] for r in rnd.reports}
    assert targets == {"creator/m1.py", "creator/m2.py", "creator/m3.py"}           # never two on the same module
    for i in (1, 2, 3):
        text = (cfg.repo / f"creator/m{i}.py").read_text(encoding="utf-8")
        assert "import os" not in text and f"_dead_{i}" not in text
    led = Ledger(cfg.ledger_path, evidence_root=cfg.repo)
    assert led.verify() and O.claude_dependence(led)["self_share"] == 1.0


def test_little_ram_means_one_worker_at_a_time(cfg: K.KernelConfig) -> None:
    gov = W.Governor(start_gb=2.0, per_worker_gb=5.0, low_gb=0.5, max_workers=3, free=lambda: 3.0)
    rnd = W.run_round(cfg, own, gov, max_packages=2, poll_s=0.2)
    assert rnd.peak_parallel == 1 and len(rnd.reports) == 2


def test_tight_ram_pulls_the_youngest_back_and_adopts_nothing_of_it(cfg: K.KernelConfig) -> None:
    state = {"n": 0}

    def free() -> float:
        state["n"] += 1
        return 10.0 if state["n"] < 6 else 0.1                          # plenty at first, then tight

    gov = W.Governor(start_gb=0, per_worker_gb=0, low_gb=1.0, max_workers=3, free=free)
    rnd = W.run_round(cfg, own, gov, max_packages=3, poll_s=0.2)
    assert rnd.pulled_back >= 1
    cancelled = [r for r in rnd.reports if r.outcome == "CANCELLED"]
    assert cancelled and all("pulled back" in r.reason and not r.merge_commit for r in cancelled)
    assert Ledger(cfg.ledger_path, evidence_root=cfg.repo).verify()


def test_hard_pull_back_kills_only_that_workers_processes(tmp_path: Path) -> None:
    """Regression (1 Oct): a cooperative pull-back came too late and the host stopped the whole swarm at critical RAM."""
    import json
    import sys
    import time
    box, other = tmp_path / "scratch" / "sb1", tmp_path / "elsewhere"
    box.mkdir(parents=True)
    other.mkdir()
    (box / ".creator_sandbox.json").write_text(json.dumps({"label": "CP9"}), encoding="utf-8")
    sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]
    inside = subprocess.Popen(sleeper, cwd=box)
    outside = subprocess.Popen(sleeper, cwd=other)
    try:
        time.sleep(1.0)
        assert W.stop_worker_processes(tmp_path / "scratch", "CP9") >= 1
        inside.wait(timeout=20)
        assert inside.returncode is not None and outside.poll() is None
        assert W.stop_worker_processes(tmp_path / "scratch", "NOPE") == 0
    finally:
        for p in (inside, outside):
            if p.poll() is None:
                p.kill()
