"""Engine loop DRY end-to-end (no model, no GPU, nothing live): seeded metrics -> engine decision -> fake builder (scripted team actors) ->
verification with the REAL check functions on a temporary git repo (sandbox, patch apply, static checks, affected tests, the cold protected
suite, git merge, post-merge suite) -> adopt -> mark_done -> the calibration is updated -> the class earns its next autonomy level."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from creator import adopt as A
from creator import constraints as CON
from creator import engineloop as EL
from creator import kernel as K
from creator import safety as SF
from creator import team as TM

REAL_SAFETY = True                                  # run the real safety loop (trust gate, rate limit, post-merge suite)
NOW = 1_800_000_000.0
BY = "dry-coder"
SRC = "def foo(xs):\n    return sorted(xs)[0]\n"
DIFF = ("diff --git a/creator/fastfoo.py b/creator/fastfoo.py\n--- a/creator/fastfoo.py\n+++ b/creator/fastfoo.py\n"
        "@@ -1,2 +1,2 @@\n def foo(xs):\n-    return sorted(xs)[0]\n+    return min(xs)\n")
TEST = "from creator.fastfoo import foo\n\n\ndef test_foo():\n    assert foo([3, 1, 2]) == 1\n"
BENCH = {"argv": ["python", "bench_foo.py"], "unit": "cpu_s/event", "claim": "foo is faster"}


def git(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@localhost", *a], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def make_repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / "creator").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "creator" / "__init__.py").write_text("", encoding="utf-8", newline="\n")
    (repo / "creator" / "fastfoo.py").write_text(SRC, encoding="utf-8", newline="\n")
    (repo / "tests" / "test_fastfoo.py").write_text(TEST, encoding="utf-8", newline="\n")
    git(root, "init", "-q", str(repo))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def seed_metrics(state: Path) -> None:
    d = state / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"t": NOW - 3600, "outcome": "ok", "in_tok": 0, "out_tok": 0, "wall_s": 1.0, "cache_hit": False, "actor": "indexer", "cpu_s": 40.0}
            for _ in range(400)] + [{"t": NOW - 3600, "outcome": "ok", "in_tok": 0, "out_tok": 0, "wall_s": 1.0, "cache_hit": False,
                                    "actor": "filler", "cpu_s": 1.0} for _ in range(100)]
    (d / "events-20270101.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def make_team(tmp: Path, patch: str = DIFF) -> TM.Team:
    def note(text: str) -> Any:
        return lambda env, team: text

    def coder(env: TM.Envelope, team: TM.Team) -> str:
        return json.dumps(BENCH) if env.step == "BENCH" else patch

    return TM.Team(tmp / "team", [TM.actor("CHECKER", note("spec: min instead of sorted")), TM.actor("THINKER", note("plan: min()")),
                                  TM.actor("locate", note("creator/fastfoo.py:1-2")), TM.actor("CODER", coder)])


def setup(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, K.KernelConfig]:
    repo = make_repo(tmp)
    state = tmp / "state"
    state.mkdir()
    recs = tmp / "recs.jsonl"
    recs.write_text("\n".join(json.dumps({"cls": c, "passed": True, "confidence": 0.97, "task": f"{c}{i}"})
                              for c in ("tests_only", "docs", "refactor", "bugfix", "feature") for i in range(30)) + "\n", encoding="utf-8")
    (state / SF.POLICY_FILE).write_text(json.dumps({"full_suite": True, "supervised": [], "max_adoptions_per_hour": 50,
                                                    "trust_records": {BY: str(recs)}}), encoding="utf-8")
    monkeypatch.setattr(SF, "suite_files", lambda: ("tests/test_fastfoo.py",))          # the temp repo's own "protected suite"
    seed_metrics(state)
    cfg = K.KernelConfig(repo=repo, state=state, scratch=tmp / "scratch", hide=(), omit=())
    cfg.pytest.python = sys.executable
    return state, cfg


def bench(spec: Any, which: str) -> list[float]:
    return [40.0, 41.0, 39.0, 40.0, 40.5] if which == "before" else [10.0, 10.5, 9.5, 10.0, 10.2]


def test_decision_to_verified_adoption_to_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, cfg = setup(tmp_path, monkeypatch)
    A.grant(state, "tool", 1, "test: earned A1")
    out = EL.engine_step(state, cfg, make_team(tmp_path), BY, now=NOW, bench=bench, warm=False)
    dec, res = out["decision"], out["result"]
    assert dec["key"] == "hot_path:indexer" and dec["kind"] == "hot_path"                       # the engine decided
    assert res["outcome"] == "ADOPTED", res                                                    # real checks + real merge
    assert "min(xs)" in (cfg.repo / "creator" / "fastfoo.py").read_text(encoding="utf-8")        # the change is on main
    assert not CON.in_flight(state, NOW + 1)                                                    # mark_done freed the WIP slot
    cal = CON.calibration(state)
    assert "hot_path" in cal                                                                    # predicted vs actual recorded
    hist = CON._jsonl(CON._hist_file(state))
    assert hist[-1]["key"] == "hot_path:indexer" and hist[-1]["actual_saving_day"] == pytest.approx(out["actual_saving_day"])
    assert out["actual_saving_day"] > 0 and A.history(state)[-1]["outcome"] == "ADOPTED"
    assert out["level"] == 2 and A.level(state, "tool") == 2                                    # verified adoption: the class climbed
    ev = SF.events(state)[-1]
    assert ev["outcome"] == "ADOPTED" and ev["supervised"] is False


def test_a_failing_change_is_never_merged_and_still_frees_the_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, cfg = setup(tmp_path, monkeypatch)
    A.grant(state, "tool", 1, "test: earned A1")
    broken = DIFF.replace("min(xs)", "max(xs)")                                                  # the repo's test fails on it
    out = EL.engine_step(state, cfg, make_team(tmp_path, broken), BY, now=NOW, bench=bench, warm=False)
    assert out["result"]["outcome"] == "BUILD_FAILED" and out["level"] is None
    assert "sorted(xs)[0]" in (cfg.repo / "creator" / "fastfoo.py").read_text(encoding="utf-8")   # main untouched
    assert not CON.in_flight(state, NOW + 1) and out["actual_saving_day"] is None               # slot freed, nothing recalibrated
    assert "hot_path" not in CON.calibration(state)


def test_nothing_runs_while_the_stop_file_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, cfg = setup(tmp_path, monkeypatch)
    (state / EL.STOP_FILE).write_text("stop", encoding="utf-8")
    out = EL.engine_step(state, cfg, make_team(tmp_path), BY, now=NOW, bench=bench, warm=False)
    assert out == {"skipped": f"{EL.STOP_FILE} exists"}
    assert not (state / "engine").exists() and not (state / "adopt").exists()                   # not even a decision record


def test_the_start_load_does_not_include_the_loop() -> None:
    code = ("import sys\nsys.path[:0] = [sys.argv[1], sys.argv[1] + '/scripts']\nimport creator_swarm\n"
            "bad = [m for m in ('creator.engineloop', 'creator.adopt', 'creator.slowpath', 'creator.quickloop', 'creator.ladder') if m in sys.modules]\n"
            "assert not bad, bad\n")
    root = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, "-c", code, str(root)], cwd=root, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
