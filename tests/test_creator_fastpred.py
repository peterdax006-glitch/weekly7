"""Fast-resolving prospective topics: prediction before outcome (append-only), idempotence, the SAME gate on planted data, hooks that are no-ops
when disabled and never raise, and no measurable slowdown of the hooked path."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from creator import fastpred as F
from creator import thinking as T


@pytest.fixture()
def st(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("NUPEN_FASTPRED_STATE", str(tmp_path))
    monkeypatch.delenv("NUPEN_FASTPRED", raising=False)
    return tmp_path


def lines(st: Path) -> list[dict]:
    return [json.loads(x) for x in F.store(st).read_text(encoding="utf-8").splitlines()]


def test_prediction_precedes_outcome_and_is_idempotent(st: Path) -> None:
    pid = F.begin(st, "test_file_passes", "r1:tests/a.py", "tests/a.py", now=100.0)
    assert F.begin(st, "test_file_passes", "r1:tests/a.py", "tests/a.py", now=200.0) == pid     # idempotent: no second line
    assert len(lines(st)) == 1 and lines(st)[0]["made_at"] == 100.0
    assert F.end(st, pid, 1, now=150.0)
    assert not F.end(st, pid, 0, now=160.0)                                                      # an outcome is final
    assert [x["kind"] for x in lines(st)] == ["pred", "out"]


def test_outcome_without_or_before_prediction_is_rejected(st: Path) -> None:
    assert not F.end(st, "test_file_passes:ghost", 1, now=5.0)                                  # no prediction: nothing written
    assert not F.store(st).exists()
    pid = F.begin(st, "stage_slow", "s1", "worker", now=100.0)
    assert not F.end(st, pid, 1, now=99.0)                                                       # time-travelling outcome
    # a hand-written outcome line that precedes its prediction line is ignored by the fold
    with F.store(st).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "out", "id": "stage_slow:s2", "y": 1, "at": 500.0}) + "\n")
        f.write(json.dumps({"kind": "pred", "id": "stage_slow:s2", "topic": "stage_slow", "subject": "s2", "key": "worker", "made_at": 400.0,
                            "p": 0.5, "base": 0.5, "last": 0.5, "mode": "live", "k": 1}) + "\n")
    fo = F.fold(st)
    assert "stage_slow:s2" not in fo.outs and fo.rejected >= 1


def test_learner_uses_only_earlier_outcomes(st: Path) -> None:
    for i in range(30):
        pid = F.begin(st, "test_file_slow", f"x{i}", "slow.py" if i % 2 else "fast.py", now=1000.0 + 10 * i)
        F.end(st, pid, i % 2, now=1000.0 + 10 * i + 1)
    p_slow = F.begin(st, "test_file_slow", "late1", "slow.py", now=5000.0)
    p_fast = F.begin(st, "test_file_slow", "late2", "fast.py", now=5000.0)
    recs = {r["id"]: r for r in lines(st) if r["kind"] == "pred"}
    assert recs[p_slow or ""]["p"] > 0.7 > 0.3 > recs[p_fast or ""]["p"]
    early = recs["test_file_slow:x0"]                                                            # made when nothing had resolved yet: the prior
    assert abs(early["p"] - 0.5) < 1e-6


def test_gate_on_planted_data(st: Path) -> None:
    rnd = random.Random(7)
    t = 1000.0
    for i in range(200):                                                                          # key "a" nearly always passes, key "b" half the time
        k = "a.py" if i % 2 else "b.py"
        y = int(rnd.random() < (0.95 if k == "a.py" else 0.1))
        pid = F.begin(st, "test_file_passes", f"p{i}", k, now=t)
        F.end(st, pid, y, now=t + 1)
        t += 5
    sec = F.trust_section(st)["test_file_passes"]
    assert sec["score"]["n"] == 200 and sec["score"]["n_live"] == 200
    assert sec["trusted"], sec["why_not"]                                                        # skill + calibration + prospective counts
    assert sec["score"]["gain_ci95"][0] > 0
    none = F.trust_section(st)["stage_slow"]
    assert not none["trusted"] and none["why_not"] == ["no scored predictions"]
    for i in range(60):                                                                          # no skill: a coin, keys carry nothing
        pid = F.begin(st, "stage_slow", f"c{i}", "k%d" % (i % 3), now=t)
        F.end(st, pid, rnd.randint(0, 1), now=t + 1)
        t += 5
    assert not F.trust_section(st)["stage_slow"]["trusted"]


def test_thinking_trust_reports_fast_topics_through_the_same_gate(st: Path) -> None:
    from creator import anticipation as A
    rep = T.trust(st, write=False, owner_dir=st)
    assert set(F.TOPICS) <= set(rep["fast_topics"])
    assert T.independent(st, "plan_first") is False
    assert A  # imported on demand by trust


def test_plan_first_race(st: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    F.plan_begin([("PK1", "EFF.size"), ("PK2", "K07.exists"), ("PK3", "EFF.size")])
    recs = {r["subject"]: r for r in lines(st)}
    assert all(abs(r["base"] - 1 / 3) < 1e-6 for r in recs.values())                             # honest baseline: 1/k
    F.plan_verdict("PK2")
    F.plan_verdict("PK1")
    fo = F.fold(st)
    assert [fo.outs[f"plan_first:{p}"]["y"] for p in ("PK1", "PK2", "PK3")] == [0, 1, 0]


def test_hooks_are_noops_when_disabled(st: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUPEN_FASTPRED", "0")
    assert F.begin_safe("stage_slow", "z", "worker") is None
    F.end_safe("stage_slow:z", 1)
    F.plan_begin([("A", "x"), ("B", "y")])
    F.plan_verdict("A")
    assert not F.store(st).exists()
    monkeypatch.delenv("NUPEN_FASTPRED")
    monkeypatch.delenv("NUPEN_FASTPRED_STATE")                                                   # under pytest without a state dir: still disabled
    assert not F.enabled()


def test_hooks_never_raise(st: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(F, "_append", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert F.begin_safe("stage_slow", "z", "worker") is None
    F.end_safe("stage_slow:z", 1)
    F.tests_end([("a.py", "", "")], {}, 1.0, False)


def test_testrun_hook_scores_files(st: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUPEN_TEST_SLOTS_DIR", str(tmp_path / "slots"))
    from creator import testrun as TR
    (tmp_path / "test_x.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (tmp_path / "test_y.py").write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    run = TR.run_pytest(tmp_path, ["test_x.py", "test_y.py"], tmp_path / "j.xml", label="candidate",
                        config=TR.PytestConfig(python=__import__("sys").executable, timeout=120))
    assert run.trustworthy
    fo = F.fold(st)
    ys = {p["key"]: fo.outs[i]["y"] for i, p in fo.preds.items() if p["topic"] == "test_file_passes"}
    assert ys == {"test_x.py": 1, "test_y.py": 0}


def test_hook_overhead_is_small(st: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for i in range(300):                                                                         # a store with history, as in real use
        pid = F.begin(st, "stage_slow", f"h{i}", f"s{i % 5}", now=1000.0 + i)
        F.end(st, pid, i % 3 == 0, now=1000.5 + i)
    n = 200
    t0 = time.perf_counter()
    for i in range(n):
        pid = F.begin_safe("stage_slow", f"b{i}", f"s{i % 5}")
        F.end_safe(pid, 0)
    per_call = (time.perf_counter() - t0) / n
    assert per_call < 0.015, per_call                                                            # a predict+resolve pair costs under 15 ms (stages last seconds)
    monkeypatch.setenv("NUPEN_FASTPRED", "0")
    t0 = time.perf_counter()
    for i in range(2000):
        F.end_safe(F.begin_safe("stage_slow", f"d{i}", "s"), 0)
    assert (time.perf_counter() - t0) / 2000 < 0.0001                                            # disabled: effectively free
