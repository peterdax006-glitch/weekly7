"""Shadow evaluation + the pre-registered switch rule (fakes, no model)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from creator import action_student as A
from creator import shadow as SH
from creator.curriculum import Lesson, LessonLog
from tests.test_creator_action_student import PKG, PLAN, FakeLLM, work  # noqa: F401


class Boom:
    trained = True

    def pick(self, *a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("chooser exploded")


class Fixed:
    trained = True

    def __init__(self, n: int) -> None:
        self.n = n

    def pick(self, *a, **k) -> int:  # type: ignore[no-untyped-def]
        return self.n


def _run(work: Path, state: Path, ch, llm_reply="CHOICE: 2\nWHY: x", use=False):  # type: ignore[no-untyped-def]
    st = A.ActionStudent(work / "n.jsonl", llm=FakeLLM(llm_reply), chooser=ch, use_chooser=use, state_dir=state)
    return st, st(PLAN, PKG, work)


def test_shadow_row_written_without_changing_decision(work: Path, tmp_path: Path) -> None:
    import shutil
    w2 = tmp_path / "w2"
    shutil.copytree(work, w2)
    _, base = _run(work, tmp_path / "s0", None)
    _, res = _run(w2, tmp_path / "s1", Fixed(1))
    assert base.claimed_done and res.claimed_done and base.notes == res.notes
    assert (work / "app/u.py").read_text() == (w2 / "app/u.py").read_text()
    row = json.loads((tmp_path / "s1" / "shadow_choices.jsonl").read_text().splitlines()[0])
    assert row["chooser_pick"] == 1 and row["model_pick"] == 2 and row["package_id"] == "P9" and row["applied"] == "default"
    assert row["default_pick"] is not None and row["candidates"]


def test_chooser_crash_does_not_affect_student(work: Path, tmp_path: Path) -> None:
    st, res = _run(work, tmp_path / "s", Boom())
    assert res.claimed_done
    assert not (tmp_path / "s" / "shadow_choices.jsonl").exists() and st.last_shadow_error


def test_unwritable_state_does_not_crash(work: Path, tmp_path: Path) -> None:
    f = tmp_path / "file"
    f.write_text("x")
    _, res = _run(work, f / "sub", Fixed(1))
    assert res.claimed_done


def _lesson(i: int, adopted, verdict="") -> Lesson:  # type: ignore[no-untyped-def]
    return Lesson(f"L{i}", f"P{i}", "c", "gap", "o", solver=A.ActionStudent.name, adopted=adopted, verdict=verdict)


def test_no_signal_outcomes_ignored(tmp_path: Path) -> None:
    for i in range(3):
        SH.record(tmp_path, f"P{i}", [], 1, 2, None, "default")
    lessons = [_lesson(0, True, "ADOPTED: ok"), _lesson(1, False, "cancelled: ram"), _lesson(2, None)]
    res = SH.resolve(tmp_path, lessons)
    assert [r["package_id"] for r in res] == ["P0"] and res[0]["adopted"] is True


def _s(n, c, d, ct=0, ca=0, dt=0, da=0):  # type: ignore[no-untyped-def]
    return {"n": n, "chooser_hits": c, "default_hits": d, "disagree_tested": ct + dt, "disagree_chooser_tested": ct,
            "disagree_chooser_adopted": ca, "disagree_default_tested": dt, "disagree_default_adopted": da}


def test_rule_flips_only_past_thresholds_and_back() -> None:
    assert SH.decide(_s(29, 29, 0), False) is None                 # n < 30
    assert SH.decide(_s(30, 20, 20), False) is None                # lower CI < default point
    assert SH.decide(_s(30, 30, 20), False) is True                # wilson_lower(30/30)=0.886 >= 0.667
    assert SH.decide(_s(30, 20, 20, ct=12, ca=12, dt=12, da=2), False) is True    # branch B: rates 1.0 vs 0.17
    assert SH.decide(_s(30, 20, 20, ct=9, ca=9, dt=12, da=0), False) is None      # chooser arm too few attempts
    assert SH.decide(_s(30, 20, 20, ct=12, ca=12, dt=9, da=0), False) is None     # default arm too few attempts
    assert SH.decide(_s(30, 25, 25), True) is None
    assert SH.decide(_s(100, 40, 80), True) is False               # 0.40 < wilson_lower(80/100)=0.71
    assert SH.decide(_s(10, 0, 10), True) is None


def test_update_policy_persists_event_and_student_reads_it(work: Path, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(SH, "stats", lambda state, lessons: {**_s(40, 40, 20), "shadow_resolved": 0})
    out = SH.update_policy(tmp_path, tmp_path / "lessons.jsonl")
    assert out["use_chooser"] is True and SH.read_policy(tmp_path) is True
    ev = json.loads((tmp_path / "policy_events.jsonl").read_text().splitlines()[0])
    assert ev["to"] == "chooser" and ev["evidence"]["n"] == 40
    assert A.ActionStudent(work / "n.jsonl", llm=FakeLLM(""), state_dir=tmp_path).use_chooser is True
    assert A.ActionStudent(work / "n.jsonl", llm=FakeLLM(""), state_dir=tmp_path, use_chooser=False).use_chooser is False
    monkeypatch.setattr(SH, "stats", lambda state, lessons: {**_s(100, 40, 80), "shadow_resolved": 0})
    assert SH.update_policy(tmp_path, tmp_path / "lessons.jsonl")["use_chooser"] is False
    assert len((tmp_path / "policy_events.jsonl").read_text().splitlines()) == 2


def test_loo_runs_on_real_lessons_only(tmp_path: Path) -> None:
    log = LessonLog(tmp_path / "l.jsonl")
    assert SH.loo(log.lessons()) == {"n": 0, "chooser_hits": 0, "default_hits": 0}
    assert SH.wilson(0, 0) == (0.0, 1.0) and SH.wilson(30, 30)[0] > 0.88


def test_a_retried_package_does_not_feed_one_outcome_to_every_attempt(tmp_path: Path) -> None:
    """Validator 6: shadow rows carry no lesson id, so every row of a retried package was joined to the package's LATEST lesson; the
    first attempt's rejected row was scored with the second attempt's adoption (evidence for the pre-registered switch rule)."""
    rows = [{"package_id": "P1", "lesson_id": "", "candidates": [], "default_pick": 1, "chooser_pick": 2, "model_pick": None,
             "applied": "chooser", "at": "2026-10-02T10:00:00"},
            {"package_id": "P1", "lesson_id": "", "candidates": [], "default_pick": 1, "chooser_pick": 2, "model_pick": None,
             "applied": "default", "at": "2026-10-02T11:00:00"}]
    (tmp_path / "shadow_choices.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    l1 = Lesson("L1", "P1", "c", "gap", "o", solver=A.ActionStudent.name, adopted=False, verdict="REJECTED: worse", at="2026-10-02T10:00:05")
    l2 = Lesson("L2", "P1", "c", "gap", "o", solver=A.ActionStudent.name, adopted=True, verdict="ADOPTED: ok", at="2026-10-02T11:00:05")
    res = SH.resolve(tmp_path, [l1, l2])
    assert [(r["applied"], r["adopted"]) for r in res] == [("chooser", False), ("default", True)]


def test_rule_b_uses_attempt_rates_not_raw_adopted_counts() -> None:
    # chooser applied 60 times, adopted 30 (50%); default applied 12 times, adopted 6 (50%): equal rates, raw counts favour chooser
    assert SH.decide(_s(30, 20, 20, ct=60, ca=30, dt=12, da=6), False) is None
    # chooser rate clearly worse but with far more adopted outcomes
    assert SH.decide(_s(30, 20, 20, ct=100, ca=40, dt=10, da=9), False) is None


def test_stats_counts_attempts_per_arm(tmp_path: Path) -> None:
    import creator.shadow as M
    rows = []
    for i in range(4):
        rows.append((f"P{i}", "chooser", i < 1))
    for i in range(4, 6):
        rows.append((f"P{i}", "default", True))
    for pid, app, _ in rows:
        SH.record(tmp_path, pid, [], 1, 2, None, app)
    lessons = [_lesson(int(pid[1:]), ad, "ADOPTED: ok" if ad else "REJECTED: no") for pid, _, ad in rows]
    for les, (pid, _, _) in zip(lessons, rows):
        les.package_id = pid
    res = M.resolve(tmp_path, lessons)
    assert len(res) == 6
    import unittest.mock as mk
    with mk.patch.object(M, "loo", lambda lessons: {"n": 0, "chooser_hits": 0, "default_hits": 0}),             mk.patch.object(M, "resolve", lambda st, ls: res):
        s = M.stats(tmp_path, [])
    assert (s["disagree_chooser_tested"], s["disagree_chooser_adopted"]) == (4, 1)
    assert (s["disagree_default_tested"], s["disagree_default_adopted"]) == (2, 2)
