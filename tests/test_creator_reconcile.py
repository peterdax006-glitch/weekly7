"""Lesson reconciliation from the kernel's records, and the skip rule for hopeless students (planted logs, no kernel run)."""
from __future__ import annotations

import json
from pathlib import Path

from creator import constraints as CON
from creator import curriculum as CUR


def lesson(i: str, pkg: str, solver: str = "stu", kind: str = "shrink", done: bool = False, adopted=None, verdict: str = "",
           at: str = "2026-10-01T10:00:00") -> CUR.Lesson:
    return CUR.Lesson(lesson_id=i, package_id=pkg, component="c", task_kind=kind, objective="o", solver=solver,
                      claimed_done=done, adopted=adopted, verdict=verdict, at=at)


def plant(tmp: Path, lessons: list[CUR.Lesson], kernel: dict[str, str]) -> CUR.LessonLog:
    log = CUR.LessonLog(tmp / "lessons.jsonl")
    for les in lessons:
        log.add(les)
    with (tmp / "kernel_log.jsonl").open("w", encoding="utf-8") as fh:
        for pkg, oc in kernel.items():
            fh.write(json.dumps({"package": pkg, "outcome": oc, "reason": f"why {oc}"}) + "\n")
    return log


def test_every_pending_lesson_gets_the_right_outcome_and_rerun_changes_nothing(tmp_path: Path) -> None:
    log = plant(tmp_path, [lesson(f"l{i}", f"P{i}", done=True) for i in range(1, 7)] + [lesson("t", "TEACH-1", "claude", verdict="teacher lesson: pending kernel")],
                {"P1": "ADOPTED", "P2": "REJECTED", "P3": "ROLLED_BACK", "P4": "CANCELLED", "P5": "ERROR"})
    cyc = tmp_path / "cycles" / "P6"
    cyc.mkdir(parents=True)
    (cyc / "cycle.json").write_text(json.dumps({"package": "P6", "outcome": "REJECTED", "reason": "from cycle file"}), encoding="utf-8")
    rep = CUR.reconcile_lessons(log, tmp_path)
    got = {x.lesson_id: (x.adopted, x.verdict) for x in log.lessons()}
    assert got["l1"][0] is True and got["l1"][1].startswith("ADOPTED")
    assert got["l2"][0] is False and got["l2"][1].startswith("REJECTED")
    assert got["l3"][0] is False and got["l3"][1].startswith("ROLLED_BACK")
    assert got["l4"][0] is False and got["l4"][1].startswith("cancelled:")
    assert got["l5"][0] is False and got["l5"][1].startswith("error:")
    assert got["l6"][1].startswith("REJECTED") and "from cycle file" in got["l6"][1]
    assert got["t"][0] is None                                      # a teacher lesson written ahead of the kernel stays pending
    assert rep["pending"] == 7 and rep["skipped"] == 1
    before = log.path.read_text(encoding="utf-8")
    again = CUR.reconcile_lessons(log, tmp_path)
    assert log.path.read_text(encoding="utf-8") == before and again["settled"] == {}


def test_interrupted_cycle_becomes_no_signal_and_in_flight_is_left_alone(tmp_path: Path) -> None:
    log = plant(tmp_path, [lesson("a", "DEAD", done=True), lesson("b", "LIVE", done=True)], {})
    CUR.reconcile_lessons(log, tmp_path, in_flight={"LIVE"})
    got = {x.lesson_id: x for x in log.lessons()}
    assert got["a"].adopted is False and got["a"].verdict.startswith("interrupted")
    assert not CUR.is_skill_signal(got["a"])                        # no skill signal: excluded from student scores
    assert CUR.student_scores(got.values())["cells"] == {}
    assert got["b"].adopted is None
    assert CON._unmeasured_class(got["a"]) == "cancelled_unrecorded"


def test_reconcile_never_raises(tmp_path: Path) -> None:
    log = CUR.LessonLog(tmp_path / "nope" / "lessons.jsonl")
    assert CUR.reconcile_lessons(log, tmp_path / "nowhere")["pending"] == 0
    (tmp_path / "kernel_log.jsonl").write_text("garbage\n{bad\n", encoding="utf-8")
    assert CUR.reconcile_lessons(log, tmp_path)["settled"] == {}


def hopeless(n: int, kind: str = "shrink", solver: str = "stu") -> list[CUR.Lesson]:
    return [lesson(f"{solver}{kind}{i}", f"Q{i}", solver, kind, adopted=False, verdict="not claimed done: no usable edit",
                   at=f"2026-10-01T10:{i:02d}:00") for i in range(n)]


def test_skip_rule_per_kind_and_unskips_on_new_evidence(tmp_path: Path) -> None:
    r = CUR.Router()
    ls = hopeless(10) + hopeless(4, "gap")
    assert "skipped on shrink" in r.skip_reason(ls, "stu", "shrink")
    assert r.skip_reason(ls, "stu", "gap") == ""                    # too few attempts there
    assert r.skip_reason(ls, "other", "shrink") == ""               # another student is unaffected
    assert r.skip_reason(hopeless(9), "stu", "shrink") == ""
    assert r.skip_reason(ls + [lesson("w", "Z", "stu", done=True, adopted=True, verdict="ADOPTED: x")], "stu", "shrink") == ""
    # cancelled cycles say nothing about skill and do not count towards the 10
    assert r.skip_reason(hopeless(9) + [lesson("c", "C", adopted=False, verdict="cancelled: ram")], "stu", "shrink") == ""
    # 80% no-output share: 8 of 10 still skipped, 7 of 10 not
    mix = hopeless(8) + [lesson(f"m{i}", f"M{i}", done=True, adopted=False, verdict="REJECTED: x") for i in range(2)]
    assert r.skip_reason(mix, "stu", "shrink")
    assert r.skip_reason(hopeless(7) + [lesson(f"n{i}", f"N{i}", done=True, adopted=False, verdict="REJECTED: x") for i in range(3)], "stu", "shrink") == ""
    assert r.skip_reason(ls, "claude", "shrink") == ""
    # a reenable event (retrain) ignores the older attempts
    log = CUR.LessonLog(tmp_path / "l.jsonl")
    for x in ls:
        log.add(x)
    log.reenable("stu", "shrink", "retrained")
    since = log.reenabled()[("stu", "shrink")]
    assert since and r.skip_reason(log.lessons(), "stu", "shrink", since) == ""
    assert len(log.lessons()) == len(ls)                            # the event is not a lesson


def test_learning_signal_reports_per_student_and_kind() -> None:
    import datetime as dt
    now = dt.datetime(2026, 10, 1, 12, 0, 0)
    ls = hopeless(3) + [lesson("ok", "OK", done=True, adopted=True, verdict="ADOPTED: m", at="2026-10-01T11:00:00")]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        log = CUR.LessonLog(Path(td) / "lessons.jsonl")
        for x in ls:
            log.add(x)
        m = [x for x in CON.learning_metrics(Path(td), now, 24.0) if x.name == "learning_signal"][0]
    assert m.detail["per_student_kind"]["stu|shrink"] == {"attempts": 4, "with_verdict": 1, "adopted": 1}


def test_an_orphan_lesson_is_not_credited_with_a_retry_by_another_worker(tmp_path: Path) -> None:
    """Validator 6: package ids are reused by retries. A student's lesson whose process died (no outcome) was settled ADOPTED because a
    LATER cycle of the same package, by another worker, was adopted - that credit entered the student's skill record."""
    log = plant(tmp_path, [lesson("orphan", "P9", "stu", done=True), lesson("other", "P8", "stu2", done=True, adopted=True, verdict="ADOPTED: x"),
                           lesson("mine", "P7", "stu", done=True)], {})
    with (tmp_path / "kernel_log.jsonl").open("w", encoding="utf-8") as fh:
        for pkg, by in (("P9", "stu2"), ("P7", "stu")):
            fh.write(json.dumps({"package": pkg, "outcome": "ADOPTED", "reason": "ok", "details": {"worker": {"by": by}}}) + "\n")
    CUR.reconcile_lessons(log, tmp_path)
    got = {x.lesson_id: x for x in log.lessons()}
    assert got["orphan"].adopted is False and got["orphan"].verdict.startswith("interrupted")
    assert not CUR.is_skill_signal(got["orphan"])
    assert got["mine"].adopted is True                               # the worker that was judged is the lesson's own
