"""The skip rule counts only judged attempts and can never become permanent without evidence."""
from __future__ import annotations

import datetime as dt

from creator import curriculum as CUR

NOW = dt.datetime(2026, 10, 2, 12, 0, 0)


def les(i: int, verdict: str, adopted: bool | None = False, done: bool = False, version: str = "", at: str = "2026-10-02T10:00:00",
        solver: str = "stu") -> CUR.Lesson:
    return CUR.Lesson(lesson_id=f"L{i}", package_id=f"P{i}", component="c", task_kind="shrink", objective="o", solver=solver,
                      adopted=adopted, verdict=verdict, claimed_done=done, at=at, version=version)


def nothing(n: int, **kw: object) -> list[CUR.Lesson]:
    return [les(i, "not claimed done: no edit applied", **kw) for i in range(n)]  # type: ignore[arg-type]


def test_unjudged_and_runtime_failures_never_count() -> None:
    r = CUR.Router()
    assert r.skip_reason(nothing(10), "stu", "shrink", now=NOW)
    junk = [les(100 + i, "not claimed done: model call failed: Timeout") for i in range(30)]
    junk += [les(200, "", adopted=None, done=True), les(201, "interrupted: no kernel record", done=True),
             les(202, "cancelled: pulled back before evaluation", done=True), les(203, "error: boom", done=True)]
    assert r.skip_reason(nothing(9) + junk, "stu", "shrink", now=NOW) == ""
    assert not CUR.is_judged(junk[0]) and not CUR.is_judged(junk[-1])
    assert CUR.is_judged(les(1, "REJECTED: x", done=True)) and CUR.is_judged(les(2, "ROLLED_BACK: x", done=True))


def test_skip_lifts_on_new_version() -> None:
    r = CUR.Router()
    old = nothing(12, version="v1")
    assert r.skip_reason(old, "stu", "shrink", version="v1", now=NOW)
    assert r.skip_reason(old, "stu", "shrink", version="v2", now=NOW) == ""        # new code/model: fresh start
    assert r.skip_reason(nothing(12), "stu", "shrink", version="v2", now=NOW) == ""  # legacy unversioned evidence is not evidence
    assert r.skip_reason(old + nothing(10, version="v2"), "stu", "shrink", version="v2", now=NOW)   # re-skipped only on new evidence


def test_skip_cooldown_allows_one_probe() -> None:
    r = CUR.Router(skip_cooldown_s=3600)
    ls = nothing(12, at="2026-10-02T10:00:00")
    assert r.skip_reason(ls, "stu", "shrink", now=dt.datetime(2026, 10, 2, 10, 30))
    assert r.skip_reason(ls, "stu", "shrink", now=dt.datetime(2026, 10, 2, 11, 5)) == ""      # cool-down over: probe
    probe = ls + [les(99, "", adopted=None, done=True, at="2026-10-02T11:05:00")]               # the probe is in flight
    assert r.skip_reason(probe, "stu", "shrink", now=dt.datetime(2026, 10, 2, 11, 6))          # clock restarted
    assert CUR.Curriculum.__init__ and CUR.SKIP_COOLDOWN_S > 0


def test_lesson_version_roundtrip_and_student_version() -> None:
    class S:
        name = "s"
        version = "m1"
    a = CUR.student_version(S())
    assert a.startswith("m1|")
    S.version = "m2"
    assert CUR.student_version(S()) != a
    assert CUR.Lesson.from_dict(les(1, "x", version="q").to_dict()).version == "q"


def test_infra_failure_matches_the_note_prefix_not_a_mention() -> None:
    for note in ("model call failed: TimeoutError: x", "local model server did not become healthy", "pulled back before evaluation"):
        assert not CUR.is_judged(les(1, f"not claimed done: {note}")), note
    for note in ("no edit applied: server.py unchanged", "model reply held no usable edit about a timeout", "prescreen: no candidate (ram use)"):
        assert CUR.is_judged(les(1, f"not claimed done: {note}")), note
    assert CUR.Router().skip_reason([les(i, "not claimed done: no edit applied to server.py timeouts") for i in range(10)], "stu", "shrink", now=NOW)
