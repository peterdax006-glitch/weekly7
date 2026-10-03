"""K29: every student attempt should reach a measured verdict - the learning-signal report, its causes and remedies."""
from __future__ import annotations

import json
from pathlib import Path

from creator import goal_constraint_learning_signal as LS


def les(verdict: str = "", reasoning: str = "", claimed: object = None, solver: str = "s1", pkg: str = "CP1", at: str = "2026-10-02T10:00") -> dict:
    return {"verdict": verdict, "reasoning": reasoning, "claimed_done": claimed, "solver": solver, "package_id": pkg, "at": at}


def test_each_way_an_attempt_can_end_is_classified() -> None:
    assert LS.classify(les("ADOPTED: improvement")) == LS.JUDGED
    assert LS.classify(les("REJECTED: claim REGRESSION")) == LS.JUDGED
    assert LS.classify(les("ROLLED_BACK: audit red")) == LS.JUDGED
    assert LS.classify(les("teacher lesson: pending kernel")) == LS.NOT_AN_ATTEMPT
    assert LS.classify(les("interrupted: no kernel record")) == LS.INTERRUPTED
    assert LS.classify(les("cancelled: pulled back")) == LS.CANCELLED
    assert LS.classify(les("error: git timeout")) == LS.ERROR
    assert LS.classify(les(reasoning="model call failed: TimeoutError: timed out")) == LS.MODEL_TIMEOUT
    assert LS.classify(les(reasoning="model reply held no usable search/replace edit")) == LS.UNUSABLE_REPLY
    assert LS.classify(les(reasoning="no edit applied: creator/kernel.py: SEARCH text not found")) == LS.UNUSABLE_REPLY
    assert LS.classify(les(reasoning="prescreen: no candidate can improve the targeted metric")) == LS.NO_CANDIDATE
    assert LS.classify(les(reasoning="teacher absent: deferred until the teacher is back")) == LS.DEFERRED
    assert LS.classify(les(claimed=True)) == LS.UNJUDGED                                  # claimed, never judged
    assert LS.classify(les(claimed=False, reasoning="gave up")) == LS.NOT_CLAIMED


def test_the_report_measures_the_signal_rate_and_ranks_what_it_lost() -> None:
    rows = ([les("ADOPTED: x", solver="a")] + [les("REJECTED: y", solver="b")]
            + [les("interrupted: z", solver="a")] * 3 + [les(reasoning="model call failed: timed out", solver="b")] * 2
            + [les("teacher lesson: pending kernel", solver="claude")])
    r = LS.signal_report(rows)
    assert r["attempts"] == 7 and r["judged"] == 2 and r["signal_rate"] == round(2 / 7, 4)   # the teacher lesson is not an attempt
    assert r["lost_by_cause"][0] == (LS.INTERRUPTED, 3) and r["lost_by_cause"][1] == (LS.MODEL_TIMEOUT, 2)
    assert r["per_solver"]["a"] == {"attempts": 4, "judged": 1, "signal_rate": 0.25}
    top = LS.remedies(r, top=1)
    assert top[0]["cause"] == LS.INTERRUPTED and top[0]["lost"] == 3 and "drain" in top[0]["action"]


def test_a_window_counts_only_recent_attempts() -> None:
    rows = [les("ADOPTED: x", at="2026-10-01T09:00"), les("interrupted: y", at="2026-10-02T12:00")]
    assert LS.signal_report(rows, since="2026-10-02")["attempts"] == 1


def test_cancelled_cycles_without_a_lesson_are_found() -> None:
    lessons = [les("cancelled: x", pkg="CP1")]
    kernel = [{"package": "CP1", "outcome": "CANCELLED"}, {"package": "CP2", "outcome": "CANCELLED"},
              {"package": "CP3", "outcome": "ADOPTED"}, {"package": "CP4", "outcome": "ERROR"}, {"package": "CP4", "outcome": "ERROR"}]
    assert LS.unrecorded_cycles(lessons, kernel) == ["CP2", "CP4"]                        # recorded, judged and duplicate rows skipped


def test_empty_and_broken_inputs_never_raise(tmp_path: Path) -> None:
    empty = LS.signal_report([])
    assert empty["attempts"] == 0 and empty["signal_rate"] is None and LS.remedies(empty) == []
    assert LS.assess(tmp_path)["attempts"] == 0                                             # no files at all
    (tmp_path / "lessons.jsonl").write_text('{"verdict": "ADOPTED: x"}\nnot json\n[1, 2]\n', encoding="utf-8")
    (tmp_path / "kernel_log.jsonl").write_text(json.dumps({"package": "CP9", "outcome": "ERROR"}) + "\n", encoding="utf-8")
    got = LS.assess(tmp_path)
    assert got["attempts"] == 1 and got["judged"] == 1 and got["unrecorded_cycles"] == ["CP9"]
    lines = LS.summary_lines(got)
    assert lines[0].startswith("learning signal: 1 of 1") and any("left no lesson" in ln for ln in lines)
    assert LS.summary_lines(LS.signal_report([]))[0].endswith("(no attempts)")
