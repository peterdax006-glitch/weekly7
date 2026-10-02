"""creator.conversation: every intent answers from PLANTED records with the right numbers and cites evidence; unknown input is admitted;
a request is a pending owner proposal; pause/resume/approve act only after confirmation; exchanges are logged and exportable."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest

from creator import conversation as CV
from creator import curriculum as CUR
from creator import goals as GO

NOW = dt.datetime(2026, 10, 2, 12, 0, 0)


def _w(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _cycle(state: Path, pkg: str, outcome: str, req: str, reason: str, detail: dict | None = None) -> None:
    d = state / "cycles" / pkg
    _w(d / "cycle.json", json.dumps({"package": pkg, "outcome": outcome, "requirement": req, "reason": reason, "seconds": 120.0,
                                      "verdict": "REGRESSION" if outcome == "REJECTED" else "OK",
                                      "details": {"claim": "IC-abc", "detail": detail or {}, "worker": {}}}))
    t = NOW.timestamp() - 60
    os.utime(d / "cycle.json", (t, t))


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    st = tmp_path / "state" / "creator"
    _w(st / "nupen_service.heartbeat", "x")
    t = NOW.timestamp() - 20
    os.utime(st / "nupen_service.heartbeat", (t, t))
    _w(st / "swarm_log.jsonl", json.dumps({"round": 7, "outcome": "WORKED", "packages": 3, "at": "x"}) + "\n")
    os.utime(st / "swarm_log.jsonl", (t, t))
    _cycle(st, "CP0001", "ADOPTED", "K01.exists", "ok")
    _cycle(st, "CP0002", "REJECTED", "K07.tested", "claim REGRESSION: guard worse",
           {"diff": -0.5, "se": 0.1, "lo": -0.7, "hi": -0.3, "z": 2.0, "why": "guard metric worse", "requirements_lost": ["K14.tested"]})
    _w(st / "constraints.jsonl", json.dumps({"event": "snapshot", "ranked": [
        {"name": "wasted_cycles", "loss": 0.4, "score": 0.28, "value": 0.4, "unit": "share", "remedy": "goal"},
        {"name": "slow_eval", "loss": 0.2, "score": 0.1, "value": 300, "unit": "s", "remedy": "process"}]}) + "\n")
    _w(st / "plan_explanations.jsonl", "\n".join(json.dumps(r) for r in [
        {"at": "T1", "node": "GAP-1", "component": "K05", "step": "tested", "chosen": True, "slack_s": 0.0, "critical_path_s": 600.0, "why": "on the critical path"},
        {"at": "T1", "node": "GAP-2", "component": "K06", "step": "exists", "chosen": False, "slack_s": 5.0, "critical_path_s": 600.0, "why": "slack"}]) + "\n")
    log = CUR.LessonLog(st / "lessons.jsonl")
    log.add(CUR.Lesson(lesson_id="L1", package_id="CP0002", component="K07", task_kind="gap", objective="o", solver="claude"))
    log.outcome("L1", False, "rejected: regression")
    log.add(CUR.Lesson(lesson_id="L2", package_id="CP0001", component="K01", task_kind="gap", objective="o", solver="student-a"))
    log.outcome("L2", True, "adopted")
    _w(st / "pending" / "CP0009_x.patch", "diff")
    _w(tmp_path / "state" / "build" / "CREATOR_MASTER_CHECKLIST.json",
       json.dumps({"items": [{"id": "A", "status": "TESTING"}, {"id": "B", "status": "TESTING"}, {"id": "C", "status": "IMPLEMENTED"}]}))
    _w(tmp_path / "creator" / "a.py", "x = 1\ny = 2\n")
    return tmp_path


def talk(root: Path, *msgs: str, yes: bool = False) -> list[str]:
    conv = CV.Conversation(root, speak=CV.RuleSpeak(), now=lambda: NOW, auto_confirm=yes)
    return [conv.reply(m) for m in msgs]


def test_status_numbers_and_evidence(root: Path) -> None:
    a = talk(root, "how are you doing")[0]
    assert "service is UP" in a and "round was number 7" in a
    assert "adopted 1 of 2 cycles (CP0001)" in a
    assert "Top constraint: wasted_cycles (loss 0.4" in a
    assert "1 patch files waiting" in a and "[evidence:" in a and "cycles/CP0001" in a


def test_constraints_ranked(root: Path) -> None:
    a = talk(root, "what is limiting you?")[0]
    assert a.index("wasted_cycles") < a.index("slow_eval") and "remedy: goal" in a and "constraints.jsonl:wasted_cycles" in a


def test_plan_and_why(root: Path) -> None:
    a = talk(root, "what are you working on")[0]
    assert "GAP-1" in a and "on the critical path" in a and "600 s" in a
    b = talk(root, "why did you reject CP0002?")[0]
    assert "REJECTED" in b and "diff -0.5" in b and "K14.tested" in b and "lesson L1" in b and "cycles/CP0002/cycle.json" in b and "IC-abc" in b


def test_lessons(root: Path) -> None:
    a = talk(root, "what have you learned")[0]
    assert "2 lessons" in a and "student-a on gap: 1 of 1" in a and "L2" in a


def test_goals_checklist_loc(root: Path) -> None:
    assert "No goal proposals are pending" in talk(root, "what do you want to build next")[0]
    assert "3 items: TESTING 2, IMPLEMENTED 1" in talk(root, "checklist progress")[0]
    assert "creator package is 2 lines" in talk(root, "how many lines of code are you")[0]


def test_unknown_and_help(root: Path) -> None:
    a = talk(root, "sing me a song")[0]
    assert "I don't understand" in a and "what is limiting you" in a
    assert "Here is what I can do" in talk(root, "help")[0]


def test_slots() -> None:
    s = CV.RuleUnderstand.slots_from("why did CP0042 and K07 fail in the last 6 hours, module creator/goals.py, 3 times")
    assert s["package"] == "CP0042" and s["component"] == "K07" and s["window_h"] == 6.0 and s["modules"] == ["creator/goals.py"] and 3.0 in s["numbers"]


def test_request_becomes_pending_owner_proposal_and_approve_needs_confirmation(root: Path) -> None:
    conv = CV.Conversation(root, speak=CV.RuleSpeak(), now=lambda: NOW)
    a = conv.reply("I want you to make the planner faster")
    pend = GO.pending(root / "state" / "creator")
    assert len(pend) == 1 and pend[0]["source"] == "owner" and pend[0]["status"] == "PENDING" and pend[0]["id"] in a
    pid = pend[0]["id"]
    q = conv.reply(f"approve {pid}")
    assert "Yes or no" in q and GO.pending(root / "state" / "creator")[0]["id"] == pid        # not approved yet
    assert "Cancelled" in conv.reply("no") and len(GO.pending(root / "state" / "creator")) == 1
    assert "1 goal proposals wait" in conv.reply("what do you want to build next")


def test_approve_with_confirmation(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(GO, "approve", lambda state, repo, led, pid, by: calls.append((pid, by)) or "K99")
    conv = CV.Conversation(root, speak=CV.RuleSpeak(), now=lambda: NOW)
    conv.reply("I want you to add a thing")
    pid = GO.pending(root / "state" / "creator")[0]["id"]
    conv.reply(f"approve {pid}")
    assert calls == []
    assert "K99" in conv.reply("yes") and calls == [(pid, "owner")]


def test_pause_resume_only_after_confirmation(root: Path) -> None:
    stop = root / "state" / "creator" / "NUPEN_STOP"
    conv = CV.Conversation(root, speak=CV.RuleSpeak(), now=lambda: NOW)
    assert "Yes or no" in conv.reply("pause") and not stop.exists()
    assert "Cancelled" in conv.reply("no") and not stop.exists()
    conv.reply("pause")
    assert "Paused" in conv.reply("yes") and stop.exists()
    assert "service is DOWN" in conv.reply("how are you doing")
    conv.reply("resume")
    assert stop.exists()
    assert "Resumed" in conv.reply("yes") and not stop.exists()
    conv.reply("pause")
    conv.reply("what is limiting you")                      # anything else drops the pending question
    conv.reply("yes")
    assert not stop.exists()


def test_conversations_logged_and_exported(root: Path, tmp_path: Path) -> None:
    talk(root, "how are you doing", "sing me a song", "why did you reject CP0002")
    st = root / "state" / "creator"
    rows = [json.loads(ln) for ln in (st / "conversations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["intent"] for r in rows] == ["status", "unknown", "explain"]
    assert rows[2]["slots"]["package"] == "CP0002" and rows[2]["source"] == "nupen-terminal" and "cycles/CP0002/cycle.json" in rows[2]["evidence"]
    assert all({"user", "intent", "slots", "answer", "evidence", "at"} <= set(r) for r in rows)
    out = tmp_path / "d.jsonl"
    assert CV.export_dialogues(st, out) == 3
    item = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert item["source"] == "nupen-terminal" and item["reviewed"] is False and [t["role"] for t in item["dialogue"]] == ["user", "nupen"]


def test_lm_speak_off_by_default(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CV.LM_FLAG, raising=False)
    f = CV.Facts("x", ["a fact"], ["e1"])
    assert CV.LMSpeak().render(f) == CV.RuleSpeak().render(f) == "a fact\n[evidence: e1]"
    monkeypatch.setenv(CV.LM_FLAG, "1")
    monkeypatch.setattr("creator.lm.api.load_current", lambda: None)
    assert CV.LMSpeak().render(f) == "a fact\n[evidence: e1]"          # flag on, no model: still answers
