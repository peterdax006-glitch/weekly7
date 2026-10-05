"""Conversation learning: each detector on a synthetic log, the candidate shape, the backlog ranking, the voice rows, the engine hook."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from creator import constraints as CON
from creator import convlearn as CL

T0 = time.mktime((2026, 10, 5, 10, 0, 0, 0, 0, -1))


def turn(i: float, text: str, reply: str = "Certainly, sir.", intent: str = "chat", action: Optional[dict] = None, device: str = "iphone") -> dict[str, Any]:
    return {"ts": T0 + i * 20, "t": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(T0 + i * 20)), "device": device, "in": text, "reply": reply,
            "intent": intent, "action": action, "ms": 100}


def subs(sess: list[dict[str, Any]], now: Optional[float] = None) -> set[str]:
    return {f["subtype"] for f in CL.failures(sess, now if now is not None else T0 + 10)}


def test_repeat_within_two_turns() -> None:
    s = [turn(0, "what is the weather in town"), turn(1, "ok"), turn(2, "what's the weather in town?")]
    assert "repeat" in subs(s)
    far = [turn(0, "what is the weather in town"), turn(1, "a"), turn(2, "b"), turn(3, "c"), turn(4, "what is the weather in town")]
    assert "repeat" not in subs(far)


def test_correction_blames_previous_answer() -> None:
    s = [turn(0, "play some jazz", "Here is a poem."), turn(1, "that's not what I asked")]
    f = [x for x in CL.detect_correction(s)]
    assert len(f) == 1 and f[0]["turn"] == 0
    for phrase in ("this is stupid", "no", "wrong", "you didn't do it"):
        assert CL.detect_correction([turn(0, "x"), turn(1, phrase)])
    assert not CL.detect_correction([turn(0, "x"), turn(1, "nothing else, thanks")])
    assert not CL.detect_correction([turn(0, "x"), turn(1, "no problem at all")])


def test_unsupported_and_failed_actions() -> None:
    assert "unsupported" in subs([turn(0, "order a pizza", "I cannot do that yet, sir.")])
    assert "no_url" in subs([turn(0, "open the news", "Opening.", "action", {"type": "open"})])
    assert "no_url" not in subs([turn(0, "set a timer", "Done.", "action", {"type": "timer", "minutes": 5})])
    assert "no_url" not in subs([turn(0, "open the news", "Opening.", "action", {"type": "open", "url": "https://x"})])
    assert "action_failed" in subs([turn(0, "open the news", "Opening."), turn(1, "it didn't work")])


def test_leaked_internals() -> None:
    for bad in ("According to [doc:12] it is so.", "See C:\\Users\\x\\notes.txt", "Look in /Users/x/b", '{"answer": 42}', "edit creator/kernel.py"):
        assert "leak" in subs([turn(0, "q", bad)]), bad
    assert "leak" not in subs([turn(0, "q", "All is well, sir.")])


def test_long_answer_to_short_question() -> None:
    assert "long" in subs([turn(0, "why is the sky blue", "word " * 200)])
    assert "long" not in subs([turn(0, "explain in detail " + "x " * 20, "word " * 200)])
    assert "long" not in subs([turn(0, "hi", "short reply")])


def test_abandon_only_after_quiet_and_bad_last_answer() -> None:
    s = [turn(0, "order a pizza", "I cannot do that yet.")]
    assert "abandon" in subs(s, T0 + 7200)
    assert "abandon" not in subs(s, T0 + 60)                       # session still open
    assert "abandon" not in subs([turn(0, "hello", "Good day, sir.")], T0 + 7200)


def test_sessions_split_by_gap_and_device_and_synthetic_dropped(tmp_path: Path) -> None:
    rows = [turn(0, "a"), turn(1, "b"), turn(500, "c"), turn(0, "p", device="probe"), turn(1, "w", device="warmup")]
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps({k: v for k, v in r.items() if k != "ts"}) for r in rows) + "\nnot json\n", encoding="utf-8")
    loaded = CL.load_log(p)
    assert [r["in"] for r in loaded] == ["a", "b", "c"]
    assert [len(x) for x in CL.sessions(loaded)] == [2, 1]
    assert CL.load_log(tmp_path / "missing.jsonl") == []


def test_candidate_shape_and_anonymised_evidence() -> None:
    s = [turn(0, "call 555 1234 at bob@example.com via https://x.y/z", "I cannot do that yet.")]
    cs = CL.candidates(CL.failures(s, T0 + 10), days=2.0)
    c = next(x for x in cs if x["subtype"] == "unsupported")
    assert c["kind"] == "conv_failure" and c["key"] == "conv_failure:unsupported" and c["fix_kind"] == "new_action" == c["evidence"]["fix_kind"]
    assert c["f_per_day"] == 0.5 and c["risk"] == "low"
    ex = c["evidence"]["excerpt"]
    assert "555" not in ex and "example.com" not in ex and "http" not in ex and "#" in ex
    assert {"f_per_day", "c0", "c1", "p", "fix", "risk", "build_s", "key", "kind", "evidence"} <= set(c)
    assert all(x["fix_kind"] in CL.FIX_KINDS for x in cs)
    v = CON.value(c)                                               # the engine accepts it as-is
    assert v["saving_day"] > 0


def test_backlog_ranked_by_frequency(tmp_path: Path) -> None:
    s = [turn(0, "Order a pizza!", "I cannot do that yet."), turn(1, "book a taxi", "I cannot do that yet."), turn(2, "order a pizza", "I cannot do that yet."),
         turn(3, "ORDER A PIZZA.", "I cannot do that yet."), turn(4, "book a taxi", "I cannot do that yet."), turn(5, "tell me a joke")]
    items = CL.backlog(CL.failures(s, T0 + 10))
    assert [(i["request"], i["count"]) for i in items] == [("order a pizza", 3), ("book a taxi", 2)]
    p = tmp_path / "phone" / "capability_backlog.json"
    CL.write_backlog(p, items)
    CL.write_backlog(p, items)                                     # same window again: no double counting
    CL.write_backlog(p, [{"request": "book a taxi", "count": 9, "subtypes": ["unsupported"], "last": 1.0}])
    got = json.loads(p.read_text(encoding="utf-8"))["items"]
    assert [(i["request"], i["count"]) for i in got] == [("book a taxi", 9), ("order a pizza", 3)]


def test_voice_rows_good_and_corrected(tmp_path: Path) -> None:
    s = [turn(0, "what time is it", "Ten o'clock, sir."), turn(1, "thanks, and the date", "Monday, the fifth."),
         turn(2, "tell me about my day", "I cannot do that yet."), turn(3, "that's not what I asked, tell me about my day please"),
         turn(4, "tell me about my day please", "You have two meetings, sir."), turn(5, "great")]
    rows = CL.voice_rows(s, T0 + 10)
    kinds = [r["kind"] for r in rows]
    assert kinds.count("corrected") == 1 and kinds.count("good") >= 2
    c = next(r for r in rows if r["kind"] == "corrected")
    assert c["messages"][-1]["content"] == "You have two meetings, sir." and c["rejected"] == "I cannot do that yet." and c["owner_only"] is True
    assert [m["role"] for m in c["messages"]] == ["system", "user", "assistant"]
    assert not any(r["messages"][-1]["content"] == "I cannot do that yet." for r in rows)      # a bad answer is never a target
    p = tmp_path / "voice" / "conv_rows.jsonl"
    assert CL.write_voice_rows(p, rows) == len(rows)
    assert CL.write_voice_rows(p, rows) == 0                       # idempotent


def test_engine_hook_and_no_log_is_noop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NUPEN_PHONE_DIR", raising=False)
    monkeypatch.delenv("NUPEN_VOICE_DIR", raising=False)
    state = tmp_path / "state"
    state.mkdir()
    assert CL.detect(state) == [] and CON.detect_all(state) == []
    phone = tmp_path / "phone"
    phone.mkdir()
    now = time.time()
    rows = [turn(0, "order a pizza", "I cannot do that yet.")]
    rows[0]["t"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - 7200))
    (phone / "conversations.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    cands = CON.detect_all(state, 7.0, now)
    assert {c["subtype"] for c in cands if c["kind"] == "conv_failure"} >= {"unsupported", "abandon"}
    assert json.loads((phone / "capability_backlog.json").read_text(encoding="utf-8"))["items"][0]["request"] == "order a pizza"
