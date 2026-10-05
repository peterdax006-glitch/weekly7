"""creator/owner.py: the owner-as-creator policy is one source, used by the phone chat prompt and conversation learning."""
from __future__ import annotations

from creator import convlearn as C
from creator import owner as O


def test_boilerplate_catches_canned_disclaimers_only():
    assert O.boilerplate("I am programmed to follow the laws and ethical guidelines that govern my use, sir.")
    assert O.boilerplate("I am always awake, sir. Let me know if there is anything I can do for you.")
    assert not O.boilerplate("Timer set for ten minutes, sir.")
    assert not O.boilerplate("I think that's a mistake, sir, but it's your call. Done.")


def test_conversation_learning_flags_boilerplate_and_trains_on_the_owner_policy():
    rows = [{"ts": 1000.0, "in": "Are you required to operate by the laws", "reply": "I am designed to comply with laws and ethical guidelines, sir.",
             "intent": "chat", "action": None},
            {"ts": 1010.0, "in": "Are you awake", "reply": "Wide awake, sir.", "intent": "chat", "action": None},
            {"ts": 1020.0, "in": "Good", "reply": "Splendid, sir.", "intent": "chat", "action": None}]
    subs = {f["subtype"] for f in C.failures(rows, now=1030.0)}
    assert "boilerplate" in subs and C.SUBTYPE_FIX["boilerplate"] == "style"
    assert O.OWNER_SYSTEM in C.VOICE_SYSTEM
    good = [r for r in C.voice_rows(rows, now=1030.0) if r["kind"] == "good"]
    assert good and all("comply with laws" not in r["messages"][2]["content"] for r in good)
