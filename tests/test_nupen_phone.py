"""scripts/nupen_phone.py: bearer auth, size cap, rate limit, LAN-bind default, reply shape (a fake talk model; no real voice)."""
from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("nupen_phone", Path(__file__).resolve().parents[1] / "scripts" / "nupen_phone.py")
P = importlib.util.module_from_spec(SPEC)
sys.modules["nupen_phone"] = P
SPEC.loader.exec_module(P)

TOKEN = "t" * 32


class FakeConv:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def reply(self, text: str) -> str:
        self.seen.append(text)
        return "I am fine. " + " ".join(["word"] * 100) + "\n[evidence: x]"


@pytest.fixture()
def server():
    core = P.Core(Path("."), make_conv=FakeConv)
    srv = P.make_server("127.0.0.1", 0, TOKEN, core, per_min=5)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, core
    srv.shutdown()
    srv.server_close()


def call(srv, path="/talk", body=None, token=TOKEN, raw=None, method="POST"):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}{path}", data=data, method=method)
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    for attempt in range(6):                       # retry only a failed connection setup (a loaded PC aborts it); bounded, no fixed timing
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:        # the server answered: that answer is final
            return e.code, json.loads(e.read())
        except (ConnectionError, urllib.error.URLError, socket.timeout):
            if attempt == 5:
                raise
            time.sleep(0.2 * (attempt + 1))
    raise AssertionError("unreachable")


def test_auth_required_and_wrong_token_rejected(server):
    srv, _ = server
    assert call(srv, body={"text": "hi"}, token=None)[0] == 401
    assert call(srv, body={"text": "hi"}, token="x" * 32)[0] == 401
    assert call(srv, "/health", token=None, method="GET")[0] == 401
    assert call(srv, "/health", method="GET")[0] == 200


def test_reply_shape_short_and_say_more(server):
    srv, _ = server
    code, d = call(srv, body={"text": "how are you", "device": "iphone"})
    assert code == 200 and set(d) == {"reply", "action", "actions", "end", "ms", "more"} and d["action"] is None and d["end"] is False
    assert "evidence" not in d["reply"] and len(d["reply"].split()) <= 62 and d["more"] is True
    code, d2 = call(srv, body={"text": "say more", "device": "iphone"})
    assert code == 200 and d2["reply"] and d2["reply"] != d["reply"]


def test_size_cap_and_bad_input(server):
    srv, _ = server
    assert call(srv, raw=b"x" * (P.MAX_BODY + 1))[0] == 413
    assert call(srv, raw=b"not json")[0] == 400
    assert call(srv, body={"text": "a" * (P.MAX_TEXT + 1)})[0] == 400


def test_rate_limit(monkeypatch):
    now = [1000.0]                                  # frozen clock: the one-minute window cannot expire however slow the machine is
    real = P.RateLimit
    monkeypatch.setattr(P, "RateLimit", lambda per_min=P.RATE_PER_MIN: real(per_min, clock=lambda: now[0]))
    core = P.Core(Path("."), make_conv=FakeConv)
    srv = P.make_server("127.0.0.1", 0, TOKEN, core, per_min=5)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        codes = [call(srv, body={"text": "hi"})[0] for _ in range(7)]
        assert codes[:5] == [200] * 5 and codes[5:] == [429, 429]
        now[0] += 61                                # and the window does reopen
        assert call(srv, body={"text": "hi"})[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_devices_have_own_memory(server):
    srv, core = server
    call(srv, body={"text": "a", "device": "one"})
    call(srv, body={"text": "b", "device": "two"})
    assert set(core.convs) == {"one", "two"}


def test_low_ram_refuses_voice():
    core = P.Core(Path("."), ram_free=lambda: 3.0)
    code, d = core.talk("d", "hi")
    assert code == 503 and "memory" in d["reply"]


def test_idle_unload():
    t = [0.0]

    class V:
        closed = False

        def close(self):
            V.closed = True
    core = P.Core(Path("."), idle_min=1, make_conv=FakeConv, clock=lambda: t[0])
    core.voice = V()
    t[0] = 30
    assert not core.reap()
    t[0] = 100
    assert core.reap() and V.closed and core.voice is None


def test_lan_bind_default_and_all_interfaces_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("CREATOR_RUNTIME", str(tmp_path))
    monkeypatch.delenv("NUPEN_RUNTIME", raising=False)
    # missing token file: refuses
    with pytest.raises(SystemExit):
        P.main([])
    P.make_token(tmp_path / "phone")
    assert P.load_token(tmp_path / "phone")
    with pytest.raises(SystemExit):
        P.main(["--host", "0.0.0.0"])
    seen = {}
    monkeypatch.setattr(P, "lan_ip", lambda: "lan-host")
    monkeypatch.setattr(P, "tailscale_ips", lambda: [])

    def fake_make(host, port, *a, **k):
        seen["host"], seen["port"] = host, port
        raise KeyboardInterrupt
    monkeypatch.setattr(P, "make_server", fake_make)
    with pytest.raises(KeyboardInterrupt):
        P.main([])
    assert seen == {"host": "lan-host", "port": 8765}


def test_firewall_command_is_lan_only():
    s = P.FIREWALL.format(port=8765, cgnat=P.CGNAT)
    assert "-RemoteAddress LocalSubnet," + P.CGNAT in s and "8765" in s


def test_persona_default_butler_no_real_names(tmp_path):
    t = P.persona_text(tmp_path)
    assert "British" in t and "sir" in t and "jarvis" not in t.lower()
    (tmp_path / "persona.json").write_text('{"custom": "Be terse."}', encoding="utf-8")
    assert P.persona_text(tmp_path) == "Be terse."
    m = [{"role": "system", "content": "You are Nupen, x"}, {"role": "user", "content": "q"}]
    assert P.with_persona(m, "S")[0]["content"].endswith("Style: S") and m[0]["content"] == "You are Nupen, x"
    j = [{"role": "system", "content": "You route"}]
    assert P.with_persona(j, "S") == j


def test_talk_audio_stub_is_501(server):
    srv, _ = server
    assert call(srv, "/talk_audio", body={"text": "hi"})[0] == 501


# ---- phone actions, goodbye, unknown requests (code-parsed allow-list) ----
NOW = datetime(2026, 10, 5, 10, 0)             # a Monday, 10:00


def act(text):
    r = P.interpret(text, NOW)
    assert r is not None, text
    return r


@pytest.mark.parametrize("text,want", [
    ("set a timer for five minutes", {"type": "timer", "minutes": 5}),
    ("Timer for 10 minutes", {"type": "timer", "minutes": 10}),
    ("can you start a 20 minute timer please", {"type": "timer", "minutes": 20}),
    ("set a timer for thirty seconds", {"type": "timer", "seconds": 30}),
    ("set a timer for an hour and a half", {"type": "timer", "minutes": 90}),
    ("set a timer for half an hour", {"type": "timer", "minutes": 30}),
])
def test_timer(text, want):
    assert act(text)[1] == want
    assert act("set a timer for five minutes")[0] == "Timer set for five minutes, sir."


@pytest.mark.parametrize("text,when", [
    ("set an alarm for 7 am", "07:00"), ("set alarm for 6:30 pm", "18:30"), ("wake me up at 7:30", "07:30"),
    ("wake me at half past six", "06:30"), ("set an alarm for seven thirty am", "07:30"),
])
def test_alarm(text, when):
    assert act(text)[1] == {"type": "alarm", "time": when}


def test_alarm_without_time_asks():
    r, a, e = act("set an alarm")
    assert a is None and "time" in r and e is False


@pytest.mark.parametrize("text,want", [
    ("remind me to call the dentist tomorrow at 9 am", {"type": "reminder", "text": "call the dentist", "when": "2026-10-06T09:00"}),
    ("remind me to buy milk", {"type": "reminder", "text": "buy milk"}),
    ("remind me in ten minutes to take the washing out", {"type": "reminder", "text": "take the washing out", "when": "2026-10-05T10:10"}),
    ("set a reminder to ring mum at 5", {"type": "reminder", "text": "ring mum", "when": "2026-10-05T17:00"}),
    ("add pay the rent to my reminders", {"type": "reminder", "text": "pay the rent"}),
])
def test_reminder(text, want):
    assert act(text)[1] == want


@pytest.mark.parametrize("text,want", [
    ("make a note that the boiler code is 4412", {"type": "note", "text": "The boiler code is 4412"}),
    ("note down buy stamps", {"type": "note", "text": "Buy stamps"}),
    ("take a note: ideas for the shed", {"type": "note", "text": "Ideas for the shed"}),
    ("jot down call the plumber", {"type": "note", "text": "Call the plumber"}),
])
def test_note(text, want):
    assert act(text)[1] == want


@pytest.mark.parametrize("text,want", [
    ("add lunch with Sam to my calendar tomorrow at 1 pm", {"type": "calendar", "title": "Lunch with Sam", "start": "2026-10-06T13:00"}),
    ("schedule a meeting with Anna on friday at 3 pm for an hour",
     {"type": "calendar", "title": "Meeting with Anna", "start": "2026-10-09T15:00", "duration": 60}),
    ("create an event called dentist tomorrow at 10 am for 45 minutes",
     {"type": "calendar", "title": "Dentist", "start": "2026-10-06T10:00", "duration": 45}),
    ("schedule dentist today at 4 pm", {"type": "calendar", "title": "Dentist", "start": "2026-10-05T16:00"}),
])
def test_calendar(text, want):
    assert act(text)[1] == want


def test_calendar_without_time_asks():
    r, a, _ = act("schedule a meeting with Anna")
    assert a is None and "time" in r


@pytest.mark.parametrize("text,to,msg", [
    ("text mum I'll be late", "mum", "I'll be late"),
    ("send a message to Sarah saying dinner is ready", "Sarah", "Dinner is ready"),
    ("message my wife that I am on my way", "wife", "I am on my way"),
    ("send a text to John Smith: see you at eight", "John Smith", "See you at eight"),
])
def test_message(text, to, msg):
    a = act(text)[1]
    assert a["type"] == "message" and a["to"] == to and a["text"].lower() == msg.lower() and "confirm" in act(text)[0]


@pytest.mark.parametrize("text,to", [("call mum", "mum"), ("phone my brother", "brother"), ("ring the dentist", "dentist"), ("give Sarah a call", "Sarah")])
def test_call(text, to):
    assert act(text)[1] == {"type": "call", "to": to}                      # a name: no url, the Shortcut looks the contact up


@pytest.mark.parametrize("text,q", [("play some jazz", "jazz"), ("play Bohemian Rhapsody by Queen", "Bohemian Rhapsody by Queen"),
                                    ("put on the beatles", "the beatles"), ("play songs by Adele", "Adele")])
def test_music(text, q):
    a = act(text)[1]
    assert a["type"] == "music" and a["query"] == q and a["app"] == "Apple Music"
    assert a["url"].startswith("music://music.apple.com/search?term=")


@pytest.mark.parametrize("text,app,url", [
    ("play Adele on Spotify", "Spotify", "spotify:search:Adele"),
    ("play some jazz on spotify", "Spotify", "spotify:search:jazz"),
    ("play cat videos on youtube", "YouTube", "https://www.youtube.com/results?search_query=cat%20videos"),
    ("play Hello by Adele on Apple Music", "Apple Music", "music://music.apple.com/search?term=Hello%20by%20Adele"),
])
def test_music_app_and_url(text, app, url):
    a = act(text)[1]
    assert a["app"] == app and a["url"] == url


APP_URLS = {
    "open Spotify": ("Spotify", "spotify:"), "open YouTube": ("YouTube", "youtube://"), "open instagram": ("Instagram", "instagram://"),
    "launch WhatsApp": ("WhatsApp", "whatsapp://"), "open maps": ("Maps", "maps://"), "open the messages app": ("Messages", "sms:"),
    "open facetime": ("FaceTime", "facetime://"), "open mail": ("Mail", "mailto:"), "open apple music": ("Music", "music://"),
    "open photos": ("Photos", "photos-redirect://"), "open the calendar": ("Calendar", "calshow://"),
    "open reminders": ("Reminders", "x-apple-reminderkit://"), "open settings": ("Settings", "App-prefs:"), "open notes": ("Notes", "mobilenotes://"),
}


@pytest.mark.parametrize("text", list(APP_URLS))
def test_open_app_mapped_with_url(text):
    name, url = APP_URLS[text]
    r, a, e = act(text)
    assert a == {"type": "open_app", "name": name, "url": url} and r == f"Opening {name}, sir."


@pytest.mark.parametrize("text,name", [("open Safari", "Safari"), ("launch the camera app", "Camera"), ("open Tinder", "Tinder")])
def test_open_unknown_app_no_url_polite(text, name):
    r, a, e = act(text)
    assert a == {"type": "open_app", "name": name} and "url" not in a and "do not know" in r and r.endswith("sir.")


def test_app_table_only_known_schemes():
    assert all(u.endswith((":", "://")) for _, u in P.APPS.values())
    assert not any("camera" in k for k in P.APPS)        # no verified scheme: never guessed


@pytest.mark.parametrize("text,to", [("directions to the airport", "the airport"), ("navigate to Heathrow", "Heathrow"),
                                     ("take me to the train station", "the train station"), ("how do I get to Bath", "Bath")])
def test_directions(text, to):
    a = act(text)[1]
    assert a["type"] == "directions" and a["to"] == to
    assert a["url"] == "maps://?daddr=" + P.quote(to, safe="") + "&dirflg=d"


def test_call_and_message_urls_only_for_numbers():
    assert act("call 555 0100")[1] == {"type": "call", "to": "555 0100", "url": "tel:5550100"}
    a = act("send a message to 07700900123 saying I am outside")[1]
    assert a["url"] == "sms:07700900123&body=I%20am%20outside"
    assert "url" not in act("call mum")[1] and "url" not in act("text mum hello")[1]


@pytest.mark.parametrize("text,state", [("turn on the flashlight", "on"), ("torch on", "on"), ("turn off the torch", "off"), ("flashlight off", "off")])
def test_flashlight(text, state):
    assert act(text)[1] == {"type": "flashlight", "state": state}


@pytest.mark.parametrize("text,name,state", [
    ("turn on do not disturb", "Do Not Disturb", "on"), ("turn off do not disturb", "Do Not Disturb", "off"),
    ("switch on sleep mode", "Sleep", "on"), ("enable work focus", "Work", "on"), ("disable the driving focus", "Driving", "off")])
def test_focus(text, name, state):
    assert act(text)[1] == {"type": "focus", "name": name, "state": state}


@pytest.mark.parametrize("text,dev,state", [
    ("turn on the living room lights", "living room lights", "on"), ("turn off the kettle", "kettle", "off"),
    ("switch the bedroom lamp off", "bedroom lamp", "off"), ("turn the hall light on", "hall light", "on")])
def test_home(text, dev, state):
    assert act(text)[1] == {"type": "home", "device": dev, "state": state}


@pytest.mark.parametrize("text", ["goodbye", "thanks, that's all", "that will be all, thank you", "no thanks", "thank you", "bye", "ok see you later"])
def test_end_flag(text):
    r, a, e = act(text)
    assert e is True and a is None and "Goodbye" in r


def test_end_with_action_and_not_without():
    r, a, e = act("set a timer for five minutes, that's all thanks")
    assert a == {"type": "timer", "minutes": 5} and e is True
    assert act("set a timer for five minutes")[2] is False
    assert act("please play some jazz")[2] is False


@pytest.mark.parametrize("text", ["how are you", "what time is it", "tell me a joke", "what is the weather in Paris",
                                  "how long does a timer on an oven take"])
def test_conversation_goes_to_the_model(text):
    assert P.interpret(text, NOW) is None


@pytest.mark.parametrize("text", ["book me a flight to Rome", "order a pizza", "turn off the alarm", "play a game", "email my boss", "set a timer"])
def test_unknown_requests_no_action_polite(text):
    r, a, e = act(text)
    assert a is None and e is False and r.endswith(("sir.", "sir?"))


@pytest.mark.parametrize("text", ["shut down the computer", "restart my pc", "turn off the laptop", "open the nupen server", "delete the kernel"])
def test_never_touches_the_pc(text):
    r, a, e = act(text)
    assert a is None and "computer" in r


def test_actions_only_from_allow_list_and_http_shape(server):
    srv, _ = server
    code, d = call(srv, body={"text": "set a timer for five minutes", "device": "iphone"})
    assert code == 200 and d["action"] == {"type": "timer", "minutes": 5} and d["reply"] == "Timer set for five minutes, sir." and d["end"] is False
    assert isinstance(d["ms"], int)
    code, d = call(srv, body={"text": "goodbye"})
    assert code == 200 and d["end"] is True and d["action"] is None
    allowed = {"timer", "alarm", "reminder", "note", "calendar", "message", "call", "music", "open_app", "directions", "flashlight", "focus", "home"}
    for t in ("set an alarm for 7", "text mum hello", "call dad", "play jazz", "open maps", "turn on the torch", "turn on the hall light"):
        assert P.interpret(t, NOW)[1]["type"] in allowed


def test_auth_rules_unchanged_for_actions(server):
    srv, core = server
    assert call(srv, body={"text": "set a timer for five minutes"}, token=None)[0] == 401
    assert call(srv, body={"text": "set a timer for five minutes"}, token="x" * 32)[0] == 401
    assert call(srv, "/health", token=None, method="GET")[0] == 401
    assert call(srv, raw=b"x" * (P.MAX_BODY + 1))[0] == 413
    assert core.convs == {}                       # unauthenticated or action requests never even build a conversation


# ---- routing: chat vs about-Nupen, history, leftovers, conversation log ----
DUMP = "[doc:creator/resources.py] " + " ".join(["word"] * 120) + " [evidence: x]"
PROBES = ["how's your day going", "tell me a joke", "what should I make for dinner", "help me with my homework"]


class Stub:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def __call__(self, messages, more=False):
        self.calls.append(messages)
        return "Quite well, sir. [doc:x] Thank you for asking." if messages[0]["content"].startswith("You are Nupen, the owner") else "We are training, sir."


class DumpConv:
    def reply(self, text):
        return DUMP


def make_core(tmp_path=None, **kw):
    stub = Stub()
    core = P.Core(Path("."), make_conv=DumpConv, chat=stub, log_path=(tmp_path / "c.jsonl") if tmp_path else None, now=lambda: NOW, **kw)
    return core, stub


def test_probe_questions_never_dump_docs_or_leave_leftovers():
    core, stub = make_core()
    for q in PROBES:
        code, d = core.talk("iphone", q)
        assert code == 200 and "[doc" not in d["reply"] and "evidence" not in d["reply"] and d["more"] is False, q
    assert core.convs == {} and core.rest["iphone"] == ""          # general chat never touches the doc/status conversation
    assert len(stub.calls) == 4


def test_about_nupen_is_rewritten_never_raw():
    core, stub = make_core()
    for q in ("what are you working on", "give me a status report", "who are you", "what are your plans"):
        code, d = core.talk("iphone", q)
        assert d["reply"] == "We are training, sir." and "[doc" not in d["reply"], q
    assert stub.calls[-1][0]["content"].startswith("You are Nupen. Answer")
    assert "Facts:" in stub.calls[-1][1]["content"] and "[doc" not in stub.calls[-1][1]["content"]


def test_leftover_only_on_explicit_more():
    class Long(Stub):
        def __call__(self, messages, more=False):
            return " ".join(["word"] * 100) + "."
    core = P.Core(Path("."), chat=Long(), now=lambda: NOW)
    core.make_conv = None
    code, d1 = core.talk("d", "tell me about the moon")
    assert d1["more"] is True and core.rest["d"]
    code, d2 = core.talk("d", "tell me a joke")                      # a new question must not continue the old answer
    assert core.rest["d"] and d2["reply"].split()[:3] == ["word"] * 3 and len(d2["reply"].split()) <= 62
    assert core.talk("d", "say more")[1]["reply"]
    core.rest["d"] = "leftover words here"
    core.talk("d", "what is two plus two")
    assert core.rest["d"] != "leftover words here"
    core.rest["d"] = "leftover words here"
    assert core.talk("d", "go on")[1]["reply"] == "leftover words here"


def test_history_used_per_device_and_capped():
    core, stub = make_core()
    for i in range(10):
        core.talk("a", f"question number {i}")
    last = stub.calls[-1]
    assert last[0]["role"] == "system" and last[-1]["content"] == "question number 9"
    assert len(last) == 1 + 2 * P.HIST_TURNS + 1 and last[1]["content"] == "question number 3"
    core.talk("b", "hello there")
    assert [m["content"] for m in stub.calls[-1][1:]] == ["hello there"]


def test_conversation_log_written_and_rotated(tmp_path):
    core, _ = make_core(tmp_path)
    core.talk("iphone", "how's your day going")
    core.talk("iphone", "set a timer for five minutes")
    rows = [json.loads(x) for x in (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["intent"] for r in rows] == ["chat", "action"]
    assert rows[0]["in"] == "how's your day going" and rows[0]["device"] == "iphone" and rows[0]["reply"] and isinstance(rows[0]["ms"], int)
    assert rows[1]["action"] == {"type": "timer", "minutes": 5} and rows[0]["t"]
    (tmp_path / "c.jsonl").write_text("x" * (P.LOG_MAX + 1), encoding="utf-8")
    core.talk("iphone", "hello")
    assert (tmp_path / "c.jsonl.1").is_file() and (tmp_path / "c.jsonl").stat().st_size < 10000


def test_add_sir_only_when_missing_and_once():
    assert P.add_sir("It is raining.") == "It is raining, sir."
    assert P.add_sir("Very good, sir.") == "Very good, sir."
    assert P.add_sir("Is that so?") == "Is that so, sir?"
    assert P.add_sir("Why did the chicken cross? To arrive.") == "Why did the chicken cross? To arrive, sir."
    assert P.add_sir(" ".join(["word"] * 60) + ".").count("sir") == 0     # long 'say more' text is left alone


def test_chat_reply_gets_sir_and_varied_seed_flag():
    seen = []

    def chat(messages, more):
        seen.append(messages[0]["content"])
        return "It is a lovely day."
    core = P.Core(Path("."), chat=chat)
    assert core.talk("d", "what a day")[1]["reply"] == "It is a lovely day, sir."
    assert "sir" in seen[0] and "two short" in seen[0]


def test_two_sentences_cap():
    assert P.two_sentences("One. Two! Three? Four.") == "One. Two!"
    assert P.two_sentences("One. Two.") == "One. Two."
    assert P.two_sentences("No stop") == "No stop"


def test_clean_reply_no_space_before_punctuation():
    assert P.clean_reply("Hamlet was written by *Shakespeare* , sir.") == "Hamlet was written by Shakespeare, sir."
