"""Phone server: the two universal doors (action.url, action.shortcut) and the optional phone context."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("nupen_phone", Path(__file__).resolve().parents[1] / "scripts" / "nupen_phone.py")
P = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("nupen_phone_doors_copy", P)
SPEC.loader.exec_module(P)

NOW = datetime(2026, 10, 5, 10, 0)


def act(text):
    r = P.interpret(text, NOW)
    assert r is not None, text
    return r[0], P.route(r[1])


# ---- door 1: URL generation per kind ----
@pytest.mark.parametrize("text,url", [
    ("google best pizza", "https://www.google.com/search?q=best%20pizza"),
    ("search the web for rust lifetimes", "https://www.google.com/search?q=rust%20lifetimes"),
    ("duckduckgo open source maps", "https://duckduckgo.com/?q=open%20source%20maps"),
    ("search youtube for cat videos", "https://www.youtube.com/results?search_query=cat%20videos"),
    ("search spotify for Adele", "spotify:search:Adele"),
    ("search apple music for jazz", "music://music.apple.com/search?term=jazz"),
    ("find coffee near me", "maps://?q=coffee%20near%20me"),
    ("search the app store for chess", "itms-apps://search.itunes.apple.com/WebObjects/MZSearch.woa/wa/search?media=software&term=chess"),
    ("directions to the airport", "maps://?daddr=the%20airport&dirflg=d"),
    ("walking directions to the park", "maps://?daddr=the%20park&dirflg=w"),
    ("directions to the airport by transit", "maps://?daddr=the%20airport&dirflg=r"),
    ("how do i get to the museum on foot", "maps://?daddr=the%20museum&dirflg=w"),
    ("call 555 0100", "tel:5550100"),
    ("facetime 555 0100", "facetime:5550100"),
    ("facetime audio 555 0100", "facetime-audio:5550100"),
    ("send a message to 07700900123 saying hi there", "sms:07700900123&body=Hi%20there"),
    ("email bob@example.com about lunch saying see you at one", "mailto:bob@example.com?subject=Lunch&body=See%20you%20at%20one"),
    ("email bob@example.com", "mailto:bob@example.com"),
    ("open settings", "App-prefs:"), ("open calendar", "calshow://"), ("open photos", "photos-redirect://"),
    ("open notes", "mobilenotes://"), ("open reminders", "x-apple-reminderkit://"), ("open things", "things://"),
])
def test_url_per_kind(text, url):
    assert act(text)[1]["url"] == url


@pytest.mark.parametrize("text", ["open wifi settings", "search the app store for chess"])
def test_uncertain_links_are_marked(text):
    assert act(text)[1]["uncertain"] is True


def test_certain_links_not_marked():
    assert "uncertain" not in act("google best pizza")[1]


def test_unknown_kinds_get_no_guessed_url():
    assert "url" not in act("call mum")[1] and "url" not in act("open Camera")[1]


def test_every_url_scheme_is_in_the_allow_list():
    ok = ("https://", "spotify:", "music://", "maps://", "itms-apps://", "sms:", "tel:", "facetime:", "facetime-audio:", "mailto:", "App-prefs:")
    for t in ("google x", "search youtube for x", "search spotify for x", "directions to x", "call 5550100", "email a@b.co", "open wifi settings"):
        assert act(t)[1]["url"].startswith(ok)


# ---- door 2: shortcut routing per ability ----
@pytest.mark.parametrize("text,name,inp", [
    ("set a timer for five minutes", "Nupen Timer", "300"),
    ("set a timer for thirty seconds", "Nupen Timer", "30"),
    ("set an alarm for 7 am", "Nupen Alarm", "07:00"),
    ("make a note that the boiler code is 4412", "Nupen Note", "The boiler code is 4412"),
    ("turn on the torch", "Nupen Flashlight", "on"),
    ("turn off the flashlight", "Nupen Flashlight", "off"),
    ("turn on do not disturb", "Nupen Focus", '{"name": "Do Not Disturb", "state": "on"}'),
    ("turn off the living room lights", "Nupen Home Device", '{"device": "living room lights", "state": "off"}'),
    ("set volume to fifty percent", "Nupen Volume", "50"),
    ("turn the volume down", "Nupen Volume", "down"),
    ("set the brightness to 30", "Nupen Brightness", "30"),
    ("turn the brightness up", "Nupen Brightness", "up"),
    ("turn off wifi", "Nupen WiFi", "off"),
    ("turn on bluetooth", "Nupen Bluetooth", "on"),
    ("turn on low power mode", "Nupen Low Power Mode", "on"),
    ("switch to dark mode", "Nupen Dark Mode", "on"),
    ("switch to light mode", "Nupen Dark Mode", "off"),
    ("play jazz from my library", "Nupen Play Music", "jazz"),
    ("copy hello there to my clipboard", "Nupen Clipboard", "hello there"),
    ("text mum I'll be late", "Nupen Message", '{"to": "mum", "text": "I\'ll be late"}'),
    ("call mum", "Nupen Call", '{"to": "mum"}'),
    ("facetime mum", "Nupen Call", '{"to": "mum", "via": "facetime"}'),
])
def test_shortcut_routing(text, name, inp):
    a = act(text)[1]
    assert "shortcut" not in a and a["type"]                       # one shortcut: the action is self-contained, no helper door
    assert P.shortcut_for(a) == {"name": name, "input": inp} and name in P.HELPERS     # the legacy mapping still agrees with the inline params


def test_reminder_and_calendar_inputs_are_json():
    a = act("remind me to call the dentist tomorrow at nine")[1]
    assert a["type"] == "reminder" and a["text"] == "call the dentist" and "shortcut" not in a
    sc = P.shortcut_for(a)
    assert sc["name"] == "Nupen Reminder" and json.loads(sc["input"]) == {"text": "call the dentist", "when": "2026-10-06T09:00"}
    a = act("schedule dentist today at 4 pm for 45 minutes")[1]
    assert a["type"] == "calendar" and a["title"] == "Dentist" and a["duration"] == 45 and "shortcut" not in a
    sc = P.shortcut_for(a)
    assert sc["name"] == "Nupen Calendar Event" and json.loads(sc["input"]) == {"title": "Dentist", "start": "2026-10-05T16:00", "duration": 45}


def test_url_door_actions_have_no_helper_when_the_link_covers_them():
    assert "shortcut" not in act("call 555 0100")[1] and "shortcut" not in act("play jazz")[1] and "shortcut" not in act("google x")[1]


def test_catalogue_complete_and_every_helper_routable():
    want = {"Timer", "Alarm", "Reminder", "Note", "Calendar Event", "Flashlight", "Focus", "Home Device", "Volume", "Brightness", "WiFi",
            "Bluetooth", "Low Power Mode", "Dark Mode", "Play Music", "Clipboard", "Battery", "Weather", "Location"}
    assert {"Nupen " + w for w in want} <= set(P.HELPERS)
    assert set(P._HELPER_OF.values()) | set(P._REPORTERS.values()) <= set(P.HELPERS)


def test_phone_toggles_do_not_eat_home_devices():
    assert act("turn on the kitchen lights")[1]["type"] == "home"


# ---- context answers ----
def ask(text, ctx):
    return P.answer_context(text, P.safe_context(ctx))


def test_battery_answer():
    assert ask("how much battery do I have", {"battery": 83})[0] == "Your battery is at 83 percent, sir."
    assert ask("how much battery do I have", {"battery": {"level": "40%", "charging": True}})[0] == "Your battery is at 40 percent, and charging, sir."
    assert ask("what is my battery level", {"battery": 0.5})[0] == "Your battery is at 50 percent, sir."


def test_location_answer_is_city_level():
    r = ask("where am I", {"location": "Provo, Utah"})[0]
    assert r == "You are in Provo, Utah, sir."
    r = ask("where am I", {"location": "123 Main Street, Provo, UT 84601"})[0]
    assert "123" not in r and "Provo" in r
    r = ask("where am I", {"location": "40.2338,-111.6585"})[0]
    assert "40.2338" not in r and "40.2" in r


def test_calendar_answer():
    assert ask("what's on my calendar today", {"calendar_today": []})[0] == "Your calendar is clear today, sir."
    r = ask("what's on my calendar today", {"calendar_today": [{"title": "Dentist", "start": "2026-10-05T16:00"}, "Lunch with Sam"]})[0]
    assert r == "You have 2 events today, sir: Dentist at 4 PM; Lunch with Sam."
    assert ask("do I have any meetings", {"calendar_today": ["Standup"]})[0] == "You have one event today, sir: Standup."


def test_other_context_answers():
    assert ask("what song is this", {"now_playing": {"title": "Hello", "artist": "Adele"}})[0] == "That is Hello by Adele, sir."
    assert ask("what's on my clipboard", {"clipboard": "abc"})[0] == "Your clipboard says: abc, sir."
    assert ask("what's the weather like", {"weather": {"temp": "14 C", "condition": "cloudy"}})[0] == "It is 14 C, cloudy, sir."


def test_missing_context_asks_helper_or_says_so():
    r, a = ask("how much battery do I have", {})
    a == {"type": "report", "what": "battery"}
    assert ask("where am I", {})[1] == {"type": "report", "what": "location"}
    assert ask("what's the weather like", {})[1] == {"type": "report", "what": "weather"}
    r, a = ask("what song is this", {})
    assert a is None and "do not have" in r
    r, a = ask("where am I", {"location": None})                 # tried and failed: no loop back to the helper
    assert a is None and "could not read" in r


def test_ordinary_questions_are_not_context():
    for t in ("who invented the battery", "what is the capital of France", "tell me a joke", "what happened in the event of 1066"):
        assert P.answer_context(t, {}) is None


def test_context_is_sanitised():
    c = P.safe_context({"battery": "x", "evil": "y", "clipboard": "z" * 1000, "calendar_today": ["a"] * 50, "location": {"lat": 40.123456, "lon": -111.98765}})
    assert "evil" not in c and c["battery"] is None and len(c["clipboard"]) == 300 and len(c["calendar_today"]) == 8
    assert c["location"] == "near 40.1, -112.0"
    assert P.safe_context("nope") == {}


# ---- privacy ----
def test_scrub_cuts_precise_coordinates():
    assert P.scrub("at 40.233812, -111.658534 now") == "at 40.2, -111.7 now"
    assert P.scrub("version 1.5 and 3.14") == "version 1.5 and 3.14"


def test_log_has_city_not_coordinates_clipboard_or_calendar(tmp_path):
    log = tmp_path / "log.jsonl"
    core = P.Core(Path("."), make_conv=lambda: None, chat=lambda m, more: "x", log_path=log)
    core.talk("d", "where am I", {"location": {"lat": 40.233812, "lon": -111.658534}, "clipboard": "secret-password", "calendar_today": ["Dentist"]})
    core.talk("d", "what's on my clipboard", {"clipboard": "secret-password"})
    core.talk("d", "where am I", {"location": "Provo, Utah"})
    text = log.read_text(encoding="utf-8")
    assert "40.233812" not in text and "111.658534" not in text and "secret-password" not in text and "Dentist" not in text
    rows = [json.loads(x) for x in text.splitlines()]
    assert rows[0]["ctx"]["keys"] == ["calendar_today", "clipboard", "location"] and rows[2]["ctx"]["city"] == "Provo, Utah"
    assert rows[0]["intent"] == "context"


def test_context_not_kept_between_requests():
    core = P.Core(Path("."), make_conv=lambda: None, chat=lambda m, more: "Fine, sir.")
    assert core.talk("d", "where am I", {"location": "Provo, Utah"})[1]["reply"] == "You are in Provo, Utah, sir."
    r = core.talk("d", "where am I")[1]
    assert r["action"].pop("id") and r["action"] == {"type": "report", "what": "location"} and "Provo" not in json.dumps(r)
    assert not any("Provo" in json.dumps(v, default=str) for v in vars(core).values())


def test_http_passes_context(tmp_path):
    import threading
    import urllib.request
    core = P.Core(Path("."), make_conv=lambda: None, chat=lambda m, more: "x")
    srv = P.make_server("127.0.0.1", 0, "t" * 32, core)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/talk", method="POST",
                                     data=json.dumps({"text": "how much battery do I have", "context": {"battery": 61}}).encode())
        req.add_header("Authorization", "Bearer " + "t" * 32)
        with urllib.request.urlopen(req, timeout=30) as r:
            assert json.loads(r.read())["reply"] == "Your battery is at 61 percent, sir."
    finally:
        srv.shutdown()
        srv.server_close()


def test_helper_door_is_off_by_default_and_flag_restores_it(monkeypatch):
    assert P.HELPER_DOOR is False
    assert "shortcut" not in P.route(act("set a timer for five minutes")[1])
    monkeypatch.setattr(P, "HELPER_DOOR", True)
    assert P.route(act("set a timer for five minutes")[1])["shortcut"]["name"] == "Nupen Timer"
