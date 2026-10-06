"""creator/shortcutgen.py: request -> plan -> Jelly -> validator -> store -> ntfy 'Confirm' flow (fake ntfy server, never the real one)."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from creator import decisions as D
from creator import shortcutgen as S

SPEC = importlib.util.spec_from_file_location("nupen_phone_sg", Path(__file__).resolve().parents[1] / "scripts" / "nupen_phone.py")
P = importlib.util.module_from_spec(SPEC)
sys.modules["nupen_phone_sg"] = P
SPEC.loader.exec_module(P)
TOKEN = "t" * 32


class Fake:
    def __init__(self):
        self.got: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.got.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": self.rfile.read(n).decode()})
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    f = Fake()
    monkeypatch.setenv("NUPEN_RUNTIME", str(tmp_path))
    monkeypatch.setenv("NUPEN_NTFY_SERVER", f.url)
    monkeypatch.setenv("NUPEN_NTFY_TOPIC", "testtopic123")
    monkeypatch.setenv("NUPEN_PHONE_URL", "http://phone.invalid:8765")
    monkeypatch.delenv("NUPEN_SHORTCUT", raising=False)
    (tmp_path / "phone").mkdir()
    (tmp_path / "phone" / "token.txt").write_text(TOKEN)
    yield f, tmp_path / "phone"
    f.srv.shutdown()
    f.srv.server_close()


EXAMPLES = [
    ("make me a shortcut that texts 555 123 4567 saying running late and then opens maps to 12 Main Street",
     ['openURL(url: "sms:5551234567&body=running%20late")', 'openURL(url: "maps://?daddr=12%20Main%20Street&dirflg=d")']),
    ("create a shortcut for my morning routine",
     ["speakText(", "getCurrentConditions() >> weather", "setBrightness(value: 0.7)", "setDND(state: false)", "batteryLevel() >> battery", "play(behavior: Play)"]),
    ("make a shortcut that starts a 10 minute timer and plays some jazz",
     ['timer(duration: "10 min")', "term=jazz", "play(behavior: Play)"]),
    ("build me a shortcut that lets me speak a reminder",
     ["dictateText() >> said", 'createNote(text: "${said}", show: false)']),
    ("make a shortcut that tells me my battery level",
     ["batteryLevel() >> battery", 'speakText(text: "Battery is at ${battery} percent")']),
]


@pytest.mark.parametrize("request_text,needles", EXAMPLES)
def test_five_examples_generate_valid_code(request_text, needles):
    r = S.make(request_text)
    assert r["ok"], r["errors"]
    assert S.validate(r["code"]) == []
    assert r["code"].splitlines()[1].startswith("import Shortcuts #Color: ")
    for n in needles:
        assert n in r["code"], (n, r["code"])


def test_named_shortcut_and_unknown_request():
    r = S.make('make a shortcut called "Bedtime" that turns on do not disturb and sets brightness to 20%')
    assert r["ok"] and r["name"] == "Bedtime" and "setDND(state: true)" in r["code"] and "setBrightness(value: 0.2)" in r["code"]
    assert not S.make("make a shortcut that does the dishes")["ok"]
    assert S.handle("what is the weather", None) is None


def test_validator_rejects_unknown_forbidden_and_malformed():
    head = "import Shortcuts #Color: blue, #Icon: shortcuts\n"
    assert S.validate(head + 'showResult(text: "hi")\n') == []
    bad = {
        'urlContents(url: "https://evil.example/x")': "forbidden",
        'deleteFile(file: ShortcutInput)': "forbidden",
        'deletePhotos(photos: ShortcutInput)': "forbidden",
        'xCallbackURL(url: "x://y")': "forbidden",
        'runShortcut(name: "x")': "forbidden",
        'frobnicate(text: "x")': "unknown action",
        'showResult(txt: "x")': "no parameter",
        'showResult()': "needs text",
        'wait(seconds: "soon")': "bad value",
        'timer(duration: tenminutes)': "bad value",
        'setDND(state: maybe)': "bad value",
        'showResult(text: "${nothing}")': "before it is made",
        'openURL(url: "https://evil.example/")': "openURL only",
        'showResult text "x"': "not a function call",
    }
    for line, why in bad.items():
        errs = S.validate(head + line + "\n")
        assert errs and why in " ".join(errs), (line, errs)
    assert any("import" in e for e in S.validate('showResult(text: "x")\n'))
    # data can never ride out inside a URL
    assert S.validate(head + 'dictateText() >> said\nopenURL(url: "https://music.apple.com/search?term=${said}")\n')


def test_no_plan_can_leak_or_inject():
    # quotes, backslashes and ${} in the owner's words cannot break out of the string literal
    plan = {"name": "x", "steps": [{"kind": "speak", "text": 'hi" ) openURL(url: "https://evil.example") ${x} \\'}]}
    code = S.to_jelly(plan)
    assert S.validate(code) == [] and not any(l.startswith("openURL") for l in code.splitlines()) and "${" not in code
    with pytest.raises(ValueError):
        S.to_jelly({"steps": [{"kind": "delete_everything"}]})
    with pytest.raises(ValueError):
        S.to_jelly({"steps": []})


def test_model_plan_is_filtered_by_the_same_table():
    llm = lambda p: 'Sure {"name": "Sneaky", "steps": [{"kind": "run_shell", "cmd": "rm -rf"}, {"kind": "vibrate"}, {"kind": "speak", "text": "ok"}]}'  # noqa: E731
    r = S.make("make a shortcut that does something odd", llm)
    assert r["ok"] and "vibrate()" in r["code"] and "rm" not in r["code"]
    assert not S.make("make a shortcut that does something odd", lambda p: "no json here")["ok"]


def test_store_fetch_and_revalidation(env):
    _, rt = env
    r = S.make(EXAMPLES[4][0])
    sid = S.store(rt, r["name"], r["code"], now=1.0)
    rec = S.fetch(rt, sid)
    assert rec and rec["code"] == r["code"] and rec["deliver"] == "compile_jelly_text"
    assert S.fetch(rt, "../../x") is None and S.fetch(rt, "0" * 12) is None
    f = rt / "shortcuts" / f"{sid}.json"
    d = json.loads(f.read_text())
    d["code"] += 'urlContents(url: "https://evil.example")\n'
    f.write_text(json.dumps(d))
    assert S.fetch(rt, sid) is None                       # a tampered file is never served


def test_notification_has_confirm_and_not_now(env):
    f, rt = env
    res = S.handle(EXAMPLES[0][0], rt)
    assert res and res["pushed"] and res["action"]["type"] == "build_shortcut" and res["action"]["deliver"] == "compile_jelly_text"
    got = f.got[0]
    assert got["path"] == "/testtopic123"
    assert got["body"] == "Sir, your new shortcut 'Text and Maps' is ready."
    acts = [a.strip() for a in got["headers"]["actions"].split(";")]
    assert acts[0] == ("view, Confirm, shortcuts://run-shortcut?name=Nupen&input=text&text=" + urllib.parse.quote("build_shortcut " + res["id"]))
    assert acts[1].startswith("http, Not now, http://phone.invalid:8765/decision, method=POST")
    assert f"shortcut-{res['id']}" in acts[1] and "headers.X-Decision-Key=" in acts[1] and TOKEN not in got["headers"]["actions"]
    assert [p["id"] for p in D.pending(rt)] == [f"shortcut-{res['id']}"]


def test_not_now_records_a_decision_and_changes_nothing_else(env):
    f, rt = env
    res = S.handle(EXAMPLES[2][0], rt)
    key = D.decision_key(TOKEN, "shortcut-" + res["id"])
    srv = P.make_server("127.0.0.1", 0, TOKEN, _NoCore(), None, decisions_dir=rt)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/decision", data=json.dumps({"id": "shortcut-" + res["id"], "choice": "later"}).encode(),
                                     headers={"X-Decision-Key": key, "Content-Type": "application/json"}, method="POST")
        assert json.loads(urllib.request.urlopen(req).read())["ok"] is True
        assert D.answers(rt) == {"shortcut-" + res["id"]: "later"}
        assert S.fetch(rt, res["id"]) is not None          # the stored shortcut stays; nothing was installed or deleted
        # the phone fetches the code by id with the bearer token, and only with it
        g = urllib.request.Request(f"http://127.0.0.1:{port}/shortcut?id={res['id']}", headers={"Authorization": "Bearer " + TOKEN})
        body = json.loads(urllib.request.urlopen(g).read())
        assert body["name"] == res["action"]["name"] and body["code"].startswith("// ")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/shortcut?id={res['id']}")
        assert e.value.code == 401
    finally:
        srv.shutdown()
        srv.server_close()


class _NoCore:
    voice = None


def test_next_pending_hands_out_the_newest_once_and_expires(env):
    _, rt = env
    r = S.make(EXAMPLES[4][0])
    old = S.store(rt, "Old", r["code"], now=1000.0)
    new = S.store(rt, "New", r["code"], now=1500.0)
    assert S.next_pending(rt, now=1600.0)["id"] == new
    assert S.next_pending(rt, now=1601.0)["id"] == old       # the newest was handed out; the older one is still waiting
    assert S.next_pending(rt, now=1602.0) is None
    late = S.store(rt, "Late", r["code"], now=2000.0)
    assert S.next_pending(rt, now=2000.0 + S.PENDING_S + 1) is None and late


def test_shortcut_next_endpoint_needs_the_token(env):
    _, rt = env
    res = S.handle(EXAMPLES[4][0], rt)
    srv = P.make_server("127.0.0.1", 0, TOKEN, _NoCore(), None, decisions_dir=rt)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/shortcut/next"
        got = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Bearer " + TOKEN})).read())
        assert got["id"] == res["id"] and got["code"].startswith("// ")
        assert json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Bearer " + TOKEN})).read()) == {}
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(url)
        assert e.value.code == 401
    finally:
        srv.shutdown()
        srv.server_close()


def test_shortcut_report_is_logged_with_the_token_only(env):
    _, rt = env
    res = S.handle(EXAMPLES[4][0], rt)
    srv = P.make_server("127.0.0.1", 0, TOKEN, _NoCore(), None, decisions_dir=rt)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/shortcut_report"
        body = json.dumps({"id": res["id"], "result_type": "File", "has_value": "yes", "junk": "x"}).encode()
        req = urllib.request.Request(url, data=body, headers={"Authorization": "Bearer " + TOKEN}, method="POST")
        assert json.loads(urllib.request.urlopen(req).read())["ok"] is True
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST"))
        assert e.value.code == 401
        bad = urllib.request.Request(url, data=json.dumps({"id": "../x"}).encode(), headers={"Authorization": "Bearer " + TOKEN}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
        reps = S.reports(rt)
        assert len(reps) == 1 and reps[0]["id"] == res["id"] and reps[0]["result_type"] == "File" and "junk" not in reps[0]
    finally:
        srv.shutdown()
        srv.server_close()


def test_refusal_is_spoken_and_nothing_is_stored_or_sent(env):
    f, rt = env
    res = S.handle("make me a shortcut that deletes all my photos", rt)
    assert res and res["action"] is None and "cannot build" in res["reply"]
    assert f.got == [] and not (rt / "shortcuts").exists()


def test_no_ntfy_configured_is_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("NUPEN_RUNTIME", str(tmp_path))
    monkeypatch.delenv("NUPEN_NTFY_TOPIC", raising=False)
    res = S.handle(EXAMPLES[4][0], tmp_path / "phone")
    assert res and res["pushed"] is False and res["action"]["type"] == "build_shortcut"


def test_trusted_records_skip_the_validator_only_when_stored_trusted(env):
    _, rt = env
    code = 'import Shortcuts\nurlContents(url: "http://192.0.2.1:8765/talk")\n'
    assert S.validate(code)                                   # a model-written shortcut with web calls is refused
    plain = S.store(rt, "Plain", code, now=1.0)
    assert S.fetch(rt, plain) is None
    sid = S.store(rt, "Nupen 2", code, now=2.0, trusted=True)
    assert S.fetch(rt, sid)["code"] == code
