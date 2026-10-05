"""creator/notify.py + creator/decisions.py + POST /decision: fake ntfy server, never the real one."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from creator import decisions as D
from creator import notify as N

SPEC = importlib.util.spec_from_file_location("nupen_phone_n", Path(__file__).resolve().parents[1] / "scripts" / "nupen_phone.py")
P = importlib.util.module_from_spec(SPEC)
sys.modules["nupen_phone_n"] = P
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
    monkeypatch.delenv("NUPEN_PHONE_URL", raising=False)
    yield f, tmp_path
    f.srv.shutdown()
    f.srv.server_close()


def test_headers_and_default_action(env):
    f, _ = env
    assert N.notify("Hello", "body text", "high", ("tada",), click="https://example.org/x", now=1000.0)
    r = f.got[0]
    assert r["path"] == "/testtopic123" and r["body"] == "body text"
    assert r["headers"]["title"] == "Hello" and r["headers"]["priority"] == "high" and r["headers"]["tags"] == "tada"
    assert r["headers"]["click"] == "https://example.org/x"
    assert r["headers"]["actions"] == "view, Talk to Nupen, shortcuts://run-shortcut?name=Nupen"


def test_shortcut_name_configurable(env, monkeypatch):
    f, tmp = env
    (tmp / "phone").mkdir()
    (tmp / "phone" / "shortcut_name.txt").write_text("My Shortcut", encoding="utf-8")
    N.notify("T", "one", now=1.0)
    assert f.got[0]["headers"]["actions"] == "view, Talk to Nupen, shortcuts://run-shortcut?name=My%20Shortcut"
    monkeypatch.setenv("NUPEN_SHORTCUT", "Other")
    assert N.shortcut_name() == "Other"


def test_decision_actions_format(env, monkeypatch):
    f, tmp = env
    monkeypatch.setenv("NUPEN_PHONE_URL", "http://phone.invalid:8765")
    (tmp / "phone").mkdir()
    (tmp / "phone" / "token.txt").write_text(TOKEN, encoding="utf-8")
    assert N.notify_decision("adopt-x1", "Need you", "pick one", ) is True
    acts = f.got[0]["headers"]["actions"].split("; ")
    assert len(acts) == 3 and acts[0].startswith("http, Approve, http://phone.invalid:8765/decision, method=POST")
    assert "body='{\"id\":\"adopt-x1\",\"choice\":\"approve\"}'" in acts[0] and "body='{\"id\":\"adopt-x1\",\"choice\":\"later\"}'" in acts[1]
    assert f"headers.X-Decision-Key={D.decision_key(TOKEN, 'adopt-x1')}" in acts[0]
    assert TOKEN not in f.got[0]["headers"]["actions"]                      # the master token never goes through ntfy
    assert acts[2].startswith("view, Talk to Nupen")
    assert [r["id"] for r in D.pending(tmp / "phone")] == ["adopt-x1"]


def test_not_configured_is_silent_noop(env, monkeypatch):
    f, _ = env
    monkeypatch.delenv("NUPEN_NTFY_TOPIC")
    assert N.notify("a", "b") is False and f.got == []
    assert N.notify_decision("d1", "a", "b") is False and f.got == []
    assert N.adopt_outcome({"outcome": "ADOPTED", "id": "z"}) is False


def test_unreachable_server_never_raises(env, monkeypatch):
    monkeypatch.setenv("NUPEN_NTFY_SERVER", "http://127.0.0.1:1")
    assert N.notify("a", "b") is False


def test_dedupe_within_ten_minutes(env):
    f, _ = env
    assert N.notify("T", "same", now=1000.0)
    assert not N.notify("T", "same", now=1500.0)
    assert N.notify("T", "same", now=1000.0 + N.DEDUPE_S + 1)
    assert len(f.got) == 2


def test_burst_and_hourly_rate_limit(env):
    f, _ = env
    sent = [N.notify("T", f"m{i}", now=1000.0 + i) for i in range(5)]
    assert sent == [True] * N.BURST + [False] * (5 - N.BURST)
    t = 1000.0
    for i in range(30):
        t += 70.0                                                            # spaced: only the hourly cap bites
        N.notify("T", f"h{i}", now=t)
    hour = [r for r in N._recent(t) if r["status"] == "sent"]
    assert len(hour) <= N.MAX_PER_HOUR


def test_scrub_removes_paths_hosts_secrets(env):
    f, _ = env
    drive, ip, bs = "C" + ":", ".".join(["10", "1", "2", "3"]), chr(92)
    path = bs.join([drive, "Users", "someone", "secret dir", "train.py"])
    N.notify("Job", f"failed in {path} on {ip} root@gpu.example.com key " + "a" * 40, now=1.0)
    body = f.got[0]["body"]
    assert "someone" not in body and ip not in body and "root@" not in body and "a" * 40 not in body
    assert "train.py" in body


def test_log_is_outside_repo_and_has_no_topic(env):
    _, tmp = env
    N.notify("T", "logged", now=5.0)
    p = tmp / "notify" / "sent.jsonl"
    assert p.is_file() and "testtopic123" not in p.read_text(encoding="utf-8")


def test_pytest_without_fake_server_never_pushes(monkeypatch):
    monkeypatch.delenv("NUPEN_NTFY_SERVER", raising=False)
    monkeypatch.setenv("NUPEN_NTFY_TOPIC", "realtopic")
    assert N.topic() is None


# ---- decision endpoint
@pytest.fixture()
def phone(tmp_path):
    core = P.Core(Path("."), make_conv=lambda: None)
    dd = tmp_path / "dec"
    srv = P.make_server("127.0.0.1", 0, TOKEN, core, decisions_dir=dd)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", dd
    srv.shutdown()
    srv.server_close()


def post(url, body, headers):
    req = urllib.request.Request(url + "/decision", data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_decision_records_but_does_not_execute(phone, tmp_path):
    url, dd = phone
    assert D.ask("d-1", "Adopt it?", dd)
    st, body = post(url, {"id": "d-1", "choice": "approve"}, {"X-Decision-Key": D.decision_key(TOKEN, "d-1")})
    assert (st, body["ok"]) == (200, True)
    assert D.answers(dd) == {"d-1": "approve"} and D.pending(dd) == []
    assert sorted(p.name for p in dd.iterdir()) == ["decisions.jsonl"]       # only the queue file was written; nothing else happened
    st, _ = post(url, {"id": "d-1", "choice": "later"}, {"X-Decision-Key": D.decision_key(TOKEN, "d-1")})
    assert st == 409 and D.answers(dd) == {"d-1": "approve"}                 # first answer wins


def test_decision_auth_and_validation(phone):
    url, dd = phone
    D.ask("d-2", "x", dd)
    assert post(url, {"id": "d-2", "choice": "approve"}, {})[0] == 401
    assert post(url, {"id": "d-2", "choice": "approve"}, {"X-Decision-Key": D.decision_key(TOKEN, "other")})[0] == 401
    assert post(url, {"id": "d-2", "choice": "delete-everything"}, {"Authorization": "Bearer " + TOKEN})[0] == 400
    assert post(url, {"id": "never-asked", "choice": "approve"}, {"Authorization": "Bearer " + TOKEN})[0] == 400
    assert post(url, {"id": "../x", "choice": "approve"}, {"Authorization": "Bearer " + TOKEN})[0] == 400
    assert D.answers(dd) == {}
    assert post(url, {"id": "d-2", "choice": "later"}, {"Authorization": "Bearer " + TOKEN})[0] == 200
    assert D.answers(dd) == {"d-2": "later"}
