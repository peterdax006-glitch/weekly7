"""scripts/nupen_phone.py: bearer auth, size cap, rate limit, LAN-bind default, reply shape (a fake talk model; no real voice)."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
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
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_auth_required_and_wrong_token_rejected(server):
    srv, _ = server
    assert call(srv, body={"text": "hi"}, token=None)[0] == 401
    assert call(srv, body={"text": "hi"}, token="x" * 32)[0] == 401
    assert call(srv, "/health", token=None, method="GET")[0] == 401
    assert call(srv, "/health", method="GET")[0] == 200


def test_reply_shape_short_and_say_more(server):
    srv, _ = server
    code, d = call(srv, body={"text": "how are you", "device": "iphone"})
    assert code == 200 and set(d) == {"reply", "ms", "more"}
    assert "evidence" not in d["reply"] and len(d["reply"].split()) <= 62 and d["more"] is True
    code, d2 = call(srv, body={"text": "say more", "device": "iphone"})
    assert code == 200 and d2["reply"] and d2["reply"] != d["reply"]


def test_size_cap_and_bad_input(server):
    srv, _ = server
    assert call(srv, raw=b"x" * (P.MAX_BODY + 1))[0] == 413
    assert call(srv, raw=b"not json")[0] == 400
    assert call(srv, body={"text": "a" * (P.MAX_TEXT + 1)})[0] == 400


def test_rate_limit(server):
    srv, _ = server
    codes = [call(srv, body={"text": "hi"})[0] for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5:] == [429, 429]


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
