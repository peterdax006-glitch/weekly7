"""creator/shortcutsign + the /install route: signed .shortcut files built on the PC (no network, no WSL: compiler and signer are fakes)."""
from __future__ import annotations

import base64
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from creator import shortcutgen as SG
from creator import shortcutsign as SS

SPEC = importlib.util.spec_from_file_location("nupen_phone_sign", Path(__file__).resolve().parents[1] / "scripts" / "nupen_phone.py")
P = importlib.util.module_from_spec(SPEC)
sys.modules["nupen_phone_sign"] = P
SPEC.loader.exec_module(P)

TOKEN = "k" * 32
OK_CODE = 'import Shortcuts #Color: green, #Icon: timer\ntimer(duration: "5 min")\n'
SIGNED = b"AEA1" + b"\x00" * 60


def fake_compile(code: str) -> bytes:
    return b"<plist>" + code.encode() + b"</plist>"


class FakeHttp:
    def __init__(self, reply: bytes | None = None, nested: bool = True) -> None:
        self.calls: list[tuple[str, bytes]] = []
        payload = {"content_base64": base64.b64encode(SIGNED).decode(), "format": "AEA1"}
        self.reply = reply if reply is not None else json.dumps({"file": payload, "success": True} if nested else payload).encode()

    def __call__(self, url: str, body: bytes, headers: dict, timeout: int) -> bytes:
        self.calls.append((url, body))
        return self.reply


@pytest.fixture()
def shortcuty(monkeypatch):
    monkeypatch.setenv("NUPEN_SIGNER", "shortcuty")
    http = FakeHttp()
    monkeypatch.setattr(SS, "compile_plist", fake_compile)
    monkeypatch.setattr(SS, "_http_post", http)
    return http


# ---- what may leave the PC ----
@pytest.mark.parametrize("code", [
    'openURL(url: "sms:07700900123&body=hi")', 'openURL(url: "tel:5550100")', 'showResult(text: "call 0770 090 0123")',
    'showResult(text: "bob@example.com")', 'openURL(url: "maps://?daddr=12%20High%20Street&dirflg=d")',
    'showResult(text: "http://192.168.1.5:8765")', 'showResult(text: "my password is x")',
])
def test_private_data_is_never_sent_to_a_signing_service(code):
    assert SS.remote_ok({"code": code, "name": "x"})[0] is False


@pytest.mark.parametrize("code", [OK_CODE, 'openURL(url: "https://music.apple.com/search?term=jazz")', 'setVolume(level: 0.5)',
                                  'openURL(url: "maps://?daddr=the%20airport&dirflg=d")'])
def test_ordinary_generated_shortcuts_may_be_signed(code):
    assert SS.remote_ok({"code": code, "name": "Timer"}) == (True, "ok")


def test_trusted_records_and_secrets_never_leave():
    assert SS.remote_ok({"code": OK_CODE, "trusted": True})[0] is False
    assert SS.remote_ok({"code": OK_CODE + "// abc123secret"}, secrets=("abc123secret",))[0] is False


def test_signer_config(tmp_path, monkeypatch):
    monkeypatch.delenv("NUPEN_SIGNER", raising=False)
    assert SS.signer_config(tmp_path) == ("shortcuty", [])
    (tmp_path / "signer.txt").write_text("local\n/root/k.der\n/root/auth.bin\n", encoding="utf-8")
    assert SS.signer_config(tmp_path) == ("local", ["/root/k.der", "/root/auth.bin"])
    (tmp_path / "signer.txt").write_text("nonsense\n", encoding="utf-8")
    assert SS.signer_config(tmp_path)[0] == "off"
    monkeypatch.setenv("NUPEN_SIGNER", "off")
    assert SS.signer_config(None)[0] == "off"


def test_multipart_shape():
    body, ctype = SS.multipart("file", 'My "Timer".shortcut', b"PLIST")
    b = ctype.split("boundary=")[1]
    assert body.startswith(f"--{b}\r\n".encode()) and body.endswith(f"\r\n--{b}--\r\n".encode())
    assert b'name="file"; filename="My Timer.shortcut"' in body and b"\r\n\r\nPLIST\r\n" in body


@pytest.mark.parametrize("nested", [True, False])
def test_shortcuty_reply_flat_or_nested(nested):
    http = FakeHttp(nested=nested)
    assert SS.sign_shortcuty(b"<plist/>", "T", http=http) == SIGNED
    assert http.calls[0][0] == SS.SHORTCUTY_URL + "?response=json"


def test_shortcuty_errors_are_runtime_errors():
    with pytest.raises(RuntimeError):
        SS.sign_shortcuty(b"x", "T", http=FakeHttp(reply=b'{"error":"signer_unavailable"}'))

    def boom(*a):
        raise urllib.error.URLError("down")
    with pytest.raises(RuntimeError, match="unreachable"):
        SS.sign_shortcuty(b"x", "T", http=boom)


# ---- build + cache ----
def test_build_signs_once_and_caches(tmp_path, shortcuty):
    sid = SG.store(tmp_path, "Five minute timer", OK_CODE)
    rec = json.loads((tmp_path / "shortcuts" / f"{sid}.json").read_text(encoding="utf-8"))
    r1 = SS.build(tmp_path / "shortcuts", rec, tmp_path)
    assert r1["ok"] and r1["signer"] == "shortcuty" and Path(r1["path"]).read_bytes() == SIGNED
    assert b"timer(duration: \"5 min\")" in shortcuty.calls[0][1]           # the compiled plist was what got uploaded
    r2 = SS.build(tmp_path / "shortcuts", rec, tmp_path)
    assert r2["signer"] == "cache" and len(shortcuty.calls) == 1


def test_build_refuses_private_and_unsigned_results(tmp_path, shortcuty):
    rec = {"id": "a" * 12, "name": "Text mum", "code": 'openURL(url: "sms:07700900123&body=hi")'}
    r = SS.build(tmp_path, rec, tmp_path)
    assert not r["ok"] and "phone" in r["error"] and shortcuty.calls == []
    bad = FakeHttp(reply=json.dumps({"content_base64": base64.b64encode(b"<plist/>").decode()}).encode())
    r = SS.build(tmp_path, {"id": "b" * 12, "name": "T", "code": OK_CODE}, tmp_path, http=bad)
    assert not r["ok"] and "AEA1" in r["error"] and not (tmp_path / ("b" * 12 + ".shortcut")).exists()


def test_build_off_and_bad_id(tmp_path, monkeypatch):
    monkeypatch.setenv("NUPEN_SIGNER", "off")
    assert SS.build(tmp_path, {"id": "c" * 12, "code": OK_CODE}, tmp_path)["error"].startswith("signing is off")
    assert SS.build(tmp_path, {"id": "../x", "code": OK_CODE}, tmp_path)["error"] == "bad id"


def test_local_signer_uses_owner_key_and_never_the_network(tmp_path, monkeypatch):
    monkeypatch.delenv("NUPEN_SIGNER", raising=False)
    (tmp_path / "signer.txt").write_text("local\n/root/key.der\n/root/auth.bin\n", encoding="utf-8")
    seen = []

    def run(cmd, timeout):
        seen.append(cmd)
        out = cmd[cmd.index("-o") + 1]
        Path("C:/" + out[len("/mnt/c/"):] if out.startswith("/mnt/c/") else out).write_bytes(SIGNED)
        return 0, "ok"

    def no_net(*a):
        raise AssertionError("network used")
    rec = {"id": "d" * 12, "name": "T", "code": 'openURL(url: "sms:07700900123")'}     # private data is fine locally
    r = SS.build(tmp_path, rec, tmp_path, compile_=fake_compile, http=no_net, run=run)
    assert r["ok"] and r["signer"] == "local"
    assert seen[0][1:2] == ["sign"] and "-k" in seen[0] and "/root/key.der" in seen[0] and "/root/auth.bin" in seen[0]


# ---- shortcutgen: what /shortcut/next tells Nupen 2 ----
def test_next_pending_says_signed_with_a_url_ready_path(tmp_path, shortcuty):
    sid = SG.store(tmp_path, "Five minute timer!", OK_CODE)
    nxt = SG.next_pending(tmp_path)
    assert nxt["install"] == "signed" and nxt["install_path"] == f"/install/{sid}/Five%20minute%20timer.shortcut"


def test_trusted_and_signing_off_keep_the_jelly_path(tmp_path, monkeypatch, shortcuty):
    sid = SG.store(tmp_path, "Nupen 2", OK_CODE, trusted=True)
    assert SG.fetch(tmp_path, sid)["install"] == "jelly"
    monkeypatch.setenv("NUPEN_SIGNER", "off")
    sid2 = SG.store(tmp_path, "Timer", OK_CODE)
    assert SG.fetch(tmp_path, sid2)["install"] == "jelly"


# ---- the server route ----
@pytest.fixture()
def server(tmp_path):
    core = P.Core(Path("."), make_conv=lambda: None)
    srv = P.make_server("127.0.0.1", 0, TOKEN, core, per_min=50, voice_rt=tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def get(srv, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}{path}", timeout=30) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_install_route_sends_the_signed_file(tmp_path, server, shortcuty):
    sid = SG.store(tmp_path, "Five minute timer", OK_CODE)
    path = SG.fetch(tmp_path, sid)["install_path"]
    code, hdr, body = get(server, path + "?key=" + TOKEN)
    assert code == 200 and body == SIGNED
    assert hdr["Content-Disposition"] == 'attachment; filename="Five minute timer.shortcut"'
    assert get(server, path)[0] == 401 and get(server, path + "?key=wrong")[0] == 401


def test_install_route_refuses_trusted_and_unknown(tmp_path, server, shortcuty):
    sid = SG.store(tmp_path, "Nupen 2", OK_CODE, trusted=True)
    code, _, body = get(server, f"/install/{sid}/Nupen%202.shortcut?key={TOKEN}")
    assert code == 409 and "token" in json.loads(body)["error"] and shortcuty.calls == []
    assert get(server, f"/install/{'f' * 12}/x.shortcut?key={TOKEN}")[0] == 409
    assert get(server, f"/install/..%2F..%2Fx?key={TOKEN}")[0] == 404


# ---- the phone side (Jelly) ----
def test_nupen2_installs_signed_files_and_keeps_the_old_path():
    t = P.served_script("http://pc.example:8765", TOKEN)
    assert 'if(installMode == "signed") {' in t
    assert 'text(text: "${base}${installPath}?key=' + TOKEN + '") >> installUrl' in t
    assert 'runShortcut(name: "Nupen Net", input: installUrl, show: false) >> newShortcut' in t
    assert 'openIn(input: newShortcut, app: "com.apple.shortcuts", ask: false)' in t
    assert t.index("installMode") < t.index("jellyCode")


def test_menus_become_the_apps_form():
    src = 'menu "Pick" {\ncase "A x":\n    vibrate()\ncase "B":\n    vibrate()\n}\n'
    assert P._menus_for_app(src) == 'menu("Pick", ["A x", "B"]) {\ncase("A x"):\n    vibrate()\ncase("B"):\n    vibrate()\n}\n'


def test_install_probe_fills_every_id(tmp_path):
    signed = []
    out, ok = P.make_install_probe(tmp_path, TOKEN, sign=lambda sid: signed.append(sid) or {"ok": True})
    text = Path(out).read_text(encoding="utf-8")
    assert ok and len(signed) == 4 and "PROBE_" not in text and all(s in text for s in signed)
    assert TOKEN not in text                                       # the key is only added when the setup link serves it
    assert SG.next_pending(tmp_path) is None                       # probe shortcuts are never offered to Nupen 2
    served = P.served_script("http://pc.example:8765", TOKEN, Path(out))
    assert 'menu("Nupen install probe", [' in served and "grabJellycut() >> lastExport" in served


def test_install_probe_stops_on_a_signing_failure(tmp_path):
    out, ok = P.make_install_probe(tmp_path, TOKEN, sign=lambda sid: {"ok": False, "error": "signer_unavailable"})
    assert not ok and "signer_unavailable" in out
