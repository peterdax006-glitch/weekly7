"""Nupen 2 (the Jelly shortcut): template passes the static checker, the checker catches the compiler pitfalls, action ids + by_index,
query-string /talk, POST /action_report, the one-time /setup/<key> link, raw uploads without a name, and the convlearn signal."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = sys.modules.get("nupen_phone") or _load("nupen_phone", "scripts/nupen_phone.py")
J = _load("jelly_check", "scripts/jelly_check.py")
from creator import convlearn as CL  # noqa: E402

TOKEN = "k" * 32
TEMPLATE = ROOT / "scripts" / "nupen2.jelly"


class FakeConv:
    def reply(self, text: str) -> str:
        return "Fine, sir."


@pytest.fixture()
def srv(tmp_path):
    core = P.Core(Path("."), make_conv=FakeConv)
    s = P.make_server("127.0.0.1", 0, TOKEN, core, log_path=tmp_path / "access.log", per_min=40, voice_rt=tmp_path)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield s, tmp_path
    s.shutdown()
    s.server_close()


def req(s, path, body=None, token=TOKEN, method="POST", raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(f"http://127.0.0.1:{s.server_address[1]}{path}", data=data, method=method)
    if token:
        r.add_header("Authorization", "Bearer " + token)
    for k, v in (headers or {}).items():
        r.add_header(k, v)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


# ------------------------------------------------------------------------------------------------ the template and the checker
def test_template_passes_checker_and_has_only_placeholders():
    src = TEMPLATE.read_text(encoding="utf-8")
    errs, warns = J.check(src)
    assert errs == [] and warns == []
    assert P.PLACE_BASE in src and P.PLACE_TOKEN in src and "192.168." not in src and "Bearer PASTE_TOKEN" in src
    served = P.served_script("http://pc.example:8765", TOKEN)
    errs, _ = J.check(served, served=True)
    assert errs == [] and "PASTE_" not in served and f"key={TOKEN}" in served and 'var base = "http://pc.example:8765"' in served
    assert J.check(src, served=True)[0]                                 # the unfilled template is refused as a served copy


def test_template_covers_the_blocks():
    src = TEMPLATE.read_text(encoding="utf-8")
    for t in ("timer", "reminder", "alarm", "calendar", "note", "flashlight", "focus", "volume", "brightness", "wifi", "bluetooth", "low_power",
              "dark_mode", "clipboard", "home", "music", "message", "call", "report", "send_media"):
        assert f'if(atype == "{t}")' in src, t
    for frag in ("/shortcut/next", "/talk?device=iphone", "if(said == nil)", "/action_report?id=${aid}&type=${atype}&phase=start",
                 "phase=done&ok=${okv}", 'if(nextStep == "stop")', "requestType: File, requestVar: RepeatItem", "repeat(20)"):
        assert frag in src, frag


@pytest.mark.parametrize("bad,why", [
    ("import Shortcuts\nfoo(x: 1)\n", "unknown"),
    ("import Shortcuts\nif(x == 1) {\n", "not defined"),
    ("import Shortcuts\nvar x = 1\nif(x == 1) {\n", "not closed"),
    ('import Shortcuts\nvar u = "a"\ndownloadURL(url: "${u}", headers: {"A": "${u}"})\n', "dictionary literal"),
    ("import Shortcuts\nbatteryLevel() >> b\nbatteryLevel() >> b\ntext(text: \"${b}\")\n", "already used"),
    ("import Shortcuts\nbatteryLevel() >> b\nrepeat(2) {\nrepeatEach(b) {\ntext(text: \"${RepeatItem}\")\n}\n}\n", "nested"),
    ("import Shortcuts\nbatteryLevel() >> b\ntimer(duration: b)\n", "literal"),
    ("import Shortcuts\nx = 1\n", "assigned before"),
    ("import Shortcuts\nbatteryLevel() >> b\nlocationDetail(detail: Street Address, location: b)\n", "enum"),
    ("import Shortcuts\nspeakText(\"hi\")\n", "unnamed"),
])
def test_checker_catches_compiler_pitfalls(bad, why):
    errs, _ = J.check(bad)
    assert any(why in e for e in errs), errs


# ------------------------------------------------------------------------------------------------ ids, by_index, next
def test_tag_actions_ids_index_and_next():
    n = iter(range(100))
    body = {"reply": "x", "actions": [{"type": "timer", "minutes": 5}, {"type": "note", "id": "keep_me"}], "end": False}
    P.tag_actions(body, new_id=lambda: f"a{next(n)}")
    assert [a["id"] for a in body["actions"]] == ["a0", "keep_me"] and body["action"] is body["actions"][0]
    assert body["by_index"] == {"1": body["actions"][0], "2": body["actions"][1]} and body["n"] == 2 and body["next"] == "listen"
    end = P.tag_actions({"actions": [], "end": True})
    assert end["next"] == "stop" and end["n"] == 0 and end["action"] is None and end["by_index"] == {}
    big = P.tag_actions({"actions": [{"type": "note"}] * 0 + [{"type": "note"} for _ in range(12)]})
    assert big["n"] == P.MAX_ACTIONS and len({a["id"] for a in big["actions"]}) == P.MAX_ACTIONS


def test_talk_query_string_form_carries_context_and_ids(srv):
    s, _ = srv
    code, _, raw = req(s, "/talk?device=iphone&battery=81&location=Leeds&text=set%20a%20timer%20for%20five%20minutes")
    d = json.loads(raw)
    assert code == 200 and d["action"]["type"] == "timer" and d["action"]["id"] == d["by_index"]["1"]["id"] and d["next"] == "listen"
    code, _, raw = req(s, "/talk?device=iphone&battery=81&text=how%20much%20battery%20do%20I%20have", raw=b"{}")
    assert code == 200 and "81" in json.loads(raw)["reply"]
    code, _, raw = req(s, "/talk?device=iphone&text=goodbye")
    assert code == 200 and json.loads(raw)["next"] == "stop"
    code, _, raw = req(s, "/talk", {"text": "turn on the torch", "device": "iphone"})          # the JSON body form still works
    assert code == 200 and json.loads(raw)["action"]["id"]
    assert req(s, "/talk?device=iphone&text=hi", token="wrong")[0] == 401
    log = (srv[1] / "access.log").read_text(encoding="utf-8")
    assert "battery" in log and "timer%20for" not in log and "text=<text>" in log


# ------------------------------------------------------------------------------------------------ /action_report
def test_action_report_query_and_json_whitelisted(srv):
    s, rt = srv
    assert req(s, "/action_report?id=a1b2&type=timer&phase=start")[0] == 200
    code, _, raw = req(s, "/action_report?id=a1b2&type=timer&phase=done&ok=false&detail=no%20preset%20for%20this%20length&junk=1")
    assert code == 200 and json.loads(raw)["ok"] is True
    assert req(s, "/action_report", {"id": "a9", "type": "note", "ok": True, "detail": "x" * 999, "ms": "1200", "evil": "y"})[0] == 200
    assert req(s, "/action_report?id=../etc&type=timer")[0] == 400
    assert req(s, "/action_report?id=a1&type=Timer;rm")[0] == 400
    assert req(s, "/action_report?id=a1&type=timer&phase=boom")[0] == 400
    assert req(s, "/action_report?id=a1&type=timer", token=None)[0] == 401
    rows = [json.loads(x) for x in (rt / P.ACTION_REPORTS).read_text(encoding="utf-8").splitlines()]
    assert [r["phase"] for r in rows] == ["start", "done", "done"]
    assert rows[1]["ok"] is False and rows[1]["detail"] == "no preset for this length" and "junk" not in rows[1]
    assert rows[2]["ok"] is True and len(rows[2]["detail"]) <= 200 and rows[2]["ms"] == 1200 and "evil" not in rows[2]
    assert set(rows[0]) == {"t", "id", "type", "phase", "ok", "detail", "ms"}


# ------------------------------------------------------------------------------------------------ /setup/<key>
def test_setup_link_serves_filled_script_three_times(srv, monkeypatch):
    s, rt = srv
    url = P.make_setup_link(rt, "http://pc.example:8765")
    key = url.rsplit("/", 1)[1]
    assert len(key) == P.SETUP_KEY_LEN and key.isalnum() and key in json.loads((rt / P.SETUP_FILE).read_text())
    for _ in range(P.SETUP_USES):
        code, ctype, raw = req(s, f"/setup/{key}", token=None, method="GET")
        txt = raw.decode("utf-8")
        assert code == 200 and ctype.startswith("text/plain") and f"key={TOKEN}" in txt and 'var base = "http://pc.example:8765"' in txt
        assert "PASTE_" not in txt and J.check(txt, served=True)[0] == []
    assert req(s, f"/setup/{key}", token=None, method="GET")[0] == 404                 # used up
    assert req(s, "/setup/" + "A" * P.SETUP_KEY_LEN, token=None, method="GET")[0] == 404  # unknown
    assert req(s, "/setup/short", token=None, method="GET")[0] == 404
    assert key not in (rt / "access.log").read_text(encoding="utf-8")
    old = P.make_setup_link(rt, "http://pc.example:8765", now=time.time() - P.SETUP_TTL_S - 1).rsplit("/", 1)[1]
    assert req(s, f"/setup/{old}", token=None, method="GET")[0] == 404                 # expired


def test_setup_link_validation_and_cli(tmp_path, monkeypatch, capsys):
    with pytest.raises(ValueError):
        P.make_setup_link(tmp_path, "javascript:alert(1)")
    monkeypatch.setattr(P, "runtime_dir", lambda: tmp_path)
    (tmp_path / "token.txt").write_text(TOKEN, encoding="utf-8")
    assert P.main(["--make-setup-link", "--setup-base", "http://pc.example:8765"]) == 0
    out = capsys.readouterr().out
    assert "http://pc.example:8765/setup/" in out and TOKEN not in out
    assert P.use_setup_link(tmp_path, out.split("/setup/")[1].split()[0]) == "http://pc.example:8765"


def test_setup_rate_limited(tmp_path):
    core = P.Core(Path("."), make_conv=FakeConv)
    s = P.make_server("127.0.0.1", 0, TOKEN, core, per_min=2, voice_rt=tmp_path)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        codes = [req(s, "/setup/" + "B" * P.SETUP_KEY_LEN, token=None, method="GET")[0] for _ in range(3)]
        assert codes == [404, 404, 429]
    finally:
        s.shutdown()
        s.server_close()


# ------------------------------------------------------------------------------------------------ raw uploads without a name
def test_sniff_and_raw_upload_without_extension(srv):
    assert P.sniff_media_ext(b"\x00\x00\x00\x14ftypqt  \x00") == ".mov"
    assert P.sniff_media_ext(b"\x00\x00\x00\x18ftypmp42") == ".mp4"
    assert P.sniff_media_ext(b"RIFF\x00\x00\x00\x00WAVEfmt ") == ".wav"
    assert P.sniff_media_ext(b"\x89PNG\r\n") == ""
    s, rt = srv
    clip = b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00" + b"\x00" * 64
    code, _, raw = req(s, "/voice_upload?consent=friend-agreed&speaker=friend&filename=nupen2", raw=clip, headers={"Content-Type": "video/quicktime"})
    assert code == 200, raw
    stored = list((rt / "voice_inbox").glob("*_nupen2.mov"))
    assert len(stored) == 1 and stored[0].read_bytes() == clip


# ------------------------------------------------------------------------------------------------ convlearn: action_failed_report
def test_convlearn_action_reports_signal(tmp_path, monkeypatch):
    now = time.time()
    reps = [{"t": now - 50, "id": "a1", "type": "timer", "phase": "start"},
            {"t": now - 49, "id": "a1", "type": "timer", "phase": "done", "ok": False, "detail": "no preset for 7 min"},
            {"t": now - 900, "id": "a2", "type": "alarm", "phase": "start"},                         # never finished: stopped the shortcut
            {"t": now - 40, "id": "a3", "type": "note", "phase": "start"},
            {"t": now - 39, "id": "a3", "type": "note", "phase": "done", "ok": True, "detail": "note created"},
            {"t": now - 10, "id": "a4", "type": "volume", "phase": "start"}]                          # still running: not yet a failure
    fails = CL.detect_action_reports(reps, now)
    assert {(f["atype"], f["subtype"]) for f in fails} == {("timer", "action_failed_report"), ("alarm", "action_failed_report")}
    assert next(f for f in fails if f["atype"] == "alarm")["detail"].startswith("started but never finished")
    cands = CL.candidates(fails, 7.0)
    assert {c["key"] for c in cands} == {"conv_failure:action_failed_report:timer", "conv_failure:action_failed_report:alarm"}
    t = next(c for c in cands if c["key"].endswith(":timer"))
    assert t["fix_kind"] == "bug" and t["evidence"]["action_type"] == "timer" and "payload" in t["evidence"]["proposal"]
    # engine hook: a reports file alone (no conversation log) produces candidates and a backlog entry
    monkeypatch.delenv("NUPEN_PHONE_DIR", raising=False)
    monkeypatch.delenv("NUPEN_VOICE_DIR", raising=False)
    state, phone = tmp_path / "state", tmp_path / "phone"
    state.mkdir()
    phone.mkdir()
    (phone / CL.REPORTS_NAME).write_text("".join(json.dumps(r) + "\n" for r in reps) + "not json\n", encoding="utf-8")
    got = CL.detect(state, 7.0, now)
    assert {c["key"] for c in got} == {"conv_failure:action_failed_report:timer", "conv_failure:action_failed_report:alarm"}
    assert CL.side_effects(state, 7.0, now)["backlog"] == 2


def test_copy_page_escapes_and_has_a_copy_button():
    page = P.copy_page('a < b & "c"\n</textarea><script>x</script>')
    assert "Copy the whole script" in page and "execCommand" in page
    assert "&lt;/textarea&gt;&lt;script&gt;" in page and page.count("</textarea>") == 1


def test_app_dialect_matches_what_the_jellycuts_app_accepts():
    src = ('downloadURL(url: "x", method: GET, headers: "{\\"Authorization\\": \\"Bearer abc\\"}") >> r\n'
           'timer(duration: "9 min")\nsetBluetooth(value: false)\n')
    out = P.app_dialect(src)
    assert "downloadURL" not in out and 'text(text: "x?key=abc") >> nnUrl' in out
    lines = out.split("\n")
    assert 'runShortcut(name: "Nupen Net", input: nnUrl' in lines[1] and lines[1].endswith(", show: false) >> rRaw")
    assert lines[2] == "getDictionaryFrom(input: rRaw) >> r"                     # the reply is read as a dictionary
    assert "timer(duration: 9 min)" in out and 'okv = "false"' in out and "setBluetooth" not in out
    served = "\n".join(ln for ln in P.served_script("http://pc.invalid:8765", "tok123").splitlines() if not ln.lstrip().startswith("//"))
    assert "headers:" not in served and "requestJSON" not in served                # the app rejects both literals
    assert 'duration: "' not in served and "setBluetooth(" not in served
    assert "downloadURL(" not in served                                          # broken in the app: every request goes via Nupen Net
    urls = [ln for ln in served.splitlines() if ln.lstrip().startswith("text(text: ") and ">> nnUrl" in ln]
    assert len(urls) >= 6 and all("key=tok123" in ln for ln in urls)
    # every request goes through Nupen Net: the converted downloadURLs plus the signed-shortcut download (h106, >> installUrl)
    assert served.count('runShortcut(name: "Nupen Net"') == len(urls) + served.count("input: installUrl")
    assert 'text(text: "${base}${installPath}?key=tok123") >> installUrl' in served
    import re
    assert not re.search(r"\bif\(\s*\w+\s*(?:==|!=)\s*-?\d", served)          # numbers on the right side are quoted
    assert P.app_dialect('if(tMin == 12) {') == 'if(tMin == "12") {'


def test_key_query_auth_and_get_talk(srv):
    s, rt = srv
    code, _, body = req(s, "/talk?device=iphone&text=hello&key=" + TOKEN, token=None, method="GET")
    assert code == 200 and "reply" in json.loads(body)
    assert req(s, "/talk?text=hello&key=wrong", token=None, method="GET")[0] == 401
    code, _, body = req(s, "/action_report?id=a1b2c3d4&type=timer&phase=start&key=" + TOKEN, token=None, method="GET")
    assert code == 200 and json.loads(body)["ok"] is True
    assert TOKEN not in (rt / "access.log").read_text(encoding="utf-8")       # the key never reaches the access log
