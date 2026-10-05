"""voice upload (auth, cap, consent, byte-identical storage) and creator/voiceprep.py on synthetic audio (no real voice data)."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from creator import voiceprep as VP

SPEC = importlib.util.spec_from_file_location("nupen_phone_v", Path(__file__).resolve().parents[1] / "scripts" / "nupen_phone.py")
P = importlib.util.module_from_spec(SPEC)
sys.modules["nupen_phone_v"] = P
SPEC.loader.exec_module(P)
TOKEN = "t" * 32
SR = VP.SR
RNG = np.random.default_rng(7)


def voiced(seconds: float, f0: float = 120.0, noise: float = 0.002) -> np.ndarray:
    """Speech-like: harmonics at f0 with a 4 Hz syllable envelope plus faint noise."""
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 12))
    env = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 0.7
    return (0.25 * x / 3 * env + RNG.normal(0, noise, len(t))).astype(np.float32)


def quiet(seconds: float) -> np.ndarray:
    return RNG.normal(0, 0.002, int(seconds * SR)).astype(np.float32)


def wav_bytes(x: np.ndarray, tmp: Path, name: str = "a.wav") -> bytes:
    p = tmp / name
    VP.write_wav(p, x)
    return p.read_bytes()


def multipart(file_bytes: bytes, filename: str, **fields: str) -> tuple[bytes, str]:
    b = "XBOUNDARYX"
    out = b""
    for k, v in fields.items():
        out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    out += f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: audio/x-wav\r\n\r\n'.encode()
    return out + file_bytes + f"\r\n--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


@pytest.fixture()
def srv(tmp_path):
    runs: list[Path] = []
    core = P.Core(Path("."), make_conv=lambda: None)
    s = P.make_server("127.0.0.1", 0, TOKEN, core, voice_rt=tmp_path, on_voice=runs.append)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield s, tmp_path, runs
    s.shutdown()
    s.server_close()


def post(s, body: bytes, ctype: str, path: str = "/voice_upload", token: str | None = TOKEN, headers: dict | None = None):
    req = urllib.request.Request(f"http://127.0.0.1:{s.server_address[1]}{path}", data=body, method="POST")
    req.add_header("Content-Type", ctype)
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_auth_required(srv):
    s, rt, _ = srv
    body, ct = multipart(b"RIFFxxxx", "a.wav", consent="friend-agreed")
    assert post(s, body, ct, token=None)[0] == 401
    assert post(s, body, ct, token="wrong")[0] == 401
    assert not (rt / "voice_inbox").exists()


def test_consent_required_then_recorded_once(srv):
    s, rt, runs = srv
    body, ct = multipart(b"RIFFxxxx", "a.wav")
    assert post(s, body, ct)[0] == 403
    assert not list((rt / "voice_inbox").glob("*.wav"))
    body, ct = multipart(b"RIFFxxxx", "a.wav", consent="friend-agreed", speaker="sam", statement="Owner says Sam agreed")
    code, out = post(s, body, ct)
    assert code == 200 and out["reply"].startswith("Received, sir.")
    c = json.loads((rt / "voice_inbox" / "CONSENT.json").read_text())
    assert c["sam"]["owner_statement"] == "Owner says Sam agreed" and c["sam"]["date"]
    body, ct = multipart(b"RIFFyyyy", "b.wav", speaker="sam")          # later upload for a consented speaker: no flag needed
    assert post(s, body, ct)[0] == 200
    assert len(runs) == 2
    body, ct = multipart(b"RIFFyyyy", "c.wav", speaker="other")        # a new speaker needs its own consent
    assert post(s, body, ct)[0] == 403


def test_stored_byte_identical_multipart_and_raw(srv, tmp_path):
    s, rt, _ = srv
    data = bytes(range(256)) * 50 + b"\r\n--NOTTHEBOUNDARY\r\n\x00\xff"
    body, ct = multipart(data, "my clip!.m4a", consent="friend-agreed")
    code, out = post(s, body, ct)
    assert code == 200
    assert (rt / "voice_inbox" / out["stored"]).read_bytes() == data
    assert " " not in out["stored"] and "!" not in out["stored"]
    code, out = post(s, data, "application/octet-stream", headers={"X-Filename": "../../evil.mp3", "X-Consent": "friend-agreed"})
    assert code == 200 and (rt / "voice_inbox" / out["stored"]).read_bytes() == data
    assert ".." not in out["stored"] and (rt / "voice_inbox" / out["stored"]).parent == rt / "voice_inbox"


def test_type_and_size_cap(srv, monkeypatch):
    s, rt, _ = srv
    body, ct = multipart(b"abc", "a.exe", consent="friend-agreed")
    assert post(s, body, ct)[0] == 415
    for ext in ("m4a", "mp3", "wav", "aac", "caf"):
        body, ct = multipart(b"abc", f"a.{ext}", consent="friend-agreed")
        assert post(s, body, ct)[0] == 200
    monkeypatch.setattr(VP, "MAX_FILE", 1000)
    body, ct = multipart(b"x" * 200000, "big.wav", consent="friend-agreed")
    assert post(s, body, ct)[0] == 413
    assert VP.MAX_FILE == 1000 and VP.store_upload(rt, "big.wav", b"x" * 1001, {"consent": "friend-agreed"})[0] == 413


def test_cap_is_200mb():
    src = Path(VP.__file__).read_text()
    assert "MAX_FILE = 200 * 1024 * 1024" in src


def test_spoken_minutes():
    assert VP.spoken_minutes(2.2) == "Two minutes of speech so far."
    assert VP.spoken_minutes(0.1).startswith("Less than a minute")
    assert VP.spoken_minutes(1.0) == "One minute of speech so far."


def test_split_and_filters(tmp_path):
    other = voiced(4, f0=260.0)
    clipped = np.clip(voiced(3) * 12, -1, 1)
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(4 * SR) / SR)).astype(np.float32)
    noise = RNG.normal(0, 0.05, 4 * SR).astype(np.float32)
    parts = []
    for _ in range(6):
        parts += [voiced(3.5), quiet(0.8)]
    x = np.concatenate(parts + [clipped, quiet(0.8), tone, quiet(0.8), noise, quiet(0.8)] + [other, quiet(0.8)])
    inbox = tmp_path / "phone" / "voice_inbox"
    inbox.mkdir(parents=True)
    (inbox / "20261005-000000-000000_friend_t.wav").write_bytes(wav_bytes(x, tmp_path))
    out = tmp_path / "out"
    r = VP.process_inbox(tmp_path / "phone", out, transcribe=lambda p: "hello there friend", tool_label="fake")
    assert r["pieces_kept"] >= 5 and r["usable_minutes"] > 0.2
    d = r["dropped_by_reason"]
    assert d.get("clipping", 0) >= 1
    assert d.get("not_speech_or_music", 0) + d.get("noisy_or_silent", 0) >= 2     # tone and noise
    assert r["decoder"] in ("ffmpeg", "python-wav")
    rows = [ln.split("|") for ln in (out / "metadata.csv").read_text().splitlines()]
    assert len(rows) == r["pieces_kept"] and all(len(c) == 3 and c[1] == "hello there friend" for c in rows)
    for c in rows:
        with VP.wave.open(str(out / "wavs" / (c[0] + ".wav"))) as w:
            assert w.getframerate() == SR and w.getnchannels() == 1 and VP.MIN_UTT <= w.getnframes() / SR <= VP.MAX_UTT
    again = VP.process_inbox(tmp_path / "phone", out, transcribe=lambda p: "x", tool_label="fake")     # idempotent
    assert again["pieces_kept"] == r["pieces_kept"]
    assert json.loads((out / "REPORT.json").read_text())["transcriber"] == "fake"


def test_long_speech_is_cut_at_15s():
    x = voiced(40.0)
    spans = VP.split_utterances(x)
    assert spans and all((b - a) / SR <= VP.MAX_UTT + 0.01 for a, b in spans)


def test_mp3_decoded_by_ffmpeg(tmp_path):
    ff = VP._tool("ffmpeg")
    if not ff:
        pytest.skip("no ffmpeg")
    import subprocess
    src = tmp_path / "s.wav"
    VP.write_wav(src, voiced(3.0))
    mp3 = tmp_path / "s.mp3"
    subprocess.run([ff, "-y", "-v", "error", "-i", str(src), str(mp3)], check=True)
    dst = tmp_path / "d.wav"
    assert VP.decode(mp3, dst) == "ffmpeg"
    x, sr = VP.read_wav(dst)
    assert sr == SR and 2.5 < len(x) / SR < 3.6
    assert shutil.which("ffprobe") is None or VP.duration_seconds(mp3) > 2.5


# ---- one Nupen: the voice upload is a normal /talk request ----

def _core(tmp_path):
    c = P.Core(Path("."), chat=lambda m, more: "chat reply")
    c.voice_rt = tmp_path
    return c


def test_talk_asks_consent_then_returns_upload_action(tmp_path):
    c = _core(tmp_path)
    code, b = c.talk("iphone", "I want to send you my friend's voice clips")
    assert b["reply"] == "Has your friend agreed to lend their voice, sir?" and b["action"] is None
    assert not (tmp_path / "voice_inbox" / "CONSENT.json").exists()
    code, b = c.talk("iphone", "yes")
    assert b["reply"] == "Of course, sir. Choose the clips."
    a = b["action"]
    assert a["type"] == "send_media" and a["purpose"] == "voice" and a["endpoint"] == "/voice_upload" and "url" not in a
    assert a["fields"] == {"consent": "friend-agreed", "speaker": "friend"}
    assert "shortcut" not in a
    rec = json.loads((tmp_path / "voice_inbox" / "CONSENT.json").read_text())["friend"]
    assert rec["date"] and "yes" in rec["owner_statement"]
    code, b = c.talk("iphone", "take these voice clips")             # consent already on record: no question
    assert b["action"]["type"] == "send_media"


def test_talk_consent_no_cancels_and_off_without_runtime(tmp_path):
    c = _core(tmp_path)
    c.talk("iphone", "send you some voice recordings")
    code, b = c.talk("iphone", "no")
    assert b["action"] is None and not (tmp_path / "voice_inbox" / "CONSENT.json").exists()
    code, b = c.talk("iphone", "yes")                                # a stray yes later records nothing
    assert not (tmp_path / "voice_inbox" / "CONSENT.json").exists()
    off = P.Core(Path("."), chat=lambda m, more: "chat reply")
    assert off.talk("iphone", "take these voice clips")[1]["action"] is None


# ---- recompiling keeps the pinned voice entries ----

def test_recompile_keeps_pinned_voice_entries(tmp_path, monkeypatch):
    from creator import gpucompile as GC
    from creator import pinned_extras as PX
    pins = PX.load()
    assert {"p2.voice_tts", "p2.voice_persona_data", "p2.voice_persona_17b"} <= set(pins)
    wl = GC.phase2_wishlist(tmp_path)                                  # the compiler's own wishlist, merged with the pins
    ids = {w["id"] for w in wl}
    assert "p2.voice_tts" not in ids                                   # external: not a compiled job
    out = tmp_path / "pkg"
    (out / "specs").mkdir(parents=True)
    (out / "wishlist.json").write_text(json.dumps({"p2.voice_tts": {"id": "p2.voice_tts", "stale": True}}))
    monkeypatch.setattr(GC, "render_timeline", lambda *x, **k: "timeline")
    GC.write_package({"jobs": [], "schedule": {"order": []}}, out, [{"id": "p2.x", "name": "x"}])
    written = json.loads((out / "wishlist.json").read_text())
    assert "p2.x" in written and written["p2.voice_tts"].get("target_minutes") is None
    assert written["p2.voice_tts"]["extra"]["target_minutes"] == "30-60" and "stale" not in written["p2.voice_tts"]
    assert "external" not in written["p2.voice_tts"] and "p2.voice_persona_data" in written


def test_merge_replaces_by_id_and_drops_entries_that_need_skipped_pins():
    from creator import pinned_extras as PX
    pins = {"a": {"id": "a", "kind": "infer"}, "b": {"id": "b", "deps": ["a"]}, "c": {"id": "c", "x": 1}}
    out = PX.merge([{"id": "c", "x": 0}, {"id": "z"}], pins, lambda e: e.get("kind") != "infer")
    assert [w["id"] for w in out] == ["c", "z"] and out[0]["x"] == 1
    assert [w["id"] for w in PX.merge([], pins)] == ["a", "b", "c"]


# ---- send_media: parsing, video accept, audio extraction ----

@pytest.mark.parametrize("text,src,kind,count,album,since", [
    ("I want to send you my last 3 videos", "latest", "video", 3, "", ""),
    ("pull the latest video from my photos", "latest", "video", 1, "", ""),
    ("take my last two voice clips", "latest", "any", 2, "", ""),
    ("send you everything in my Nupen Voice album", "album", "any", 0, "Nupen Voice", ""),
    ("upload all the videos in the Friend Clips album", "album", "video", 0, "Friend Clips", ""),
    ("grab the videos from today", "latest", "video", 50, "", "today"),
    ("take these voice clips", "latest", "any", 5, "", ""),
    ("send you the last 4 photos", "latest", "photo", 4, "", ""),
])
def test_parse_media_request(text, src, kind, count, album, since):
    assert VP.parse_media_request(text) == {"source": src, "kind": kind, "count": count, "album": album, "since": since}


def test_parse_ignores_other_talk():
    for t in ("set a timer for five minutes", "what's the weather", "tell me a joke about videos"):
        assert VP.parse_media_request(t) is None


def test_talk_returns_send_media_action_after_consent(tmp_path):
    c = _core(tmp_path)
    code, b = c.talk("iphone", "pull my last 3 videos")
    assert b["reply"] == "Has your friend agreed to lend their voice, sir?"
    code, b = c.talk("iphone", "yes he has")
    a = b["action"]
    assert b["reply"] == "Of course, sir. Choose the clips." or b["reply"].startswith("Of course")
    assert a["type"] == "send_media" and a["source"] == "latest" and a["kind"] == "video" and a["count"] == 3
    assert a["endpoint"] == "/voice_upload" and a["fields"]["consent"] == "friend-agreed" and a["purpose"] == "voice" and "url" not in a and "shortcut" not in a
    code, b = c.talk("iphone", "send you everything in my Nupen Voice album")
    assert b["action"]["source"] == "album" and b["action"]["album"] == "Nupen Voice" and b["action"]["count"] == 0


def test_video_accepted_stored_exact_and_audio_extracted(srv, tmp_path):
    ff = VP._tool("ffmpeg")
    if not ff:
        pytest.skip("no ffmpeg")
    import subprocess
    s, rt, _ = srv
    wav = tmp_path / "v.wav"
    VP.write_wav(wav, voiced(3.0))
    mov = tmp_path / "clip.mov"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=3", "-i", str(wav), "-shortest", "-pix_fmt", "yuv420p", str(mov)], check=True)
    data = mov.read_bytes()
    for ext in ("mov", "mp4", "m4v", "3gp"):
        body, ct = multipart(data, f"IMG_1.{ext}", consent="friend-agreed")
        code, out = post(s, body, ct)
        assert code == 200 and (rt / "voice_inbox" / out["stored"]).read_bytes() == data
    dst = tmp_path / "d.wav"
    assert VP.decode(mov, dst) == "ffmpeg"
    x, sr = VP.read_wav(dst)
    assert sr == SR and 2.5 < len(x) / SR < 3.6 and np.abs(x).max() > 0.05


def test_video_cap_is_500mb_and_audio_200mb(monkeypatch):
    assert VP.MAX_FILE == 200 * 1024 * 1024 and VP.MAX_VIDEO == 500 * 1024 * 1024
    monkeypatch.setattr(VP, "MAX_FILE", 1000)
    monkeypatch.setattr(VP, "MAX_VIDEO", 5000)
    rt = Path(__file__).parent / "_unused"
    f = {"consent": "friend-agreed"}
    assert VP.store_upload(rt, "a.m4a", b"x" * 1001, f)[0] == 413
    assert VP.store_upload(rt, "a.mov", b"x" * 5001, f)[0] == 413
