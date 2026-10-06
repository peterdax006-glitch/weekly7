"""creator/phonescreen.py: PNG encoder, pixel conversion, mirror-window choice, grab and loop with fake windows (no phone, no Win32)."""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

from creator import phonescreen as PS


def _decode_png(b: bytes) -> tuple[int, int, bytes]:
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    i, w, h, idat = 8, 0, 0, b""
    while i < len(b):
        n = struct.unpack(">I", b[i:i + 4])[0]
        tag, data, crc = b[i + 4:i + 8], b[i + 8:i + 8 + n], struct.unpack(">I", b[i + 8 + n:i + 12 + n])[0]
        assert zlib.crc32(tag + data) & 0xFFFFFFFF == crc
        if tag == b"IHDR":
            w, h = struct.unpack(">II", data[:8])
            assert data[8:10] == b"\x08\x02"                                            # 8-bit RGB
        elif tag == b"IDAT":
            idat += data
        i += 12 + n
    raw = zlib.decompress(idat)
    rows = [raw[y * (w * 3 + 1):(y + 1) * (w * 3 + 1)] for y in range(h)]
    assert all(r[0] == 0 for r in rows)                                                  # filter type None
    return w, h, b"".join(r[1:] for r in rows)


def test_png_roundtrip_and_size_check():
    rgb = bytes(range(256)) * 3 + bytes(48)                                              # 3x? arbitrary bytes, 816 = 16x17x3
    w, h = 16, 17
    out = PS.encode_png(w, h, rgb)
    assert _decode_png(out) == (w, h, rgb)
    try:
        PS.encode_png(4, 4, b"\x00" * 10)
        raise AssertionError("short buffer accepted")
    except ValueError:
        pass


def test_bgra_to_rgb_and_blank():
    assert PS.bgra_to_rgb(bytes([1, 2, 3, 255, 10, 20, 30, 0])) == bytes([3, 2, 1, 30, 20, 10])
    assert PS.is_blank(b"\x00" * 3000) and PS.is_blank(b"")
    assert not PS.is_blank(b"\x00" * 3000 + b"\x05")


def test_pick_window_prefers_receiver_video_window():
    wins = [
        {"hwnd": 1, "title": "Stock picker - airplay notes - Chrome", "w": 1900, "h": 1100, "exe": "chrome.exe"},
        {"hwnd": 2, "title": "uxplay-windows", "w": 120, "h": 80, "exe": "uxplay-windows.exe"},             # too small: settings/tray
        {"hwnd": 3, "title": "AirPlay notes.txt", "w": 900, "h": 900, "exe": "notepad.exe"},
        {"hwnd": 4, "title": "Direct3D11 renderer", "w": 600, "h": 1200, "exe": "uxplay-windows.exe"},
        {"hwnd": 5, "title": "AirPlay Video Stream (ALT+ENTER for Fullscreen)", "w": 590, "h": 1250, "exe": "uxplay-windows.exe"},
    ]
    assert PS.pick_window(wins)["hwnd"] == 5
    assert PS.pick_window(wins[:4])["hwnd"] == 4
    assert PS.pick_window(wins[:3])["hwnd"] == 3                                          # a foreign look-alike only when nothing else
    assert PS.pick_window(wins[:2]) is None
    assert PS.pick_window([]) is None


def _fake(rgb: bytes, w: int = 4, h: int = 2):
    wins = [{"hwnd": 7, "title": "AirPlay Video Stream (ALT+ENTER for Fullscreen)", "w": 400, "h": 800, "exe": "uxplay-windows.exe"}]
    return (lambda: wins), (lambda hwnd: (w, h, rgb, "fake"))


def test_grab_writes_png_or_explains(tmp_path: Path):
    rgb = bytes(range(24))
    lister, cap = _fake(rgb)
    r = PS.grab(rt=tmp_path, lister=lister, capturer=cap)
    assert r["ok"] and r["method"] == "fake" and (r["w"], r["h"]) == (4, 2)
    assert Path(r["path"]) == tmp_path / "phone" / "screen" / "latest.png"
    assert _decode_png(Path(r["path"]).read_bytes()) == (4, 2, rgb)
    none = PS.grab(rt=tmp_path, lister=lambda: [], capturer=cap)
    assert not none["ok"] and "Screen Mirroring" in none["why"]
    blank = PS.grab(rt=tmp_path, lister=lister, capturer=_fake(b"\x00" * 24)[1])
    assert not blank["ok"] and "blank" in blank["why"]


def test_loop_saves_prunes_and_waits_when_not_mirroring(tmp_path: Path):
    lister, cap = _fake(bytes(range(24)))
    state = {"t": 1_700_000_000.0, "n": 0}
    sleeps: list[float] = []

    def clock():
        return state["t"]

    def sleep(s):
        sleeps.append(s)
        state["t"] += s

    def grabber(out, rt=None):
        state["n"] += 1
        if state["n"] == 2:                                                              # mirroring paused once
            return PS.grab(out, rt=rt, lister=lambda: [], capturer=cap)
        return PS.grab(out, rt=rt, lister=lister, capturer=cap)

    saved = PS.loop(every=5, keep=3, rt=tmp_path, max_frames=5, grabber=grabber, sleep=sleep, clock=clock)
    d = tmp_path / "phone" / "screen"
    assert saved == 5
    assert len(list(d.glob("frame_*.png"))) == 3                                         # pruned to --keep
    assert (d / "latest.png").exists()
    st = json.loads((d / "status.json").read_text(encoding="utf-8"))
    assert st["ok"] and st["saved"] == 5
    assert 5.0 in sleeps and len(sleeps) == 5                                             # waits between frames, also when no window


def test_prune_keeps_newest(tmp_path: Path):
    for i in range(6):
        (tmp_path / f"frame_2026100{i}.png").write_bytes(b"x")
    assert PS.prune(tmp_path, 2) == 4
    assert sorted(p.name for p in tmp_path.glob("frame_*.png")) == ["frame_20261004.png", "frame_20261005.png"]
