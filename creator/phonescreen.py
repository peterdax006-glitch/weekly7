"""Nupen's eyes on the iPhone: grab the AirPlay mirroring window (uxplay-windows receiver) to a PNG.

The iPhone mirrors its screen to this PC (Control Center -> Screen Mirroring -> the PC's AirPlay name); the receiver shows it in a
video window. This module finds that window by title, captures it with PrintWindow (falls back to a screen BitBlt of the window's
rectangle) and writes a PNG. No third-party packages: ctypes + a small PNG encoder (zlib), so it runs in any Python here.

    python -m creator.phonescreen list                 # visible windows that look like the mirror
    python -m creator.phonescreen grab [out.png]       # one frame (default <runtime>/phone/screen/latest.png)
    python -m creator.phonescreen loop --every 5       # a frame every 5 s while mirroring is active; keeps the newest --keep files

Frames live outside the repo, in <runtime>/phone/screen/. When no mirror window is open the loop just waits (no files written).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Title fragments of the mirror window, best first. uxplay-windows 2.x renames the GStreamer video window to
# "AirPlay Video Stream (ALT+ENTER for Fullscreen)"; the D3D11/D3D12 sink default titles are fallbacks. Windows owned by the receiver
# process (exe name contains RECEIVER_EXE) beat look-alikes from other programs; small windows and EXCLUDE titles never match.
TITLES = ("airplay video stream", "direct3d11 renderer", "direct3d12 renderer", "uxplay", "airplay")
RECEIVER_EXE = "uxplay"
EXCLUDE = ("uxplay-windows settings", "log viewer", "visual studio code", "chrome", "edge", "firefox", "explorer", "powershell", "terminal")
MIN_SIDE = 200


def screen_dir(rt: Optional[Path] = None) -> Path:
    if rt is None:
        from creator import device as DEV
        rt = Path(DEV.runtime_dir())
    return Path(rt) / "phone" / "screen"


# --------------------------------------------------------------------------------------------- PNG (pure python)
def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def encode_png(width: int, height: int, rgb: bytes, level: int = 6) -> bytes:
    """8-bit RGB rows, top to bottom -> PNG bytes."""
    stride = width * 3
    if len(rgb) != stride * height:
        raise ValueError(f"need {stride * height} bytes of RGB, got {len(rgb)}")
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, level)) + _chunk(b"IEND", b"")


def bgra_to_rgb(bgra: bytes) -> bytes:
    """Windows DIB pixels (B,G,R,A) -> R,G,B."""
    out = bytearray(len(bgra) // 4 * 3)
    out[0::3] = bgra[2::4]
    out[1::3] = bgra[1::4]
    out[2::3] = bgra[0::4]
    return bytes(out)


def is_blank(rgb: bytes, sample: int = 4096) -> bool:
    """True when the frame is all one value (black/blank: a GPU window PrintWindow could not read)."""
    if not rgb:
        return True
    step = max(1, len(rgb) // sample)
    s = rgb[::step]
    return min(s) == max(s)


# --------------------------------------------------------------------------------------------- picking the mirror window
def pick_window(wins: Iterable[dict[str, Any]], titles: tuple[str, ...] = TITLES) -> Optional[dict[str, Any]]:
    """wins: [{hwnd, title, w, h, exe?}] -> the best mirror candidate or None. Receiver-owned first, then earlier TITLES, then larger."""
    best, key = None, None
    for w in wins:
        t = str(w.get("title", "")).lower()
        if not t or any(x in t for x in EXCLUDE) or min(int(w.get("w", 0)), int(w.get("h", 0))) < MIN_SIDE:
            continue
        rank = next((i for i, frag in enumerate(titles) if frag in t), None)
        if rank is None:
            continue
        foreign = 0 if RECEIVER_EXE in str(w.get("exe", "")).lower() else 1
        k = (foreign, rank, -int(w["w"]) * int(w["h"]))
        if key is None or k < key:
            best, key = w, k
    return best


# --------------------------------------------------------------------------------------------- Win32 (ctypes)
def _win32():
    import ctypes
    from ctypes import wintypes as W
    u32, g32 = ctypes.windll.user32, ctypes.windll.gdi32
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)                  # real pixels, not scaled ones
    except Exception:
        try:
            u32.SetProcessDPIAware()
        except Exception:
            pass
    return ctypes, W, u32, g32


def _exe_of(ctypes, W, u32, hwnd) -> str:
    """Owning process's exe file name ('' when it cannot be read)."""
    pid = W.DWORD()
    u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid.value)                                    # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf, n = ctypes.create_unicode_buffer(520), W.DWORD(520)
        return buf.value.rsplit("\\", 1)[-1] if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) else ""
    finally:
        k32.CloseHandle(h)


def list_windows() -> list[dict[str, Any]]:
    ctypes, W, u32, _ = _win32()
    out: list[dict[str, Any]] = []
    proto = ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)

    def cb(hwnd, _l):
        if not u32.IsWindowVisible(hwnd) or u32.IsIconic(hwnd):
            return True
        n = u32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        u32.GetWindowTextW(hwnd, buf, n + 1)
        r = W.RECT()
        u32.GetWindowRect(hwnd, ctypes.byref(r))
        out.append({"hwnd": int(hwnd), "title": buf.value, "x": r.left, "y": r.top, "w": r.right - r.left, "h": r.bottom - r.top,
                    "exe": _exe_of(ctypes, W, u32, hwnd)})
        return True

    u32.EnumWindows(proto(cb), 0)
    return out


def capture_window(hwnd: int) -> tuple[int, int, bytes, str]:
    """-> (width, height, rgb, method). Client area via PrintWindow(PW_RENDERFULLCONTENT); blank result -> screen BitBlt of the same area."""
    ctypes, W, u32, g32 = _win32()
    r = W.RECT()
    u32.GetClientRect(hwnd, ctypes.byref(r))
    w, h = r.right - r.left, r.bottom - r.top
    if w <= 0 or h <= 0:
        raise RuntimeError("mirror window has no client area")

    class BIH(ctypes.Structure):
        _fields_ = [("biSize", W.DWORD), ("biWidth", W.LONG), ("biHeight", W.LONG), ("biPlanes", W.WORD), ("biBitCount", W.WORD),
                    ("biCompression", W.DWORD), ("biSizeImage", W.DWORD), ("biXPelsPerMeter", W.LONG), ("biYPelsPerMeter", W.LONG),
                    ("biClrUsed", W.DWORD), ("biClrImportant", W.DWORD)]

    def grab(src_dc_fn) -> bytes:
        hdc_screen = u32.GetDC(0)
        mem = g32.CreateCompatibleDC(hdc_screen)
        bmp = g32.CreateCompatibleBitmap(hdc_screen, w, h)
        old = g32.SelectObject(mem, bmp)
        try:
            src_dc_fn(mem)
            bih = BIH(ctypes.sizeof(BIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)            # negative height = top-down rows
            buf = ctypes.create_string_buffer(w * h * 4)
            g32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bih), 0)
            return buf.raw
        finally:
            g32.SelectObject(mem, old)
            g32.DeleteObject(bmp)
            g32.DeleteDC(mem)
            u32.ReleaseDC(0, hdc_screen)

    rgb = bgra_to_rgb(grab(lambda mem: u32.PrintWindow(hwnd, mem, 1 | 2)))           # PW_CLIENTONLY | PW_RENDERFULLCONTENT
    if not is_blank(rgb):
        return w, h, rgb, "printwindow"
    pt = W.POINT(0, 0)
    u32.ClientToScreen(hwnd, ctypes.byref(pt))

    def blit(mem):
        sdc = u32.GetDC(0)
        g32.BitBlt(mem, 0, 0, w, h, sdc, pt.x, pt.y, 0x00CC0020 | 0x40000000)        # SRCCOPY | CAPTUREBLT
        u32.ReleaseDC(0, sdc)
    return w, h, bgra_to_rgb(grab(blit)), "bitblt"


# --------------------------------------------------------------------------------------------- grab / loop
def grab(out: Optional[Path] = None, rt: Optional[Path] = None, lister: Callable[[], list] = list_windows,
         capturer: Callable[[int], tuple] = capture_window) -> dict[str, Any]:
    """One frame -> PNG. Returns {ok, path, w, h, title, method} or {ok: False, why}."""
    win = pick_window(lister())
    if win is None:
        return {"ok": False, "why": "no mirror window open (on the iPhone: Control Center -> Screen Mirroring -> Nupen-PC)"}
    w, h, rgb, method = capturer(win["hwnd"])
    if is_blank(rgb):
        return {"ok": False, "why": "mirror window is blank (phone not mirroring yet, or the window is covered/minimised)", "title": win["title"]}
    out = Path(out) if out else screen_dir(rt) / "latest.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(encode_png(w, h, rgb, level=3))
    tmp.replace(out)
    return {"ok": True, "path": str(out), "w": w, "h": h, "title": win["title"], "method": method, "t": time.time()}


def prune(d: Path, keep: int) -> int:
    files = sorted(d.glob("frame_*.png"))
    gone = files[:-keep] if keep > 0 else files
    for f in gone:
        f.unlink(missing_ok=True)
    return len(gone)


def loop(every: float = 5.0, keep: int = 200, rt: Optional[Path] = None, max_frames: Optional[int] = None,
         grabber: Callable[..., dict] = grab, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time) -> int:
    """Save a frame every `every` s while mirroring is active: frame_<time>.png + latest.png + status.json. Returns frames saved."""
    d = screen_dir(rt)
    d.mkdir(parents=True, exist_ok=True)
    saved = 0
    while max_frames is None or saved < max_frames:
        t0 = clock()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(t0)) + f"_{int(t0 * 1000) % 1000:03d}"
        r = grabber(d / f"frame_{stamp}.png", rt=rt)
        if r.get("ok"):
            saved += 1
            (d / "latest.png").write_bytes(Path(r["path"]).read_bytes())
            prune(d, keep)
        (d / "status.json").write_text(json.dumps({**r, "saved": saved, "checked": t0}), encoding="utf-8")
        if max_frames is not None and saved >= max_frames:
            break
        sleep(max(0.2, every - (clock() - t0)) if r.get("ok") else max(every, 3.0))
    return saved


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="phonescreen", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("grab")
    g.add_argument("out", nargs="?")
    lp = sub.add_parser("loop")
    lp.add_argument("--every", type=float, default=5.0)
    lp.add_argument("--keep", type=int, default=200)
    lp.add_argument("--frames", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        wins = list_windows()
        pick = pick_window(wins)
        for w in wins:
            if w["w"] * w["h"] > 0:
                print(("* " if pick is w else "  ") + json.dumps({k: w.get(k) for k in ("hwnd", "title", "exe", "w", "h")}))
        return 0 if pick else 1
    if a.cmd == "grab":
        r = grab(Path(a.out) if a.out else None)
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    return 0 if loop(a.every, a.keep, max_frames=a.frames) >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
