"""Signed .shortcut files for the iPhone, built on this PC: Jelly source -> Open-Jellycore plist (WSL) -> signed AEA1 file.

Why: iOS 15+ imports only Apple-signed shortcut files (or iCloud links), and Apple's signer (`shortcuts sign`) runs only on a Mac.
The phone side then needs no Jellycuts at all: Nupen 2 downloads the signed file through Nupen Net and hands it to Shortcuts,
which shows Apple's Add Shortcut sheet (the owner's one tap). See <runtime>/phone/INSTALL_ROUTE.md.

Signers (chosen by <runtime>/phone/signer.txt, first word; default "shortcuty"):
  shortcuty  free public signing API (sign.shortcuty.app, "anyone" mode, no account). THIRD PARTY: it sees the whole shortcut.
             Only records that pass `remote_ok` go there: never trusted records (Nupen 2 holds the token), never anything with
             the token, an address, a phone number, an e-mail or a street number in it.
  local      shortcut-sign (github.com/0xilis/shortcut-sign) inside WSL with the owner's own Apple ID key + auth data (lines 2-3
             of signer.txt: WSL paths). Zero third parties, but the key can only be dumped from a jailbroken iPhone or an
             AMFI-disabled Mac today, so it is off until the owner has one.
  off        never sign; Nupen 2 falls back to speaking "a new shortcut is ready" (old Jellycuts path).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

SHORTCUTY_URL = "https://sign.shortcuty.app/api/v1/sign"
SIGNERS = ("shortcuty", "local", "off")
DEFAULT_SIGNER = "shortcuty"
SIGNED_MAGIC = b"AEA1"                     # Apple Encrypted Archive: what iOS 15+ accepts (checked on a real Shortcuty reply, 6 Oct)
MAX_PLIST = 2_000_000
HTTP_TIMEOUT_S = 90
LOCAL_BIN = "/usr/local/bin/shortcut-sign"

Http = Callable[[str, bytes, dict[str, str], int], bytes]
Compiler = Callable[[str], bytes]
Runner = Callable[[list[str], int], tuple[int, str]]

# What must never reach a third-party signer. Generated shortcuts never contain web requests (shortcutgen.FORBIDDEN), so the
# token/address checks are a second wall; the rest is the owner's private data that steps like text_to/directions can carry.
_PRIVATE = [
    ("an IP address", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
    ("a web address", re.compile(r"https?://(?!(?:music|maps)\.apple\.com/)", re.I)),    # public Apple search links are fine
    ("a phone or message link", re.compile(r"\b(?:sms|tel|facetime(?:-audio)?|mailto):", re.I)),
    ("a phone number", re.compile(r"\d(?:[\s().-]?\d){6,}")),
    ("an e-mail address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("a street address", re.compile(r"daddr=[^\"&]*\d", re.I)),
    ("a password or token", re.compile(r"\b(?:token|password|passcode|bearer)\b", re.I)),
]


def remote_ok(rec: dict[str, Any], secrets: tuple[str, ...] = ()) -> tuple[bool, str]:
    """May this stored shortcut record be sent to a third-party signer? (ok, reason)."""
    if rec.get("trusted"):
        return False, "trusted records (Nupen 2 itself) carry the token and are never sent to a signing service"
    code = urllib.parse.unquote(str(rec.get("code", "")) + "\n" + str(rec.get("name", "")))      # %20 is not a street number
    for s in secrets:
        if s and s in code:
            return False, "it contains a secret"
    for what, rx in _PRIVATE:
        if rx.search(code):
            return False, f"it contains {what}, which would leave this PC"
    return True, "ok"


def signer_config(rt: Optional[Path]) -> tuple[str, list[str]]:
    """(signer, extra lines) from <rt>/signer.txt; env NUPEN_SIGNER overrides the first word."""
    lines: list[str] = []
    if rt is not None:
        try:
            lines = [ln.strip() for ln in (Path(rt) / "signer.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
        except OSError:
            lines = []
    name = (os.environ.get("NUPEN_SIGNER") or (lines[0].split()[0] if lines else DEFAULT_SIGNER)).lower()
    return (name if name in SIGNERS else "off"), lines[1:]


# ---------------------------------------------------------------------------------------------------------------- compile
def compile_plist(code: str) -> bytes:
    """Jelly -> unsigned shortcut plist with the local Open-Jellycore (WSL). Raises RuntimeError with the diagnostics."""
    from creator import jellyc as J
    if not J.available():
        raise RuntimeError("jelly compiler not installed (WSL + Open-Jellycore)")
    d = Path(tempfile.mkdtemp(prefix="jsign_"))
    try:
        src, out = d / "in.jelly", d / "out.shortcut"
        src.write_text(code.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
        r = J.compile_file(src, export=out)
        if not r["ok"] or not out.is_file():
            raise RuntimeError("compile failed: " + "; ".join(r.get("errors") or ["no output"])[:400])
        data = out.read_bytes()
        if not data or len(data) > MAX_PLIST:
            raise RuntimeError("compiled plist is empty or too large")
        return data
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------- signers
def _http_post(url: str, body: bytes, headers: dict[str, str], timeout: int) -> bytes:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310 - fixed https URL
        return r.read()


def multipart(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = "nupen" + uuid.uuid4().hex
    fn = re.sub(r'["\r\n\\]', "", filename) or "shortcut.shortcut"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{fn}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def sign_shortcuty(plist: bytes, name: str, http: Http = _http_post) -> bytes:
    body, ctype = multipart("file", f"{name}.shortcut", plist)
    try:
        raw = http(SHORTCUTY_URL + "?response=json", body, {"Content-Type": ctype, "User-Agent": "nupen-phone"}, HTTP_TIMEOUT_S)
    except Exception as e:                                       # network, HTTP 4xx/5xx
        raise RuntimeError(f"signing service unreachable or refused ({type(e).__name__}: {str(e)[:160]})") from e
    try:
        d = json.loads(raw.decode("utf-8"))
        f = d.get("file") if isinstance(d.get("file"), dict) else d          # documented flat; live reply nests it under "file"
        import base64
        out = base64.b64decode(f["content_base64"])
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as e:
        raise RuntimeError(f"signing service answered something unexpected: {raw[:160]!r}") from e
    return out


def _wsl_run(cmd: list[str], timeout: int) -> tuple[int, str]:
    from creator import jellyc as J
    return J._run(J._wsl(cmd), timeout)


def sign_local(plist: bytes, key: str, auth: str, run: Runner = _wsl_run, binary: str = "") -> bytes:
    """shortcut-sign in WSL with the owner's own key (ASN1 ECDSA-P256) and auth data. Nothing leaves the PC."""
    from creator import jellyc as J
    d = Path(tempfile.mkdtemp(prefix="jsign_"))
    try:
        src, out = d / "in.shortcut", d / "out.shortcut"
        src.write_bytes(plist)
        rc, msg = run([binary or os.environ.get("NUPEN_SHORTCUT_SIGN_BIN", LOCAL_BIN), "sign", "-i", J.wsl_path(src),
                       "-o", J.wsl_path(out), "-k", key, "-a", auth], 120)
        if rc != 0 or not out.is_file():
            raise RuntimeError(f"shortcut-sign failed ({rc}): {msg[-300:]}")
        return out.read_bytes()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------- build + cache
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(sid: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(sid, threading.Lock())


def signed_path(dir_: Path, sid: str) -> Path:
    return Path(dir_) / f"{sid}.shortcut"


def build(dir_: Path, rec: dict[str, Any], rt: Optional[Path] = None, secrets: tuple[str, ...] = (),
          compile_: Optional[Compiler] = None, http: Optional[Http] = None, run: Optional[Runner] = None) -> dict[str, Any]:
    """Signed file for one stored record, cached next to it as <id>.shortcut. -> {ok, path?, signer, error?}. Thread-safe per id."""
    sid = str(rec.get("id", ""))
    if not re.fullmatch(r"[0-9a-f]{12}", sid):
        return {"ok": False, "signer": "-", "error": "bad id"}
    compile_, http, run = compile_ or compile_plist, http or _http_post, run or _wsl_run     # looked up per call (tests patch them)
    out = signed_path(dir_, sid)
    with _lock(sid):
        if out.is_file() and out.read_bytes()[:4] == SIGNED_MAGIC:
            return {"ok": True, "path": str(out), "signer": "cache"}
        signer, extra = signer_config(rt)
        if signer == "off":
            return {"ok": False, "signer": signer, "error": "signing is off (signer.txt)"}
        if signer == "shortcuty":
            ok, why = remote_ok(rec, secrets)
            if not ok:
                return {"ok": False, "signer": signer, "error": "not sent to the signing service: " + why}
        try:
            plist = compile_(str(rec.get("code", "")))
            if signer == "local":
                if len(extra) < 2:
                    raise RuntimeError("signer 'local' needs the key and auth-data paths on lines 2 and 3 of signer.txt")
                data = sign_local(plist, extra[0], extra[1], run=run)
            else:
                data = sign_shortcuty(plist, str(rec.get("name") or "Nupen shortcut"), http=http)
        except RuntimeError as e:
            return {"ok": False, "signer": signer, "error": str(e)[:400]}
        if data[:4] != SIGNED_MAGIC:
            return {"ok": False, "signer": signer, "error": "the result is not a signed shortcut (no AEA1 header)"}
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, out)
        return {"ok": True, "path": str(out), "signer": signer}
