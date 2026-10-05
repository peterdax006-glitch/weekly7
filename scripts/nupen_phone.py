"""Talk to Nupen from a phone (home Wi-Fi). A tiny HTTP server on the PC; the phone's Shortcut POSTs what you said and speaks the reply.

    python scripts/nupen_phone.py                 # binds the PC's LAN address + Tailscale address if present (detected at start), port 8765
    python scripts/nupen_phone.py --host 127.0.0.1   # local only (smoke tests)
    python scripts/nupen_phone.py --print-firewall   # the admin command to allow the port (printed, never run)
    python scripts/nupen_phone.py --print-startup    # write the optional Windows startup .cmd (not installed)

    POST /talk   {"text": "...", "device": "iphone"}  ->  {"reply": "...", "ms": N, "more": bool}      GET /health
    Header  Authorization: Bearer <token>   (token: <runtime>/phone/token.txt, generated once, outside the repo)

Talk only: pause/resume/approve/request are refused from the phone (they need the terminal); status and open questions are read-only.
It works while Nupen is halted (NUPEN_STOP): this process only loads the voice model (refused when < 2 GB RAM would stay free; unloaded
after --idle-min idle minutes). Never binds 0.0.0.0 unless --allow-all is given. Access log: <runtime>/phone/access.log."""
from __future__ import annotations

import argparse
import hmac
import json
import re
import secrets
import socket
import sys
import threading
import time
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PORT = 8765
MAX_BODY = 4096
MAX_TEXT = 600
MAX_WORDS = 60
RATE_PER_MIN = 30
MIN_FREE_GB = 2.0
CGNAT = "100." + "64.0.0/10"                 # Tailscale's address range
MAX_DEVICES = 8
BLOCKED = ("pause", "resume", "approve", "request")
MORE = re.compile(r"^\s*(?:say|tell me|go on|continue|more|and|keep going)\b.{0,20}$", re.I)
_EVID = re.compile(r"\n?\[evidence:.*?\]", re.S)


def runtime_dir() -> Path:
    from creator import device as DEV
    return Path(DEV.runtime_dir()) / "phone"


def token_path(rt: Optional[Path] = None) -> Path:
    return (rt or runtime_dir()) / "token.txt"


def make_token(rt: Optional[Path] = None) -> Path:
    """Generate the token once (never overwrites)."""
    p = token_path(rt)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
    return p


def load_token(rt: Optional[Path] = None) -> str:
    p = token_path(rt)
    t = p.read_text(encoding="utf-8").strip() if p.is_file() else ""
    if len(t) < 16:
        raise SystemExit(f"no usable token file at {p}; run with --make-token once")
    return t


def lan_ip() -> str:
    """This PC's LAN address (the interface a UDP socket to a private address would use; nothing is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((".".join(("10", "255", "255", "255")), 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


PERSONAS = {
    "butler": ("Speak as a calm, witty British AI assistant: precise, polite, understated, with light dry wit. Address the owner as "
               "\"sir\". Keep it to one to three short spoken sentences, no lists, no brackets, no markdown."),
    "plain": "Speak plainly and briefly, in one to three short spoken sentences, no lists or markdown.",
}


def persona_text(rt: Optional[Path] = None) -> str:
    """<runtime>/phone/persona.json {"persona": "butler"|"plain", "custom": "<own style text>"}; default butler; custom text wins."""
    try:
        d = json.loads(((rt or runtime_dir()) / "persona.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    custom = str(d.get("custom") or "").strip()[:500] if isinstance(d, dict) else ""
    name = str(d.get("persona") or "butler") if isinstance(d, dict) else "butler"
    return custom or PERSONAS.get(name, PERSONAS["butler"])


def with_persona(messages: list[dict[str, str]], persona: str) -> list[dict[str, str]]:
    """The persona is added to the speaking prompt only (understanding stays JSON-only); facts and grounding checks are untouched."""
    if not persona or not messages or messages[0].get("role") != "system" or not messages[0]["content"].startswith("You are Nupen"):
        return messages
    return [dict(messages[0], content=messages[0]["content"] + "\nStyle: " + persona)] + list(messages[1:])


def tailscale_ips() -> list[str]:
    """This PC's Tailscale addresses (the CGNAT range Tailscale uses), detected at start; [] when Tailscale is absent."""
    import ipaddress
    found: list[str] = []
    try:
        import psutil
        found += [a.address for v in psutil.net_if_addrs().values() for a in v if a.family == socket.AF_INET]
    except Exception:  # noqa: BLE001
        pass
    try:
        found += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    net = ipaddress.ip_network(CGNAT)
    out: list[str] = []
    for a in found:
        try:
            if ipaddress.ip_address(a) in net and a not in out:
                out.append(a)
        except ValueError:
            pass
    return out


FACT_CHARS = 1000
SPOKEN_TOKENS = 60                      # ~40 words; 'say more' re-asks with MORE_TOKENS
MORE_TOKENS = 150
THREADS = 6
CTX = 2048


def pick_model(name: str, free_gb: Optional[float]) -> str:
    """The 1.7B voice when memory is comfortable, else the 0.6B base model (smaller and ~3x faster)."""
    if name != "1.7b" or free_gb is None or free_gb >= 4.0:
        return name
    return "0.6b"


def make_voice(T: Any, model: str, persona: str, ram_free: Optional[Callable[[], Optional[float]]] = None) -> Any:
    """A Voice over a llama.cpp server OWNED by this process: Normal priority (interactive service), 6 threads, kept warm, one slot,
    cache_prompt so the fixed system prompt (persona included) is not re-read; output capped at ~60 words."""
    import subprocess

    from creator import generator as G
    if ram_free is None:
        free = G._free_ram_gb()
    else:
        free = ram_free()
    name = pick_model(model, free)
    MODELS_06 = Path(T.resolve_model("1.7b")).with_name("Qwen3-0.6B-Q4_K_M.gguf")
    path = MODELS_06 if name == "0.6b" else T.resolve_model(name)

    class PhoneVoice(T.Voice):
        timings: list = []
        more = False

        def open(self) -> bool:
            if self.lm is not None:
                return True
            try:
                port = G.free_port()
                flags = 0x00000020 if sys.platform == "win32" else 0         # NORMAL_PRIORITY_CLASS, not inherited BelowNormal
                job = G._KillOnCloseJob()
                proc = subprocess.Popen([str(G.SERVER_EXE), "-m", str(path), "--host", "127.0.0.1", "--port", str(port), "-c", str(CTX),
                                         "-t", str(THREADS), "-np", "1", "--log-disable"], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, creationflags=flags)
                job.adopt(proc)
                t0 = time.monotonic()
                while time.monotonic() - t0 < 180:
                    try:
                        import urllib.request
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                            if r.status == 200:
                                break
                    except OSError:
                        time.sleep(0.5)
                else:
                    proc.kill()
                    raise TimeoutError("voice server did not start")
                import types
                self.lm = types.SimpleNamespace(port=port, proc=proc, job=job, leased=False,
                                                __exit__=lambda *a: (proc.kill(), job.close()))
                self.load_s = time.monotonic() - t0
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {str(e)[:160]}"
                return False
            return True

        def close(self) -> None:
            lm, self.lm = self.lm, None
            if lm is not None:
                try:
                    lm.proc.kill()
                finally:
                    lm.job.close()

        def ask(self, messages: list[dict[str, str]], max_tokens: int, temperature: float = 0.2) -> Any:
            cap = MORE_TOKENS if self.more else SPOKEN_TOKENS
            r = super().ask(with_persona(messages, persona), min(max_tokens, cap), temperature)
            if r.tokens_out >= cap - 1:                       # cut at the last whole sentence rather than mid-word
                cut = max(r.text.rfind(". "), r.text.rfind("! "), r.text.rfind("? "), r.text.rstrip().rfind(".") if r.text.rstrip().endswith(".") else -1)
                if cut > len(r.text) // 3:
                    r.text = r.text[:cut + 1]
            self.timings.append((round(r.seconds, 2), r.tokens_in, r.tokens_out))
            del self.timings[:-20]
            return r

    v = PhoneVoice(path, ctx=CTX, timeout_s=120)
    v.model_name = name
    return v


def spoken(text: str) -> str:
    return re.sub(r"\s+", " ", _EVID.sub("", text)).strip()


def chunk(text: str, words: int = MAX_WORDS) -> tuple[str, str]:
    """(first ~60 words ending at a sentence if possible, the rest)."""
    w = text.split()
    if len(w) <= words:
        return text, ""
    head = " ".join(w[:words])
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut > len(head) // 2:
        head = head[:cut + 1]
    return head, text[len(head):].strip()


class RateLimit:
    def __init__(self, per_min: int = RATE_PER_MIN, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_min, self.clock, self.hits = per_min, clock, {}
        self.lock = threading.Lock()

    def ok(self, who: str) -> bool:
        now = self.clock()
        with self.lock:
            q = self.hits.setdefault(who, deque())
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_min:
                return False
            q.append(now)
            return True


class Core:
    """Conversations per device over one lazily loaded voice. `make_conv(voice)` and `ram_free()` are injectable (tests)."""

    def __init__(self, root: Path, model: str = "1.7b", idle_min: float = 60.0, make_conv: Optional[Callable[[], Any]] = None,
                 ram_free: Optional[Callable[[], Optional[float]]] = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.root, self.model, self.idle_s, self.clock = Path(root), model, idle_min * 60, clock
        self.make_conv, self.ram_free = make_conv, ram_free
        self.convs: OrderedDict[str, Any] = OrderedDict()
        self.rest: dict[str, str] = {}
        self.prev: dict[str, str] = {}
        self.voice: Any = None
        self.last = clock()
        self.last_split: dict[str, Any] = {}
        self.lock = threading.Lock()

    def _ram_ok(self) -> bool:
        if self.ram_free is None:
            from creator import conversation as CV
            free = CV.ram_free_gb()
        else:
            free = self.ram_free()
        return free is None or free - 2.0 >= MIN_FREE_GB     # ~2 GB for the 1.7B voice; keep 2 GB free after

    def _build(self) -> Any:
        if self.make_conv is not None:
            return self.make_conv()
        from creator import conversation as CV
        from creator import talk as T

        def refuse(c: Any, s: dict[str, Any]) -> Any:
            return CV.Facts("help", ["I only talk from the phone. Pausing, resuming, approving and requests need the terminal."])
        if self.voice is None:
            T.FACT_CHARS = FACT_CHARS                          # lean grounding: ~300 tokens of facts, chosen by the handlers' own ranking
            self.voice = make_voice(T, self.model, persona_text(), self.ram_free)
        conv = T.conversation(self.root, self.voice, log=False, mode="rules", handlers={k: refuse for k in BLOCKED})
        conv.speak.retries = 0                                 # no second model call to repair a status; the rule report speaks instead
        return conv

    def talk(self, device: str, text: str) -> tuple[int, dict[str, Any]]:
        t0 = time.monotonic()
        with self.lock:
            self.last = self.clock()
            if MORE.match(text) and self.rest.get(device):
                head, self.rest[device] = chunk(self.rest[device])
                return 200, {"reply": head, "ms": int((time.monotonic() - t0) * 1000), "more": bool(self.rest[device])}
            if MORE.match(text) and self.prev.get(device):       # nothing left over: ask the same question again with room to say more
                text = self.prev[device]
                if self.voice is not None:
                    self.voice.more = True
            conv = self.convs.get(device)
            if conv is None:
                if self.make_conv is None and self.voice is None and not self._ram_ok():
                    return 503, {"reply": "I cannot load my voice now, the PC is short of memory. Try again later.", "ms": 0, "more": False}
                conv = self._build()
                self.convs[device] = conv
                while len(self.convs) > MAX_DEVICES:
                    old, _ = self.convs.popitem(last=False)
                    self.rest.pop(old, None)
            self.convs.move_to_end(device)
            try:
                v0 = len(getattr(self.voice, "timings", []))
                t1 = time.monotonic()
                raw = conv.reply(text)
                tot = time.monotonic() - t1
                calls = list(getattr(self.voice, "timings", []))[v0:] if self.voice is not None else []
                vs = sum(c[0] for c in calls)
                self.last_split = {"load_s": round(t1 - t0, 2), "voice_s": round(vs, 2), "grounding_s": round(max(tot - vs, 0), 2),
                                   "prompt_tokens": sum(c[1] for c in calls), "out_tokens": sum(c[2] for c in calls)}
            except Exception as e:  # noqa: BLE001
                return 500, {"reply": f"Something went wrong ({type(e).__name__}).", "ms": 0, "more": False}
            if self.voice is not None:
                self.voice.more = False
            if not MORE.match(text):
                self.prev[device] = text
            head, self.rest[device] = chunk(spoken(raw))
            return 200, {"reply": head, "ms": int((time.monotonic() - t0) * 1000), "more": bool(self.rest[device])}

    def reap(self) -> bool:
        """Unload the voice after the idle time."""
        with self.lock:
            if self.voice is not None and self.clock() - self.last > self.idle_s:
                try:
                    self.voice.close()
                finally:
                    self.voice, self.convs = None, OrderedDict()
                    self.rest.clear()
                return True
        return False


def make_server(host: str, port: int, token: str, core: Core, log_path: Optional[Path] = None, per_min: int = RATE_PER_MIN) -> ThreadingHTTPServer:
    limiter = RateLimit(per_min)
    tok = token.encode()

    class H(BaseHTTPRequestHandler):
        server_version = "nupen-phone"

        def log_message(self, fmt: str, *args: Any) -> None:   # access log goes outside the repo
            if log_path is None:
                return
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {self.client_address[0]} {fmt % args}\n")
            except OSError:
                pass

        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            h = self.headers.get("Authorization", "")
            got = h[7:].strip().encode() if h.lower().startswith("bearer ") else b""
            return hmac.compare_digest(got, tok)

        def do_GET(self) -> None:  # noqa: N802
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            if self.path.split("?")[0] == "/health":
                return self._send(200, {"ok": True, "voice_loaded": core.voice is not None})
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            if self.path.split("?")[0] == "/talk_audio":
                self.rfile.read(min(max(int(self.headers.get("Content-Length") or 0), 0), MAX_BODY))      # design stub: local Piper TTS (en_GB male) returning audio/wav; not built yet
                return self._send(501, {"error": "talk_audio is not installed yet (see IPHONE_SETUP.md, next step)"})
            if self.path.split("?")[0] != "/talk":
                return self._send(404, {"error": "not found"})
            if not limiter.ok(self.client_address[0]):
                return self._send(429, {"error": "slow down"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                return self._send(413, {"error": "request too large"})
            try:
                d = json.loads(self.rfile.read(n).decode("utf-8"))
                text = str(d["text"]).strip()
                device = re.sub(r"[^A-Za-z0-9_-]", "", str(d.get("device") or "phone"))[:24] or "phone"
            except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                return self._send(400, {"error": 'expected JSON {"text": "..."}'})
            if not text or len(text) > MAX_TEXT:
                return self._send(400, {"error": "text empty or too long"})
            code, body = core.talk(device, text)
            self._send(code, body)

    return ThreadingHTTPServer((host, port), H)


FIREWALL = ('New-NetFirewallRule -DisplayName "Nupen phone (LAN + Tailscale only)" -Direction Inbound -Protocol TCP -LocalPort {port} '
            '-RemoteAddress LocalSubnet,{cgnat} -Profile Any -Action Allow')


def startup_cmd(rt: Path) -> Path:
    p = rt / "start_nupen_phone.cmd"
    py = Path(sys.executable)
    p.write_text(f'@echo off\r\nrem Optional: copy a shortcut to this file into shell:startup. Not installed by Nupen.\r\n'
                 f'cd /d "{ROOT}"\r\nstart "Nupen phone" /normal /wait "{py}" scripts/nupen_phone.py\r\npause\r\n', encoding="utf-8")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="", help="bind address (default: this PC's LAN address)")
    ap.add_argument("--allow-all", action="store_true", help="permit 0.0.0.0 (owner's explicit choice; default refuses it)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--model", default="1.7b")
    ap.add_argument("--idle-min", type=float, default=60.0)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--make-token", action="store_true", help="create the token file if missing, print its path")
    ap.add_argument("--print-firewall", action="store_true")
    ap.add_argument("--print-startup", action="store_true")
    a = ap.parse_args(argv)
    rt = runtime_dir()
    if a.print_firewall:
        print(FIREWALL.format(port=a.port, cgnat=CGNAT))
        return 0
    if a.make_token:
        print(make_token(rt))
        return 0
    if a.print_startup:
        rt.mkdir(parents=True, exist_ok=True)
        print(startup_cmd(rt))
        return 0
    token = load_token(rt)                       # refuses when the token file is missing
    hosts = [a.host] if a.host else [lan_ip()] + tailscale_ips()
    if any(h in ("0.0.0.0", "") for h in hosts) and not a.allow_all:
        raise SystemExit("refusing to bind 0.0.0.0 without --allow-all")
    core = Core(Path(a.root), a.model, a.idle_min)
    servers = []
    for h in dict.fromkeys(hosts):
        try:
            servers.append(make_server(h, a.port, token, core, rt / "access.log"))
        except OSError as e:
            for sv in servers:
                sv.server_close()
            raise SystemExit(f"cannot listen on {h}:{a.port} ({e.strerror or e}); is Nupen phone already running? Close the other window and retry")

    def reaper() -> None:
        while True:
            time.sleep(60)
            core.reap()
    threading.Thread(target=reaper, daemon=True).start()
    threading.Thread(target=lambda: core.talk("warmup", "hello"), daemon=True).start()     # load + prime the cache before the first request
    for sv in servers[1:]:
        threading.Thread(target=sv.serve_forever, daemon=True).start()
    print("Nupen phone server on " + ", ".join(f"http://{h}:{a.port}/talk" for h in dict.fromkeys(hosts)) +
          f" (token in {token_path(rt)}); Ctrl+C stops", flush=True)
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if core.voice is not None:
            core.voice.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
