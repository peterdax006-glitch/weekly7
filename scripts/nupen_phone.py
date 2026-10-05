"""Talk to Nupen from a phone (home Wi-Fi). A tiny HTTP server on the PC; the phone's Shortcut POSTs what you said and speaks the reply.

    python scripts/nupen_phone.py                 # binds the PC's LAN address + Tailscale address if present (detected at start), port 8765
    python scripts/nupen_phone.py --host 127.0.0.1   # local only (smoke tests)
    python scripts/nupen_phone.py --print-firewall   # the admin command to allow the port (printed, never run)
    python scripts/nupen_phone.py --print-startup    # write the optional Windows startup .cmd (not installed)

    POST /talk   {"text": "...", "device": "iphone"}  ->  {"reply": "...", "action": null | {"type": ...}, "end": bool, "ms": N, "more": bool}
    GET /health.  "action" is from a fixed allow-list (timer alarm reminder note calendar message call music open_app directions flashlight focus
    home) parsed by code; "end" is true on goodbye so the Shortcut stops listening. Nothing the phone says can touch the PC.
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
from datetime import datetime, timedelta
from urllib.parse import quote
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


# ---- phone actions: an allow-list parsed by CODE from the owner's words; never free-form commands; nothing here touches the PC ----
_W1 = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                                  "seventeen eighteen nineteen".split())}
_W10 = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_NUM_RE = re.compile(r"\b(?:(?:%s)(?:[ -](?:%s))?|%s)\b" % ("|".join(_W10), "|".join(list(_W1)[1:10]), "|".join(_W1)), re.I)
_SAY = list(_W1)[:1] + list(_W1)[1:] + ["twenty"]
_UNIT = r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?)"
_ONE = rf"(?:(?:\d+(?:\.\d+)?\s*|an?\s+|half\s+an?\s+|a\s+quarter\s+of\s+an?\s+){_UNIT}|\d+\s+and\s+a\s+half\s+{_UNIT})"
_CHAIN = rf"{_ONE}(?:(?:\s*,?\s*and\s+|\s*,\s*|\s+)(?:{_ONE}|a\s+half))*"
_MER = r"(a\.?m\.?|p\.?m\.?)"
_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_FOCUS = ("do not disturb", "sleep", "work", "personal", "driving", "reading", "fitness", "gaming", "mindfulness")
_LEAD = re.compile(r"^(?:(?:hey|hi|ok|okay|right|so|well|and|now|then|nupen|please|kindly)\b[ ,]*)+", re.I)
_MODAL = re.compile(r"^(?:(?:can|could|would|will) you(?: possibly| please)?|i(?:'d| would) like (?:you )?to|i want (?:you )?to|i need you to|i need to|"
                    r"go ahead and)\b[ ,]*(?:please\b[ ,]*)?", re.I)
_TRAIL = re.compile(r"(?:[ ,.!?]+(?:please|sir|nupen|thanks|thank you|for me))+[ .!?]*$|[ ,.!?]+$", re.I)
_Q = re.compile(r"^(?:what|what's|whats|how|when|why|who|whom|whose|where|which|is|are|was|were|did|does|do(?! not disturb)|have|has|had|am)\b", re.I)
_DO = re.compile(r"^(?:set|turn|switch|call|ring|phone|text|message|send|play|open|launch|remind|schedule|book|order|buy|delete|cancel|shut|restart|"
                 r"reboot|lock|unlock|navigate|email|post|pay|install|download|enable|disable|activate|dim|increase|decrease|"
                 r"(?:stop|pause|skip|resume)\s+(?:the\s+)?(?:music|song|track|playback))\b", re.I)
_PC = re.compile(r"\b(?:computer|pc|laptop|desktop|server|kernel|gpu|nupen)\b", re.I)
_NO_HOME = re.compile(r"\b(?:alarm|timer|wi-?fi|bluetooth|airplane|volume|brightness|phone|data|silent|ringer)\b", re.I)
_END = re.compile(r"\b(?:good ?bye|bye(?: bye)?|see you|that'?s all|that is all|that will be all|that'?ll be all|that'?s it|nothing else|"
                  r"no,? thanks|no thank you|i'?m done|we'?re done|stop listening|talk later|speak later)\b", re.I)
_THANKS = re.compile(r"^(?:ok(?:ay)?[ ,]+)?(?:thanks|thank you|cheers|ta)(?:[ ,]+(?:very much|a lot|so much|nupen|sir|then))*[ .!,]*$", re.I)


def _numwords(s: str) -> str:
    """Spoken numbers to digits ('seven thirty' -> '7 30'); only for time-like fields, never for message or note text."""
    return _NUM_RE.sub(lambda m: str(sum({**_W1, **_W10}[w] for w in re.split(r"[ -]", m.group(0).lower()))), s)


def _say(n: int) -> str:
    return _SAY[n] if 0 <= n <= 20 else str(n)


def _usec(u: str) -> int:
    return {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[u[0].lower()]


def _dur_seconds(s: str) -> int:
    """Sum of every duration in the words ('an hour and a half', '2 hours 10 minutes', 'half an hour', '45 seconds'); 0 when none."""
    tot = [0.0]
    s = s.lower()

    def take(pat: str, f: Callable[[Any], float]) -> None:
        nonlocal s

        def rep(m: Any) -> str:
            tot[0] += f(m)
            return " "
        s = re.sub(pat, rep, s)

    def num(x: str) -> float:
        x = x.strip()
        return 1.0 if x in ("a", "an") else float(x)
    n = r"((?:\d+(?:\.\d+)?\s*|an?\s+))"
    take(rf"\b{n}({_UNIT})\s+and\s+a\s+half\b", lambda m: num(m[1]) * _usec(m[2]) * 1.5)
    take(rf"\b(\d+)\s+and\s+a\s+half\s+({_UNIT})\b", lambda m: (float(m[1]) + 0.5) * _usec(m[2]))
    take(rf"\bhalf\s+an?\s+({_UNIT})\b", lambda m: 0.5 * _usec(m[1]))
    take(rf"\ba\s+quarter\s+of\s+an?\s+({_UNIT})\b", lambda m: 0.25 * _usec(m[1]))
    take(rf"\b{n}({_UNIT})\b", lambda m: num(m[1]) * _usec(m[2]))
    return int(round(tot[0]))


def _find_time(s: str) -> Optional[tuple[Any, int, int, Optional[str]]]:
    """First clock time in the words: (match, hour, minute, 'am'|'pm'|None)."""
    pats: list[tuple[str, Callable[[Any], tuple[int, int, Optional[str]]]]] = [
        (r"\bhalf\s+past\s+(\d{1,2})\b", lambda m: (int(m[1]), 30, None)),
        (r"\bquarter\s+past\s+(\d{1,2})\b", lambda m: (int(m[1]), 15, None)),
        (r"\bquarter\s+to\s+(\d{1,2})\b", lambda m: (int(m[1]) - 1 or 12, 45, None)),
        (rf"\b(?:(?:at|by|for)\s+)?(\d{{1,2}})(?:[:.\s]\s*(\d{{2}}))?\s*{_MER}(?![A-Za-z])", lambda m: (int(m[1]), int(m[2] or 0), m[3].lower())),
        (r"\b(?:at|by)\s+(\d{1,2})(?:[:.\s]\s*(\d{2}))?(?:\s*o'?clock)?(?![\d:])(?!\s*(?:minutes?|mins?|hours?|hrs?|seconds?|secs?|days?|weeks?|times))",
         lambda m: (int(m[1]), int(m[2] or 0), None)),
        (r"\b(\d{1,2}):(\d{2})\b", lambda m: (int(m[1]), int(m[2]), None)),
        (r"\b(\d{1,2})\s*o'?clock\b", lambda m: (int(m[1]), 0, None)),
        (r"\bnoon\b", lambda m: (12, 0, None)),
        (r"\bmidnight\b", lambda m: (0, 0, None)),
    ]
    for pat, f in pats:
        m = re.search(pat, s, re.I)
        if not m:
            continue
        h, mi, mer = f(m)
        if mi < 60 and ((mer and 1 <= h <= 12) or (not mer and 0 <= h <= 24)):
            return m, h, mi, mer
    return None


def _extract_when(s: str, now: datetime, pm_default: bool = False) -> tuple[Optional[datetime], bool, str]:
    """(when, has_time, the words without the time phrase). Relative ('in ten minutes'), today/tomorrow/tonight, weekdays, clock times."""
    s = _numwords(s)
    m = re.search(rf"\bin\s+({_CHAIN})(?!\w)", s, re.I)
    if m and _dur_seconds(m[1]):
        return now + timedelta(seconds=_dur_seconds(m[1])), True, s[:m.start()] + " " + s[m.end():]
    date = None
    period = None
    m = re.search(r"\b(?:on\s+)?(tomorrow|today|tonight)\b", s, re.I)
    if m:
        w = m[1].lower()
        date = (now + timedelta(days=1 if w == "tomorrow" else 0)).date()
        period = "evening" if w == "tonight" else None
        s = s[:m.start()] + " " + s[m.end():]
    else:
        m = re.search(r"\b(?:on\s+|next\s+|this\s+)?(" + "|".join(_DAYS) + r")\b", s, re.I)
        if m:
            d = (_DAYS.index(m[1].lower()) - now.weekday()) % 7
            date = (now + timedelta(days=d or 7)).date()
            s = s[:m.start()] + " " + s[m.end():]
    m = re.search(r"\b(?:in\s+the\s+|this\s+)?(morning|afternoon|evening|night)\b", s, re.I)
    if m:
        period = period or m[1].lower()
        s = s[:m.start()] + " " + s[m.end():]
    t = _find_time(s)
    has_time = t is not None
    if t:
        m, h, mi, mer = t
        s = s[:m.start()] + " " + s[m.end():]
        if mer:
            h = (h % 12) + (12 if mer.startswith("p") else 0)
        elif period in ("afternoon", "evening", "night") and 1 <= h < 12:
            h += 12
        elif pm_default and 1 <= h <= 7:
            h += 12
        h %= 24
    elif date is not None or period:
        h, mi = {"morning": (9, 0), "afternoon": (15, 0), "evening": (19, 0), "night": (21, 0)}.get(period or "", (9, 0))
        has_time = period is not None
    else:
        return None, False, s
    if date is None:
        dt = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if dt <= now:
            dt += timedelta(days=1)
    else:
        dt = datetime(date.year, date.month, date.day, h, mi)
    return dt, has_time, s


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


def _tidy(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip(" ,.;:-")
    t = re.sub(r"\s+(?:at|on|by|for|in|to|and)$", "", t, flags=re.I)
    return re.sub(r"^(?:to|that|about|of)\s+", "", t.strip(" ,.;:-"), flags=re.I).strip(" ,.;:-")


def _clock12(h: int, mi: int) -> str:
    return f"{(h % 12) or 12}{':%02d' % mi if mi else ''} {'AM' if h < 12 else 'PM'}"


def _who(t: str) -> str:
    return re.sub(r"^(?:my|the)\s+", "", _tidy(t), flags=re.I)


def _clean(raw: str) -> str:
    s = raw.strip()
    while True:
        s2 = _TRAIL.sub("", _MODAL.sub("", _LEAD.sub("", s))).strip()
        if s2 == s:
            return s
        s = s2


def _a_timer(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    if not re.search(r"\btimer\b", n, re.I) or re.match(r"(?:cancel|stop|delete|turn off)", n, re.I):
        return None
    sec = _dur_seconds(n)
    if not sec:
        return "For how long, sir?", None
    if sec % 60 == 0:
        return f"Timer set for {_say(sec // 60)} minute{'s' if sec != 60 else ''}, sir.", {"type": "timer", "minutes": sec // 60}
    return f"Timer set for {_say(sec)} seconds, sir.", {"type": "timer", "seconds": sec}


def _a_alarm(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    if not (re.search(r"\balarm\b", n, re.I) or re.match(r"wake me(?: up)?\b", n, re.I)) or re.match(r"(?:cancel|stop|delete|turn off|switch off)", n, re.I):
        return None
    dt, has_time, _ = _extract_when(re.sub(r"\balarm for (\d)", r"alarm at \1", n, flags=re.I), now)
    if dt is None or not has_time:
        return "For what time, sir?", None
    return f"Alarm set for {_clock12(dt.hour, dt.minute)}, sir.", {"type": "alarm", "time": dt.strftime("%H:%M")}


def _a_reminder(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    m = re.match(r"(?:remind me|(?:set|create|add|make)\s+(?:a\s+|another\s+)?reminder|reminder)\b[ ,:]*(.*)$", n, re.I)
    m2 = re.match(r"add\s+(.+?)\s+to\s+(?:my\s+)?(?:reminders?|reminder list|to-?do list)$", n, re.I)
    if not (m or m2):
        return None
    dt, _, rest = _extract_when((m or m2)[1], now, pm_default=True)
    text = _tidy(rest)
    if not text:
        return "What shall I remind you about, sir?", None
    a: dict[str, Any] = {"type": "reminder", "text": text[:200]}
    if dt is not None:
        a["when"] = _iso(dt)
    return "Reminder noted, sir.", a


def _a_calendar(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    m = re.match(r"(?:add|put)\s+(.+?)\s+(?:to|in|on|into)\s+(?:my\s+)?calendar\b(.*)$", n, re.I)
    body = (m[1] + " " + m[2]) if m else None
    if body is None:
        m = re.match(r"(?:schedule|set up|create|make|add)\s+(?:an?\s+|the\s+)?(.+)$", n, re.I)
        if not m or not (n.lower().startswith("schedule") or re.search(r"\b(?:event|meeting|appointment|calendar)\b", n, re.I)):
            return None
        body = m[1]
    dur = None
    d = re.search(rf"\bfor\s+({_CHAIN})(?!\w)", _numwords(body), re.I)
    if d:
        dur = max(_dur_seconds(d[1]) // 60, 1)
        body = (_numwords(body)[:d.start()] + " " + _numwords(body)[d.end():])
    dt, has_time, rest = _extract_when(body, now, pm_default=True)
    title = _tidy(re.sub(r"^(?:calendar\s+)?(?:event|entry)\s+(?:called|named|titled|for|about)\s+", "", _tidy(rest), flags=re.I))
    title = re.sub(r"\s+(?:to|in|on|into)\s+(?:my\s+)?calendar$", "", title, flags=re.I)
    if not title:
        return "What shall I call the event, sir?", None
    if dt is None or not has_time:
        return "At what time, sir?", None
    a: dict[str, Any] = {"type": "calendar", "title": _cap(title)[:120], "start": _iso(dt)}
    if dur:
        a["duration"] = dur
    return "Added to your calendar, sir.", a


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _a_note(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    m = re.match(r"(?:(?:make|take|create|write|add|save)\s+(?:an?\s+)?(?:quick\s+)?note|note(?:\s+down)?|jot\s+down|write\s+down)\b"
                 r"(?:\s+(?:saying|that|to say|to|of))?[ ,:]*(.*)$", s, re.I)
    m2 = re.match(r"add\s+(.+?)\s+to\s+(?:my\s+)?notes?$", s, re.I)
    if not (m or m2):
        return None
    text = _tidy((m or m2)[1])
    if not text:
        return "What shall I note, sir?", None
    return "Noted, sir.", {"type": "note", "text": _cap(text)[:300]}


def _a_message(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    pre = r"(?:send\s+(?:an?\s+)?(?:text|message|imessage|text message)\s+to|(?:text|message|imessage)(?:\s+to)?)"
    m = re.match(pre + r"\s+(.+?)\s*(?:,|:|\bsaying\s+that\b|\bsaying\b|\bthat\b|\bto\s+say\b)\s*(.+)$", s, re.I)
    if not m:
        m = re.match(r"(?:text|message|imessage)\s+(\w+)\s+(.+)$", s, re.I)
    if m and _who(m[1]).lower() not in ("me", "back"):
        to, text = _who(m[1]), _tidy(m[2])
        if to and text:
            return f"Message to {to} ready, sir. Your phone will ask you to confirm.", {"type": "message", "to": to, "text": _cap(text)[:300]}
    m = re.match(pre + r"\s+(.+)$", s, re.I)
    if m:
        return f"What shall I say to {_who(m[1])}, sir?", None
    return None


def _a_call(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    m = re.match(r"(?:call|phone|ring|dial|facetime)\s+(?:up\s+)?(.+)$", s, re.I) or re.match(r"give\s+(.+?)\s+a\s+(?:call|ring)$", s, re.I)
    if not m:
        return None
    to = _who(m[1])
    if not to or to.lower() in ("me", "it", "back", "them", "a taxi", "a cab") or len(to.split()) > 5:
        return None
    return f"Calling {to}, sir.", {"type": "call", "to": to}


def _a_flashlight(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    if not re.search(r"\b(?:flash ?light|torch)\b", s, re.I):
        return None
    off = bool(re.search(r"\b(?:off|disable|stop|kill)\b", s, re.I))
    return f"Flashlight {'off' if off else 'on'}, sir.", {"type": "flashlight", "state": "off" if off else "on"}


def _a_focus(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    low = s.lower()
    name = next((f for f in _FOCUS if re.search(r"\b" + f + r"\b", low) and (f == "do not disturb" or re.search(r"\b(?:focus|mode)\b", low))), None)
    if name is None and re.search(r"\bdnd\b", low):
        name = "do not disturb"
    if name is None or not re.search(r"\b(?:on|off|enable|disable|activate|deactivate|start|stop|end|exit|turn|switch|put|set)\b", low):
        return None
    off = bool(re.search(r"\b(?:off|disable|deactivate|stop|end|exit)\b", low))
    nice = "Do Not Disturb" if name == "do not disturb" else _cap(name)
    return f"{nice} {'off' if off else 'on'}, sir.", {"type": "focus", "name": nice, "state": "off" if off else "on"}


def _a_directions(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    m = re.match(r"(?:(?:give|get|show)\s+me\s+(?:the\s+)?(?:directions?|route|way)\s+(?:to|for)|directions?\s+(?:to|for)|navigate\s+(?:me\s+)?to|"
                 r"take\s+me\s+to|drive\s+(?:me\s+)?to|(?:how\s+do\s+i|how\s+can\s+i|how\s+to)\s+get\s+to|(?:show|find)\s+(?:me\s+)?(?:the\s+)?(?:way|route)\s+to|"
                 r"(?:i\s+need|i\s+want)\s+directions\s+to|let's\s+go\s+to)\s+(.+)$", s, re.I)
    if not m:
        return None
    to = _tidy(m[1])
    return (f"Directions to {to}, sir.", {"type": "directions", "to": to}) if to else ("Where to, sir?", None)


def _a_music(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    m = re.match(r"(?:play|put\s+on|listen\s+to|start\s+playing)\s+(.+)$", s, re.I)
    if not m:
        return None
    q = re.sub(r"^(?:some\s+|the\s+(?:song|album|playlist|track)\s+|(?:songs?|music|tracks?)\s+(?:by|from)\s+)", "", _tidy(m[1]), flags=re.I)
    if re.match(r"(?:a\s+|the\s+)?(?:game|video|movie|film|podcast)\b", q, re.I):
        return "I can only play music from here, sir.", None
    if not q or q.lower() in ("music", "something", "a song", "songs"):
        return "What shall I play, sir?", None
    app = "Apple Music"
    on = re.search(r"\s+(?:on|in|with|using)\s+(?:the\s+)?(spotify|youtube(?:\s+music)?|apple\s+music|music(?:\s+app)?)$", q, re.I)
    if on:
        q = _tidy(q[:on.start()])
        app = {"spotify": "Spotify", "youtube": "YouTube", "youtube music": "YouTube"}.get(on[1].lower(), "Apple Music")
    return f"Playing {q} on {app}, sir." if on else f"Playing {q}, sir.", {"type": "music", "query": q[:120], "app": app}


def _a_home(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    m = re.match(r"(?:turn|switch|put)\s+(on|off)\s+(?:the\s+|my\s+)?(.+)$", s, re.I)
    if m:
        state, dev = m[1].lower(), m[2]
    else:
        m = re.match(r"(?:turn|switch)\s+(?:the\s+|my\s+)?(.+?)\s+(on|off)$", s, re.I)
        if not m:
            return None
        state, dev = m[2].lower(), m[1]
    dev = _tidy(dev).lower()
    if _NO_HOME.search(dev):
        return "I cannot change that from here yet, sir.", None
    if not dev or len(dev.split()) > 5:
        return None
    return f"Turning the {dev} {state}, sir.", {"type": "home", "device": dev, "state": state}


def _a_open(s: str, n: str, now: datetime) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    m = re.match(r"(?:open|launch|start|go to|switch to|run)\s+(?:up\s+)?(?:the\s+)?(.+?)(?:\s+app(?:lication)?)?$", s, re.I)
    if not m:
        return None
    name = _tidy(m[1])
    if not name or len(name.split()) > 3:
        return None
    name = name.title() if name.islower() else name
    hit = APPS.get(re.sub(r"\s+", " ", name.lower()))
    if hit is None:                                        # not in the table: no guessed link, the per-app If blocks in the Shortcut may still cover it
        return f"I do not know how to open {name} directly, sir.", {"type": "open_app", "name": name}
    return f"Opening {hit[0]}, sir.", {"type": "open_app", "name": hit[0], "url": hit[1]}


# App name (lower case, as spoken) -> (display name, iOS deep link). Only schemes known to work; nothing guessed.
# Not mapped on purpose (no verified public scheme): Camera, Clock, Phone app, Wallet, Health, Files, Safari by name.
# Uncertain: Settings 'App-prefs:' opens Settings on current iOS but Apple may restrict it in a future version.
APPS: dict[str, tuple[str, str]] = {
    "spotify": ("Spotify", "spotify:"),
    "youtube": ("YouTube", "youtube://"),
    "instagram": ("Instagram", "instagram://"),
    "whatsapp": ("WhatsApp", "whatsapp://"),
    "maps": ("Maps", "maps://"), "apple maps": ("Maps", "maps://"),
    "messages": ("Messages", "sms:"), "imessage": ("Messages", "sms:"), "texts": ("Messages", "sms:"),
    "facetime": ("FaceTime", "facetime://"),
    "mail": ("Mail", "mailto:"), "email": ("Mail", "mailto:"),
    "music": ("Music", "music://"), "apple music": ("Music", "music://"),
    "photos": ("Photos", "photos-redirect://"),
    "calendar": ("Calendar", "calshow://"),
    "reminders": ("Reminders", "x-apple-reminderkit://"),
    "settings": ("Settings", "App-prefs:"),
    "notes": ("Notes", "mobilenotes://"),
}


def _url_for(a: dict[str, Any]) -> Optional[str]:
    """The iOS deep link for an action (None when there is no verified one); the Shortcut opens it with ONE Open URLs step."""
    q = quote
    t = a.get("type")
    if t == "open_app":
        return a.get("url")
    if t == "music":
        s = q(str(a["query"]), safe="")
        return {"Spotify": "spotify:search:" + s, "YouTube": "https://www.youtube.com/results?search_query=" + s}.get(
            str(a.get("app")), "music://music.apple.com/search?term=" + s)
    if t == "directions":
        return "maps://?daddr=" + q(str(a["to"]), safe="") + "&dirflg=d"
    num = re.sub(r"[ ()-]", "", str(a.get("to", "")))
    if t == "call" and re.fullmatch(r"\+?\d{3,15}", num):
        return "tel:" + num
    if t == "message" and re.fullmatch(r"\+?\d{3,15}", num):
        return "sms:" + num + "&body=" + q(str(a["text"]), safe="")
    return None


_ORDER = (_a_timer, _a_alarm, _a_reminder, _a_calendar, _a_note, _a_message, _a_call)
_ORDER2 = (_a_flashlight, _a_focus, _a_directions, _a_music, _a_home, _a_open)


def interpret(text: str, now: Optional[datetime] = None) -> Optional[tuple[str, Optional[dict[str, Any]], bool]]:
    """(spoken reply, action or None, end) when the words are a phone command, a goodbye or a refused request; None = hand to the voice model.
    Every action comes from the allow-list above; unknown or PC-touching requests give a polite reply and no action."""
    now = now or datetime.now()
    raw = text.strip().replace("’", "'")
    s = _clean(raw)
    n = _numwords(s)
    end = len(raw.split()) <= 12 and bool(_END.search(raw))
    res: Optional[tuple[str, Optional[dict[str, Any]]]] = None
    if s:
        res = _a_directions(s, n, now)
        if res is None and not _Q.match(s):
            for f in _ORDER:
                res = f(s, n, now)
                if res:
                    break
            if res is None and _DO.match(s) and _PC.search(s):
                res = ("I am afraid I cannot touch the computer from the phone, sir.", None)
            for f in (() if res else _ORDER2):
                res = f(s, n, now)
                if res:
                    break
            if res is None and _DO.match(s):
                res = ("I am afraid I cannot do that yet, sir.", None)
    if res is not None:
        a = res[1]
        if a is not None and (u := _url_for(a)):
            a["url"] = u
        return res[0], a, end
    if end or _THANKS.match(raw):
        return "Very good, sir. Goodbye.", None, True
    return None


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
                 ram_free: Optional[Callable[[], Optional[float]]] = None, clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = datetime.now) -> None:
        self.now = now
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
        cmd = interpret(text, self.now())                       # phone actions and goodbyes: code only, no voice model, no RAM needed
        if cmd is not None:
            return 200, {"reply": cmd[0], "action": cmd[1], "end": cmd[2], "ms": int((time.monotonic() - t0) * 1000), "more": False}
        with self.lock:
            self.last = self.clock()
            if MORE.match(text) and self.rest.get(device):
                head, self.rest[device] = chunk(self.rest[device])
                return 200, {"reply": head, "action": None, "end": False, "ms": int((time.monotonic() - t0) * 1000), "more": bool(self.rest[device])}
            if MORE.match(text) and self.prev.get(device):       # nothing left over: ask the same question again with room to say more
                text = self.prev[device]
                if self.voice is not None:
                    self.voice.more = True
            conv = self.convs.get(device)
            if conv is None:
                if self.make_conv is None and self.voice is None and not self._ram_ok():
                    return 503, {"reply": "I cannot load my voice now, the PC is short of memory. Try again later.", "action": None, "end": False, "ms": 0, "more": False}
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
                return 500, {"reply": f"Something went wrong ({type(e).__name__}).", "action": None, "end": False, "ms": 0, "more": False}
            if self.voice is not None:
                self.voice.more = False
            if not MORE.match(text):
                self.prev[device] = text
            head, self.rest[device] = chunk(spoken(raw))
            return 200, {"reply": head, "action": None, "end": False, "ms": int((time.monotonic() - t0) * 1000), "more": bool(self.rest[device])}

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
