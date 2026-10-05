"""Phone brain: plan -> tools -> answer and/or a LIST of phone actions, for the phone server (scripts/nupen_phone.py).

Owner, 5 Oct: Nupen should be more capable than Siri: it finds a way, thinks on its own, and can use the computer for help.
Design (small, safe, fast):
  * The PLAN comes from rules over the owner's own words (never from fetched page text), so a web page cannot steer the tools.
  * Tools run on the PC: web_search / wikipedia / fetch_page (free, no key), calculator (AST whitelist), nupen_status, file_search and
    read_file (READ-ONLY, only inside Nupen's own folders), queue_coding_task (writes ONE request file in the phone dir; runs nothing).
  * Every question goes through a FALLBACK CHAIN of routes; the last route hands the question to the phone (a web search link).
  * Risky phone actions (message, call, mail) are held behind a confirm prompt; the server releases them only on "confirm".
  * The voice model only phrases the final answer (<= 2 spoken sentences, source named); the injected `summarize` may be None.
Every tool call is recorded as {tool, ok, ms, note}: never a page body."""
from __future__ import annotations

import ast
import html
import ipaddress
import json
import math
import operator
import os
import re
import socket
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

HttpGet = Callable[[str, float], tuple[int, str]]            # (url, timeout_s) -> (status, text); injected in tests
UA = "NupenPhone/1.0 (personal assistant; contact: owner)"
BUDGET_S = 9.0                                               # whole web question
HTTP_TIMEOUT = 4.0
MAX_BYTES = 300_000
RISKY_TYPES = frozenset({"message", "call", "mail"})         # held for 'confirm' (the phone also asks once more on its own)
CONFIRM_TTL = 300.0
FOLDER_REFUSAL = "I only look inside Nupen's own folders, sir."
_SECRET = re.compile(r"(?:\.env|token|secret|password|passwd|credential|id_rsa|\.pem|\.key|\.pfx)", re.I)
_OUTSIDE = re.compile(r"\bmy (?:documents|files|pictures|photos|downloads|desktop)\b|\b(?:documents|downloads|desktop|pictures|photos|music|videos|onedrive) folder\b|"
                      r"\bdownloads?\b|\bdesktop\b|\bonedrive\b|\b[a-z]:[\\/]|\busers?\b[\\/]", re.I)


@dataclass
class Call:
    tool: str
    ok: bool
    ms: int
    note: str = ""


@dataclass
class Result:
    reply: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    confirm: bool = False                                     # True: actions are held; the server stores them until 'confirm'
    intent: str = "brain"


# ---------------------------------------------------------------- network

def _public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


def http_get(url: str, timeout: float = HTTP_TIMEOUT) -> tuple[int, str]:
    """Plain GET. https/http only, public hosts only (no LAN, no localhost), size and time capped."""
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname or not _public_host(u.hostname):
        return 0, ""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(MAX_BYTES).decode(r.headers.get_content_charset() or "utf-8", "replace")
    except Exception:  # noqa: BLE001 - any network failure is just "this route failed"
        return 0, ""


def page_text(raw: str, limit: int = 4000) -> str:
    t = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)\b.*?</\1>", " ", raw)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", html.unescape(t)).strip()[:limit]


# ---------------------------------------------------------------- text helpers

def sentences(text: str, n: int = 2, max_words: int = 45) -> str:
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    out: list[str] = []
    for p in parts:
        if p and len(" ".join(out + [p]).split()) <= max_words:
            out.append(p)
        if len(out) == n:
            break
    return " ".join(out) or " ".join(text.split()[:max_words])


def _lower_first(t: str) -> str:
    return t[:1].lower() + t[1:] if t[:2] != t[:2].upper() else t


# ---------------------------------------------------------------- calculator (AST whitelist)

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10,
          "floor": math.floor, "ceil": math.ceil}
_CONST = {"pi": math.pi, "e": math.e}
MAX_EXPR = 120


def safe_eval(expr: str) -> float:
    """Arithmetic only: numbers, + - * / // % **, parentheses, a few math functions. No names but those, no attributes, no imports, no strings.
    Exponents are capped so '9**9**9' cannot eat the PC."""
    if len(expr) > MAX_EXPR:
        raise ValueError("too long")

    def ev(n: ast.AST, depth: int = 0) -> Any:
        if depth > 20:
            raise ValueError("too deep")
        if isinstance(n, ast.Expression):
            return ev(n.body, depth + 1)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.Name) and n.id in _CONST:
            return _CONST[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand, depth + 1)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left, depth + 1), ev(n.right, depth + 1)
            if isinstance(n.op, ast.Pow) and (abs(b) > 1000 or abs(a) > 1e12):
                raise ValueError("power too large")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords and len(n.args) <= 2:
            return _FUNCS[n.func.id](*[ev(x, depth + 1) for x in n.args])
        raise ValueError("not arithmetic")

    v = ev(ast.parse(expr.strip(), mode="eval"))
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        raise ValueError("no result")
    return v


_WORD_OPS = (
    (r"\bsquare root of\s+(\d+(?:\.\d+)?)", r"sqrt(\1)"), (r"(\d+(?:\.\d+)?)\s+squared\b", r"(\1)**2"), (r"(\d+(?:\.\d+)?)\s+cubed\b", r"(\1)**3"),
    (r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)", r"(\1/100*\2)"), (r"\bto the power of\b", "**"),
    (r"\b(?:multiplied by|times|x)\b", "*"), (r"\b(?:divided by|over)\b", "/"), (r"\b(?:plus|and)\b", "+"), (r"\b(?:minus|less)\b", "-"),
    (r"\bmod(?:ulo)?\b", "%"), (r"\^", "**"), (r"(?<=\d),(?=\d{3}\b)", ""),
)


def math_expression(text: str) -> Optional[str]:
    """'what is 15 percent of 240' -> '(15/100*240)'; None when the words are not plain arithmetic."""
    t = re.sub(r"^(?:what(?:'s| is)|calculate|compute|work out|how much is|whats)\s+", "", text.strip().rstrip("?.! "), flags=re.I).lower()
    t = re.sub(r"^(?:the|a|an)\s+", "", t)
    for pat, rep in _WORD_OPS:
        t = re.sub(pat, rep, t)
    t = t.strip()
    if not re.search(r"\d", t) or not re.fullmatch(r"[\d\s.+\-*/%()a-z,]*", t):
        return None
    try:
        safe_eval(t)
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError):
        return None
    return t


def say_number(v: float) -> str:
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        v = int(v)
    return f"{v:,}" if isinstance(v, int) else f"{v:,.6g}"


# ---------------------------------------------------------------- the brain

class Brain:
    """`interpret(text, now)` and `route(action)` are the phone server's own (rules for phone actions). `summarize(question, facts, source)`
    phrases an answer (None = extractive). `about(text)` is the existing Nupen grounding. All paths and the network are injectable."""

    def __init__(self, interpret: Callable[..., Any], route: Callable[[Optional[dict[str, Any]]], Optional[dict[str, Any]]],
                 summarize: Optional[Callable[[str, str, str], str]] = None, about: Optional[Callable[[str], str]] = None,
                 http: Optional[HttpGet] = None, roots: Optional[list[Path]] = None, queue_dir: Optional[Path] = None,
                 now: Callable[[], datetime] = datetime.now, clock: Callable[[], float] = time.monotonic) -> None:
        self.interpret, self.route, self.summarize, self.about = interpret, route, summarize, about
        self.http = http or http_get
        self.roots = [Path(r).resolve() for r in (roots if roots is not None else default_roots())]
        self.queue_dir = Path(queue_dir) if queue_dir else None
        self.now, self.clock = now, clock
        self.calls: list[Call] = []
        self.tools: dict[str, Callable[..., tuple[bool, str, str]]] = {          # name -> fn(arg) -> (ok, facts, source)
            "wikipedia": self.t_wikipedia, "ddg_instant": self.t_ddg_instant, "web_search": self.t_web_search, "fetch_page": self.t_fetch_page,
            "calculator": self.t_calculator, "nupen_status": self.t_status, "file_search": self.t_file_search, "read_file": self.t_read_file,
            "queue_coding_task": self.t_queue}

    # ---- bookkeeping
    def call(self, tool: str, arg: str) -> tuple[bool, str, str]:
        t0 = self.clock()
        try:
            ok, facts, source = self.tools[tool](arg)
            note = "" if ok else (facts[:80] or "no result")
        except PermissionError as e:
            ok, facts, source, note = False, str(e), "", "refused"
        except Exception as e:  # noqa: BLE001 - a failing tool must not kill the request; the chain goes on
            ok, facts, source, note = False, "", "", type(e).__name__
        self.calls.append(Call(tool, ok, int((self.clock() - t0) * 1000), note if not ok else (source or "")[:40]))
        return ok, facts, source

    # ---- web tools
    def _get(self, url: str, left: float) -> tuple[int, str]:
        return self.http(url, max(0.5, min(HTTP_TIMEOUT, left)))

    def t_wikipedia(self, q: str) -> tuple[bool, str, str]:
        st, body = self._get("https://en.wikipedia.org/w/api.php?action=opensearch&limit=1&format=json&search=" + urllib.parse.quote(q), self.left)
        if st != 200:
            return False, "", ""
        titles = json.loads(body)[1]
        if not titles:
            return False, "", ""
        st, body = self._get("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(titles[0].replace(" ", "_"), safe=""), self.left)
        d = json.loads(body) if st == 200 else {}
        ext = str(d.get("extract") or "")
        return (True, ext, "Wikipedia") if ext and d.get("type") != "disambiguation" else (False, "", "")

    def t_ddg_instant(self, q: str) -> tuple[bool, str, str]:
        st, body = self._get("https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1&q=" + urllib.parse.quote(q), self.left)
        d = json.loads(body) if st == 200 and body.strip() else {}
        ab = str(d.get("AbstractText") or d.get("Answer") or "")
        return (True, ab, str(d.get("AbstractSource") or "DuckDuckGo")) if ab else (False, "", "")

    def t_web_search(self, q: str) -> tuple[bool, str, str]:
        st, body = self._get("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q), self.left)
        if st != 200:
            return False, "", ""
        hits = parse_ddg(body)
        if not hits:
            return False, "", ""
        top = hits[0]
        facts = " ".join(h["snippet"] for h in hits[:3] if h["snippet"])
        self.top_url = top["url"]
        return (True, facts, top["host"]) if facts else (False, "", "")

    def t_fetch_page(self, url: str) -> tuple[bool, str, str]:
        u = urllib.parse.urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return False, "", ""
        rp = urllib.robotparser.RobotFileParser()
        st, robots = self._get(f"{u.scheme}://{u.netloc}/robots.txt", min(2.0, self.left))
        if st == 200:
            rp.parse(robots.splitlines())
            if not rp.can_fetch(UA, url):
                return False, "robots.txt disallows", ""
        st, raw = self._get(url, self.left)
        txt = page_text(raw) if st == 200 else ""
        return (True, txt, u.hostname.removeprefix("www.")) if len(txt) > 80 else (False, "", "")

    # ---- PC tools
    def t_calculator(self, expr: str) -> tuple[bool, str, str]:
        return True, say_number(safe_eval(expr)), "calculator"

    def t_status(self, text: str) -> tuple[bool, str, str]:
        if self.about is None:
            return False, "", ""
        out = self.about(text)
        return (True, out, "Nupen") if out else (False, "", "")

    def allowed(self, p: str | Path) -> Path:
        """Resolve a path and require it to sit inside an allowed root (symlinks and .. resolved first). Never opens anything."""
        rp = Path(p).expanduser().resolve()
        if not any(rp == r or r in rp.parents for r in self.roots):
            raise PermissionError(FOLDER_REFUSAL)
        if _SECRET.search(rp.name):
            raise PermissionError("I will not open files that look like secrets, sir.")
        return rp

    def t_file_search(self, q: str) -> tuple[bool, str, str]:
        if _OUTSIDE.search(q):
            raise PermissionError(FOLDER_REFUSAL)
        words = [w for w in re.findall(r"[\w.-]{3,}", q.lower()) if w not in ("file", "files", "the", "for", "called", "named", "with", "about", "find")]
        if not words:
            return False, "", ""
        found: list[tuple[int, str]] = []
        t_end = self.clock() + 3.0
        skip = {".git", ".venv", "__pycache__", "node_modules", "state", "evidence"}
        for root in self.roots:
            for dirpath, dirs, files in os.walk(root):
                dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
                for f in files:
                    score = sum(w in f.lower() for w in words)
                    if score and not _SECRET.search(f):
                        found.append((score, f"{f} in {Path(dirpath).name}"))
                if self.clock() > t_end or len(found) >= 60:
                    break
            if self.clock() > t_end:
                break
        found.sort(key=lambda x: -x[0])
        top = [f for _, f in found[:3]]
        return (True, "I found " + "; ".join(top) + ".", "Nupen's folders") if top else (False, "", "")

    def t_read_file(self, path: str) -> tuple[bool, str, str]:
        p = self.allowed(path)
        if not p.is_file():
            return False, "", ""
        return True, page_text(p.read_text(encoding="utf-8", errors="replace"), 1500), p.name

    def t_queue(self, task: str) -> tuple[bool, str, str]:
        """Writes ONE request file in the phone dir (never elsewhere). Nothing is executed from the phone."""
        if self.queue_dir is None:
            return False, "", ""
        d = self.queue_dir.resolve()
        dest = (d / f"{self.now():%Y%m%d-%H%M%S}-{abs(hash(task)) % 10**6:06d}.json").resolve()
        if d not in dest.parents:
            raise PermissionError("Writing there needs your confirmation, sir.")
        d.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"kind": "coding_task", "text": task[:600], "from": "phone", "t": self.now().isoformat(timespec="seconds"),
                                    "status": "queued"}, ensure_ascii=False), encoding="utf-8")
        return True, "queued", "Nupen"

    # ---- answering
    left = BUDGET_S

    def phrase(self, q: str, facts: str, source: str) -> str:
        out = ""
        if self.summarize is not None:
            try:
                out = self.summarize(q, facts[:1200], source).strip()
            except Exception:  # noqa: BLE001
                out = ""
        out = sentences(out or facts)
        if source.lower() not in out.lower():
            out = f"According to {source}, {_lower_first(out)}"
        out = out.rstrip(".") + "."
        return out if out.endswith("sir.") else out[:-1] + ", sir."

    def web_answer(self, q: str, recent: bool) -> tuple[str, list[dict[str, Any]]]:
        """Fallback chain; the last route is the phone's own web search."""
        t_end = self.clock() + BUDGET_S
        chain = ["ddg_instant", "web_search", "wikipedia"] if recent else ["wikipedia", "ddg_instant", "web_search"]
        for tool in chain:
            self.left = t_end - self.clock()
            if self.left < 0.6:
                break
            ok, facts, source = self.call(tool, q)
            if tool == "web_search" and ok and len(facts) < 120 and getattr(self, "top_url", ""):    # thin snippets: read the top page
                self.left = t_end - self.clock()
                ok2, facts2, src2 = self.call("fetch_page", self.top_url)
                if ok2:
                    facts, source = facts2, src2
            if ok and facts:
                return self.phrase(q, facts, source), []
        a = self.interpret(f"google {q}", self.now())
        act = self.route(a[1]) if a and a[1] else None
        return "I could not reach the web from here just now, sir. I have opened a search on your phone instead.", ([act] if act else [])

    # ---- planning (rules over the owner's words)
    def clause(self, text: str) -> Optional[tuple[str, Any]]:
        """One clause -> ('phone', (reply, action)) | ('tool', name+args) | None. Phone words win; tool intents are the owner's explicit asks."""
        s = text.strip()
        if m := re.match(r"(?:please\s+)?(?:queue|add|submit|file)\s+(?:a\s+)?(?:new\s+)?(?:coding\s+)?(?:task|request|job)\s*(?:to|for|:|-)?\s*(.+)$", s, re.I) or \
                re.match(r"(?:please\s+)?(?:ask|tell|have|get)\s+nupen\s+to\s+(.+)$", s, re.I):
            return "queue", m[1].strip()
        if re.match(r"(?:find|search for|look for|locate|where is|which)\b.*\b(?:files?|documents?|scripts?)\b", s, re.I):
            return "files", s
        if (e := math_expression(s)) is not None and re.match(r"(?:what(?:'s| is)|calculate|compute|work out|how much is|whats)\b", s, re.I):
            return "calc", e
        r = self.interpret(s, self.now())
        if r is not None and r[1] is not None:
            return "phone", r
        if recent_cue(s):
            return "web_recent", s
        if fact_cue(s):
            return "web", s
        return None

    def run(self, text: str) -> Optional[Result]:
        """None = not for the brain (the server carries on with its normal paths)."""
        self.calls = []
        parts = [p for p in re.split(r"\s*(?:,\s*)?\b(?:and then|and also|after that|then|and)\b\s+", text.strip(), flags=re.I) if p.strip()]
        self.top_url = ""
        plan: list[tuple[str, Any]] = []
        if len(parts) >= 2:
            cl = [self.clause(p) for p in parts]
            if all(cl):
                plan = [c for c in cl if c]
        if not plan:
            one = self.clause(text)
            if one is None or one[0] == "phone":
                return None                                      # plain phone commands stay on the fast rules path
            plan = [one]
        replies: list[str] = []
        actions: list[dict[str, Any]] = []
        for kind, arg in plan:
            if kind == "phone":
                replies.append(arg[0])
                actions.append(arg[1])
            elif kind == "calc":
                ok, facts, _ = self.call("calculator", arg)
                replies.append(f"That is {facts}, sir." if ok else "I could not work that out, sir.")
            elif kind == "queue":
                try:
                    ok, _, _ = self.call("queue_coding_task", arg)
                except PermissionError:
                    ok = False
                replies.append("Queued for Nupen's coding pipeline, sir. Nothing runs until it picks it up." if ok else "I could not queue that, sir.")
            elif kind == "files":
                if _OUTSIDE.search(arg):
                    self.calls.append(Call("file_search", False, 0, "refused"))
                    replies.append(FOLDER_REFUSAL)
                    continue
                ok, facts, _ = self.call("file_search", arg)
                replies.append(facts.rstrip(".") + ", sir." if ok else "I found nothing like that in Nupen's folders, sir.")
            elif kind in ("web", "web_recent"):
                ans, acts = self.web_answer(arg, kind == "web_recent")
                replies.append(ans)
                actions.extend(acts)
        actions = [self.route(a) or a for a in actions]
        risky = [a for a in actions if a.get("type") in RISKY_TYPES]
        reply = " ".join(replies)
        if risky:
            return Result(f"{_confirm_line(risky)} Say confirm to go ahead, sir.", actions, list(self.calls), confirm=True, intent="confirm")
        return Result(reply, actions, list(self.calls), intent="brain")


def _confirm_line(risky: list[dict[str, Any]]) -> str:
    bits = []
    for a in risky:
        t = a["type"]
        bits.append({"message": f"message {a.get('to', '')}", "call": f"call {a.get('to', '')}", "mail": f"email {a.get('to', '')}"}[t].strip())
    return "I am about to " + " and ".join(bits) + "."


_RECENT = re.compile(r"\b(?:latest|news|headlines?|today'?s|tonight'?s|current(?:ly)?|right now|price of|stock price|score|scores|who won|results? of|"
                     r"release date|weather (?:in|for|at)|exchange rate|trending)\b", re.I)
_FACT = re.compile(r"^(?:(?:please\s+)?(?:look (?:it|that) up|check online|search online|find out)\b|who (?:is|was|are|were)\b|"
                   r"what(?:'s| is| was) the (?:population|capital|height|tallest|largest|biggest|speed|distance|age|boiling|melting)\b|"
                   r"when (?:is|was|did|does)\b|how (?:tall|old|far|big|long|many|much) (?:is|are|was|were|do|does|did)\b|what year\b|"
                   r"(?:what|who) (?:is|are|was) .{3,60}(?:on the web|online)$)", re.I)
_NOTFACT = re.compile(r"\b(?:you|your|nupen|my|me|i|we)\b|\bjoke\b|\bstory\b", re.I)


def recent_cue(s: str) -> bool:
    return bool(_RECENT.search(s)) and len(s.split()) <= 18


def fact_cue(s: str) -> bool:
    return bool(_FACT.search(s)) and not _NOTFACT.search(s) and len(s.split()) <= 18


# Quick pre-check used by the server so ordinary chat never even imports this module's heavier paths.
HINT = re.compile(r"\b(?:and then|and also|after that|then|and|nupen to|queue|coding task|files?|documents?|scripts?|latest|news|headlines?|"
                  r"today'?s|current(?:ly)?|right now|price of|score|who won|results? of|release date|weather|exchange rate|trending|"
                  r"look (?:it|that) up|check online|search online|find out|who (?:is|was|are|were)|population|capital|when (?:is|was|did|does)|"
                  r"how (?:tall|old|far|big|long|many|much)|what year)\b|^(?:what(?:'s| is)|calculate|compute|work out|how much is|whats)\s+\S*\d", re.I)


def parse_ddg(body: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    links = list(re.finditer(r'(?s)class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', body))
    for i, m in enumerate(links):
        href = html.unescape(m[1])
        if "uddg=" in href:
            href = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0])
        host = (urllib.parse.urlparse(href).hostname or "").removeprefix("www.")
        seg = body[m.end(): links[i + 1].start() if i + 1 < len(links) else len(body)]
        sn = re.search(r'(?s)class="result__snippet"[^>]*>(.*?)</a>', seg)
        if host and href.startswith(("http://", "https://")):
            out.append({"url": href, "host": host, "title": page_text(m[2], 120), "snippet": page_text(sn[1] if sn else "", 300)})
        if len(out) >= 5:
            break
    return out


def default_roots() -> list[Path]:
    """Nupen's own folders only: the project checkout and the runtime dir. Nothing else of the owner's PC is ever searched."""
    root = Path(__file__).resolve().parents[1]
    home = Path.home()
    cands = [root, home / "weekly7", home / "creator_runtime"]
    return [c for c in cands if c.is_dir()]
