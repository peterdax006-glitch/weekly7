"""Nupen terminal conversation, LAYER 2: a small local model as Nupen's VOICE (owner, 3 Oct 2026: "Turn Nupen into a Language model so it
can talk to me and have a two way conversation with me").

The model is a swappable part (Voice: any llama.cpp GGUF through creator.generator.LocalModel - Qwen3-1.7B by default, 4B on request; a
GPU pod answers when creator.pulseroute routes that model, never required). Layer 1 (creator.conversation) stays the skeleton: its handlers
read the real records, and nothing here invents a fact.

    ModelUnderstand   free text -> Parsed(intent, slots). Hybrid by default: an unambiguous rule hit costs no model call; anything else
                      (ambiguous, unknown, a follow-up) asks the voice for JSON over conversation.INTENTS + 'ask' + 'smalltalk';
                      a bad or missing answer falls back to RuleUnderstand.
    h_ask             an open question: BM25 over Nupen's own docs (module docstrings, docs/*.md) and records (lessons, cycles, goal
                      proposals, plan explanations); the snippets are the Facts. Private text (gpuday.private_reason) is never indexed.
    ModelSpeak        the voice writes a SHORT reply from the Facts only, with the last turns as history. Every number and id in the reply
                      must occur in the facts or the question (`ungrounded`), or the rule speaker's report is used instead. The evidence
                      line is appended by us, so it can be neither dropped nor invented. Actions (pause/resume/approve/request) and
                      help stay literal.

Efficiency (priority 1 on the home PC): one warm server is leased for the session (creator.modelpool) instead of loading weights; replies are
capped at 150 tokens, understanding at 40; the retrieval index is built once and cached by file signature under the runtime dir."""
from __future__ import annotations

import dataclasses
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from creator import conversation as CV

LAYER = 2
TALK_INTENTS = {"ask": "an open question about Nupen, its code, docs or records (what is X, how does Y work, explain a module)",
                "smalltalk": "greetings, thanks, who are you, chit-chat"}
GREETING = re.compile(r"^\s*(?:hi|hello|hey|good (?:morning|evening|afternoon|night)|thanks?|thank you|who are you|what are you|"
                      r"what(?:'s| is) your name|nice|cool|great|ok(?:ay)?)\b", re.I)
LITERAL = ("help", "unknown", "pause", "resume", "approve", "request", "confirm_yes", "confirm_no")   # spoken exactly as the rules say
MODELS = {"1.7b": ("Qwen3-1.7B-Q4_K_M.gguf",), "4b": ("Qwen3-4B-Q4_K_M.gguf", "dl/Qwen3-4B-Q4_K_M.gguf"),
          "8b": ("Qwen3-8B-Q4_K_M.gguf", "dl/Qwen3-8B-Q4_K_M.gguf")}
SPEAK_TOKENS = 150
UNDERSTAND_TOKENS = 40
FACT_CHARS = 1800                     # facts shown to the voice (prompt processing is most of a CPU reply's time)
HISTORY_TURNS = 2


# ------------------------------------------------------------------------------------------------ the voice (a swappable model)

@dataclasses.dataclass
class Reply:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0


def resolve_model(name: str, runtime: Optional[Path] = None) -> Path:
    """'1.7b' | '4b' | '8b' | a GGUF path -> the weights file (models/ first, then models/dl/)."""
    if name.lower() not in MODELS:
        return Path(name)
    if runtime is None:
        from creator import device as DEV
        runtime = DEV.runtime_dir()
    cands = [Path(runtime) / "models" / f for f in MODELS[name.lower()]]
    return next((p for p in cands if p.is_file()), cands[0])


class Voice:
    """One model for the session. `factory` builds the client (default: a LocalModel of `model`, which leases a warm pool server, or starts
    one, or attaches to a routed pod); anything with .chat(messages, max_tokens=, temperature=) works (tests use a stub)."""

    def __init__(self, model: Path, ctx: int = 8192, slot_wait_s: float = 60.0, factory: Optional[Callable[[], Any]] = None,
                 timeout_s: float = 180.0, servers: Optional[int] = None, startup_s: float = 300.0, route: str = "") -> None:
        self.route = route                   # 'auto': use a healthy pod route (creator.pulseroute) even where the device setting is off
        self.startup_s = startup_s           # a busy PC (the swarm at full CPU) loads a server slowly
        self.model, self.ctx, self.slot_wait_s, self.timeout_s, self.servers = Path(model), ctx, slot_wait_s, timeout_s, servers
        self.factory = factory
        self.lm: Any = None
        self.error = ""
        self.calls = 0

    @property
    def name(self) -> str:
        return self.model.name

    def open(self) -> bool:
        if self.lm is not None:
            return True
        if self.error:
            return False
        try:
            if self.factory is not None:
                self.lm = self.factory()
            else:
                from creator import generator as G
                lm = G.LocalModel(model=self.model, ctx=self.ctx, slot_wait_s=self.slot_wait_s, servers=self.servers,
                                  startup_s=self.startup_s)
                if self.route == "auto" and not self._attach_route(lm):
                    self.lm = lm.__enter__()
                elif self.route != "auto":
                    self.lm = lm.__enter__()
                else:
                    self.lm = lm
        except Exception as e:  # noqa: BLE001 - no voice means layer 1 answers
            self.error = f"{type(e).__name__}: {str(e)[:160]}"
            return False
        return True

    def _attach_route(self, lm: Any) -> bool:
        """Attach `lm` to a healthy routed pod server of this model (as LocalModel._route_attach does when the setting is 'auto')."""
        try:
            from creator import pulseroute as PR
            got = PR.attach(self.model, {"pulse_route": "auto"})
        except Exception:  # noqa: BLE001
            return False
        if got is None:
            return False
        lm.port, lm.pulse = got
        lm.routed = True
        return True

    def close(self) -> None:
        lm, self.lm = self.lm, None
        if lm is not None and hasattr(lm, "__exit__"):
            try:
                lm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    def where(self) -> str:
        lm = self.lm
        if lm is None:
            return "none"
        if getattr(lm, "pulse", ""):
            return "gpu-pod"
        return "pool-lease" if getattr(lm, "leased", False) else ("local" if getattr(lm, "port", 0) else "stub")

    def ask(self, messages: list[dict[str, str]], max_tokens: int, temperature: float = 0.2) -> Reply:
        """The model's answer (scratch <think> removed) with token counts: exact from the server's usage when we talk to it directly,
        else estimated (4 characters a token)."""
        if not self.open():
            raise RuntimeError(f"no voice: {self.error}")
        from creator import generator as G
        msgs = G.prepare_messages(messages, self.model)
        t0 = time.monotonic()
        lm = self.lm
        tin = tout = 0
        direct = bool(getattr(lm, "port", 0)) and not getattr(lm, "pulse", "")
        if getattr(lm, "routed", False):                     # the overflow route: direct only when the prompt may leave this PC
            from creator import pulseroute as PR
            direct = PR.may_leave(msgs)
        got = None
        if direct:
            try:
                got = self._post(lm.port, msgs, max_tokens, temperature)
            except OSError:                                  # (URLError, timeouts, resets) a routed pod that fails: LocalModel falls back
                if not getattr(lm, "routed", False):
                    raise
                from creator import pulseroute as PR
                PR.invalidate()
        if got is not None:
            text, tin, tout = got
        else:
            text = str(lm.chat(msgs, max_tokens=max_tokens, temperature=temperature))
        text = G.THINK_BLOCK.sub("", text).strip()
        self.calls += 1
        if not tin:
            tin = sum(len(m.get("content", "")) for m in msgs) // 4
        if not tout:
            tout = max(1, len(text) // 4)
        return Reply(text, tin, tout, time.monotonic() - t0)

    def _post(self, port: int, msgs: list[dict[str, str]], max_tokens: int, temperature: float) -> tuple[str, int, int]:
        import urllib.request
        body = json.dumps({"messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "seed": 0, "cache_prompt": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            d = json.loads(r.read().decode("utf-8"))
        u = d.get("usage") or {}
        return str(d["choices"][0]["message"].get("content") or ""), int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


# ------------------------------------------------------------------------------------------------ grounding: no invented numbers or ids

_NUM = re.compile(r"(?<![A-Za-z0-9_])\d+(?:[.,]\d+)*(?![A-Za-z_])")
_ID = re.compile(r"\b(?:CP\d{3,5}|K\d{2}|GP-[0-9a-f]{4,}|[0-9a-f]{12})\b", re.I)
_LIST_MARK = re.compile(r"(?m)^\s*\d{1,2}[.)]\s")


def _norm_num(x: str) -> Optional[float]:
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return None


def ungrounded(reply: str, source: str) -> list[str]:
    """Numbers and record ids in `reply` that do not occur in `source` (the facts + the owner's words). [] = grounded.
    List markers ('1. ') are not claims; '0.857' and '0.86' are different numbers (rounding is a change of fact, kept strict)."""
    text = _LIST_MARK.sub(" ", reply)
    have_ids = {m.lower() for m in _ID.findall(source)}
    bad = [m for m in _ID.findall(text) if m.lower() not in have_ids]
    nums_src = {v for v in (_norm_num(x) for x in _NUM.findall(source)) if v is not None}
    for m in _NUM.findall(_ID.sub(" ", text)):
        v = _norm_num(m)
        if v is not None and v not in nums_src:
            bad.append(m)
    return list(dict.fromkeys(bad))


def facts_text(f: CV.Facts) -> str:
    return "\n".join(f.lines) + "\n" + " ".join(f.evidence) + "\n" + f.user


# State verbs (teacher, 3 Oct: "I am working on K28" when the plan chose nothing is an overstatement): a verb of state in the reply must match
# the status the record states. Each rule: (claim in the reply, what the facts must say for it, what the record says instead).
_WORKING = re.compile(r"\b(?:i(?:'m| am)\s+(?:currently\s+|now\s+)?(?:working on|busy with|building|doing|running)|i(?:'ve| have)\s+(?:chosen|picked)"
                      r"|my (?:current )?(?:work|task) is)\b\s*(?P<what>[^.;,\n]*)", re.I)
_BLOCKED = re.compile(r"\b(?:i(?:'m| am)\s+(?:blocked|stuck)|blocked by|stuck on)\b", re.I)
_LEARNED = re.compile(r"\bi(?:'ve| have)?\s+learn(?:ed|t)\b", re.I)
_DONE = re.compile(r"\bi(?:'ve| have)?\s+(?:finished|completed|adopted|merged|approved)\b", re.I)
_NOTHING_CHOSEN = re.compile(r"Chosen now: nothing", re.I)
_CHOSEN = re.compile(r"Chosen now: (?P<list>[^.\n]+)", re.I)


def state_claims(reply: str, source: str) -> list[str]:
    """Verbs of state the record does not support ([] = consistent):
      working on / busy with X   needs a chosen plan entry (and X, when it names an id, among the chosen) or a running swarm
      blocked / stuck            needs a constraint, a deferral or a waiting dependency in the facts
      learned                    needs a lesson in the facts;   finished / adopted / merged / approved   needs that outcome in the facts."""
    bad: list[str] = []
    for m in _WORKING.finditer(reply):
        what = m.group("what")
        ch = _CHOSEN.search(source)
        if _NOTHING_CHOSEN.search(source):
            bad.append(f"'{m.group(0).strip()}' but the plan chose nothing")
        elif ch:
            ids = [x.lower() for x in _ID.findall(what)]
            chosen = ch.group("list").lower()
            if ids and not any(i in chosen for i in ids):
                bad.append(f"'{m.group(0).strip()}' but the plan chose {ch.group('list').strip()}")
        elif not re.search(r"\bswarm is running\b", source, re.I):
            bad.append(f"'{m.group(0).strip()}' but no record says work is chosen or running")
    if _BLOCKED.search(reply) and not re.search(r"limit|constraint|deferred|waiting on|blocked|stuck|bottleneck", source, re.I):
        bad.append("claims to be blocked; no record says so")
    if _LEARNED.search(reply) and not re.search(r"\blessons?\b|\blearn", source, re.I):
        bad.append("claims to have learned; no lesson in the facts")
    for m in _DONE.finditer(reply):
        verb = m.group(0).split()[-1].lower()
        if not re.search(rf"(?<!not )\b{verb[:5]}", source, re.I):     # 'not adopted' does not support 'I adopted'
            bad.append(f"claims '{verb}'; the facts do not")
    return list(dict.fromkeys(bad))


# ------------------------------------------------------------------------------------------------ retrieval (BM25 over docs + records)

_TOK = re.compile(r"[a-z][a-z0-9_]+|\d+")
STOP = frozenset("the a an and or of to in on for is are was were be been it its this that with as at by from what how why who which do does "
                 "did you your i me my we our can could would should will about into than then there their them they not no yes have has "
                 "had so if but just also any all more most some such only".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in STOP and len(t) > 1]


def _private(text: str) -> bool:
    from creator import gpuday as GD
    return bool(GD.private_reason(text))


def _docstring(path: Path) -> str:
    """The module docstring, read without importing (and without parsing the whole file when it is long)."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.match(r'\s*(?:#.*\n\s*)*[rubRUB]*("""|\'\'\')(.*?)\1', src, re.S)
    return m.group(2).strip() if m else ""


def corpus(root: Path, max_chars: int = 900) -> list[dict[str, Any]]:
    """Every document worth finding again: (id, kind, text). Private text is dropped (gpuday.private_reason), never indexed."""
    root, st = Path(root), Path(root) / "state" / "creator"
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(did: str, kind: str, text: str, key: str = "") -> None:
        text = re.sub(r"\s+", " ", text).strip()
        key = key or text
        if len(text) < 20 or key in seen or _private(text):
            return
        seen.add(key)
        docs.append({"id": did, "kind": kind, "text": text[:max_chars]})
    for sub in ("creator", "scripts"):
        for p in sorted((root / sub).rglob("*.py")) if (root / sub).is_dir() else []:
            if "lm" in p.relative_to(root).parts[1:-1] or p.name.startswith("test"):
                continue
            ds = _docstring(p)
            if ds:
                add(f"doc:{p.relative_to(root).as_posix()}", "module", f"{p.stem}: {ds}")
    for md in [*sorted((root / "docs").glob("*.md")), root / "creator" / "ARCHITECTURE.md"]:
        try:
            body = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, sec in enumerate(re.split(r"(?m)^#{1,3} ", body)):
            add(f"doc:{md.relative_to(root).as_posix()}#{i}", "doc", sec)
    for r in CV._jsonl(st / "lessons.jsonl"):
        obj, ver = str(r.get("objective") or "").strip(), str(r.get("verdict") or "").strip()
        if r.get("lesson_id") and len(obj) > 2:
            add(f"lesson:{r['lesson_id']}", "lesson", f"lesson {r['lesson_id']} {r.get('task_kind', '')} {r.get('package_id', '')}: "
                f"{obj[:400]} verdict: {ver[:200] or 'none yet'}", key=f"lesson|{obj[:400]}|{ver[:200]}")
    for r in CV._jsonl(st / "goal_proposals.jsonl"):
        if r.get("event") == "proposal":
            add(f"goal:{r.get('id')}", "goal", f"goal proposal {r.get('id')}: {r.get('title')} - {r.get('rationale')}")
    for r in CV._jsonl(st / "plan_explanations.jsonl")[-300:]:
        add(f"plan:{r.get('node')}", "plan", f"plan {r.get('component')} {r.get('step')} ({r.get('node')}): {r.get('why')}")
    try:
        from creator import constraints as CON
        for x in CON.read_cycles(st)[-200:]:
            add(f"cycle:{x.package}", "cycle", f"cycle {x.package} {x.requirement} {x.outcome} after {x.seconds:.0f} s: {x.reason}")
    except Exception:  # noqa: BLE001 - a broken cycle record removes it from search, nothing else
        pass
    return docs


class Index:
    """Okapi BM25 (k1 1.2, b 0.75) over `corpus`; postings in dicts - a few thousand short documents answer in milliseconds."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.tf = [Counter(tokens(d["text"])) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - k + 0.5) / (k + 0.5)) for t, k in df.items()}
        self.post: dict[str, list[int]] = {}
        for i, c in enumerate(self.tf):
            for t in c:
                self.post.setdefault(t, []).append(i)

    def search(self, query: str, k: int = 4, min_score: float = 1.5) -> list[dict[str, Any]]:
        sc: dict[int, float] = {}
        for t in set(tokens(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                sc[i] = sc.get(i, 0.0) + idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * self.len[i] / self.avg))
        best = sorted(sc.items(), key=lambda kv: -kv[1])[:k]
        return [dict(self.docs[i], score=round(s, 2)) for i, s in best if s >= min_score]


def _signature(root: Path) -> str:
    """Changes when any indexed source changes (sizes + mtimes of the record files and the newest code/doc file)."""
    root = Path(root)
    parts = []
    for f in ("lessons.jsonl", "goal_proposals.jsonl", "plan_explanations.jsonl", "ledger.jsonl"):
        try:
            s = (root / "state" / "creator" / f).stat()
            parts.append(f"{f}:{s.st_size}:{s.st_mtime_ns}")
        except OSError:
            parts.append(f"{f}:-")
    newest = 0
    for sub in ("creator", "scripts", "docs"):
        try:
            newest = max([newest] + [p.stat().st_mtime_ns for p in (root / sub).iterdir()])
        except OSError:
            pass
    return "|".join(parts) + f"|{newest}"


_INDEX: dict[str, Any] = {}


def index_for(root: Path, cache_dir: Optional[Path] = None) -> Index:
    """The index of `root`, built once per process and cached on disk (JSON documents; BM25 statistics are rebuilt, ~0.1 s)."""
    sig = _signature(root)
    key = str(Path(root).resolve())
    hit = _INDEX.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]  # type: ignore[no-any-return]
    docs: Optional[list[dict[str, Any]]] = None
    cf = None
    if cache_dir is not None:
        import hashlib
        cf = Path(cache_dir) / f"talk_index_{hashlib.sha1(key.encode()).hexdigest()[:10]}.json"
        try:
            d = json.loads(cf.read_text(encoding="utf-8"))
            if d.get("sig") == sig:
                docs = list(d["docs"])
        except (OSError, ValueError, KeyError):
            pass
    if docs is None:
        docs = corpus(root)
        if cf is not None:
            try:
                cf.parent.mkdir(parents=True, exist_ok=True)
                cf.write_text(json.dumps({"sig": sig, "docs": docs}), encoding="utf-8")
            except OSError:
                pass
    ix = Index(docs)
    _INDEX[key] = (sig, ix)
    return ix


def _cache_dir() -> Optional[Path]:
    try:
        from creator import device as DEV
        return Path(DEV.runtime_dir()) / "talk"
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ layer-2 handlers

def h_ask(c: CV.Ctx, s: dict[str, Any]) -> CV.Facts:
    q = str(s.get("question") or "")
    hits = index_for(c.root, _cache_dir()).search(q, k=int(s.get("k", 4)))
    if not hits:
        return CV.Facts("ask", ["I found nothing in my records or docs about that."])
    return CV.Facts("ask", [f"[{h['id']}] {h['text'][:420]}" for h in hits], [h["id"] for h in hits])


def h_smalltalk(c: CV.Ctx, s: dict[str, Any]) -> CV.Facts:
    return CV.Facts("smalltalk", ["I am Nupen, a self-development engine that runs on the owner's home PC: I plan, build, test and adopt "
                                  "changes to my own code, and I learn from each attempt.",
                                  "I answer from my own records; my voice is a small local language model.",
                                  f"It is {c.now.strftime('%A %H:%M')} here."], ["identity"])


def h_unknown_ask(c: CV.Ctx, s: dict[str, Any]) -> CV.Facts:
    """Layer 2 treats what nobody understood as an open question; with nothing found, layer 1's 'what I can do' list."""
    f = h_ask(c, s)
    return f if f.evidence else CV.h_unknown(c, s)


HANDLERS: dict[str, Callable[[CV.Ctx, dict[str, Any]], CV.Facts]] = {"ask": h_ask, "smalltalk": h_smalltalk, "unknown": h_unknown_ask}


# ------------------------------------------------------------------------------------------------ the session (shared by understand and speak)

@dataclasses.dataclass
class Session:
    history: list[dict[str, str]] = dataclasses.field(default_factory=list)     # {"user", "reply", "intent"}
    turn: dict[str, Any] = dataclasses.field(default_factory=dict)              # this turn's numbers (logged, read by the eval)

    def new_turn(self) -> None:
        self.turn = {"layer": LAYER, "understand_by": "rules", "voice_s": 0.0, "tokens_in": 0, "tokens_out": 0, "spoken_by": "rules"}

    def add(self, r: Reply, kind: str = "") -> None:
        self.turn.setdefault("calls", []).append([kind, round(r.seconds, 2), r.tokens_in, r.tokens_out])
        self.turn["voice_s"] = round(self.turn.get("voice_s", 0.0) + r.seconds, 3)
        self.turn["tokens_in"] = self.turn.get("tokens_in", 0) + r.tokens_in
        self.turn["tokens_out"] = self.turn.get("tokens_out", 0) + r.tokens_out


def rule_scores(text: str) -> list[tuple[float, str]]:
    """Layer 1's scores per intent (best first), so an unambiguous rule hit can skip the model."""
    out = []
    for it in CV.INTENTS:
        sc = sum(it.weight for pat in it.patterns if re.search(pat, text, re.I))
        if sc:
            out.append((sc, it.name))
    return sorted(out, reverse=True)


def understand_messages(text: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    names = [f"{i.name}: {i.about} (e.g. \"{i.example}\")" for i in CV.INTENTS if i.name != "help"]
    names += [f"{k}: {v}" for k, v in TALK_INTENTS.items()] + ["help: what can you do", "unknown: none of these"]
    sys_ = ("You route the owner's message to Nupen (a self-improving coding system) to ONE intent. Reply with JSON only, e.g. "
            "{\"intent\": \"status\"} or {\"intent\": \"request\", \"request\": \"<what the owner wants built>\"}.\nIntents:\n- "
            + "\n- ".join(names))
    prev = ""
    if history:
        h = history[-1]
        prev = f"(Previous turn: owner said \"{h['user'][:120]}\"; topic {h['intent']}.)\n"
    return [{"role": "system", "content": sys_}, {"role": "user", "content": prev + text}]


ALLOWED = set(CV.INTENT_BY_NAME) | set(TALK_INTENTS) | {"unknown"}


def parse_intent_json(raw: str) -> Optional[dict[str, Any]]:
    m = re.search(r"\{.*?\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(d, dict) or str(d.get("intent", "")).strip().lower() not in ALLOWED:
        return None
    d["intent"] = str(d["intent"]).strip().lower()
    return d


class ModelUnderstand:
    """mode 'hybrid' (default): rules when they are unambiguous, else the voice; 'model': always the voice; 'rules': layer 1."""

    def __init__(self, voice: Optional[Voice], session: Session, mode: str = "hybrid") -> None:
        self.voice, self.session, self.mode = voice, session, mode
        self.rules = CV.RuleUnderstand()

    def _rules_sure(self, text: str, p: CV.Parsed) -> bool:
        sc = rule_scores(text)
        if p.intent == "smalltalk":
            return True
        if p.intent == "unknown" or not sc:
            return False
        if p.intent in ("request", "approve", "pause", "resume", "help"):     # strong, anchored patterns; actions confirm anyway
            return True
        return len(sc) == 1 or sc[0][0] >= sc[1][0] + 2.0

    def parse(self, text: str) -> CV.Parsed:
        self.session.new_turn()
        p = self.rules.parse(text)
        p.slots.setdefault("question", text)
        if p.intent == "unknown" and GREETING.search(text) and len(text) < 60:
            p.intent = "smalltalk"
        if self.mode == "rules" or self.voice is None or (self.mode == "hybrid" and self._rules_sure(text, p)):
            return p
        try:
            r = self.voice.ask(understand_messages(text, self.session.history), UNDERSTAND_TOKENS, temperature=0.0)
        except Exception as e:  # noqa: BLE001 - the voice is never required
            self.session.turn["understand_error"] = f"{type(e).__name__}"
            return p
        self.session.add(r, "understand")
        d = parse_intent_json(r.text)
        if d is None:
            self.session.turn["understand_by"] = "rules (model answer unusable)"
            return p
        intent = d["intent"]
        slots = dict(p.slots)
        if intent == "request":
            req = str(d.get("request") or slots.get("request") or re.sub(r"^(?:please|could you|can you|i want you to)\s+", "", text, flags=re.I))
            slots["request"] = req
        if intent == "explain" and "package" not in slots and "component" not in slots:
            last = self.session.history[-1]["user"] if self.session.history else ""
            slots.update({k: v for k, v in CV.RuleUnderstand.slots_from(last).items() if k in ("package", "component")})
            if "package" not in slots and "component" not in slots:
                intent = "ask"
        if intent == "approve" and "goal" not in slots:
            intent = "goals"
        self.session.turn["understand_by"] = "model"
        return CV.Parsed(intent, slots)


SPEAK_SYSTEM = ("You are Nupen, a self-improving coding system. Your owner is asking YOU about YOURSELF: answer as Nupen in the first person "
                "('I', 'my'); 'you' in the question means you, Nupen. At most 3 short sentences, plainly. Use ONLY the FACTS given with the "
                "question: never write a number, name or id that is not in them, and say a status exactly as they state it (considered is not "
                "working on). If the FACTS do not answer it, say \"I don't know\" and what you do know. Do not list evidence; it is added for you.")


def speak_messages(f: CV.Facts, history: list[dict[str, str]]) -> list[dict[str, str]]:
    """System + one user message: the facts, the earlier questions (context for a follow-up; earlier ANSWERS are not repeated - measured
    3 Oct: with them in the prompt the 1.7B voice re-told the previous turn's facts), and the question."""
    block = "\n".join(f.lines)[:FACT_CHARS]
    prev = [h["user"][:200] for h in history[-HISTORY_TURNS:] if h.get("user")]
    ctx = ("Earlier the owner asked: " + " / ".join(prev) + "\n") if prev else ""
    return [{"role": "system", "content": SPEAK_SYSTEM},
            {"role": "user", "content": f"FACTS:\n{block}\n\n{ctx}QUESTION TO NUPEN: {f.user}"}]


def evidence_line(f: CV.Facts) -> str:
    return "\n[evidence: " + "; ".join(dict.fromkeys(f.evidence)) + "]" if f.evidence else ""


class ModelSpeak:
    def __init__(self, voice: Optional[Voice], session: Session, fallback: Optional[CV.Speak] = None) -> None:
        self.voice, self.session = voice, session
        self.fallback: CV.Speak = fallback or CV.RuleSpeak()
        self.retries = 1                                     # repairs of an overstated status (one more model call, only when needed)

    def render(self, facts: CV.Facts) -> str:
        out = self._render(facts)
        body = out.split("\n[evidence:", 1)[0]
        self.session.history.append({"user": facts.user, "reply": body, "intent": facts.intent})
        del self.session.history[:-8]
        return out

    def _render(self, facts: CV.Facts) -> str:
        if not self.session.turn:
            self.session.new_turn()
        if self.voice is None or facts.intent in LITERAL or facts.confirm is not None:
            return self.fallback.render(facts)
        msgs = speak_messages(facts, self.session.history)
        src = facts_text(facts)
        for attempt in range(1 + self.retries):
            try:
                r = self.voice.ask(msgs, SPEAK_TOKENS)
            except Exception as e:  # noqa: BLE001
                self.session.turn["speak_error"] = type(e).__name__
                return self.fallback.render(facts)
            self.session.add(r, "speak")
            text = r.text.strip()
            bad = ungrounded(text, src)
            claims = state_claims(text, src) if text and not bad else []
            if text and not bad and not claims:
                self.session.turn["spoken_by"] = "model" if attempt == 0 else "model (corrected)"
                return text + evidence_line(facts)
            self.session.turn.setdefault("rejected", []).append(text[:240])
            if bad:
                self.session.turn["ungrounded"] = bad[:6]
            if claims:
                self.session.turn["state_claims"] = claims[:4]
            if not claims or attempt == self.retries:        # invented numbers/ids are not repaired: the record speaks
                break
            msgs = msgs + [{"role": "assistant", "content": text},          # one repair: say the status as the record says it
                           {"role": "user", "content": "That overstates the record: " + "; ".join(claims) +
                            ". Rewrite it saying the status exactly as the FACTS state it."}]
        self.session.turn["spoken_by"] = ("rules (voice not grounded)" if self.session.turn.get("ungrounded") else
                                          "rules (voice overstated)" if self.session.turn.get("state_claims") else "rules (voice empty)")
        return self.fallback.render(facts)


# ------------------------------------------------------------------------------------------------ the conversation

def conversation(root: Path, voice: Optional[Voice], mode: str = "hybrid", log: bool = True, auto_confirm: bool = False,
                 handlers: Optional[dict[str, Callable[[CV.Ctx, dict[str, Any]], CV.Facts]]] = None) -> CV.Conversation:
    """A layer-2 Conversation (voice None = layer 1 understanding + rule speaking, with the layer-2 handlers)."""
    s = Session()
    u = ModelUnderstand(voice, s, mode)
    sp = ModelSpeak(voice, s)

    def meta() -> dict[str, Any]:
        return dict(s.turn, voice=voice.name if voice else "", voice_at=voice.where() if voice else "")
    conv = CV.Conversation(root, understand=u, speak=sp, log=log, auto_confirm=auto_confirm, handlers=dict(HANDLERS, **(handlers or {})),
                           meta=meta)
    conv.talk_session = s  # type: ignore[attr-defined]
    return conv


def banner(voice: Optional[Voice]) -> str:
    v = f"voice {voice.name}" if voice else "no voice (rules)"
    return (f"Talking to Nupen (layer 2: {v}; every answer comes from my records, evidence in brackets). "
            f"'help' lists what I know; 'quit' leaves.   " + time.strftime("%Y-%m-%d %H:%M"))
