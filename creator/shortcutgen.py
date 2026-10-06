"""Owner's words -> a shortcut plan -> Jelly source (Jellycuts language) -> static validation -> stored -> owner notified.

Nupen cannot sign .shortcut files on Windows, so it writes Jelly source code and the Jellycuts iOS app (or its 'Compile Jelly Text'
Shortcuts action) builds the shortcut on the phone; Apple's own 'Add Shortcut' sheet is the owner's one confirmation.

Safety rules enforced by validate(), not by good manners:
  * only actions in ACTIONS (each one verified against docs.jellycuts.com, see docs/JELLYCUTS.md);
  * nothing that sends data off the phone: no urlContents / downloadURL / xCallbackURL / share / runShortcut / SSH / scripts, and openURL only to
    sms:, tel:, maps: or an allow-listed Apple https host with no interpolated variable;
  * nothing that deletes or overwrites data: no deleteFile, deletePhotos, removeEvents, removeReminders, saveFile ...
Stored shortcuts live in <runtime>/phone/shortcuts/<id>.json (outside the repo); the phone fetches one by id with the bearer token.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Optional

# --------------------------------------------------------------------------------------------- the verified action table
# name -> {param: kind}; first params listed are required. kinds: str | num | bool | dur (time span like '10 min') | enum:<a>|<b>
ACTIONS: dict[str, dict[str, str]] = {
    "text": {"text": "str"},                                                   # docs.jellycuts.com/Documentation/Shortcuts/text.html
    "alert": {"alert": "str", "title": "str?", "cancel": "bool?"},
    "showResult": {"text": "str"},
    "wait": {"seconds": "num"},
    "openURL": {"url": "str"},
    "timer": {"duration": "dur"},
    "speakText": {"text": "str", "wait": "bool?", "rate": "num?", "pitch": "num?"},
    "sendNotification": {"body": "str", "title": "str?", "sound": "bool?"},
    "setDND": {"state": "bool"},
    "setFlashlight": {"state": "bool", "level": "num?"},
    "setVolume": {"level": "num"},
    "setBrightness": {"value": "num"},
    "lowPowerMode": {"state": "bool"},
    "vibrate": {},
    "batteryLevel": {},
    "getCurrentConditions": {},
    "dictateText": {},
    "createNote": {"text": "str", "show": "bool?"},
    "play": {"behavior": "enum:Play|Pause|Play/Pause"},
}
FORBIDDEN = frozenset({"urlContents", "downloadURL", "getURL", "xCallbackURL", "share", "shareEXT", "runShortcut", "runShellScript", "runSSH",
                       "runAppleScript", "deleteFile", "deletePhotos", "removeEvents", "removeReminders", "saveFile", "saveFileDropbox", "moveFile",
                       "renameFile", "appendFile", "prependFile", "sendMessage", "call", "facetime", "airdrop", "uploadPhotos", "openIn"})
URL_SCHEMES = ("sms:", "tel:", "maps://", "https://maps.apple.com/", "https://music.apple.com/")
COLORS = ("red", "orange", "tangerine", "yellow", "green", "teal", "lightblue", "blue", "navy", "grape", "purple", "pink", "grayblue", "graygreen", "graybrown")
ICONS = ("shortcuts", "sendmessage", "map", "alarmclock", "timer", "musicnote", "battery", "sun", "bell", "note", "microphone", "clock")
MAX_STEPS = 24
_NAME = re.compile(r"[^A-Za-z0-9 _-]")


def esc(s: str, n: int = 200) -> str:
    """A Jelly string literal body: no quotes/backslashes/newlines, no '${' interpolation unless we put it there ourselves."""
    return re.sub(r"[\"\\\r\n$]", " ", str(s)).strip()[:n]


# --------------------------------------------------------------------------------------------- plan steps -> Jelly lines
def _num(x: Any, lo: float, hi: float, d: float) -> str:
    try:
        v = min(max(float(x), lo), hi)
    except (TypeError, ValueError):
        v = d
    return str(int(v)) if v == int(v) else f"{v:.2f}".rstrip("0").rstrip(".")


def _dur(step: dict[str, Any]) -> str:
    if step.get("seconds") is not None:
        return '"%s sec"' % _num(step["seconds"], 1, 86400, 60)          # quoted: Open-Jellycore parses "10 min", not 10 min
    return '"%s min"' % _num(step.get("minutes", 5), 1, 1440, 5)


def _onoff(step: dict[str, Any]) -> str:
    return "false" if str(step.get("state", "on")).lower() in ("off", "false", "0", "no") else "true"


def render_step(st: dict[str, Any]) -> list[str]:
    k = st.get("kind")
    if k == "text_to":                                       # opens a Messages draft; the owner taps Send on the phone
        to = re.sub(r"[^\d+]", "", str(st.get("to", "")))
        body = urllib.parse.quote(str(st.get("text", ""))[:300], safe="")
        return [f'openURL(url: "sms:{to}&body={body}")']
    if k == "directions":
        return [f'openURL(url: "maps://?daddr={urllib.parse.quote(str(st.get("to", ""))[:120], safe="")}&dirflg=d")']
    if k == "speak":
        return [f'speakText(text: "{esc(st.get("text"))}")']
    if k == "notify":
        return [f'sendNotification(body: "{esc(st.get("body"))}", title: "{esc(st.get("title", "Nupen"), 60)}")']
    if k == "show":
        return [f'showResult(text: "{esc(st.get("text"))}")']
    if k == "alert":
        return [f'alert(alert: "{esc(st.get("text"))}", title: "{esc(st.get("title", "Nupen"), 60)}", cancel: false)']
    if k == "wait":
        return [f'wait(seconds: {_num(st.get("seconds"), 1, 3600, 1)})']
    if k == "timer":
        return [f"timer(duration: {_dur(st)})"]
    if k == "music":
        q = urllib.parse.quote(str(st.get("query", ""))[:100], safe="")
        return [f'openURL(url: "https://music.apple.com/search?term={q}")', "wait(seconds: 3)", "play(behavior: Play)"]
    if k == "battery":
        return ["batteryLevel() >> battery", 'speakText(text: "Battery is at ${battery} percent")', 'showResult(text: "Battery: ${battery}")']
    if k == "weather":
        return ["getCurrentConditions() >> weather", 'showResult(text: "${weather}")']
    if k == "dictate_note":                                  # Jelly has no 'add reminder' action: speech goes to a new note
        return ["dictateText() >> said", 'createNote(text: "${said}", show: false)', 'speakText(text: "Saved, sir.")']
    if k == "dnd":
        return [f"setDND(state: {_onoff(st)})"]
    if k == "flashlight":
        return [f"setFlashlight(state: {_onoff(st)})"]
    if k == "volume":
        return [f'setVolume(level: {_num(st.get("level"), 0, 1, 0.5)})']
    if k == "brightness":
        return [f'setBrightness(value: {_num(st.get("level"), 0, 1, 0.7)})']
    if k == "low_power":
        return [f"lowPowerMode(state: {_onoff(st)})"]
    if k == "vibrate":
        return ["vibrate()"]
    raise ValueError(f"unknown step kind {k!r}")


STEP_KINDS = ("text_to", "directions", "speak", "notify", "show", "alert", "wait", "timer", "music", "battery", "weather", "dictate_note", "dnd",
              "flashlight", "volume", "brightness", "low_power", "vibrate")


def clean_name(name: str) -> str:
    return " ".join(_NAME.sub("", str(name)).split())[:40] or "Nupen Shortcut"


def to_jelly(plan: dict[str, Any]) -> str:
    """Plan {name, steps:[{kind,...}], color?, icon?} -> Jelly source. Raises ValueError on an unknown or oversized plan."""
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps or len(steps) > MAX_STEPS:
        raise ValueError("a plan needs 1-%d steps" % MAX_STEPS)
    color = plan.get("color") if plan.get("color") in COLORS else "blue"
    icon = plan.get("icon") if plan.get("icon") in ICONS else "shortcuts"
    lines = [f"// {clean_name(plan.get('name', ''))} - written by Nupen", f"import Shortcuts #Color: {color}, #Icon: {icon}", ""]
    for st in steps:
        if not isinstance(st, dict):
            raise ValueError("a step must be an object")
        lines += render_step(st)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------- static validator
_CALL = re.compile(r"^\s*(?:(?:var\s+)?\w+\s*=\s*)?([A-Za-z_]\w*)\((.*)\)(?:\s*>>\s*(\w+))?\s*$")
_IMPORT = re.compile(r"^import Shortcuts(?: #Color: (\w+), #Icon: (\w+))?\s*$")


def _split_args(s: str) -> list[str]:
    out, cur, q, depth = [], "", False, 0
    for ch in s:
        if ch == '"':
            q = not q
        if not q and ch == "(":
            depth += 1
        if not q and ch == ")":
            depth -= 1
        if ch == "," and not q and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _kind_ok(kind: str, v: str) -> bool:
    opt = kind.endswith("?")
    kind = kind.rstrip("?")
    if kind == "str":
        return len(v) >= 2 and v[0] == v[-1] == '"' and '"' not in v[1:-1] and "\\" not in v
    if kind == "num":
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", v))
    if kind == "bool":
        return v in ("true", "false")
    if kind == "dur":
        return bool(re.fullmatch(r'"?\d+(?:\.\d+)? (?:sec|min|hr)"?', v))    # quoted: Open-Jellycore parses "10 min", not 10 min
    if kind.startswith("enum:"):
        return v in kind[5:].split("|")
    return opt


def validate(code: str, compiler: Optional[Callable[[str], dict[str, Any]]] = None) -> list[str]:
    """Static check: returns a list of problems (empty = valid). Syntax shape, known actions, parameter types, safety rules.
    Optional second check: the real Jelly compiler (creator.jellyc, Open-Jellycore in WSL) when `compiler` is given or env
    NUPEN_JELLY_CHECK=1; it runs only after the static rules pass, and a missing compiler adds no error."""
    errs = _static_validate(code)
    if errs:
        return errs
    if compiler is None and os.environ.get("NUPEN_JELLY_CHECK") == "1":
        from creator import jellyc
        compiler = jellyc.compile_code
    if compiler is not None:
        r = compiler(code)
        if r.get("available") and not r.get("ok"):
            errs += [f"jelly compiler: {e[:200]}" for e in (r.get("errors") or ["rejected the code"])]
    return errs


def _static_validate(code: str) -> list[str]:
    errs: list[str] = []
    seen_import, magic = False, set()
    for no, raw in enumerate(code.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        m = _IMPORT.match(line)
        if m:
            seen_import = True
            if m[1] and (m[1] not in COLORS):
                errs.append(f"line {no}: unknown #Color {m[1]}")
            if m[2] and m[2] not in ICONS:
                errs.append(f"line {no}: icon {m[2]} not in the allowed set")
            continue
        c = _CALL.match(line)
        if not c:
            errs.append(f"line {no}: not a function call: {line[:40]}")
            continue
        name, argstr, out = c[1], c[2], c[3]
        if name in FORBIDDEN:
            errs.append(f"line {no}: {name} is forbidden (data leaves the phone or data is changed)")
            continue
        spec = ACTIONS.get(name)
        if spec is None:
            errs.append(f"line {no}: unknown action {name}")
            continue
        if not seen_import:
            errs.append(f"line {no}: 'import Shortcuts' must come first")
        args = _split_args(argstr)
        keys = list(spec)
        got: dict[str, str] = {}
        for a in args:
            if ":" not in a:
                errs.append(f"line {no}: argument without a name: {a[:30]}")
                continue
            k, v = a.split(":", 1)
            got[k.strip()] = v.strip()
        for k, v in got.items():
            if k not in spec:
                errs.append(f"line {no}: {name} has no parameter {k}")
            elif not _kind_ok(spec[k], v):
                errs.append(f"line {no}: bad value for {name}.{k}: {v[:30]}")
        for k in keys:
            if not spec[k].endswith("?") and k not in got:
                errs.append(f"line {no}: {name} needs {k}")
        for v in got.values():                                           # interpolation may only name a magic variable made earlier
            for ref in re.findall(r"\$\{(\w+)\}", v):
                if ref not in magic:
                    errs.append(f"line {no}: ${{{ref}}} used before it is made")
        if name == "openURL":
            u = got.get("url", "").strip('"')
            if not u.startswith(URL_SCHEMES) or "${" in u:
                errs.append(f"line {no}: openURL only to sms:, tel:, maps or Apple Music/Maps pages, with no variable inside")
        if out:
            magic.add(out)
    if not seen_import:
        errs.append("missing 'import Shortcuts'")
    return errs


# --------------------------------------------------------------------------------------------- request -> plan (rules, then the model)
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "sixty": 60}
_LEAD = re.compile(r"^\s*(?:(?:please|nupen|hey)[ ,]*)*(?:(?:can|could|would) you )?(?:make|create|build|write|generate|set up|give)\s+(?:me\s+)?(?:a|an|another|the)?\s*(?:new\s+)?"
                   r"(?:iphone\s+)?shortcut\b\s*(?:called|named)?\s*(?:\"([^\"]{1,40})\"|'([^']{1,40})')?\s*(?:that|which|to|for|so that|where)?\s*", re.I)
SHORTCUT_ASK = re.compile(r"\b(?:make|create|build|write|generate|set up)\b.{0,30}\bshortcut\b", re.I)


def _n(s: str) -> float:
    s = s.lower()
    return float(_NUM[s]) if s in _NUM else float(s)


def _clause(c: str) -> list[dict[str, Any]]:
    t = c.strip(" .,!?")
    if not t:
        return []
    low = t.lower()
    if "morning routine" in low or "good morning" in low:
        return [{"kind": "speak", "text": "Good morning, sir."}, {"kind": "weather"}, {"kind": "brightness", "level": 0.7}, {"kind": "dnd", "state": "off"},
                {"kind": "battery"}, {"kind": "music", "query": "morning playlist"}]
    if m := re.search(r"\b(?:texts?|messages?|sms)\s+(?:(?:my\s+)?[A-Za-z]\w*\s+)?(?:at\s+|on\s+)?(\+?[\d][\d\s().-]{6,})\s*(?:saying|that says|with|:)?\s*(.*)$", t, re.I):
        return [{"kind": "text_to", "to": m[1], "text": m[2] or "On my way"}]
    if m := re.search(r"\b(?:opens?\s+maps|directions|navigate|maps?)\b.*?\b(?:to|for)\s+(.+)$", t, re.I):
        return [{"kind": "directions", "to": m[1]}]
    if m := re.search(r"\btimer\b.*?(\w+)\s*(second|sec|minute|min|hour|hr)s?\b|\b(\w+)\s*[- ]?(second|sec|minute|min|hour|hr)s?\s+timer\b", t, re.I):
        val, unit = (m[1], m[2]) if m[1] else (m[3], m[4])
        try:
            v = _n(val)
        except (KeyError, ValueError):
            return []
        u = unit.lower()
        return [{"kind": "timer", "seconds": v} if u.startswith("s") else {"kind": "timer", "minutes": v * (60 if u.startswith("h") else 1)}]
    if m := re.search(r"\bplay(?:s)?\s+(?:some\s+)?(.*)$", t, re.I):
        q = re.sub(r"^(?:music\s+(?:by|from)\s+|the\s+song\s+)", "", m[1].strip(), flags=re.I)
        return [{"kind": "music", "query": "music" if re.fullmatch(r"(?:music|songs?|)", q, re.I) else q}]
    if re.search(r"\b(?:remind|reminder|remember|note)\b.*\b(?:speak|speech|say|saying|dictat|voice|talk)|\b(?:speak|say|dictat|voice)\w*\b.*\b(?:remind|reminder|note)", t, re.I):
        return [{"kind": "dictate_note"}]
    if re.search(r"\bbattery\b", t, re.I):
        return [{"kind": "battery"}]
    if re.search(r"\bweather\b", t, re.I):
        return [{"kind": "weather"}]
    if m := re.search(r"\b(?:turns?|switch(?:es)?)\s+(on|off)\s+(?:the\s+)?(flashlight|do not disturb|dnd|low power(?: mode)?)\b", t, re.I):
        k = {"flashlight": "flashlight", "low power": "low_power"}.get(re.sub(r" mode$", "", m[2].lower()), "dnd")
        return [{"kind": k, "state": m[1].lower()}]
    if m := re.search(r"\b(?:set\s+)?(volume|brightness)\s+(?:to\s+)?(\d{1,3})\s*%?", t, re.I):
        return [{"kind": m[1].lower(), "level": min(int(m[2]), 100) / 100}]
    if m := re.search(r"\b(?:say|speak|read out)\s+(.+)$", t, re.I):
        return [{"kind": "speak", "text": m[1]}]
    if m := re.search(r"\b(?:show|display)\s+(?:me\s+)?(?:a message\s+)?(.+)$", t, re.I):
        return [{"kind": "show", "text": m[1]}]
    return []


def rules_plan(request: str) -> Optional[dict[str, Any]]:
    lead = _LEAD.match(request)
    if not lead:
        return None
    rest = request[lead.end():]
    name = lead[1] or lead[2]
    steps: list[dict[str, Any]] = []
    for part in re.split(r"\s*(?:,|;|\band then\b|\bthen\b|\bafter that\b|\band also\b|\band\b)\s*", rest, flags=re.I):
        steps += _clause(part)
    if not steps:
        return None
    if not name:
        kinds = list(dict.fromkeys(s["kind"] for s in steps))
        name = "Morning Routine" if len(steps) > 3 and steps[0]["kind"] == "speak" else " and ".join({"text_to": "Text", "directions": "Maps", "dictate_note": "Voice Note"}.get(k, k.title()) for k in kinds[:2])
    return {"name": clean_name(name), "steps": steps[:MAX_STEPS]}


MODEL_PROMPT = ('Turn the request into a shortcut plan. Reply with ONLY JSON: {"name": "...", "steps": [{"kind": ...}]}. Allowed kinds and fields: '
                'text_to(to,text), directions(to), speak(text), notify(body,title), show(text), alert(text,title), wait(seconds), timer(minutes|seconds), '
                'music(query), battery, weather, dictate_note, dnd(state on|off), flashlight(state), volume(level 0-1), brightness(level 0-1), '
                'low_power(state), vibrate. Nothing else. Request: ')


def model_plan(request: str, llm: Optional[Callable[[str], str]]) -> Optional[dict[str, Any]]:
    if llm is None:
        return None
    try:
        raw = llm(MODEL_PROMPT + request[:400])
        m = re.search(r"\{.*\}", raw, re.S)
        plan = json.loads(m[0]) if m else None
    except Exception:
        return None
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        return None
    steps = [s for s in plan["steps"] if isinstance(s, dict) and s.get("kind") in STEP_KINDS][:MAX_STEPS]
    return {"name": clean_name(plan.get("name", "Nupen Shortcut")), "steps": steps} if steps else None


def make(request: str, llm: Optional[Callable[[str], str]] = None) -> dict[str, Any]:
    """Request -> {ok, name, plan, code, errors}. Rules first; the local model only when the rules find nothing; the validator has the last word."""
    plan = rules_plan(request) or model_plan(request, llm)
    if plan is None:
        return {"ok": False, "errors": ["I could not turn that into steps I know are safe"], "plan": None, "code": "", "name": ""}
    try:
        code = to_jelly(plan)
    except ValueError as e:
        return {"ok": False, "errors": [str(e)], "plan": plan, "code": "", "name": plan.get("name", "")}
    errs = validate(code)
    return {"ok": not errs, "name": plan["name"], "plan": plan, "code": code if not errs else "", "errors": errs}


# --------------------------------------------------------------------------------------------- storage + delivery
DELIVER = "compile_jelly_text"        # Nupen shortcut: fetch code -> 'Compile Jelly Text' (Jellycuts) -> Add Shortcut. Fallback: 'clipboard'
DELIVERIES = ("compile_jelly_text", "clipboard")


def _dir(rt: Optional[Path]) -> Path:
    if rt is None:
        from creator import device as DEV
        rt = Path(DEV.runtime_dir()) / "phone"
    return Path(rt) / "shortcuts"


def new_id(name: str, code: str, now: Optional[float] = None) -> str:
    return hashlib.sha256(f"{name}\n{code}\n{time.time() if now is None else now}".encode()).hexdigest()[:12]


def store(rt: Optional[Path], name: str, code: str, now: Optional[float] = None, trusted: bool = False) -> str:
    """trusted=True only from the owner's PC (CLI --queue-nupen2): skips the generated-code validator, e.g. for Nupen 2 itself,
    which needs web requests the validator forbids for model-written shortcuts. Nothing reachable from the phone sets it."""
    sid = new_id(name, code, now)
    d = _dir(rt)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"id": sid, "name": name, "code": code, "at": time.time() if now is None else now, "deliver": DELIVER}
    if trusted:
        rec["trusted"] = True
    (d / f"{sid}.json").write_text(json.dumps(rec), encoding="utf-8")
    return sid


def fetch(rt: Optional[Path], sid: str) -> Optional[dict[str, Any]]:
    """What GET /shortcut?id=<id> returns: {id, name, code, deliver}. Re-validated on every read; a record that fails is never served."""
    if not re.fullmatch(r"[0-9a-f]{12}", sid or ""):
        return None
    try:
        rec = json.loads((_dir(rt) / f"{sid}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or (rec.get("trusted") is not True and _static_validate(str(rec.get("code", "")))):
        return None
    return {k: rec[k] for k in ("id", "name", "code", "deliver") if k in rec}


def build_action(name: str, code: str, deliver: str = DELIVER) -> dict[str, Any]:
    return {"type": "build_shortcut", "name": name, "code": code, "deliver": deliver if deliver in DELIVERIES else DELIVER}


def confirm_url(sid: str) -> str:
    """The 'Confirm' button: runs the one Nupen shortcut with input 'build_shortcut <id>' (a shortcuts:// URL, no data leaves the phone)."""
    from creator import notify as N
    return ("shortcuts://run-shortcut?name=" + urllib.parse.quote(N.shortcut_name()) + "&input=text&text=" + urllib.parse.quote("build_shortcut " + sid))


def notify_ready(name: str, sid: str, rt: Optional[Path] = None) -> bool:
    """Push: "Sir, your new shortcut '<name>' is ready." with [Confirm] (view: starts the build on the phone) and [Not now] (http: records 'later')."""
    try:
        from creator import decisions as D
        from creator import notify as N
        did = "shortcut-" + sid
        if N.topic() is None or not D.ask(did, f"shortcut {name}"):
            return False
        acts = [N.view_action("Confirm", confirm_url(sid))]
        acts += [a.replace(", Later,", ", Not now,", 1) for a in N.decision_actions(did) if a.startswith("http, Later,")]
        return N.notify("Nupen shortcut", f"Sir, your new shortcut '{clean_name(name)}' is ready.", "high", ("sparkles",), acts)
    except Exception:
        return False


PENDING_S = 900.0     # a shortcut waits this long for the owner's Confirm before it is no longer offered


def next_pending(rt: Optional[Path], now: Optional[float] = None, max_age: float = PENDING_S) -> Optional[dict[str, Any]]:
    """GET /shortcut/next: the newest stored shortcut younger than max_age that was never handed out, marked as handed out.
    The Nupen shortcut asks this every time it runs, so the Confirm button only has to START it (iOS drops run-shortcut text input
    on some setups); Apple's Add sheet is still the owner's final yes. None when nothing is waiting."""
    now = time.time() if now is None else now
    best: Optional[tuple[float, Path, dict[str, Any]]] = None
    for p in _dir(rt).glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        at = float(rec.get("at", 0)) if isinstance(rec, dict) else 0.0
        if not isinstance(rec, dict) or rec.get("delivered_at") or now - at > max_age:
            continue
        if best is None or at > best[0]:
            best = (at, p, rec)
    if best is None:
        return None
    _, p, rec = best
    out = fetch(rt, str(rec.get("id", "")))
    if out is None:
        return None
    rec["delivered_at"] = now
    p.write_text(json.dumps(rec), encoding="utf-8")
    return out


REPORT_FIELDS = ("id", "result_type", "has_value", "error")


def record_report(rt: Optional[Path], d: dict[str, Any], now: Optional[float] = None) -> tuple[bool, str]:
    """POST /shortcut_report: the Nupen shortcut says what 'Compile Jelly Text' returned on the phone (type, empty or not, error text),
    so Nupen learns whether Jellycuts shows Apple's Add sheet itself or needs the extra open step. Only appends a log line."""
    sid = str(d.get("id", ""))
    if not re.fullmatch(r"[0-9a-f]{12}", sid):
        return False, "bad id"
    rec = {k: esc(str(d[k]), 120) for k in REPORT_FIELDS if k in d}
    rec["at"] = time.time() if now is None else now
    p = _dir(rt) / "reports.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return True, "recorded"


def reports(rt: Optional[Path]) -> list[dict[str, Any]]:
    try:
        lines = (_dir(rt) / "reports.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def handle(text: str, rt: Optional[Path], llm: Optional[Callable[[str], str]] = None) -> Optional[dict[str, Any]]:
    """Phone-server entry: None when the words are not a shortcut request; otherwise {reply, action}. Stores, validates, notifies."""
    if not SHORTCUT_ASK.search(text):
        return None
    r = make(text, llm)
    if not r["ok"]:
        return {"reply": "I cannot build that shortcut safely, sir. " + r["errors"][0].rstrip(".") + ".", "action": None}
    sid = store(rt, r["name"], r["code"])
    pushed = notify_ready(r["name"], sid, rt)
    reply = f"Your shortcut {r['name']} is ready, sir." + (" Tap Confirm on the notification." if pushed else " Your Nupen shortcut will install it now.")
    return {"reply": reply, "action": build_action(r["name"], r["code"]), "id": sid, "pushed": pushed}
