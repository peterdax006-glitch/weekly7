"""Nupen terminal conversation, LAYER 1: rule-based understanding, answers only from Nupen's own records (owner 2 Oct 2026).

North star: talk to Nupen in a terminal like talking to a development model. This layer has NO language model. It is built so layers 2/3 can
replace the two halves independently:

    Understand.parse(text) -> Parsed(intent, slots)        RuleUnderstand: keyword/regex intent table + slot extraction
    Speak.render(facts)    -> str                          RuleSpeak: plain English from Facts; LMSpeak (flag NUPEN_LM_SPEAK=1, off by default,
                                                           never required) tries creator.lm.api.load_current() and falls back to the rules

Every handler reads real state under `<root>/state/...` and returns Facts(lines, evidence). Nothing is invented: a number or id in an answer
came from a record, and the evidence ids are listed. Every exchange is appended to state/creator/conversations.jsonl (source 'nupen-terminal');
`export_dialogues` writes the same exchanges in the creator.lm.dialogue item format (NOT teacher-reviewed).
Owner-affecting actions (pause, resume, approve) always need a confirmation turn; a request only drafts a PENDING owner goal proposal."""
from __future__ import annotations

import ctypes
import dataclasses
import datetime as dt
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from creator import constraints as CON
from creator import curriculum as CUR
from creator import goals as GO
from creator import schedule as SCH

CONVERSATIONS_FILE = "conversations.jsonl"
SOURCE = "nupen-terminal"
STOP_FILE = "NUPEN_STOP"
LM_FLAG = "NUPEN_LM_SPEAK"
YES = ("yes", "y", "confirm", "ok", "do it", "go ahead", "approve", "sure")
NO = ("no", "n", "cancel", "stop", "never mind", "nevermind", "abort")


# ------------------------------------------------------------------------------------------------ interfaces

@dataclasses.dataclass
class Parsed:
    intent: str
    slots: dict[str, Any]


@dataclasses.dataclass
class Facts:
    """What a handler knows: plain sentences (one fact each) and the record ids / files / numbers they came from."""
    intent: str
    lines: list[str]
    evidence: list[str] = dataclasses.field(default_factory=list)
    confirm: Optional[dict[str, Any]] = None        # an action waiting for the owner's yes/no
    user: str = ""                                  # the owner's words this answers (set by Conversation; a model speaker needs them)


class Understand(Protocol):
    def parse(self, text: str) -> Parsed: ...


class Speak(Protocol):
    def render(self, facts: Facts) -> str: ...


class Ctx:
    """Where the records are. root = a repo checkout (or a copy of one); state = root/state/creator."""

    def __init__(self, root: Path, now: Optional[dt.datetime] = None) -> None:
        self.root = Path(root)
        self.state = self.root / "state" / "creator"
        self.now = now or dt.datetime.now()

    @property
    def stop_file(self) -> Path:
        return self.state / STOP_FILE


# ------------------------------------------------------------------------------------------------ small readers

def _jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if isinstance(d, dict):
                out.append(d)
    except OSError:
        pass
    return out


def _json(path: Path) -> dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _age_s(path: Path, now: dt.datetime) -> Optional[float]:
    try:
        return now.timestamp() - path.stat().st_mtime
    except OSError:
        return None


def _ago(s: Optional[float]) -> str:
    if s is None:
        return "never"
    if s < 90:
        return f"{s:.0f}s ago"
    if s < 5400:
        return f"{s / 60:.0f} min ago"
    return f"{s / 3600:.1f} h ago"


def ram_free_gb() -> Optional[float]:
    """Free physical RAM in GB (Windows via ctypes, else /proc/meminfo); None when unknown."""
    try:
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        if sys.platform == "win32" and ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):  # type: ignore[attr-defined]
            return round(ms.ullAvailPhys / 2**30, 2)
    except (OSError, AttributeError):
        pass
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                return round(int(ln.split()[1]) / 2**20, 2)
    except (OSError, ValueError):
        pass
    return None


def _lessons(c: Ctx) -> list[CUR.Lesson]:
    return CUR.LessonLog(c.state / "lessons.jsonl").lessons()


# ------------------------------------------------------------------------------------------------ handlers (each reads real state)

def h_status(c: Ctx, s: dict[str, Any]) -> Facts:
    ev: list[str] = []
    lines: list[str] = []
    stopped = c.stop_file.exists()
    hb = c.state / "nupen_service.heartbeat"
    age = _age_s(hb, c.now)
    up = age is not None and age < 180 and not stopped
    lines.append(f"The service is {'UP' if up else 'DOWN'}" + (" (switched off: NUPEN_STOP exists)" if stopped else "")
                 + f"; its heartbeat was {_ago(age)}.")
    ev.append("state/creator/nupen_service.heartbeat")
    sw = _jsonl(c.state / "swarm_log.jsonl")
    sw_age = _age_s(c.state / "swarm_log.jsonl", c.now)
    running = up and sw_age is not None and sw_age < 1800
    if sw:
        last = sw[-1]
        lines.append(f"The swarm is {'running' if running else 'not running'}; the last round was number {last.get('round')} "
                     f"({last.get('outcome')}, {last.get('packages')} packages, {_ago(sw_age)}).")
        ev.append(f"swarm_log.jsonl round {last.get('round')}")
    else:
        lines.append("The swarm has no log yet, so I cannot say it has ever run.")
    cyc = CON.read_cycles(c.state)
    today = [x for x in cyc if x.at.date() == c.now.date()]
    ad = [x for x in today if x.outcome == "ADOPTED"]
    lines.append(f"Today I adopted {len(ad)} of {len(today)} cycles" + (f" ({', '.join(x.package for x in ad[:6])})." if ad else "."))
    ev += [f"cycles/{x.package}" for x in ad[:6]]
    sc = CUR.student_scores(_lessons(c))
    lines.append(f"Teacher share of adopted changes: {sc['teacher_share']}" + (f" ({sc['adopted_total']} adopted lessons)." if sc["adopted_total"] else
                                                                                " (no adopted lessons recorded yet)."))
    top = CON.top_constraints(c.state, 1)
    if top:
        t = top[0]
        lines.append(f"Top constraint: {t['name']} (loss {t.get('loss')}, {t.get('value')} {t.get('unit', '')}).")
        ev.append("constraints.jsonl:" + str(t["name"]))
    else:
        lines.append("No constraint snapshot exists yet.")
    nh = len(list((c.state / "pending").glob("*.patch"))) if (c.state / "pending").is_dir() else 0
    deferred = len(_jsonl(c.state / "deferred.jsonl"))
    lines.append(f"Pending handoffs: {nh} patch files waiting in state/creator/pending, {deferred} packages deferred to the teacher.")
    ev.append("state/creator/pending")
    free = ram_free_gb()
    lines.append(f"Free RAM: {free} GB." if free is not None else "Free RAM: unknown on this machine.")
    return Facts("status", lines, ev)


def h_constraints(c: Ctx, s: dict[str, Any]) -> Facts:
    top = CON.top_constraints(c.state, 5)
    if not top:
        return Facts("constraints", ["I have not measured my constraints yet (no snapshot in state/creator/constraints.jsonl)."])
    lines = ["What limits me, worst first:"]
    ev = []
    for i, t in enumerate(top, 1):
        lines.append(f"{i}. {t['name']}: loss {t.get('loss')}, score {t.get('score')}, now {t.get('value')} {t.get('unit', '')}"
                     f" - remedy: {t.get('remedy', 'work')}.")
        ev.append("constraints.jsonl:" + str(t["name"]))
    return Facts("constraints", lines, ev)


def h_plan(c: Ctx, s: dict[str, Any]) -> Facts:
    p = SCH.last_plan(c.state / "ledger.jsonl")
    if not p["at"]:
        return Facts("plan", ["I have no recorded plan yet (plan_explanations.jsonl is empty)."])
    chosen = [r for r in p["reasons"] if r["chosen"]]
    lines = [f"My last plan was made at {p['at']}. Chosen now: {', '.join(p['chosen']) or 'nothing'}.",
             f"Critical path: {p['critical_path_s']:.0f} s through {len(p['critical_path'])} steps."]
    for r in chosen[:5]:
        lines.append(f"- {r['component']} {r['step']} ({r['node']}): {r['why']}")
    if not chosen:
        lines.append("Nothing is chosen right now, so I am not working on a package; these steps were considered and not chosen:")
        for r in p["reasons"][:3]:
            lines.append(f"- not chosen: {r['component']} {r['step']}: {r['why']}")
    return Facts("plan", lines, [f"plan_explanations.jsonl@{p['at']}"] + p["chosen"][:5])


def _find_cycles(c: Ctx, s: dict[str, Any]) -> list[CON.Cycle]:
    cyc = CON.read_cycles(c.state)
    if s.get("package"):
        return [x for x in cyc if x.package == s["package"]]
    if s.get("component"):
        return [x for x in cyc if x.requirement.startswith(s["component"] + ".")]
    return []


def h_explain(c: Ctx, s: dict[str, Any]) -> Facts:
    sel = _find_cycles(c, s)
    if not sel:
        what = s.get("package") or s.get("component") or "that"
        return Facts("explain", [f"I have no cycle report for {what}. Try a package id like CP0001, or ask 'what are you working on'."])
    lines: list[str] = []
    ev: list[str] = []
    for x in sel[-3:]:
        cj = _json(c.state / "cycles" / x.package / "cycle.json")
        det = (cj.get("details") or {}).get("detail") or {}
        lines.append(f"{x.package} ({x.requirement}) was {x.outcome} after {x.seconds:.0f} s: {x.reason or 'no reason recorded'}.")
        ev.append(f"cycles/{x.package}/cycle.json")
        if cj.get("verdict"):
            lines.append(f"  verdict {cj['verdict']}" + (f", claim {cj['details'].get('claim')}" if (cj.get("details") or {}).get("claim") else ""))
            if (cj.get("details") or {}).get("claim"):
                ev.append(str(cj["details"]["claim"]))
        nums = {k: det[k] for k in ("diff", "se", "lo", "hi", "z", "min_effect") if k in det}
        if nums:
            lines.append("  claim numbers: " + ", ".join(f"{k} {v}" for k, v in nums.items()) + ".")
        if det.get("why"):
            lines.append(f"  diagnosis: {det['why']}; requirements lost: {', '.join(det.get('requirements_lost', [])) or 'none'}.")
        for les in _lessons(c):
            if les.package_id == x.package:
                lines.append(f"  lesson {les.lesson_id}: {les.task_kind} by {les.solver}, adopted={les.adopted}: {les.verdict[:160] or 'no verdict'}")
                ev.append(les.lesson_id)
    return Facts("explain", lines, ev)


def h_lessons(c: Ctx, s: dict[str, Any]) -> Facts:
    les = _lessons(c)
    sc = CUR.student_scores(les)
    lines = [f"I hold {len(les)} lessons; {sc['adopted_total']} adopted with a known skill signal; teacher share {sc['teacher_share']}."]
    ev: list[str] = []
    for cell in list(sc["cells"].values())[:6]:
        lines.append(f"- {cell['solver']} on {cell['task_kind']}: {cell['adopted']} of {cell['attempts']} adopted (rate {cell['rate']}).")
    for x in [l for l in les if l.adopted is not None][-3:]:
        lines.append(f"- recent {x.lesson_id}: {x.task_kind} {x.package_id} {'adopted' if x.adopted else 'not adopted'}: {x.verdict[:100]}")
        ev.append(x.lesson_id)
    try:
        from creator import shadow as SH
        pol = SH.read_policy(c.state)
        st = SH.stats(c.state, les)
        lines.append(f"Chooser shadow: policy {'chooser' if pol else 'default'}, {st['n']} real decisions (rule needs 30), "
                     f"chooser accuracy {st['chooser_acc']} vs default {st['default_acc']}.")
        ev.append("state/creator/chooser.json")
    except Exception as e:  # noqa: BLE001 - an optional section never kills the answer
        lines.append(f"Chooser shadow status unavailable ({type(e).__name__}).")
    return Facts("lessons", lines, ev)


def h_goals(c: Ctx, s: dict[str, Any]) -> Facts:
    pend = GO.pending(c.state)
    if not pend:
        return Facts("goals", ["No goal proposals are pending. Tell me 'I want you to <something>' and I will draft one."])
    lines = [f"{len(pend)} goal proposals wait for approval:"]
    for p in pend[:6]:
        lines.append(f"- {p['id']} ({p['source']}, value {p['value']}): {p['title']} - because {p['rationale']}")
    lines.append("Say 'approve GP-...' to approve one.")
    return Facts("goals", lines, [p["id"] for p in pend[:6]])


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "request"


def h_request(c: Ctx, s: dict[str, Any]) -> Facts:
    text = str(s["request"]).strip().rstrip(".!")
    key = f"owner:{_slug(text)}"
    pid = GO._pid(key)
    if any(p["id"] == pid for p in GO.listing(c.state)):
        return Facts("request", [f"I already have that request as {pid}."], [pid])
    slug = f"owner_{_slug(text)}"
    rec = {"event": "proposal", "id": pid, "key": key, "source": "owner", "title": text[0].upper() + text[1:],
           "rationale": f"the owner asked for this in the terminal: \"{text}\"", "evidence": [f"{CONVERSATIONS_FILE}:{c.now.isoformat(timespec='seconds')}"],
           "spec": {"name": text[0].upper() + text[1:], "modules": [f"creator/goal_{slug}.py"], "tests": [f"tests/test_creator_goal_{slug}.py"],
                    "floor": 0, "ladder": ["exists", "tested", "no_stubs", "integrated", "validated"]},
           "value": 1.0, "cost": {"packages": 5, "modules": 1, "tests": 1, "lines_estimate": 250}, "created": c.now.isoformat(timespec="seconds")}
    GO._append(c.state, rec)
    return Facts("request", [f"I drafted your request as goal proposal {pid}: \"{rec['title']}\" (source owner, status PENDING, 5 packages, about 250 lines).",
                             f"Nothing is approved yet. Say 'approve {pid}' to approve it, or leave it pending."], [pid])


def h_approve(c: Ctx, s: dict[str, Any]) -> Facts:
    pid = "GP-" + re.sub(r"^gp[-_]", "", str(s["goal"]), flags=re.I).lower()
    p = next((x for x in GO.listing(c.state) if x["id"] == pid), None)
    if p is None:
        return Facts("approve", [f"I have no proposal {pid}."])
    if p["status"] != "PENDING":
        return Facts("approve", [f"{pid} is already {p['status']}."], [pid])
    return Facts("approve", [f"Approve {pid} \"{p['title']}\" as owner? That creates a new capability and its requirement ladder. Yes or no?"],
                 [pid], confirm={"action": "approve", "goal": pid})


def h_pause(c: Ctx, s: dict[str, Any]) -> Facts:
    if c.stop_file.exists():
        return Facts("pause", ["I am already paused (NUPEN_STOP exists)."], [str(c.stop_file.name)])
    return Facts("pause", ["Pause me? That creates state/creator/NUPEN_STOP and the service stops the swarm. Yes or no?"], confirm={"action": "pause"})


def h_resume(c: Ctx, s: dict[str, Any]) -> Facts:
    if not c.stop_file.exists():
        return Facts("resume", ["I am not paused (no NUPEN_STOP file)."])
    return Facts("resume", ["Resume me? That removes state/creator/NUPEN_STOP so the service may start the swarm again. Yes or no?"],
                 confirm={"action": "resume"})


def h_checklist(c: Ctx, s: dict[str, Any]) -> Facts:
    d = _json(c.root / "state" / "build" / "CREATOR_MASTER_CHECKLIST.json")
    items = d.get("items") or []
    if not items:
        return Facts("checklist", ["I cannot read state/build/CREATOR_MASTER_CHECKLIST.json."])
    counts: dict[str, int] = {}
    for it in items:
        counts[str(it.get("status"))] = counts.get(str(it.get("status")), 0) + 1
    body = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    return Facts("checklist", [f"The master checklist has {len(items)} items: {body}."], ["state/build/CREATOR_MASTER_CHECKLIST.json"])


def h_loc(c: Ctx, s: dict[str, Any]) -> Facts:
    out: dict[str, int] = {}
    for sub in ("creator", "scripts", "tests", "engine"):
        n = 0
        for f in (c.root / sub).rglob("*.py") if (c.root / sub).is_dir() else []:
            try:
                n += len(f.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                continue
        out[sub] = n
    cr = out["creator"]
    return Facts("loc", [f"The creator package is {cr} lines of Python; scripts {out['scripts']}, tests {out['tests']}, engine {out['engine']}; "
                         f"{sum(out.values())} in total."], [f"{k}/**/*.py={v}" for k, v in out.items()])


def h_help(c: Ctx, s: dict[str, Any]) -> Facts:
    return Facts("help", ["Here is what I can do:"] + [f"- {i.example}  ({i.about})" for i in INTENTS if i.example])


def h_unknown(c: Ctx, s: dict[str, Any]) -> Facts:
    return Facts("unknown", ["I don't understand that yet. I only answer from my own records. Here is what I can do:"] +
                 [f"- {i.example}  ({i.about})" for i in INTENTS if i.example])


# ------------------------------------------------------------------------------------------------ the intent table (extend here)

@dataclasses.dataclass
class Intent:
    name: str
    patterns: tuple[str, ...]                 # regexes (case-insensitive); each hit adds its weight (the pattern's index order = priority)
    handler: Callable[[Ctx, dict[str, Any]], Facts]
    example: str = ""
    about: str = ""
    weight: float = 1.0


INTENTS: list[Intent] = [
    Intent("request", (r"\bi want you to (?P<request>.+)", r"^(?:please |could you |can you |you should |you need to )(?P<request>(?:make|add|fix|build|speed|improve|reduce|stop|teach|write|create)\b.+)"),
           h_request, "I want you to make the planner faster", "drafts a pending goal proposal from your request", 5.0),
    Intent("approve", (r"\bapprove\s+(?P<goal>gp[-_][0-9a-f]{4,})",), h_approve, "approve GP-xxxxxxxxxx", "approve a pending goal (asks to confirm)", 6.0),
    Intent("pause", (r"^\s*(?:please )?(?:pause|stop|halt|shut ?down|switch off)\b", r"\bpause (?:yourself|nupen|the swarm)"), h_pause, "pause",
           "creates NUPEN_STOP (asks to confirm)", 5.0),
    Intent("resume", (r"^\s*(?:please )?(?:resume|continue|restart|start again|switch on)\b", r"\bresume (?:yourself|nupen|the swarm)"), h_resume, "resume",
           "removes NUPEN_STOP (asks to confirm)", 5.0),
    Intent("explain", (r"\bwhy did you\b", r"\bwhy (?:was|is|were)\b.*\b(?:reject|adopt|fail|roll)", r"\bexplain\b", r"\bcp\d{3,5}\b", r"\bwhat happened (?:to|with)\b"),
           h_explain, "why did you reject CP0001", "the cycle report: verdict, claim numbers, diagnosis, lesson", 2.0),
    Intent("constraints", (r"\blimit(?:ing|s)?\b", r"\bbottleneck", r"\bconstraint", r"\bholding you back", r"\bslow(?:ing)? you"),
           h_constraints, "what is limiting you", "ranked constraints with loss and remedy", 2.0),
    Intent("plan", (r"\bworking on\b", r"\bplan\b", r"\bcritical path", r"\bcurrent packages?", r"\bwhat are you doing"),
           h_plan, "what are you working on", "current plan, critical path, why each was chosen", 2.0),
    Intent("lessons", (r"\blearn(?:ed|t|ing)?\b", r"\blessons?\b", r"\bstudent", r"\bchooser", r"\bteacher"),
           h_lessons, "what have you learned", "lessons, student scores, chooser shadow status", 2.0),
    Intent("goals", (r"\bbuild next\b", r"\bgoals?\b", r"\bproposals?\b", r"\bwant to build", r"\bnext\b.*\bbuild"),
           h_goals, "what do you want to build next", "pending goal proposals with rationale", 2.0),
    Intent("loc", (r"\blines of code\b", r"\bhow (?:big|many lines)", r"\bloc\b", r"\bcode size"), h_loc, "how many lines of code are you", "line counts", 3.0),
    Intent("checklist", (r"\bchecklist\b", r"\bprogress\b", r"\bhow far\b", r"\bpercent done"), h_checklist, "checklist progress", "master checklist status counts", 3.0),
    Intent("status", (r"\bhow are you\b", r"\bstatus\b", r"\bhow('?s| is) it going", r"\bare you (?:up|running|alive|ok)", r"\bhow you doing"),
           h_status, "how are you doing", "service, swarm, round, adopted today, teacher share, top constraint, handoffs, RAM", 2.0),
    Intent("help", (r"^\s*(?:help|\?|what can you do|commands)\b",), h_help, "help", "this list", 4.0),
]
INTENT_BY_NAME = {i.name: i for i in INTENTS}


class RuleUnderstand:
    """Layer 1: the best-scoring intent from the table; slots are regex groups plus ids found anywhere in the text."""

    def __init__(self, table: Optional[list[Intent]] = None) -> None:
        self.table = INTENTS if table is None else table

    @staticmethod
    def slots_from(text: str) -> dict[str, Any]:
        s: dict[str, Any] = {}
        m = re.search(r"\bCP\d{3,5}\b", text, re.I)
        if m:
            s["package"] = m.group(0).upper()
        m = re.search(r"\bK\d{2}\b", text, re.I)
        if m:
            s["component"] = m.group(0).upper()
        m = re.search(r"\bGP[-_][0-9a-f]{4,}\b", text, re.I)
        if m:
            s["goal"] = "GP-" + m.group(0)[3:].lower()
        mods = re.findall(r"\b(?:creator|engine|scripts|tests)/[\w/]+\.py\b", text)
        if mods:
            s["modules"] = mods
        m = re.search(r"\b(?:last|past)\s+(\d+(?:\.\d+)?)\s*(h|hours?|d|days?|min|minutes?)\b", text, re.I)
        if m:
            n = float(m.group(1))
            s["window_h"] = n * {"h": 1.0, "d": 24.0, "m": 1 / 60}[m.group(2)[0].lower()]
        elif re.search(r"\btoday\b", text, re.I):
            s["window_h"] = 24.0
        nums = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text)
        if nums:
            s["numbers"] = [float(x) for x in nums]
        return s

    def parse(self, text: str) -> Parsed:
        slots = self.slots_from(text)
        best, score = "unknown", 0.0
        for it in self.table:
            sc = 0.0
            for pat in it.patterns:
                m = re.search(pat, text, re.I)
                if m:
                    sc += it.weight
                    slots.update({k: v for k, v in m.groupdict().items() if v})
            if sc > score:
                best, score = it.name, sc
        if best == "explain" and "package" not in slots and "component" not in slots:
            best = "unknown" if score < 3.0 else best
        if best == "approve" and "goal" not in slots:
            best = "unknown"
        return Parsed(best, slots)


class RuleSpeak:
    """Layer 1: a colleague's report. Facts are already plain sentences; evidence is cited at the end."""

    def render(self, facts: Facts) -> str:
        out = "\n".join(facts.lines)
        if facts.evidence:
            out += "\n[evidence: " + "; ".join(dict.fromkeys(facts.evidence)) + "]"
        return out


class LMSpeak:
    """Layer 3 hook: only when NUPEN_LM_SPEAK=1 AND a promoted model loads; otherwise (always the default) the rule speaker answers.
    The model sees the facts and may only rephrase; the evidence line is appended by us so it cannot be dropped or invented."""

    def __init__(self, fallback: Optional[Speak] = None) -> None:
        self.fallback: Speak = fallback or RuleSpeak()
        self._lm: Any = None

    def enabled(self) -> bool:
        return os.environ.get(LM_FLAG, "") in ("1", "true", "yes")

    def render(self, facts: Facts) -> str:
        if not self.enabled():
            return self.fallback.render(facts)
        try:
            if self._lm is None:
                from creator.lm import api
                self._lm = api.load_current() or False
            if not self._lm:
                return self.fallback.render(facts)
            from creator.lm import dialogue as D
            prompt = D.prompt_for([{"role": "user", "text": "Report these facts plainly: " + " ".join(facts.lines)}])
            reply, ok = D.extract_reply(self._lm.generate(prompt, max_new_tokens=160, temperature=0.3), prompt)
            if not ok or not reply.strip():
                return self.fallback.render(facts)
            return reply.strip() + ("\n[evidence: " + "; ".join(dict.fromkeys(facts.evidence)) + "]" if facts.evidence else "")
        except Exception:  # noqa: BLE001 - the model is never required
            return self.fallback.render(facts)


# ------------------------------------------------------------------------------------------------ the conversation

class Conversation:
    def __init__(self, root: Path, understand: Optional[Understand] = None, speak: Optional[Speak] = None,
                 now: Optional[Callable[[], dt.datetime]] = None, auto_confirm: bool = False, log: bool = True,
                 handlers: Optional[dict[str, Callable[[Ctx, dict[str, Any]], Facts]]] = None,
                 meta: Optional[Callable[[], dict[str, Any]]] = None) -> None:
        self.root = Path(root)
        self.handlers = dict(handlers or {})        # layer 2+: extra intents, or replacements of table handlers (checked first)
        self.meta = meta                            # extra fields for the log record of each exchange (e.g. which voice, latency)
        self.last: dict[str, Any] = {}              # the last exchange: parsed intent, facts, answer (evals read it)
        self.understand: Understand = understand or RuleUnderstand()
        self.speak: Speak = speak or LMSpeak()
        self.clock = now or dt.datetime.now
        self.auto_confirm = auto_confirm
        self.log = log
        self.session = uuid.uuid4().hex[:8]
        self.waiting: Optional[dict[str, Any]] = None

    def ctx(self) -> Ctx:
        return Ctx(self.root, self.clock())

    def _do(self, c: Ctx, act: dict[str, Any]) -> Facts:
        if act["action"] == "pause":
            c.state.mkdir(parents=True, exist_ok=True)
            c.stop_file.write_text(f"paused from the terminal at {c.now.isoformat(timespec='seconds')}\n", encoding="utf-8")
            return Facts("pause", ["Paused. NUPEN_STOP is in place; the service will stop the swarm."], [STOP_FILE])
        if act["action"] == "resume":
            try:
                c.stop_file.unlink()
            except FileNotFoundError:
                pass
            return Facts("resume", ["Resumed. NUPEN_STOP is gone; the service may start the swarm again."], [STOP_FILE])
        if act["action"] == "approve":
            from creator.ledger import Ledger
            try:
                cap = GO.approve(c.state, c.root, Ledger(c.state / "ledger.jsonl", evidence_root=c.root), act["goal"], "owner")
            except (GO.GoalError, ValueError, OSError) as e:
                return Facts("approve", [f"I could not approve {act['goal']}: {e}"], [act["goal"]])
            return Facts("approve", [f"Approved {act['goal']} as owner. It is now capability {cap} with its requirement ladder."], [act["goal"], cap])
        return Facts("unknown", ["I lost track of what I was asking."])

    def reply(self, text: str) -> str:
        text = text.strip()
        c = self.ctx()
        parsed: Parsed
        facts: Facts
        low = text.lower().strip(" .!")
        if self.waiting is not None and low in YES + NO:
            act, self.waiting = self.waiting, None
            parsed = Parsed("confirm_yes" if low in YES else "confirm_no", {"action": act["action"]})
            facts = self._do(c, act) if low in YES else Facts("confirm_no", [f"Cancelled; I did not {act['action']}."])
        else:
            if self.waiting is not None:
                self.waiting = None
            parsed = self.understand.parse(text)
            handler = self.handlers.get(parsed.intent) or (INTENT_BY_NAME[parsed.intent].handler if parsed.intent in INTENT_BY_NAME else h_unknown)
            try:
                facts = handler(c, parsed.slots)
            except Exception as e:  # noqa: BLE001 - a broken record must not end the conversation
                facts = Facts(parsed.intent, [f"I tried to answer but reading my records failed: {type(e).__name__}: {str(e)[:120]}"])
            if facts.confirm is not None:
                if self.auto_confirm:
                    done = self._do(c, facts.confirm)
                    facts = Facts(facts.intent, facts.lines[:1] + done.lines, facts.evidence + done.evidence)
                else:
                    self.waiting = facts.confirm
        facts.user = text
        answer = self.speak.render(facts)
        self.last = {"parsed": parsed, "facts": facts, "answer": answer}
        if self.log:
            self._append(c, text, parsed, facts, answer)
        return answer

    def _append(self, c: Ctx, user: str, p: Parsed, f: Facts, answer: str) -> None:
        rec = {"user": user, "intent": p.intent, "slots": p.slots, "answer": answer, "evidence": f.evidence,
               "at": c.now.isoformat(timespec="seconds"), "source": SOURCE, "session": self.session}
        if self.meta is not None:
            try:
                rec.update({k: v for k, v in self.meta().items() if k not in rec})
            except Exception:  # noqa: BLE001 - logging extras never break the exchange
                pass
        try:
            c.state.mkdir(parents=True, exist_ok=True)
            with open(c.state / CONVERSATIONS_FILE, "ab") as fh:
                fh.write((json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        except OSError:
            pass


def export_dialogues(state: Path, out: Path) -> int:
    """The logged exchanges as creator.lm.dialogue items (one two-turn dialogue each). NOT teacher-reviewed: reviewed=False, source nupen-terminal."""
    rows = [r for r in _jsonl(Path(state) / CONVERSATIONS_FILE) if r.get("source") == SOURCE]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"stage": 0, "split": "train", "source": SOURCE, "reviewed": False, "entities": r.get("evidence", []), "expect": [],
                                "intent": r.get("intent"), "at": r.get("at"),
                                "dialogue": [{"role": "user", "text": r["user"]}, {"role": "nupen", "text": r["answer"]}]}, ensure_ascii=False) + "\n")
    return len(rows)


def banner() -> str:
    return "Talking to Nupen (layer 1: rule-based, answers only from my records). Say 'help'. 'quit' leaves.   " + time.strftime("%Y-%m-%d %H:%M")
