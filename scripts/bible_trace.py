"""Bible Phases 37-39 (checklist state machine, master checklist, definition of done): a traceability matrix for the
WHOLE Bible. Canon C46/C50 (a file existing is not proof) and Bible section 4 (no fake completion) are why it exists.

Pipeline:  parse BIBLE.md into stable requirement ids  ->  index code (ast), tests (ast) and evidence (state/research)
-> link each requirement to implementing code / exercising tests / real-data evidence  ->  assign one of the Bible's
five states by explicit rules  ->  write state/build/BIBLE_TRACE.md and state/build/bible_trace.json.

State rules (first match wins; every state carries the paths that justify it):
  [!]  a FAILED verdict is recorded: a linked evidence line that says fail/false and shares a term with the requirement,
       or a failing test (pytest lastfailed cache) whose name shares a term with it.
  [ ]  no implementing code found.
  [~]  code but no test that exercises it; OR code+test but the Bible asks for real-data proof (blind, eras, windows,
       out-of-sample, calibration ...) and no evidence file exists; OR code+test where no test name speaks to the requirement.
  [?]  code+test+evidence exist but no evidence line says PASS about this requirement (unproven).
  [x]  code + a test that names the requirement's terms + (where real data is asked) an evidence line that says PASS.
The linking is a term-overlap heuristic (IDF-weighted stems over identifiers and docstrings), not a proof of semantics:
[x] means "traceable", the report says so, and the heuristic's thresholds are constants below with their reasons.

No look-ahead concern (pure static analysis); deterministic: sorted walks, no randomness, `now` only labels the report.
Never runs git and never imports the modules it inspects.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

STATES = ("[ ]", "[~]", "[?]", "[x]", "[!]")
STATE_NAMES = {"[ ]": "NOT STARTED", "[~]": "IMPLEMENTED / TESTING", "[?]": "UNPROVEN", "[x]": "VALIDATED", "[!]": "FAILED"}

# --- constants with reasons -------------------------------------------------------------------------------------------
PHASE_HIT_MIN = 0.34      # share of a requirement's IDF mass a phase-linked module must cover (the phase link is prior evidence)
PHASE_HIT_MIN_TERMS = 2   # ...and at least this many distinct terms, unless the requirement has <=2 terms
FREE_HIT_MIN = 0.60       # a module with no phase link must cover much more: nothing but the words vouches for it
FREE_HIT_MIN_TERMS = 3
SPEAK_MASS = 0.5          # ...or a single shared term carrying at least this share of the requirement's weight
PRIMARY_MARGIN = 0.15     # tests/evidence are taken from the best-matching modules only, not from every weak neighbour
MAX_CODE_LINKS = 4        # more links per requirement would dilute the citation into a list of everything
MAX_PER_PHASE_IN_TOP = 5  # the "20 most important missing" list must not be one phase repeated
MAX_EVIDENCE_BYTES = 400_000   # evidence files are read up to this size: reports are small, CSV/parquet are existence-only
MAX_FLAT_LINES = 4000
MAX_MISMATCH_LINES = 25

# Phase 43's priority order when compute is limited: 1 anti-leak, 2 Algorithm, 3 pattern validation, 4 pattern->mover,
# 5 analogs, 6 memory, 7 direction, 8 exit/stop, 9 outer loop, 10 dashboard, 11 live. Phases the list does not name are
# placed by what they protect.
PHASE_PRIORITY = {1: 1, 2: 1, 21: 1, 22: 1, 23: 1, 24: 1, 26: 1, 33: 1, 0: 1, 28: 11,
                  3: 2, 4: 2, 5: 2, 7: 2, 25: 3, 6: 4, 8: 5, 9: 6, 10: 6, 11: 6, 12: 7, 13: 7, 14: 7,
                  15: 8, 16: 8, 17: 9, 18: 9, 19: 9, 20: 9, 29: 10, 27: 5, 30: 9, 31: 9, 32: 9, 34: 3,
                  35: 9, 36: 9, 37: 10, 38: 10, 39: 10}
IMPORTANT_TERMS = re.compile(r"leak|future|fail|gate|blind|reproduc|time fence|point-in-time|exception|fence|scramble|"
                             r"permutation|firewall|lookahead|look-ahead|p\(real\)|fdr", re.I)

REAL_DATA = re.compile(r"blind|histor|real[- ]data|\bera|window|out-of-sample|walk-forward|validat|reproduc|parity|calibrat|"
                       r"ablation|permutation|hidden|unseen|surviv|target|accuracy|evidence|regime|sample|replay|random-date|"
                       r"stat(istic)?ically|survive|confirm", re.I)

# Introducer lines end in ':' and precede a bullet list. Blacklisted introducers list things NOT to do or examples of
# failure, which are constraints on behaviour and not deliverables.
NEGATIVE_INTRO = re.compile(r"\bnever\b|forbidden|do not|don't|avoid|must not|unacceptable|prefer|instead|examples?\b|"
                            r"expected|stop-less|must stop|not permission", re.I)
STRONG_INTRO = re.compile(r"implement|measure|required|run\b|compare|tested against|maintain|test\b|create|build", re.I)
SKIP_HEADINGS = {"BAD"}
GENERIC_STEMS = {"every", "must", "should", "each", "need", "used", "using", "with", "that", "this", "from", "into", "when",
                 "only", "have", "also", "than", "then", "same", "both", "more", "less", "implement", "verify", "check",
                 "required", "test", "tests", "must"}
STOP = set("""the and for are but not you all can any our out has had was were been being have this that with from they
their will would there what which when where who whom whose than then them these those such into over under about above
below between through during before after again further once here more most other some very just also only own same
too its itself of to in on at by an as or if is it be do does did so no nor per via vs""".split())


# --- text helpers -------------------------------------------------------------------------------------------------------
def stem(w: str) -> str:
    """Crude suffix stripper: enough to equate 'validation'/'validate'/'validated'; deterministic and dependency-free."""
    w = w.lower()
    for suf, rep in (("ations", "ate"), ("ation", "ate"), ("ings", ""), ("ing", ""), ("ies", "y"), ("ied", "y"),
                     ("ions", ""), ("ion", ""), ("ity", ""), ("ness", ""), ("ment", ""), ("ance", ""), ("ence", ""), ("ive", ""),
                     ("ers", ""), ("er", ""), ("ed", ""), ("es", ""), ("s", ""), ("ly", ""), ("al", "")):
        if w.endswith(suf) and len(w) - len(suf) + len(rep) >= 4:
            w = w[: len(w) - len(suf)] + rep
            break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


# Three-letter abbreviations that identifiers use for words the Bible spells out; mapped to the full word's stem so
# 'max_abs_err' meets 'maximum absolute error'. Deliberately short: every alias is a place two meanings could collide.
ALIAS = {"max": "maximum", "min": "minimum", "abs": "absolut", "rel": "relativ", "err": "error", "vol": "volatil",
         "cfg": "configurat", "corr": "correlat", "pval": "value", "std": "standard", "dev": "deviat", "nan": "nan"}


def terms(text: str) -> list[str]:
    """Significant stems of a text, in order, de-duplicated. Splits snake_case and camelCase so identifiers compare with prose."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    out, seen = [], set()
    for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", text.replace("_", " ")):
        lw = w.lower()
        if lw in ALIAS:
            s = stem(ALIAS[lw]) if ALIAS[lw] not in ("nan", "value") else ALIAS[lw]
            if s not in seen:
                seen.add(s)
                out.append(s)
            continue
        if len(lw) < 4 or lw in STOP:
            continue
        s = stem(lw)
        if s in GENERIC_STEMS or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_text(path: Path, limit: int | None = None) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if limit is not None:
        data = data[:limit]
    return data.decode("utf-8", errors="ignore")


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --- 1. Bible parsing ---------------------------------------------------------------------------------------------------
@dataclass
class Requirement:
    id: str
    phase: int
    phase_title: str
    section: str
    section_title: str
    text: str
    kind: str            # checkbox | list | heading | report | checklist | phase
    line: int
    claim: str = ""      # the Bible's own marker ('[ ]', '[x/~]', ...) when it has one
    label: str = ""      # A1 / V2b / T10 ... for master-checklist items
    terms: list[str] = field(default_factory=list)
    needs_real_data: bool = False


@dataclass
class PhaseInfo:
    phase: int
    title: str
    line: int
    est_lo: int = 0
    est_hi: int = 0


PHASE_RE = re.compile(r"^#\s+PHASE\s+(\d+)\s*[—–-]\s*(.+?)\s*$")
EST_RE = re.compile(r"Estimated code:\s*\**\s*([\d,]+)\s*[–-]\s*([\d,]+)")
BOX_RE = re.compile(r"^[*-]\s+\[(?P<m>[ x~?!/]+)\]\s+(?P<t>.+)$")
BUL_RE = re.compile(r"^[*-]\s+(?P<t>.+)$")
NUM_RE = re.compile(r"^\d+\.\s+(?P<t>.+)$")
LABEL_RE = re.compile(r"^([A-Z]\d+[a-z]?)\s+(.*)$")


def _clean(s: str) -> str:
    s = re.sub(r"[`*]", "", s).strip()
    return re.sub(r"\s+", " ", s).rstrip(";,. ")


def _slug(title: str) -> str:
    m = re.match(r"^(\d+)\.(\d+)\b", title)
    if m:
        return m.group(2)
    m = re.match(r"^([A-Z])\.\s", title)
    if m:
        return m.group(1)
    m = re.match(r"^(\d+)\.\s", title)
    if m:
        return m.group(1)
    m = re.match(r"^Level\s+(\d+)", title)
    if m:
        return "L" + m.group(1)
    words = re.findall(r"[A-Za-z0-9]+", title)[:2]
    return "-".join(w.upper() for w in words)[:16] or "0"


def parse_bible(text: str) -> tuple[list[PhaseInfo], list[Requirement]]:
    """Every requirement of every '# PHASE n' section: checkbox bullets, bullet/numbered lists under an introducer line
    ending in ':', all-caps-headed fenced report templates, and headings that carry a prose requirement but no bullets.
    A phase that yields nothing still gets one requirement (its title): nothing may silently disappear (Phase 4's rule)."""
    lines = text.splitlines()
    phases: list[PhaseInfo] = []
    reqs: list[Requirement] = []
    cur: PhaseInfo | None = None
    section, section_title = "0", ""
    counters: Counter = Counter()
    used_sections: dict[int, set[str]] = defaultdict(set)
    intro = ""            # last introducer line (cleaned); '' once prose intervenes
    intro_ok = False
    prose_after_heading = ""
    heading_line = 0
    heading_had_req = True
    checklist_group = ""
    in_fence = False
    fence_lines: list[tuple[int, str]] = []
    fence_intro = ""

    def new_id(kind_section: str) -> str:
        counters[(cur.phase, kind_section)] += 1
        return f"{cur.phase}.{kind_section}.{counters[(cur.phase, kind_section)]}"

    def add(kind, txt, ln, claim="", label="", sec=None, sec_title=None):
        r = Requirement(new_id(sec or section), cur.phase, cur.title, sec or section, sec_title if sec_title is not None else section_title,
                        txt, kind, ln, claim, label)
        r.terms = terms(f"{txt} {section_title}") or terms(cur.title)
        r.needs_real_data = bool(REAL_DATA.search(txt))
        reqs.append(r)

    def close_heading():
        """A heading that produced no requirement stands as one itself (Phase 26 A-J, Phase 30 questions, Phase 8.2 ...)."""
        nonlocal heading_had_req
        if cur and section_title and not heading_had_req and section_title.split(".")[-1].strip().upper() not in SKIP_HEADINGS \
                and section_title.upper() not in SKIP_HEADINGS:
            txt = _clean(section_title + (": " + prose_after_heading if prose_after_heading else ""))
            add("heading", txt, heading_line)
        heading_had_req = True

    def flush_fence():
        nonlocal fence_lines
        body = [(n, t) for n, t in fence_lines if t.strip()]
        fence_lines = []
        if not body or cur is None:
            return
        sample = [t for _, t in body]
        skip = sum(1 for t in sample if re.match(r"^\s*(\[.\]|[↓↺])", t) or "↓" in t or "↺" in t) > len(sample) / 3
        heads = [t for t in sample if t.strip().isupper() and len(t.strip()) > 2]
        if skip or not heads or not (STRONG_INTRO.search(fence_intro) or re.search(r"containing|report", fence_intro, re.I)):
            return
        sec = "REPORT"
        ctitle = ""
        for n, t in body:
            s = t.strip()
            if s.isupper():
                ctitle = s
                sec = _slug(s)
                continue
            add("report", _clean(f"{ctitle}: {s}" if ctitle else s), n, sec=sec, sec_title=ctitle)

    for i, raw in enumerate(lines, 1):
        line = raw.rstrip()
        m = re.match(r"^#\s+(.*)$", line)
        if m and not line.startswith("##") and not in_fence:
            close_heading()
            pm = PHASE_RE.match(line)
            if pm:
                cur = PhaseInfo(int(pm.group(1)), pm.group(2), i)
                phases.append(cur)
            else:
                cur = None
            section, section_title, intro, intro_ok, prose_after_heading = "0", "", "", False, ""
            heading_had_req, checklist_group = True, ""
            continue
        if cur is None:
            continue
        if line.strip().startswith("```"):
            if in_fence:
                in_fence = False
                flush_fence()
            else:
                in_fence, fence_intro = True, intro
            continue
        if in_fence:
            fence_lines.append((i, line))
            continue
        m = re.match(r"^(#{2,3})\s+(.*)$", line)
        if m:
            close_heading()
            section_title = _clean(m.group(2))
            section = _slug(section_title)
            base, k = section, 2
            while section in used_sections[cur.phase]:
                section, k = f"{base}{k}", k + 1
            used_sections[cur.phase].add(section)
            heading_line, prose_after_heading, heading_had_req = i, "", False
            intro, intro_ok = "", False
            if cur.phase == 38:
                checklist_group = section_title
            continue
        if not line.strip() or line.strip() == "---":
            continue
        em = EST_RE.search(line)
        if em and cur.est_hi == 0:
            cur.est_lo, cur.est_hi = int(em.group(1).replace(",", "")), int(em.group(2).replace(",", ""))
            continue
        bm = BOX_RE.match(line)
        if bm:
            t = _clean(bm.group("t"))
            claim = "[" + bm.group("m").strip() + "]" if bm.group("m").strip() else "[ ]"
            lm = LABEL_RE.match(t)
            if cur.phase == 38 and lm:
                add("checklist", _clean(f"{checklist_group}: {lm.group(2)}"), i, claim=claim, label=lm.group(1))
            else:
                add("checkbox", t, i, claim=claim)
            heading_had_req = True
            continue
        bm = BUL_RE.match(line)
        nm = NUM_RE.match(line)
        if (bm or nm) and intro:
            if intro_ok or (nm is None):
                add("list", _clean((bm or nm).group("t")), i)
                heading_had_req = True
            continue
        if bm or nm:
            continue
        # a prose line: candidate introducer, or the heading's explanatory sentence
        plain = _clean(line)
        if line.rstrip().endswith(":") and not NEGATIVE_INTRO.search(plain):
            intro = plain
            intro_ok = bool(STRONG_INTRO.search(plain) or re.search(r"tested against|prioriti", plain, re.I)) and not \
                re.search(r"after each completed phase", plain, re.I)
        else:
            intro, intro_ok = "", False
            if not prose_after_heading and not heading_had_req:
                prose_after_heading = plain
    close_heading()
    if in_fence:
        flush_fence()
    for ph in phases:
        if not any(r.phase == ph.phase for r in reqs):
            first = next((_clean(lines[j]) for j in range(ph.line, min(ph.line + 12, len(lines)))
                          if lines[j].strip() and not lines[j].startswith("#") and not lines[j].strip().startswith("Estimated")), "")
            cur = ph
            section, section_title = "0", ph.title
            add("phase", _clean(f"{ph.title}: {first}" if first else ph.title), ph.line)
    return phases, reqs


# --- 2. indexes: code, tests, evidence, task files -----------------------------------------------------------------------
@dataclass
class Module:
    path: str
    lines: int
    doc: str
    symbols: dict[str, list[str]]      # symbol name -> its stems
    stems: set[str]
    phases: set[int]
    planned_by: set[str] = field(default_factory=set)
    mtime: float = 0.0


PHASE_MENTION = re.compile(r"phases?\s+(\d+(?:\s*(?:,|and|&|/|-|–|to)\s*\d+)*)", re.I)


def phases_in(text: str) -> set[int]:
    """'Phase 25', 'PHASES 21, 22, 23, 24', 'Phases 3-5' -> {ints}; ranges written with - / to expand."""
    out: set[int] = set()
    for m in PHASE_MENTION.finditer(text):
        body = m.group(1)
        nums = [int(x) for x in re.findall(r"\d+", body)]
        if re.search(r"\d\s*(?:-|–|to)\s*\d", body) and len(nums) == 2 and nums[0] < nums[1] <= 60:
            out.update(range(nums[0], nums[1] + 1))
        else:
            out.update(n for n in nums if n <= 60)
    return out


def index_module(path: Path, root: Path) -> Module | None:
    src = read_text(path)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    doc = ast.get_docstring(tree) or ""
    symbols: dict[str, list[str]] = {}
    stems = set(terms(path.stem))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            st = terms(node.name)
            symbols[node.name] = st
            stems.update(st)
            d = ast.get_docstring(node)
            if d:
                stems.update(terms(d.splitlines()[0]))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper() and len(t.id) > 3:
                    stems.update(terms(t.id))
    stems.update(terms(doc))
    head = "\n".join(src.splitlines()[:40])
    return Module(rel(path, root), len(src.splitlines()), doc, symbols, stems, phases_in(doc) | phases_in(head), mtime=path.stat().st_mtime)


def index_code(root: Path) -> dict[str, Module]:
    mods: dict[str, Module] = {}
    for sub in ("engine", "scripts"):
        d = root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.py")):
            if p.name.startswith(("_", "patch_")):
                continue
            m = index_module(p, root)
            if m:
                mods[m.path] = m
    return mods


@dataclass
class TestFile:
    path: str
    imports: set[str]
    funcs: dict[str, list[str]]     # test function -> stems of name + docstring first line
    text_words: set[str]


def index_tests(root: Path) -> dict[str, TestFile]:
    out: dict[str, TestFile] = {}
    d = root / "tests"
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*.py")):
        if p.name.startswith("_") or not (p.name.startswith("test_") or p.parent.name == "integration"):
            continue
        src = read_text(p)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        imports: set[str] = set()
        funcs: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
                imports.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                d0 = (ast.get_docstring(node) or "").splitlines()[:1]
                funcs[node.name] = terms(node.name + " " + " ".join(d0))
        words = set(re.findall(r"[a-z_][a-z0-9_]{4,}", src))
        out[rel(p, root)] = TestFile(rel(p, root), imports, funcs, words)
    return out


def tests_for(mod: Module, tests: dict[str, TestFile]) -> list[str]:
    """A test file exercises a module when it imports it (engine.x / from engine import x) or, for scripts, names it."""
    stem_ = Path(mod.path).stem
    pkg = Path(mod.path).parent.name
    out = []
    for tp, tf in tests.items():
        dotted = f"{pkg}.{stem_}"
        if any(i == dotted or i.startswith(dotted + ".") for i in tf.imports) or (pkg == "scripts" and stem_ in tf.text_words):
            out.append(tp)
    return out


@dataclass
class EvidenceFile:
    path: str
    kind: str                       # text | json | table | binary
    code_files: set[str]
    mentions: set[str]              # engine/x.py, scripts/x.py mentioned in the text
    pass_lines: list[tuple[str, list[str]]] = field(default_factory=list)
    fail_lines: list[tuple[str, list[str]]] = field(default_factory=list)


PASS_KEYS = re.compile(r"(^|_)(pass|passed|passes|ok|valid|success|clean|reproduced|verdict|status|decision|outcome|result|gate)$", re.I)
PASS_WORDS = {"pass", "passed", "ok", "adopt", "adopted", "validated", "success", "clean", "true"}
FAIL_WORDS = {"fail", "failed", "false"}   # "rejected" is a lifecycle outcome (a dead feature SHOULD be rejected), not a failed gate


def _flatten(obj, prefix="", out=None):
    out = [] if out is None else out
    if len(out) >= MAX_FLAT_LINES:
        return out
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            _flatten(obj[k], f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        for j, v in enumerate(obj[:200]):
            _flatten(v, f"{prefix}[{j}]", out)
    else:
        out.append((prefix, obj))
    return out


def judge_json(obj) -> tuple[list, list, set[str]]:
    """Pass/fail lines from a JSON document. Only keys that are verdicts count: a numeric 'failures: 0' is not a failure."""
    pass_l, fail_l, code_files = [], [], set()
    for path, val in _flatten(obj):
        if path.endswith("code_files") or ".code_files[" in path or path.startswith("code_files["):
            if isinstance(val, str):
                code_files.add(val)
            continue
        last = re.split(r"[.\[]", path.rstrip("]"))[-1] if path else ""
        keyparts = re.split(r"\.", path)[-1]
        if not (PASS_KEYS.search(keyparts) or PASS_KEYS.search(last)):
            continue
        word = str(val).strip().lower() if isinstance(val, (bool, str)) else None
        if word is None:
            continue
        toks = terms(path.replace(".", " ") + " " + word)
        if word in PASS_WORDS:
            pass_l.append((f"{path} = {val}", toks))
        elif word in FAIL_WORDS or word.startswith("fail"):
            fail_l.append((f"{path} = {val}", toks))
    return pass_l, fail_l, code_files


def judge_text(text: str) -> tuple[list, list]:
    """Upper-case PASS / FAIL tokens in a report line are verdicts (the repo's reports write them that way)."""
    pass_l, fail_l = [], []
    for ln in text.splitlines()[:MAX_FLAT_LINES]:
        has_f = re.search(r"\bFAIL(ED|URE)?\b|✗", ln)
        has_p = re.search(r"\bPASS(ED)?\b|✓", ln)
        toks = terms(ln)
        if has_f:
            fail_l.append((ln.strip()[:160], toks))
        elif has_p:
            pass_l.append((ln.strip()[:160], toks))
    return pass_l, fail_l


def index_evidence(root: Path, research: Path | None = None) -> dict[str, EvidenceFile]:
    base = research or (root / "state" / "research")
    out: dict[str, EvidenceFile] = {}
    if not base.is_dir():
        return out
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        rp = rel(p, root)
        if ext in (".json", ".md", ".txt", ".log", ".err") and p.stat().st_size <= MAX_EVIDENCE_BYTES * 4:
            text = read_text(p, MAX_EVIDENCE_BYTES)
            ev = EvidenceFile(rp, "text", set(), set(re.findall(r"(?:engine|scripts)/[a-z0-9_]+\.py", text)))
            if ext == ".json":
                try:
                    pl, fl, cf = judge_json(json.loads(text))
                    ev.kind, ev.pass_lines, ev.fail_lines, ev.code_files = "json", pl, fl, cf
                except (json.JSONDecodeError, RecursionError):
                    ev.pass_lines, ev.fail_lines = judge_text(text)
            else:
                ev.pass_lines, ev.fail_lines = judge_text(text)
            out[rp] = ev
        else:
            out[rp] = EvidenceFile(rp, "table" if ext == ".csv" else "binary", set(), set())
    return out


def evidence_for(mod: Module, evidence: dict[str, EvidenceFile], root_name: str = "state/research") -> list[str]:
    """Evidence belongs to a module when its provenance lists the module's file, its text names the file, or its path
    (directory or file name) carries the module's own name."""
    stem_ = Path(mod.path).stem
    out = []
    for ep, ev in evidence.items():
        sub = ep[len(root_name) + 1:] if ep.startswith(root_name + "/") else ep
        parts = re.split(r"[/_.]", sub.lower())
        by_path = len(stem_) >= 5 and (stem_ in sub.lower() or all(x in parts for x in stem_.lower().split("_") if len(x) > 2) and len(stem_.split("_")) > 1)
        if mod.path in ev.code_files or mod.path in ev.mentions or by_path:
            out.append(ep)
    return out


TASK_PHASES = re.compile(r"Bible:\s*PHASES?\s+([\d,\sand\-–]+)", re.I)
TASK_OWN = re.compile(r"You own:\s*(.+)")


def index_tasks(root: Path) -> list[dict]:
    """Builder task files: phases each unit was given and the files it owns (some may not exist yet)."""
    out = []
    qd = root / "state" / "build" / "queue"
    if not qd.is_dir():
        return out
    for p in sorted(qd.glob("*.md")):
        text = read_text(p)
        pm, om = TASK_PHASES.search(text), TASK_OWN.search(text)
        nums = phases_in("phases " + pm.group(1)) if pm else set()
        owns = [x.strip() for x in om.group(1).split(",")] if om else []
        out.append({"unit": p.stem, "path": rel(p, root), "phases": sorted(nums), "owns": owns})
    return out


def integration_pending(root: Path) -> dict[str, list[str]]:
    """Unchecked hooks in INTEGRATION.md, by module path mentioned: code that exists but is not wired in."""
    path = root / "state" / "build" / "INTEGRATION.md"
    pend: dict[str, list[str]] = defaultdict(list)
    for ln in read_text(path).splitlines():
        if re.match(r"^- \[ \]", ln):
            for mod in re.findall(r"\b([a-z_]+)\.py\b", ln):
                pend[mod].append(_clean(ln[6:])[:140])
    return dict(pend)


def failing_tests(root: Path) -> tuple[dict[str, list[str]], float]:
    """({test file: [test names]}, cache mtime) from the pytest lastfailed cache. Entries whose test file was edited after
    the cache was written are dropped: a failure recorded against code that has since changed is not a current verdict."""
    path = root / ".pytest_cache" / "v" / "cache" / "lastfailed"
    try:
        data = json.loads(read_text(path))
        cache_mtime = path.stat().st_mtime
    except (json.JSONDecodeError, OSError):
        return {}, 0.0
    out: dict[str, list[str]] = defaultdict(list)
    for k in data:
        f, _, name = k.partition("::")
        fp = root / f
        if fp.exists() and fp.stat().st_mtime > cache_mtime:
            continue
        out[f].append(name.split("::")[-1] if name else "")
    return dict(out), cache_mtime


# --- 3. linking ---------------------------------------------------------------------------------------------------------
def idf_table(reqs: list[Requirement]) -> dict[str, float]:
    df: Counter = Counter()
    for r in reqs:
        df.update(set(r.terms))
    n = max(len(reqs), 1)
    return {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}


def phase_links(mods: dict[str, Module], tasks: list[dict]) -> None:
    """Attach the phases a task file was given to the modules it owns (docstring links were set at indexing)."""
    for t in tasks:
        for o in t["owns"]:
            if o in mods:
                mods[o].phases.update(t["phases"])
                mods[o].planned_by.add(t["unit"])


def link_code(req: Requirement, mods: dict[str, Module], idf: dict[str, float]) -> list[dict]:
    if not req.terms:
        return []
    total = sum(idf.get(t, 1.0) for t in req.terms)
    out = []
    for m in mods.values():
        hit = [t for t in req.terms if t in m.stems]
        if not hit:
            continue
        score = sum(idf.get(t, 1.0) for t in hit) / total
        linked = req.phase in m.phases
        need_n = min(PHASE_HIT_MIN_TERMS if linked else FREE_HIT_MIN_TERMS, len(req.terms))
        if len(req.terms) <= 2 and linked:
            need_n = 1
        need_s = PHASE_HIT_MIN if linked else FREE_HIT_MIN
        syms = sorted((s for s, st in m.symbols.items() if set(st) & set(req.terms)),
                      key=lambda s: (-len(set(m.symbols[s]) & set(req.terms)), s))[:3]
        named = bool(syms) or bool(set(terms(Path(m.path).stem)) & set(req.terms))
        if not named and not (linked and (len(hit) >= 3 or len(hit) == len(req.terms))):
            continue   # a docstring that merely uses the words is weak evidence unless the phase link backs it
        if len(hit) >= need_n and score >= need_s:
            out.append({"module": m.path, "score": round(score, 3), "phase_linked": linked, "terms": hit, "symbols": syms})
    out.sort(key=lambda d: (-d["score"], not d["phase_linked"], d["module"]))
    return out[:MAX_CODE_LINKS]


def overlap(a: list[str], b: list[str], idf: dict[str, float], min_idf: float = 0.4) -> list[str]:
    return sorted(t for t in set(a) & set(b) if idf.get(t, 1.0) >= min_idf)


def speaks_to(req_terms: list[str], other: list[str], idf: dict[str, float]) -> bool:
    """A test name or a verdict line 'speaks to' a requirement when it shares two terms, or half its IDF mass: one shared
    word ('feature', 'date') is how a failing dial report ended up condemning Phase 1 in the first draft."""
    ov = overlap(req_terms, other, idf)
    if not ov:
        return False
    total = sum(idf.get(t, 1.0) for t in req_terms) or 1.0
    return len(ov) >= 2 or sum(idf.get(t, 1.0) for t in ov) / total >= SPEAK_MASS


def assess(req: Requirement, code: list[dict], mods, tests, evidence, failing, idf, cache_mtime: float = 0.0) -> dict:
    """Apply the state rules to one requirement and return the full record with the paths that justify the state."""
    rec = {"id": req.id, "phase": req.phase, "section": req.section, "kind": req.kind, "label": req.label,
           "text": req.text, "bible_line": req.line, "bible_claim": req.claim, "needs_real_data": req.needs_real_data,
           "code": code, "tests": [], "test_functions": [], "evidence": [], "pass_lines": [], "fail_lines": [],
           "failing_tests": [], "notes": []}
    if not code:
        rec["state"], rec["why"] = "[ ]", "no implementing code found by name/docstring/task-file"
        return rec
    tpaths: list[str] = []
    epaths: list[str] = []
    top = max(c["score"] for c in code)
    for c in (c for c in code if c["score"] >= top - PRIMARY_MARGIN):
        for t in tests_for(mods[c["module"]], tests):
            if t not in tpaths:
                tpaths.append(t)
        for e in evidence_for(mods[c["module"]], evidence):
            if e not in epaths:
                epaths.append(e)
    rec["tests"], rec["evidence"] = sorted(tpaths), sorted(epaths)
    speaking = []
    for tp in rec["tests"]:
        for fn, st in tests[tp].funcs.items():
            if speaks_to(req.terms, st, idf):
                speaking.append((tp, fn, overlap(req.terms, st, idf)))
                fresh = all(mods[c["module"]].mtime <= cache_mtime for c in code)
                names = [n for n in failing.get(tp, []) if n]
                if fresh and tp in failing and (fn in names or (not names and "" in failing[tp])):
                    rec["failing_tests"].append(f"{tp}::{fn}")
    speaking.sort(key=lambda x: (-len(x[2]), x[0], x[1]))
    rec["test_functions"] = [f"{tp}::{fn}" for tp, fn, _ in speaking[:4]]
    for ep in rec["evidence"]:
        ev = evidence[ep]
        for ln, toks in ev.fail_lines:
            if speaks_to(req.terms, toks, idf):
                rec["fail_lines"].append({"path": ep, "line": ln})
        for ln, toks in ev.pass_lines:
            if speaks_to(req.terms, toks, idf):
                rec["pass_lines"].append({"path": ep, "line": ln})
    rec["fail_lines"], rec["pass_lines"] = rec["fail_lines"][:3], rec["pass_lines"][:3]
    if rec["fail_lines"] or rec["failing_tests"]:
        src = rec["fail_lines"][0]["path"] if rec["fail_lines"] else rec["failing_tests"][0]
        rec["state"], rec["why"] = "[!]", f"failed verdict recorded in {src}"
    elif not rec["tests"]:
        rec["state"], rec["why"] = "[~]", "code exists, no test imports it"
    elif not speaking:
        rec["state"], rec["why"] = "[~]", "tests import the code but no test name speaks to this requirement"
    elif req.needs_real_data and not rec["evidence"]:
        rec["state"], rec["why"] = "[~]", "Bible asks for real-data proof; no evidence file under state/research"
    elif req.needs_real_data and not rec["pass_lines"]:
        rec["state"], rec["why"] = "[?]", "evidence files exist but none says PASS about this requirement"
    else:
        cite = rec["pass_lines"][0]["path"] if rec["pass_lines"] else rec["test_functions"][0]
        rec["state"], rec["why"] = "[x]", f"code {code[0]['module']}, test {rec['test_functions'][0]}, cited {cite}"
    if req.claim.startswith("[x") and rec["state"] in ("[ ]", "[~]"):
        rec["notes"].append("Bible marks this built/validated; the trace cannot support that")
    elif req.claim == "[ ]" and rec["state"] == "[x]":
        rec["notes"].append("Bible checkbox is stale (unchecked but traceable)")
    return rec


# --- Phases 38 and 39 are claims ABOUT other phases: derive them, do not word-match them ------------------------------
# Master-checklist label (or Definition-of-Done wording) -> the phases (or 'phase:section' slices) whose requirements
# decide it. A wrong entry mis-derives one line, and the derivation is printed, so it can be audited.
CHECKLIST_MAP = {
    "A1": ["3:1"], "A2": ["3"], "A2b": ["3:4"], "A3": ["7"], "A4": ["5"], "A5": ["6"], "A6": ["27"], "A7": ["19"],
    "A8": ["8"], "A8b": ["8:5"], "A8c": ["8:6"], "A9": ["17"], "A10": ["10", "9"], "A10b": ["11"], "A11": ["4"],
    "A12": ["27"], "A13": ["25"], "V1": ["6", "13"], "V2": ["13"], "V2b": ["12"], "V3": ["15"], "V4": ["16"],
    "V5": ["7", "20"], "T1": ["21"], "T2": ["22"], "T3": ["20"], "T4": ["18"], "T5": ["1:1"], "T6": ["27"],
    "T7": ["29:PATTERN-EXPLORER"], "T8": ["29:SENSITIVITY-PAGE"], "T9": ["23"], "T10": ["1:3", "26:A"], "T11": ["1:2"],
    "T12": ["24"], "T13": ["33"], "T14": ["26:B"], "T15": ["26:F"], "T16": ["26:C"], "T17": ["25"], "L1": ["28"],
    "L2": ["28"], "L3": ["28"], "L4": ["28"], "L5": ["28"],
}
DOD_MAP = [  # (substring of the Definition-of-Done line, slices)
    ("every required module exists", ["ALL"]), ("every required module is tested", ["ALL"]),
    ("data source is validated", ["27"]), ("point-in-time rule", ["1"]), ("blind simulation", ["21", "22"]),
    ("re-tester", ["23"]), ("parity works", ["2"]), ("future scramble", ["1:3", "26:A"]), ("time fence", ["1:2"]),
    ("worker health", ["24"]), ("deterministic replay", ["33"]), ("pattern discovery", ["3"]),
    ("pattern validation", ["3:4", "3:5"]), ("pattern lifecycle", ["4"]), ("analog engine", ["8"]), ("memory works", ["9"]),
    ("lesson learning", ["10"]), ("anti-memorization", ["11"]), ("direction engine", ["13"]), ("exit engine", ["15"]),
    ("stop/risk", ["16"]), ("per-type trust", ["12"]), ("timeline dial", ["17"]), ("training-basis", ["19"]),
    ("planted-pattern", ["25"]), ("experiment registry", ["0:2", "30"]), ("dashboard works", ["29:DASHBOARD"]),
    ("pattern explorer", ["29:PATTERN-EXPLORER"]), ("sensitivity page", ["29:SENSITIVITY-PAGE"]),
    ("negative findings", ["30", "0:3"]), ("champion/challenger", ["35"]),
]
AGG_X_SHARE = 0.80     # [x] needs this share of the underlying requirements traceable: 'works' means all of it works
AGG_FAIL_SHARE = 0.15  # this share failed and the claim is failed, whatever else is fine
AGG_NONE_SHARE = 0.50  # this share without code and the claim is not started
AGG_SOFT_SHARE = 0.60  # [x]+[?] share needed to call a partly-proven claim unproven rather than merely implemented


def slice_requirements(recs: list[dict], slices: list[str]) -> list[dict]:
    """Requirements of phases 0-36 selected by 'phase' or 'phase:section' entries; 'ALL' is every one of them."""
    out = []
    for r in recs:
        if r["phase"] > 36 or r["state"] is None:
            continue
        for sl in slices:
            if sl == "ALL" or sl == str(r["phase"]) or sl == f"{r['phase']}:{r['section']}":
                out.append(r)
                break
    return out


def derive(rec: dict, under: list[dict]) -> dict:
    """Give a Phase 38/39 claim the state its underlying requirements justify, citing the strongest of them."""
    n = len(under)
    c = Counter(u["state"] for u in under)
    rec["derived"] = {"n": n, "counts": {s: c.get(s, 0) for s in STATES}}
    if n == 0:
        rec["state"], rec["why"] = "[ ]", "no requirement maps to this claim; nothing verifies it"
        return rec
    share = {s: c.get(s, 0) / n for s in STATES}
    if share["[!]"] >= AGG_FAIL_SHARE:
        st = "[!]"
    elif share["[ ]"] >= AGG_NONE_SHARE:
        st = "[ ]"
    elif share["[x]"] >= AGG_X_SHARE and c.get("[!]", 0) == 0:
        st = "[x]"
    elif share["[x]"] + share["[?]"] >= AGG_SOFT_SHARE:
        st = "[?]"
    else:
        st = "[~]"
    rec["state"] = st
    best = [u for u in under if u["state"] in ("[x]", "[?]", "[~]", "[!]")]
    mods = Counter(cd["module"] for u in best for cd in u["code"][:1])
    rec["code"] = [{"module": m, "score": 0.0, "phase_linked": True, "terms": [], "symbols": []}
                   for m, _ in sorted(mods.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
    rec["tests"] = sorted({t for u in best for t in u["tests"][:1]})[:5]
    rec["evidence"] = sorted({e for u in best for e in u["evidence"][:1]})[:5]
    rec["pass_lines"] = [p for u in under for p in u["pass_lines"]][:3]
    rec["fail_lines"] = [p for u in under for p in u["fail_lines"]][:3]
    rec["failing_tests"] = [t for u in under for t in u["failing_tests"]][:3]
    rec["why"] = f"derived from {n} requirements: " + ", ".join(f"{c.get(s, 0)} {s}" for s in STATES if c.get(s, 0))
    return rec


def slices_for(req: Requirement) -> list[str] | None:
    """None: word-match this requirement. A list: derive it from those slices (possibly none, meaning unverifiable)."""
    if req.phase == 38 and req.label:
        return CHECKLIST_MAP.get(req.label, [])
    if req.phase == 39 and req.kind == "checkbox":
        low = req.text.lower()
        for key, sl in DOD_MAP:
            if key in low:
                return sl
        return []
    return None


# --- 4. the whole trace --------------------------------------------------------------------------------------------------
def importance(rec: dict) -> float:
    pr = PHASE_PRIORITY.get(rec["phase"], 8)
    score = (12 - pr) * 10.0
    if IMPORTANT_TERMS.search(rec["text"]):
        score += 6
    if rec["kind"] in ("checklist", "checkbox"):
        score += 3
    if rec["kind"] in ("phase", "heading"):
        score -= 2
    return score


def top_missing(records: list[dict], n: int = 20) -> list[dict]:
    """Highest-priority requirements with no code, at most MAX_PER_PHASE_IN_TOP per phase, ties by id for stability."""
    miss = sorted((r for r in records if r["state"] == "[ ]"), key=lambda r: (-importance(r), r["phase"], r["id"]))
    out, per = [], Counter()
    for r in miss:
        if per[r["phase"]] >= MAX_PER_PHASE_IN_TOP:
            continue
        per[r["phase"]] += 1
        out.append({"id": r["id"], "phase": r["phase"], "text": r["text"], "importance": importance(r), "bible_line": r["bible_line"]})
        if len(out) == n:
            break
    return out


def summarize(records: list[dict], phases: list[PhaseInfo], mods: dict[str, Module]) -> dict:
    total = Counter(r["state"] for r in records)
    per_phase = {}
    for ph in phases:
        rs = [r for r in records if r["phase"] == ph.phase]
        linked = sorted((m for m in mods.values() if ph.phase in m.phases), key=lambda m: m.path)
        per_phase[str(ph.phase)] = {
            "title": ph.title, "requirements": len(rs), "counts": {s: sum(1 for r in rs if r["state"] == s) for s in STATES},
            "est_range": [ph.est_lo, ph.est_hi], "linked_code_lines": sum(m.lines for m in linked),
            "linked_modules": [m.path for m in linked],
        }
    by_kind: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        by_kind[r["kind"]][r["state"]] += 1
    return {"counts": {s: total.get(s, 0) for s in STATES}, "total": len(records), "per_phase": per_phase,
            "per_kind": {k: {s: c.get(s, 0) for s in STATES} for k, c in sorted(by_kind.items())}}


def check_invariants(records: list[dict], root: Path) -> list[str]:
    """The trace's own gate: ids unique; every [x] cites paths that exist; every [!] names its source; no [x] with a fail line."""
    bad = []
    seen = Counter(r["id"] for r in records)
    bad += [f"duplicate id {i}" for i, c in seen.items() if c > 1]
    for r in records:
        if r["state"] == "[x]":
            cited = [c["module"] for c in r["code"]] + r["tests"][:1] + [p["path"] for p in r["pass_lines"]]
            if (not cited and not r.get("derived")) or any(not (root / p).exists() for p in cited):
                bad.append(f"{r['id']}: [x] cites a missing path")
            if r["fail_lines"] or r["failing_tests"]:
                bad.append(f"{r['id']}: [x] with a recorded failure")
        if r["state"] == "[!]" and not (r["fail_lines"] or r["failing_tests"]) and not (r.get("derived") and r["derived"]["counts"]["[!]"]):
            bad.append(f"{r['id']}: [!] without a source")
        if r["state"] == "[ ]" and r["code"] and not r.get("derived"):
            bad.append(f"{r['id']}: [ ] but has code links")
    return bad


def build_trace(root: Path, bible: Path | None = None, now: str = "unspecified", research: Path | None = None,
                links_path: Path | None = None) -> dict:
    root = Path(root)
    bible = bible or (root / "BIBLE.md")
    text = read_text(bible)
    phases, reqs = parse_bible(text)
    mods, tests = index_code(root), index_tests(root)
    evidence = index_evidence(root, research)
    tasks = index_tasks(root)
    phase_links(mods, tasks)
    (failing, cache_mtime), pending = failing_tests(root), integration_pending(root)
    idf = idf_table(reqs)
    records = []
    for r in reqs:
        if slices_for(r) is None:
            records.append(assess(r, link_code(r, mods, idf), mods, tests, evidence, failing, idf, cache_mtime))
        else:
            records.append({"id": r.id, "phase": r.phase, "section": r.section, "kind": r.kind, "label": r.label, "text": r.text,
                            "bible_line": r.line, "bible_claim": r.claim, "needs_real_data": r.needs_real_data, "code": [],
                            "tests": [], "test_functions": [], "evidence": [], "pass_lines": [], "fail_lines": [],
                            "failing_tests": [], "notes": [], "state": None})
    links, gaps, link_problems = load_links(Path(links_path) if links_path else root / LINKS_REL, {r.id for r in reqs})
    by_req = {r.id: r for r in reqs}
    link_report = {"declared": len(links), "honoured": 0, "broken": list(link_problems), "changed": []}
    for rec in records:
        if rec["id"] not in links:
            continue
        if slices_for(by_req[rec["id"]]) is not None:
            link_report["broken"].append({"id": rec["id"], "broken": ["derived claim (Phase 38/39): state comes from the requirements it summarises, link a those instead"]})
            continue
        ver = verify_link(links[rec["id"]], root)
        if not ver["ok"]:
            rec["link"] = "rejected"
            link_report["broken"].append({"id": rec["id"], "broken": ver["broken"]})
            continue
        before = rec["state"]
        apply_link(rec, by_req[rec["id"]], ver, failing, evidence, idf)
        link_report["honoured"] += 1
        if rec["state"] != before:
            link_report["changed"].append({"id": rec["id"], "from": before, "to": rec["state"]})
    for rec, r in zip(records, reqs):   # second pass: claims about other phases are derived from the first pass
        sl = slices_for(r)
        if sl is not None:
            derive(rec, slice_requirements(records, sl))
            if r.claim.startswith("[x") and rec["state"] in ("[ ]", "[~]"):
                rec["notes"].append("Bible marks this built/validated; the trace cannot support that")
            elif r.claim == "[ ]" and rec["state"] == "[x]":
                rec["notes"].append("Bible checkbox is stale (unchecked but traceable)")
    summary = summarize(records, phases, mods)
    used = {c["module"] for r in records for c in r["code"]}
    orphans = sorted(p for p in mods if p not in used)
    mismatches = [{"id": r["id"], "label": r["label"], "claim": r["bible_claim"], "state": r["state"], "note": r["notes"][0]}
                  for r in records if r["notes"]]
    planned_missing = sorted({o for t in tasks for o in t["owns"] if o.endswith(".py") and not (root / o).exists()})
    return {
        "generated": now, "bible": rel(bible, root) if bible.exists() else str(bible), "bible_sha": sha16(text),
        "rules": ["no code -> [ ]", "failed verdict recorded -> [!]", "code without test -> [~]",
                  "code+test, Bible asks real-data proof, none found -> [~]", "evidence exists but no PASS line -> [?]",
                  "[x] only with code + speaking test + (where asked) PASS evidence cited by path"],
        "summary": summary, "top_missing": top_missing(records), "requirements": records, "claim_mismatches": mismatches,
        "orphan_modules": orphans, "tasks": tasks, "planned_files_missing": planned_missing,
        "integration_pending": {k: v for k, v in sorted(pending.items())},
        "failing_tests_cache": {k: sorted(v) for k, v in sorted(failing.items())},
        "index": {"modules": len(mods), "test_files": len(tests), "evidence_files": len(evidence)},
        "invariant_violations": check_invariants(records, root),
        "links": link_report, "genuine_gaps": genuine_gaps(records, gaps),
    }


# --- 4b. explicit, verified links (B20) -----------------------------------------------------------------------------------
# The heuristic above links by word overlap, so it has false negatives (the code says 'cooldown' in a dict key, the
# Bible says 'cooldown' in a bullet, and no function name carries either) and false positives. state/build/trace_links.json
# lets a reviewer who READ the code say which symbol implements which requirement. A link is honoured only when every
# path, symbol and test in it is proven to exist by parsing the file (ast); any broken part rejects the WHOLE link
# and reports it, because a half-honoured link would be a claim nobody checked. [x] additionally needs an evidence
# path that exists: code and a test say "implemented", only an artefact says "it worked".
LINKS_REL = "state/build/trace_links.json"
LINK_FIELDS = ("code", "tests", "evidence")
_symbol_cache: dict[tuple[str, float], set | None] = {}


def module_symbols(path: Path) -> set[str] | None:
    """Every name a link may cite in a Python file: top-level def/class/assignment, 'Class.method', 'Class.attr'.
    None when the file is missing or does not parse (a link into it is broken, not vacuously fine)."""
    try:
        key = (str(path.resolve()), path.stat().st_mtime)
    except OSError:
        return None
    if key in _symbol_cache:
        return _symbol_cache[key]
    try:
        tree = ast.parse(read_text(path))
    except SyntaxError:
        _symbol_cache[key] = None
        return None

    def targets(node):
        if isinstance(node, ast.Assign):
            return [t.id for t in node.targets if isinstance(t, ast.Name)]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return [node.target.id]
        return []

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        names.add(f"{node.name}.{sub.name}")
                    for t in targets(sub):
                        names.add(f"{node.name}.{t}")
        names.update(targets(node))
    _symbol_cache[key] = names
    return names


def _inside(root: Path, p: str) -> Path | None:
    """The resolved file for a repo-relative path, or None when it is absolute, climbs out of the repo or does not exist."""
    if not p or Path(p).is_absolute() or ".." in Path(p).parts:
        return None
    full = (root / p).resolve()
    try:
        full.relative_to(root.resolve())
    except ValueError:
        return None
    return full if full.exists() else None


def _split_ref(ref: str) -> tuple[str, str]:
    path, _, sym = str(ref).partition("::")
    return path.strip(), sym.strip()


def verify_link(link, root: Path) -> dict:
    """Check one link entry. Returns {'ok': bool, 'code': [...], 'tests': [...], 'evidence': [...], 'broken': [reasons]}.
    ok is True only with no broken part AND at least one verified code reference."""
    out = {"ok": False, "code": [], "tests": [], "evidence": [], "broken": []}
    if not isinstance(link, dict):
        out["broken"].append("link is not an object with code/tests/evidence")
        return out
    for k in link:
        if k not in LINK_FIELDS and k != "note":
            out["broken"].append(f"unknown link field '{k}'")
    for fld in LINK_FIELDS:
        v = link.get(fld, [])
        if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
            out["broken"].append(f"{fld} must be a list of strings")
            link = {**link, fld: []}
    for ref in link.get("code", []):
        path, sym = _split_ref(ref)
        f = _inside(root, path)
        if f is None or f.suffix != ".py":
            out["broken"].append(f"code file missing: {path or ref!r}")
        elif not sym:
            out["broken"].append(f"code reference has no ::symbol: {ref}")
        elif sym not in (module_symbols(f) or set()):
            out["broken"].append(f"symbol not found: {path}::{sym}")
        else:
            out["code"].append({"module": path, "symbol": sym})
    for ref in link.get("tests", []):
        path, sym = _split_ref(ref)
        f = _inside(root, path)
        leaf = sym.split(".")[-1]
        if f is None or f.suffix != ".py" or not path.startswith("tests/"):
            out["broken"].append(f"test file missing or outside tests/: {path or ref!r}")
        elif not leaf.startswith("test"):
            out["broken"].append(f"not a test function: {ref}")
        elif sym not in (module_symbols(f) or set()):
            out["broken"].append(f"test not found: {path}::{sym}")
        else:
            out["tests"].append(f"{path}::{sym}")
    for path in link.get("evidence", []):
        f = _inside(root, path)
        if f is None:
            out["broken"].append(f"evidence missing: {path}")
        elif f.is_file() and f.stat().st_size == 0:
            out["broken"].append(f"evidence is empty: {path}")
        elif f.is_dir() and not any(x.is_file() for x in f.rglob("*")):
            out["broken"].append(f"evidence directory has no files: {path}")
        else:
            out["evidence"].append(path)
    if not link.get("code"):
        out["broken"].append("link names no code")
    out["ok"] = not out["broken"] and bool(out["code"])
    return out


def load_links(path: Path, valid_ids: set[str]) -> tuple[dict, dict, list[dict]]:
    """(links, gaps, problems). Keys starting with '_' are reserved: '_gaps' maps requirement id -> the reviewer's
    reason it is genuinely missing. A link (or gap) for an id that is not in the Bible is a problem, not a silent no-op."""
    problems: list[dict] = []
    if not Path(path).exists():
        return {}, {}, problems
    try:
        raw = json.loads(read_text(Path(path)))
    except json.JSONDecodeError as e:
        return {}, {}, [{"id": "*", "broken": [f"trace_links.json is not valid JSON: {e}"]}]
    if not isinstance(raw, dict):
        return {}, {}, [{"id": "*", "broken": ["trace_links.json must be an object {req_id: link}"]}]
    gaps = raw.get("_gaps", {}) if isinstance(raw.get("_gaps", {}), dict) else {}
    links = {k: v for k, v in raw.items() if not k.startswith("_")}
    for k in list(links):
        if k not in valid_ids:
            problems.append({"id": k, "broken": ["no such requirement id in the Bible"]})
            del links[k]
    for k in gaps:
        if k not in valid_ids:
            problems.append({"id": k, "broken": ["_gaps names an id that is not in the Bible"]})
    return links, {k: v for k, v in gaps.items() if k in valid_ids}, problems


def apply_link(rec: dict, req: Requirement, ver: dict, failing: dict, evidence: dict, idf: dict) -> dict:
    """Recompute one record from a VERIFIED link. The link replaces the heuristic's code/tests/evidence for this
    requirement (an explicit human review outranks a word match) but never a recorded failure:
      failing linked test or a fail line in linked evidence that speaks to the requirement -> [!]
      code only -> [~];  code + test -> [?] (nothing shows it worked);  code + test + evidence -> [x]."""
    rec["heuristic_state"] = rec["state"]
    code: dict[str, dict] = {}
    for c in ver["code"]:
        ent = code.setdefault(c["module"], {"module": c["module"], "score": 1.0, "phase_linked": True, "terms": [], "symbols": []})
        ent["symbols"].append(c["symbol"])
    rec.update(code=list(code.values()), tests=sorted({t.split("::")[0] for t in ver["tests"]}), test_functions=list(ver["tests"]),
               evidence=list(ver["evidence"]), pass_lines=[], fail_lines=[], failing_tests=[], link="honoured")
    for t in ver["tests"]:
        f, _, fn = t.partition("::")
        names = failing.get(f, [])
        if f in failing and (fn.split(".")[-1] in names or "" in names):
            rec["failing_tests"].append(t)
    for ep in ver["evidence"]:
        ev = evidence.get(ep)
        if not ev:
            continue
        rec["fail_lines"] += [{"path": ep, "line": ln} for ln, toks in ev.fail_lines if speaks_to(req.terms, toks, idf)][:3]
        rec["pass_lines"] += [{"path": ep, "line": ln} for ln, toks in ev.pass_lines if speaks_to(req.terms, toks, idf)][:3]
    if rec["fail_lines"] or rec["failing_tests"]:
        src = rec["fail_lines"][0]["path"] if rec["fail_lines"] else rec["failing_tests"][0]
        rec["state"], rec["why"] = "[!]", f"linked; failed verdict recorded in {src}"
    elif not ver["tests"]:
        rec["state"], rec["why"] = "[~]", "linked code verified; no linked test"
    elif not ver["evidence"]:
        rec["state"], rec["why"] = "[?]", "linked code and test verified; no evidence path, so nothing shows it worked"
    else:
        rec["state"] = "[x]"
        rec["why"] = f"linked code {ver['code'][0]['module']}, test {ver['tests'][0]}, evidence {ver['evidence'][0]}"
    return rec


def genuine_gaps(records: list[dict], gaps: dict) -> list[dict]:
    """Requirements still without any code after links, each with the reviewer's reason ('' = never reviewed).
    Derived claims (Phases 38-39) are excluded: they are a consequence of other requirements, not a separate gap."""
    out = []
    for r in records:
        if r["state"] == "[ ]" and not r.get("derived"):
            out.append({"id": r["id"], "phase": r["phase"], "kind": r["kind"], "text": r["text"], "bible_line": r["bible_line"],
                        "reason": str(gaps.get(r["id"], "")), "reviewed": r["id"] in gaps})
    return out


# --- 5. rendering -------------------------------------------------------------------------------------------------------
def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def render_markdown(tr: dict) -> str:
    s = tr["summary"]
    L = ["# BIBLE TRACE", "",
         f"Generated {tr['generated']} by scripts/bible_trace.py from {tr['bible']} (sha {tr['bible_sha']}). "
         f"Indexed {tr['index']['modules']} code files, {tr['index']['test_files']} test files, {tr['index']['evidence_files']} evidence files.",
         "", "States are computed, never typed. [x] means TRACEABLE (code + a test that names the requirement + PASS evidence "
         "where the Bible asks for real data); it is a term-overlap heuristic and does not replace reading the evidence.", "",
         "## Rules", ""]
    L += [f"- {r}" for r in tr["rules"]]
    L += ["", "## Totals", "", f"{s['total']} requirements.", "", "| state | meaning | count | share |", "|---|---|---:|---:|"]
    for st in STATES:
        L.append(f"| {st} | {STATE_NAMES[st]} | {s['counts'][st]} | {100 * s['counts'][st] / max(s['total'], 1):.1f}% |")
    L += ["", "By requirement kind:", "", "| kind | " + " | ".join(STATES) + " |", "|---|" + "---:|" * len(STATES)]
    for k, c in s["per_kind"].items():
        L.append(f"| {k} | " + " | ".join(str(c[x]) for x in STATES) + " |")
    L += ["", "## Per phase", "", "| phase | title | reqs | [ ] | [~] | [?] | [x] | [!] | linked code lines | Bible range |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for k, p in s["per_phase"].items():
        c = p["counts"]
        rng = f"{p['est_range'][0]:,}-{p['est_range'][1]:,}" if p["est_range"][1] else "-"
        L.append(f"| {k} | {_trunc(p['title'], 40)} | {p['requirements']} | " + " | ".join(str(c[x]) for x in STATES) + f" | {p['linked_code_lines']:,} | {rng} |")
    L += ["", "## The 20 most important MISSING items", "",
          "Ranked by Phase 43's priority order (anti-leak first), then leak/gate/fail-closed wording; at most "
          f"{MAX_PER_PHASE_IN_TOP} per phase.", ""]
    for i, m in enumerate(tr["top_missing"], 1):
        L.append(f"{i}. `{m['id']}` (phase {m['phase']}, Bible line {m['bible_line']}): {_trunc(m['text'], 150)}")
    if not tr["top_missing"]:
        L.append("None: every requirement has at least linked code.")
    fails = [r for r in tr["requirements"] if r["state"] == "[!]"]
    L += ["", f"## Failed ({len(fails)})", ""]
    L += [f"- `{r['id']}` {_trunc(r['text'], 100)} -> {r['why']}" for r in fails] or ["None recorded."]
    L += ["", f"## Bible claims the trace cannot support ({len(tr['claim_mismatches'])})", ""]
    L += [f"- `{m['id']}` {m['label']} claim {m['claim']} vs computed {m['state']}: {m['note']}" for m in tr["claim_mismatches"][:MAX_MISMATCH_LINES]] or ["None."]
    if len(tr["claim_mismatches"]) > MAX_MISMATCH_LINES:
        L.append(f"... and {len(tr['claim_mismatches']) - MAX_MISMATCH_LINES} more (all in bible_trace.json).")
    L += ["", "## Task-file output not on disk", ""]
    L += [f"- {p}" for p in tr["planned_files_missing"]] or ["None."]
    L += ["", "## Code no requirement links to", ""]
    L += [f"- {p}" for p in tr["orphan_modules"]] or ["None."]
    L += ["", "## Integration hooks still unchecked (code exists, not wired)", ""]
    L += [f"- {k}.py: {len(v)} pending, e.g. {_trunc(v[0], 100)}" for k, v in tr["integration_pending"].items()] or ["None."]
    L += ["", f"## Trace invariants ({len(tr['invariant_violations'])} violations)", ""]
    L += [f"- {v}" for v in tr["invariant_violations"]] or ["All hold."]
    lk = tr["links"]
    L += ["", f"## Explicit links ({lk['honoured']} honoured of {lk['declared']} declared, {len(lk['broken'])} broken)", ""]
    L += [f"- BROKEN `{b['id']}`: " + "; ".join(b["broken"]) for b in lk["broken"]] or ["No broken links."]
    if lk["changed"]:
        c = Counter(f"{x['from']} -> {x['to']}" for x in lk["changed"])
        L += ["", "State changes caused by honoured links: " + ", ".join(f"{k} x{v}" for k, v in sorted(c.items())), ""]
    gp = tr["genuine_gaps"]
    L += ["", f"## Genuine gaps ({len(gp)})", "",
          "Requirements with no implementing code after explicit links were applied and the code was read. "
          "A reason means a reviewer confirmed it is missing; 'UNREVIEWED' means the word match found nothing and nobody has checked.", ""]
    for g in gp:
        L.append(f"- `{g['id']}` (phase {g['phase']}, Bible line {g['bible_line']}, {g['kind']}): {_trunc(g['text'], 110)} -- "
                 + (g["reason"] if g["reviewed"] else "UNREVIEWED"))
    if not gp:
        L.append("None.")
    L += ["", "## Detail by phase", ""]
    cur = None
    for r in tr["requirements"]:
        if r["phase"] != cur:
            cur = r["phase"]
            L += ["", f"### Phase {cur}: {s['per_phase'][str(cur)]['title']}", "", "| id | state | requirement | code | tests | evidence |", "|---|---|---|---|---:|---:|"]
        code = "<br>".join(f"{c['module']}" + (f"::{c['symbols'][0]}" if c["symbols"] else "") for c in r["code"][:2]) or "-"
        L.append(f"| {r['id']} | {r['state']} | {_trunc(r['text'].replace('|', '/'), 100)} | {code} | {len(r['tests'])} | {len(r['evidence'])} |")
    return "\n".join(L) + "\n"


def write_outputs(tr: dict, md_path: Path, json_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(tr), encoding="utf-8", newline="\n")
    json_path.write_text(json.dumps(tr, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--root", default=str(root))
    ap.add_argument("--now", default="", help="label for the report header (default: current UTC date)")
    ap.add_argument("--links", default=None, help="trace_links.json (default state/build/trace_links.json)")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    r = Path(a.root)
    if not a.now:
        import datetime as dt
        a.now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    tr = build_trace(r, now=a.now, links_path=Path(a.links) if a.links else None)
    write_outputs(tr, Path(a.out_md or r / "state" / "build" / "BIBLE_TRACE.md"), Path(a.out_json or r / "state" / "build" / "bible_trace.json"))
    c = tr["summary"]["counts"]
    print("requirements", tr["summary"]["total"], " ".join(f"{k}={v}" for k, v in c.items()))
    print("invariant violations", len(tr["invariant_violations"]), " links honoured", tr["links"]["honoured"], "of",
          tr["links"]["declared"], " broken", len(tr["links"]["broken"]), " genuine gaps", len(tr["genuine_gaps"]))
    for b in tr["links"]["broken"]:
        print("BROKEN LINK", b["id"], "; ".join(b["broken"]))
    for i, m in enumerate(tr["top_missing"], 1):
        print(f"{i:2d}. {m['id']} p{m['phase']} {_trunc(m['text'], 110)}")
    return 1 if tr["invariant_violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
