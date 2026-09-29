"""Test inventory: maps every Bible phase and every Phase-38 checklist item to the tests that exercise it, classifies
each test onto the seven pyramid levels, and reports the gaps (Bible PHASE 32 test pyramid, PHASE 31 code quality;
checklist state machine of PHASE 37/38).

Evidence for "this item is tested", strongest first:
  direct   a test file names the item (checklist id like "T11", "A13", ranges like "L2-L4", or "Phase 25"/"Phases 21-23")
           in its docstring, a comment or a test name;
  module   a test file imports an engine module that the item is mapped to (PHASE_MODULES / CHECKLIST_MAP below);
  none     neither - reported as a GAP.
The mapping tables are code, not prose, and are themselves verified: a mapped module that does not exist in engine/, a
checklist id the Bible does not contain, or a Bible checklist id missing from the map is reported as a CONFIG problem,
so the inventory cannot silently drift from the Bible it inventories.

Pyramid levels (Bible PHASE 32), assigned per test function from its name, its body and where it lives:
  1 unit  2 integration  3 historical  4 walk-forward  5 blind  6 adversarial  7 reproducibility.
Level 3 cannot be run in a unit-test file (tests may not read the caches) so it is evidenced by committed run artefacts
under state/research/ (HISTORICAL_ARTEFACTS). Each phase lists the levels appropriate to it (REQUIRED_LEVELS); a missing
level is a level gap.

Static and offline: reads the Bible and parses test files with ast; imports nothing from engine, touches no data.
Outputs: state/quality/test_inventory.json and .md (or --out DIR). Exit 1 with --fail-on-gap when any gap exists."""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Phase -> engine modules that implement it (module short names, without engine.). Phases with no module are process
# phases (checklists, definition of done, reports) whose evidence can only be direct tagging.
PHASE_MODULES = {
    0: ["baseline", "checkpoint"], 1: ["pit"], 2: ["parity", "parity_suite", "features", "candles"],
    3: ["patterns"], 4: ["pattern_lifecycle"], 5: ["pattern_bank"], 6: ["pattern_movers"], 7: [],
    8: ["analogs", "analogs_sector", "analogs_stock", "analog_weighting"], 9: ["memory"], 10: ["lessons"],
    11: ["antimemo"], 12: ["trust"], 13: ["direction", "direction_calib"], 14: ["adaptive"], 15: ["exits"],
    16: ["stops"], 17: ["timeline"], 18: ["adaptive"], 19: ["basis_search"], 20: ["objective"],
    21: ["blind_gates"], 22: ["blind_gates"], 23: ["retester"], 24: ["health"], 25: ["patterns"],
    26: ["antioverfit"], 27: ["data_sources"], 28: ["isolation", "broker"], 29: ["site_data"],
    30: ["experiment_memory", "registry"], 31: [], 32: [], 33: ["repro"], 34: ["ablation"], 35: ["champion"],
    36: ["run_report"],
}

# Checklist id -> (phase numbers, extra modules). Phases give the module set; extra modules cover items that the
# Bible files under a broader phase (e.g. T5/T6 are point-in-time data correctness = phase 1).
CHECKLIST_MAP = {
    "A1": ([], ["candles"]), "A2": ([3], []), "A2b": ([3], []), "A3": ([7], ["antioverfit"]), "A4": ([5], []),
    "A5": ([6], []), "A6": ([], []), "A7": ([19], ["adaptive"]), "A8": ([8], []), "A8b": ([], ["analogs_sector"]),
    "A8c": ([], ["analogs_stock"]), "A9": ([17], []), "A10": ([10], []), "A10b": ([11], []), "A11": ([4], []),
    "A12": ([27], []), "A13": ([25], []),
    "V1": ([6], []), "V2": ([13], []), "V2b": ([12], []), "V3": ([15], []), "V4": ([16], []), "V5": ([20], []),
    "T1": ([], ["livesim"]), "T2": ([21], []), "T3": ([20], []), "T4": ([18], []), "T5": ([1], []), "T6": ([1], []),
    "T7": ([19], []), "T8": ([], ["basis_search"]), "T9": ([23], []), "T10": ([], ["adaptive"]),
    "T11": ([], ["adaptive"]), "T12": ([24], []), "T13": ([33], ["adaptive"]), "T14": ([26], []),
    "T15": ([26], []), "T16": ([26], []), "T17": ([25], []),
    "L1": ([35], []), "L2": ([28], []), "L3": ([28], []), "L4": ([28], []), "L5": ([28], []),
}

# Levels appropriate to each phase (Bible: "the levels appropriate to it"). Default is unit + adversarial.
DEFAULT_LEVELS = (1, 6)
REQUIRED_LEVELS = {
    1: (1, 4, 6), 2: (1, 4, 6), 3: (1, 2, 4, 6, 7), 4: (1, 6), 5: (1, 6, 7), 6: (1, 2, 6), 8: (1, 4, 6),
    9: (1, 4), 13: (1, 6), 17: (1, 4, 6), 18: (1, 2, 4, 7), 20: (1, 6), 21: (1, 5, 6), 22: (1, 5, 6), 23: (1, 6, 7),
    24: (1, 6), 25: (1, 6), 26: (1, 6), 28: (1, 6), 33: (1, 7), 35: (1, 6),
}

LEVEL_NAMES = {1: "unit", 2: "integration", 3: "historical", 4: "walk-forward", 5: "blind", 6: "adversarial",
               7: "reproducibility"}

# Level 3 evidence: committed run artefacts (globs under state/) proving the component was run on real history.
HISTORICAL_ARTEFACTS = {
    3: ["research/algorithm/test_*.json"], 25: ["research/algorithm/planted_calibration.json"],
    18: ["research/tuning_summary.json"], 20: ["research/backtest_summary.json"], 16: ["research/exits_stops_run.log"],
    15: ["research/exits_stops_run.log"], 19: ["research/frontier.json"], 8: ["research/three_way.json"],
}

_LEVEL_WORDS = {
    4: ("walk", "forward", "chronolog", "expanding", "as_of", "look_ahead", "lookahead", "point_in_time", "truncat",
        "future", "causal"),
    5: ("blind", "disguise", "hidden", "sealed"),
    6: ("leak", "plant", "scramble", "shuffle", "permut", "peek", "defect", "adversar", "catch", "refuse", "reject",
        "tamper", "corrupt", "fake", "cannot_fail", "can_fail", "control"),
    7: ("reproduc", "determin", "twice", "same_seed", "identical", "idempot", "repeat"),
    3: ("historical", "known_history", "real_cache", "archive"),
}


@dataclass
class SuiteFile:
    """One parsed test file. (Not named Test*: pytest would try to collect it when this script is imported.)"""
    path: str
    doc: str
    tests: dict = field(default_factory=dict)          # test name -> sorted list of levels
    engine_imports: set = field(default_factory=set)
    phase_tags: set = field(default_factory=set)
    item_tags: set = field(default_factory=set)
    canon_tags: set = field(default_factory=set)
    location_level: int | None = None                  # 2 for tests/integration


# ------------------------------------------------------------------ Bible parsing
_PHASE_RE = re.compile(r"^#\s*PHASE\s+(\d+)\s*[—–-]\s*(.+?)\s*$", re.M)
_ITEM_RE = re.compile(r"^\*\s*\[(?P<st>[^\]]*)\]\s*(?P<id>[ATVL]\d+[a-z]?)\s+(?P<title>.+?)\s*$", re.M)
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)


def parse_bible(text: str) -> dict:
    """Phases (number -> title, line) and Phase-38 checklist items (id, state, title, section)."""
    phases = {}
    for m in _PHASE_RE.finditer(text):
        phases[int(m.group(1))] = {"title": m.group(2), "line": text.count("\n", 0, m.start()) + 1}
    start = text.find("# PHASE 38")
    end = text.find("# PHASE 39", start) if start >= 0 else -1
    block = text[start:end] if start >= 0 and end > start else ""
    items, sections = [], [(m.start(), m.group(1)) for m in _SECTION_RE.finditer(block)]
    for m in _ITEM_RE.finditer(block):
        sec = [s for pos, s in sections if pos < m.start()]
        items.append({"id": m.group("id"), "state": m.group("st").strip(), "title": m.group("title"),
                      "section": sec[-1] if sec else ""})
    return {"phases": phases, "items": items}


# ------------------------------------------------------------------ tag extraction
def _expand_range(letter: str, a: int, b: int) -> list[str]:
    lo, hi = sorted((a, b))
    return [f"{letter}{n}" for n in range(lo, min(hi, lo + 40) + 1)]


def extract_tags(text: str) -> tuple[set, set, set]:
    """(phase numbers, checklist ids, canon ids) named in free text. Understands 'Phase 25', 'Phases 21-23',
    'PHASES 6 and 34', 'L2-L4', 'A13/T17', 'T11/T13'. Lower-case ids with a letter suffix (A2b) are kept."""
    phases, items = set(), set()
    for m in re.finditer(r"phases?\s+(\d+(?:\s*(?:-|–|,|/|and|&)\s*\d+)*)", text, re.I):
        group = m.group(1)
        for a, b in re.findall(r"(\d+)\s*[-–]\s*(\d+)", group):          # 21-23 is a range ...
            phases.update(range(min(int(a), int(b)), max(int(a), int(b)) + 1))
        rest = re.sub(r"\d+\s*[-–]\s*\d+", " ", group)
        phases.update(int(n) for n in re.findall(r"\d+", rest))          # ... 6 and 34 / 6, 34 are lists
    for m in re.finditer(r"\b([ATVL])(\d{1,2})\s*(?:-|–)\s*\1?(\d{1,2})\b", text):    # L2-L4 / T14-T16
        items.update(_expand_range(m.group(1), int(m.group(2)), int(m.group(3))))
    for m in re.finditer(r"\b([ATVL]\d{1,2}[a-c]?)\b", text):
        items.add(m.group(1))
    canon = set(re.findall(r"\bC(\d{1,3})\b", text))
    return phases, items, canon


def _level_words(name: str) -> set:
    low = name.lower()
    return {lv for lv, words in _LEVEL_WORDS.items() if any(w in low for w in words)}


def scan_file(path: Path, root: Path) -> SuiteFile | None:
    """Parse one test file; None when it cannot be parsed (the quality gate reports syntax errors separately)."""
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None
    rel = path.relative_to(root).as_posix()
    doc = ast.get_docstring(tree) or ""
    sf = SuiteFile(rel, doc)
    if "/integration/" in f"/{rel}":
        sf.location_level = 2
    aliases = {}                                        # local name -> engine module short name
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "engine" and len(parts) == 1:
                for a in node.names:
                    aliases[a.asname or a.name] = a.name
            elif parts[0] == "engine" and len(parts) > 1:
                aliases[parts[1]] = parts[1]
                sf.engine_imports.add(parts[1])
        elif isinstance(node, ast.Import):
            for a in node.names:
                p = a.name.split(".")
                if p[0] == "engine" and len(p) > 1:
                    aliases[a.asname or p[1]] = p[1]
    sf.engine_imports |= set(aliases.values())
    text_tags = [doc] + re.findall(r"#\s*(.+)", src)
    for chunk in text_tags:
        p, i, c = extract_tags(chunk)
        sf.phase_tags |= p; sf.item_tags |= i; sf.canon_tags |= c
    for fn in ast.walk(tree):
        if not (isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test")):
            continue
        p, i, c = extract_tags(ast.get_docstring(fn) or "")
        sf.phase_tags |= p; sf.item_tags |= i; sf.canon_tags |= c
        used = {aliases[n.id] for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id in aliases}
        used |= {aliases[n.value.id] for n in ast.walk(fn)
                 if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases}
        levels = _level_words(fn.name) | _level_words(ast.get_docstring(fn) or "")
        if sf.location_level == 2 or len(used) >= 2:
            levels.add(2)
        if not levels & {2, 3, 4, 5, 6, 7} or not levels:
            levels.add(1)
        sf.tests[fn.name] = sorted(levels)
    return sf


def scan_suite(tests_dir: Path, root: Path) -> list[SuiteFile]:
    """Every test_*.py under tests_dir (recursively), parsed. Order is stable (sorted paths)."""
    out = []
    for p in sorted(tests_dir.rglob("test_*.py")):
        sf = scan_file(p, root)
        if sf is not None:
            out.append(sf)
    return out


# ------------------------------------------------------------------ inventory
def _modules_present(engine_dir: Path) -> set:
    return {p.stem for p in engine_dir.glob("*.py")} if engine_dir.exists() else set()


def _historical(state_dir: Path, phase: int) -> list[str]:
    hits = []
    for pat in HISTORICAL_ARTEFACTS.get(phase, []):
        hits += [p.relative_to(state_dir).as_posix() for p in sorted(state_dir.glob(pat))]
    return hits


def _evidence(files, phases, modules, item_id=None):
    direct, mod = [], []
    for f in files:
        hit = (item_id and item_id in f.item_tags) or (f.phase_tags & set(phases))
        if hit:
            direct.append(f)
        elif f.engine_imports & set(modules):
            mod.append(f)
    return direct, mod


def build_inventory(bible_text: str, files: list[SuiteFile], engine_dir: Path, state_dir: Path | None = None) -> dict:
    """The full inventory dict: phases, items, level matrix, gaps, config problems. Pure given its inputs."""
    bible = parse_bible(bible_text)
    present = _modules_present(engine_dir)
    config_problems = []
    for ph, mods in PHASE_MODULES.items():
        config_problems += [f"PHASE_MODULES[{ph}] names missing module engine/{m}.py" for m in mods if m not in present]
        if ph not in bible["phases"]:
            config_problems.append(f"PHASE_MODULES has phase {ph} which the Bible does not contain")
    bible_ids = {i["id"] for i in bible["items"]}
    config_problems += [f"CHECKLIST_MAP has {i} which the Bible checklist does not contain" for i in CHECKLIST_MAP
                        if i not in bible_ids]
    config_problems += [f"Bible checklist item {i} has no CHECKLIST_MAP entry" for i in sorted(bible_ids - set(CHECKLIST_MAP))]
    for i, (_, extra) in CHECKLIST_MAP.items():
        config_problems += [f"CHECKLIST_MAP[{i}] names missing module engine/{m}.py" for m in extra if m not in present]

    phase_rows = {}
    for ph, meta in sorted(bible["phases"].items()):
        mods = PHASE_MODULES.get(ph, [])
        direct, mod = _evidence(files, [ph], mods)
        levels = {}
        for f in direct + mod:
            for name, lv in f.tests.items():
                for l in lv:
                    levels.setdefault(l, []).append(f"{f.path}::{name}")
        if state_dir is not None and _historical(state_dir, ph):
            levels.setdefault(3, []).extend(_historical(state_dir, ph))
        required = REQUIRED_LEVELS.get(ph, DEFAULT_LEVELS) if mods else ()
        phase_rows[ph] = {
            "title": meta["title"], "modules": mods, "direct": [f.path for f in direct], "module": [f.path for f in mod],
            "n_tests": sum(len(f.tests) for f in direct + mod), "required_levels": list(required),
            "levels": {str(l): len(v) for l, v in sorted(levels.items())},
            "missing_levels": [l for l in required if l not in levels],
            "status": "direct" if direct else "module" if mod else ("process" if not mods else "none"),
        }

    item_rows = []
    for it in bible["items"]:
        phs, extra = CHECKLIST_MAP.get(it["id"], ([], []))
        mods = sorted({m for p in phs for m in PHASE_MODULES.get(p, [])} | set(extra))
        direct, mod = _evidence(files, [], mods, item_id=it["id"])
        for f in files:                                  # phase-level tagging counts for the items filed under it
            if f.phase_tags & set(phs) and f not in direct:
                direct.append(f)
        mod = [f for f in mod if f not in direct]
        status = "direct" if direct else "module" if mod else "none"
        item_rows.append({**it, "phases": phs, "modules": mods, "status": status,
                          "tests": sorted({f.path for f in direct + mod}),
                          "n_tests": sum(len(f.tests) for f in direct + mod)})

    tagged = set().union(*[set(r["direct"]) | set(r["module"]) for r in phase_rows.values()]) if phase_rows else set()
    tagged |= {t for r in item_rows for t in r["tests"]}
    orphans = sorted(f.path for f in files if f.path not in tagged)
    gaps = {
        "phases_untested": [p for p, r in phase_rows.items() if r["status"] == "none"],
        "items_untested": [r["id"] for r in item_rows if r["status"] == "none"],
        "items_module_only": [r["id"] for r in item_rows if r["status"] == "module"],
        "level_gaps": {p: r["missing_levels"] for p, r in phase_rows.items() if r["missing_levels"]},
        "orphan_test_files": orphans,
        "config_problems": config_problems,
    }
    total = [t for f in files for t in f.tests.values()]
    hist = {l: sum(l in t for t in total) for l in LEVEL_NAMES}
    return {"n_files": len(files), "n_tests": len(total), "level_counts": {LEVEL_NAMES[l]: n for l, n in hist.items()},
            "phases": phase_rows, "items": item_rows, "gaps": gaps,
            "gap_count": sum(len(gaps[k]) for k in ("phases_untested", "items_untested", "config_problems")) +
                         len(gaps["level_gaps"])}


# ------------------------------------------------------------------ report
def render_markdown(inv: dict) -> str:
    L = ["# Test inventory", "",
         f"{inv['n_files']} test files, {inv['n_tests']} tests. Level counts: " +
         ", ".join(f"{k} {v}" for k, v in inv["level_counts"].items()), "", "## Phases", "",
         "| phase | title | status | tests | levels present | missing |", "|---|---|---|---|---|---|"]
    for p, r in inv["phases"].items():
        lv = ",".join(r["levels"]) or "-"
        L.append(f"| {p} | {r['title'][:44]} | {r['status']} | {r['n_tests']} | {lv} | "
                 f"{','.join(map(str, r['missing_levels'])) or '-'} |")
    L += ["", "## Checklist items", "", "| id | item | status | tests |", "|---|---|---|---|"]
    for r in inv["items"]:
        L.append(f"| {r['id']} | {r['title'][:50]} | {r['status']} | {r['n_tests']} |")
    g = inv["gaps"]
    L += ["", "## Gaps", "", f"- phases with modules but no test: {g['phases_untested'] or 'none'}",
          f"- checklist items with no test: {g['items_untested'] or 'none'}",
          f"- checklist items evidenced only by importing the module (no explicit tag): {g['items_module_only'] or 'none'}",
          f"- missing pyramid levels by phase: {g['level_gaps'] or 'none'}",
          f"- test files matching no phase or item: {g['orphan_test_files'] or 'none'}",
          f"- inventory config problems: {g['config_problems'] or 'none'}", ""]
    return "\n".join(L)


def run(root: Path, bible: Path | None = None, tests: Path | None = None, out: Path | None = None,
        write: bool = True) -> dict:
    """Scan root/tests against root/BIBLE.md; optionally write state/quality/test_inventory.{json,md}."""
    bible = bible or root / "BIBLE.md"
    tests = tests or root / "tests"
    inv = build_inventory(bible.read_text(encoding="utf-8"), scan_suite(tests, root), root / "engine", root / "state")
    if write:
        out = out or root / "state" / "quality"
        out.mkdir(parents=True, exist_ok=True)
        (out / "test_inventory.json").write_text(json.dumps(inv, indent=1, default=list), encoding="utf-8")
        (out / "test_inventory.md").write_text(render_markdown(inv), encoding="utf-8")
    return inv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--bible", type=Path)
    ap.add_argument("--tests", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fail-on-gap", action="store_true", help="exit 1 when any gap or config problem exists")
    a = ap.parse_args(argv)
    inv = run(a.root, a.bible, a.tests, a.out)
    g = inv["gaps"]
    print(f"{inv['n_files']} files, {inv['n_tests']} tests; untested phases {g['phases_untested']}, "
          f"untested items {g['items_untested']}, level gaps in {len(g['level_gaps'])} phases, "
          f"config problems {len(g['config_problems'])}")
    return 1 if a.fail_on_gap and inv["gap_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
