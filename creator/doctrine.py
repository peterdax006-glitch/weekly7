"""Doctrine compliance (C77 sections 0-5, 7, 8, 62, 63, 65, 67, 69, 70, 81, 82, 84-87): every process rule gets a verdict.

The 28 doctrine sections of CREATOR_MASTER_PROMPT.md have no measurable member items, so each is split into its concrete rules and
each rule is ONE of:

  MAPPED  - enforced by code that already exists: every reference "path::symbol" must exist, and every cited test must pass in the
            current junit evidence (state/creator/junit); a cited test that has no junit result is UNPROVEN, never assumed passing
  CHECK   - a small computed check below (reads the code, the ledger evidence under state/creator, the capability states); it
            returns PASS / FAIL / UNPROVEN / PENDING with its evidence text; tests/test_creator_doctrine.py exercises each check
  INTENT  - a pure statement of intent with nothing to compute; it is listed honestly as NOT_CHECKABLE and caps its section at
            IMPLEMENTED

Section roll-up (roll_up): any FAIL -> FAILED; else any UNPROVEN -> IN_PROGRESS; else any PENDING / NOT_CHECKABLE -> IMPLEMENTED;
else TESTING. Nothing here ever yields VALIDATED: only an independent validator may. PENDING marks an OUTCOME the rule asks for
that has not happened yet (Claude's share has not fallen, no recursion step is in the ledger, creator/goals.py is not merged)."""
from __future__ import annotations

import json
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

PASS, FAIL, UNPROVEN, PENDING, NOT_CHECKABLE = "PASS", "FAIL", "UNPROVEN", "PENDING", "NOT_CHECKABLE"
MAPPED, CHECK, INTENT = "MAPPED", "CHECK", "INTENT"
ROOT = Path(__file__).resolve().parents[1]
JUNIT = Path("state") / "creator" / "junit"
MAX_PACKAGE_FILES = 8          # work package granularity (sec 63): one cycle's change touches at most this many files ...
MAX_PACKAGE_DIFF_LINES = 1200  # ... and at most this many added+removed lines; larger is a broad phase, not a package


@dataclass(frozen=True)
class Rule:
    section: str
    rule_id: str
    text: str
    kind: str
    refs: tuple[str, ...] = ()
    check: str = ""
    args: tuple[Any, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Result:
    rule: Rule
    status: str
    evidence: str


# ---------------------------------------------------------------------------------------------------- evidence readers
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cycles(root: Path) -> list[dict[str, Any]]:
    """One row per recorded kernel cycle package under state/creator/cycles: outcome, verdict, effect, changed files, diff size."""
    out = []
    for d in sorted((root / "state" / "creator" / "cycles").glob("*")):
        cyc = _read_json(d / "cycle.json")
        if not isinstance(cyc, dict):
            continue
        ev = _read_json(d / "evaluation.json") or {}
        patch = d / "diff.patch"
        lines = 0
        if patch.is_file():
            lines = sum(1 for ln in patch.read_text(encoding="utf-8", errors="replace").splitlines()
                        if ln[:1] in "+-" and not ln.startswith(("+++", "---")))
        detail = (cyc.get("details") or {}).get("detail") or {}
        out.append({"package": cyc.get("package", d.name), "outcome": cyc.get("outcome"), "requirement": cyc.get("requirement", ""),
                    "verdict": cyc.get("verdict", ""), "lo": detail.get("lo"), "min_effect": detail.get("min_effect", 0.0),
                    "files": sorted((ev.get("changed") or {}) if isinstance(ev, dict) else []), "diff_lines": lines,
                    "net": _net_lines(patch)})
    return out


def _net_lines(patch: Path) -> int:
    if not patch.is_file():
        return 0
    add = rem = 0
    for ln in patch.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("+") and not ln.startswith("+++") and ln[1:].strip() and not ln[1:].strip().startswith("#"):
            add += 1
        elif ln.startswith("-") and not ln.startswith("---") and ln[1:].strip() and not ln[1:].strip().startswith("#"):
            rem += 1
    return add - rem


def _status(root: Path) -> dict[str, Any]:
    st = _read_json(root / "state" / "creator" / "STATUS.json")
    return st if isinstance(st, dict) else {}


# ---------------------------------------------------------------------------------------------------- computed checks
def check_status_states(root: Path) -> tuple[str, str]:
    """Sec 7: the eleven states exist, are distinct, and VALIDATED is reachable only through the full chain."""
    from creator import model as M
    want = ["NOT_STARTED", "IN_PROGRESS", "IMPLEMENTED", "TESTED", "INTENDED_BEHAVIOR_VERIFIED", "VALIDATED", "FAILED", "BLOCKED",
            "SCIENTIFICALLY_LIMITED", "REJECTED", "ROLLED_BACK"]
    have = [s.value for s in M.Status]
    if have != want:
        return FAIL, f"states differ from sec 7: missing {sorted(set(want) - set(have))} extra {sorted(set(have) - set(want))}"
    direct = sorted(s.value for s, nxt in M.LEGAL.items() if M.Status.VALIDATED in nxt)
    if direct != ["INTENDED_BEHAVIOR_VERIFIED"]:
        return FAIL, f"VALIDATED is reachable directly from {direct}"
    if M.Status.IMPLEMENTED in M.DONE_STATES:
        return FAIL, "IMPLEMENTED counts as done"
    return PASS, "11 distinct states; VALIDATED only from INTENDED_BEHAVIOR_VERIFIED; IMPLEMENTED is not a done state"


def check_execution_order(root: Path) -> tuple[str, str]:
    """Sec 8: the legal path from NOT_STARTED to VALIDATED is the full chain; no state may be skipped."""
    from creator import model as M
    path = [s.value for s in M.reachable(M.Status.NOT_STARTED, M.Status.VALIDATED)]
    want = ["NOT_STARTED", "IN_PROGRESS", "IMPLEMENTED", "TESTED", "INTENDED_BEHAVIOR_VERIFIED", "VALIDATED"]
    if path != want:
        return FAIL, f"shortest legal path to VALIDATED is {path}, expected {want}"
    return PASS, "shortest legal path to VALIDATED: " + " -> ".join(path)


def check_limitation_recordable(root: Path) -> tuple[str, str]:
    """Sec 70: an impossible requirement is recorded as a limitation (a legal state), never silently passed."""
    from creator import model as M
    ok = [s for s in (M.Status.IN_PROGRESS, M.Status.FAILED) if M.Status.SCIENTIFICALLY_LIMITED in M.LEGAL[s]]
    if len(ok) != 2:
        return FAIL, "SCIENTIFICALLY_LIMITED is not reachable from both IN_PROGRESS and FAILED"
    if M.Status.VALIDATED in M.LEGAL[M.Status.SCIENTIFICALLY_LIMITED]:
        return FAIL, "a limitation can be turned straight into VALIDATED"
    return PASS, "SCIENTIFICALLY_LIMITED is legal from IN_PROGRESS and FAILED and cannot become VALIDATED directly"


def check_package_fields(root: Path) -> tuple[str, str]:
    """Sec 63: the WorkPackage record carries every field the section lists (STATUS is the ledger state of the stateful record)."""
    import dataclasses
    from creator import model as M
    spec = {"PACKAGE ID": "package_id", "OBJECTIVE": "objective", "WHY IT EXISTS": "why_it_exists", "PREREQUISITES": "prerequisites",
            "INPUTS": "inputs", "IMPLEMENTATION REQUIREMENTS": "implementation_requirements", "INTERFACES": "interfaces",
            "DATA FLOW": "data_flow", "DEPENDENCIES": "dependencies", "TEST REQUIREMENTS": "test_requirements",
            "VALIDATION REQUIREMENTS": "validation_requirements", "EXPECTED FAILURE MODES": "expected_failure_modes",
            "EVIDENCE REQUIREMENTS": "evidence_requirements", "ROLLBACK REQUIREMENTS": "rollback_requirements",
            "COMPLETION CRITERIA": "completion_criteria", "ANTI-PREMATURE-COMPLETION CRITERIA": "anti_premature_completion",
            "MEANINGFUL CODE DEPTH": "meaningful_code_depth"}
    fields = {f.name for f in dataclasses.fields(M.WorkPackage)}
    missing = sorted(k for k, v in spec.items() if v not in fields)
    if missing:
        return FAIL, f"WorkPackage lacks fields for: {missing}"
    if not M.WorkPackage.STATEFUL:
        return FAIL, "WorkPackage has no ledger status (STATUS)"
    return PASS, f"WorkPackage has all {len(spec)} listed fields and a ledger status"


def check_package_size(root: Path) -> tuple[str, str]:
    """Sec 63: real work is done in small packages - every recorded cycle changes few files and few lines."""
    rows = [c for c in cycles(root) if c["files"] or c["diff_lines"]]
    if not rows:
        return PENDING, "no recorded cycle with a change under state/creator/cycles"
    big = [f"{c['package']}({len(c['files'])} files, {c['diff_lines']} lines)" for c in rows
           if len(c["files"]) > MAX_PACKAGE_FILES or c["diff_lines"] > MAX_PACKAGE_DIFF_LINES]
    if big:
        return FAIL, f"packages above {MAX_PACKAGE_FILES} files / {MAX_PACKAGE_DIFF_LINES} lines: {big}"
    return PASS, (f"{len(rows)} packages; max files {max(len(c['files']) for c in rows)}, max diff lines "
                  f"{max(c['diff_lines'] for c in rows)} (bounds {MAX_PACKAGE_FILES} / {MAX_PACKAGE_DIFF_LINES})")


def check_self_target(root: Path) -> tuple[str, str]:
    """Secs 0, 2, 82: the first autonomous target is the system itself - no recorded change touches the downstream Weekly7 engine."""
    rows = [c for c in cycles(root) if c["files"]]
    if not rows:
        return PENDING, "no recorded cycle changed any file"
    downstream = sorted({f for c in rows for f in c["files"] if f.startswith(("engine/", "state/livesim/"))})
    creator = sum(1 for c in rows if any(f.startswith(("creator/", "tests/", "scripts/")) for f in c["files"]))
    if downstream:
        return FAIL, f"cycles changed downstream files: {downstream[:6]}"
    return PASS, f"{len(rows)} cycles changed files; {creator} touch the Creator (creator/ tests/ scripts/); 0 touch engine/ or state/livesim/"


def check_work_traces_to_requirements(root: Path) -> tuple[str, str]:
    """Sec 4: every cycle works on a requirement of the objective (a component step or a self-efficiency requirement)."""
    from creator import objective as O
    rows = [c for c in cycles(root) if c["requirement"]]
    if not rows:
        return PENDING, "no recorded cycle names a requirement"
    pat = re.compile(r"^(K\d{2})\.([a-z_]+)$|^EFF\.[a-z]+$")
    bad = []
    for c in rows:
        m = pat.match(str(c["requirement"]))
        if not m or (m.group(2) is not None and m.group(2) not in O.CHECKS):
            bad.append(f"{c['package']}:{c['requirement']}")
    if bad:
        return FAIL, f"cycles on requirements that are not in the objective: {bad[:6]}"
    return PASS, f"all {len(rows)} cycles name a requirement of the objective (steps {sorted(O.CHECKS)} or EFF.*)"


def check_complexity_vs_capability(root: Path) -> tuple[str, str]:
    """Sec 69: an adopted change must carry a measured gain (verdict IMPROVEMENT, lower confidence bound above the minimum
    effect); size and number of changes alone never decide. Reports the net size delta beside the measured gain."""
    adopted = [c for c in cycles(root) if c["outcome"] == "ADOPTED"]
    if not adopted:
        return PENDING, "no adopted cycle recorded"
    bad = [c["package"] for c in adopted
           if c["verdict"] != "IMPROVEMENT" or c["lo"] is None or float(c["lo"]) <= float(c["min_effect"] or 0.0)]
    net = sum(c["net"] for c in adopted)
    if bad:
        return FAIL, f"adopted without a measured gain: {bad}"
    return PASS, f"{len(adopted)} adopted cycles, each IMPROVEMENT with confidence lower bound above the minimum effect; net code lines {net:+d}"


def check_depth_is_behaviour(root: Path) -> tuple[str, str]:
    """Sec 62 + the owner's ruling of 1 Oct 2026 ('capability, not lines'): line floors are dropped; every component's depth is
    behaviour proven by its declared tests; its state is computed from evidence, never declared."""
    caps = _read_json(root / "creator" / "capabilities.json")
    if not isinstance(caps, dict):
        return FAIL, "creator/capabilities.json unreadable"
    rows = caps.get("capabilities", [])
    floors = [c["id"] for c in rows if int(c.get("floor", 0)) != 0]
    untested = [c["id"] for c in rows if not c.get("tests")]
    declared_state = [c["id"] for c in rows if "state" in c]
    if floors or untested or declared_state:
        return FAIL, f"floors != 0: {floors}; no tests: {untested}; state written in the declaration: {declared_state}"
    return PASS, f"{len(rows)} capabilities: floor 0 (lines dropped), each declares tests, none declares its own state"


def check_meaningful_ignores_padding(root: Path) -> tuple[str, str]:
    """Sec 62 'meaningful lines only': the size ruler does not move when blank lines, comments or docstrings are added."""
    from creator import efficiency as E
    from creator import selfmodel as SM
    base = "def f(x):\n    return x + 1\n\n\nclass C:\n    def m(self):\n        return f(1)\n"
    pad = ('"""module doc"""\n\n# a comment\n\ndef f(x):\n    """doc"""\n\n    # more\n    return x + 1\n\n\n\n'
           'class C:\n    """doc"""\n\n    def m(self):\n        """doc"""\n        # note\n        return f(1)\n')
    if E.ast_size(base) != E.ast_size(pad):
        return FAIL, f"ast_size moved with padding: {E.ast_size(base)} -> {E.ast_size(pad)}"
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.py", Path(td) / "b.py"
        a.write_text(base, encoding="utf-8")
        b.write_text(pad, encoding="utf-8")
        la, lb = SM.meaningful_lines(a), SM.meaningful_lines(b)
    if la != lb:
        return FAIL, f"meaningful_lines moved with padding: {la} -> {lb}"
    return PASS, f"efficiency.ast_size and selfmodel.meaningful_lines are unchanged by blanks, comments and docstrings ({la} lines)"


def check_capability_tested(root: Path, *ids: str) -> tuple[str, str]:
    """The named components exist with their own tests passing on the current source (STATUS.json, computed by creator/selfmodel.py)."""
    caps = {c["id"]: c for c in _status(root).get("capabilities", [])}
    if not caps:
        return UNPROVEN, "state/creator/STATUS.json has no capability states"
    states = {i: caps.get(i, {}).get("state", "MISSING") for i in ids}
    low = {i: s for i, s in states.items() if s != "TESTED"}
    if low:
        return UNPROVEN, f"not TESTED yet: {low}"
    return PASS, ", ".join(f"{i}=TESTED" for i in ids)


def check_claude_share(root: Path) -> tuple[str, str]:
    """Secs 3, 67, 86 (outcome): Claude's share of adopted work must fall until the students are the primary developer."""
    cur = _status(root).get("curriculum", {})
    share = cur.get("teacher_share")
    if share is None:
        return PENDING, "no adopted work yet, teacher_share is undefined"
    if float(share) < 0.5:
        return PASS, f"teacher_share {share} (< 0.5): the students adopt most of the work ({cur.get('adopted_by')})"
    return PENDING, f"teacher_share {share}: the teacher still does most adopted work ({cur.get('adopted_by')}); the rule asks it to fall"


def check_recursion_demonstrated(root: Path) -> tuple[str, str]:
    """Sec 85/87 (outcome): the Creator has actually improved its own development process at least once, recorded in the ledger."""
    from creator import recursion as R
    path = root / "state" / "creator" / "ledger.jsonl"
    n = 0
    if path.is_file():
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if R.STEP_TAG in ln and '"Finding"' in ln:
                n += 1
    if n == 0:
        return PENDING, "no RECURSION_STEP finding is in state/creator/ledger.jsonl"
    return PASS, f"{n} recursion step record(s) in the ledger"


def check_goals_pipeline(root: Path) -> tuple[str, str]:
    """Sec 81: when the written checklist runs out, the objective still produces next work. Another engineer builds the proposal
    pipeline as creator/goals.py (branch h9/goals); until it is merged here the rule is PENDING, not assumed."""
    if (root / "creator" / "goals.py").is_file():
        return PASS, "creator/goals.py exists"
    return PENDING, "creator/goals.py (the goal-proposal pipeline, branch h9/goals) is not merged into this tree"


def check_authoritative_documents(root: Path) -> tuple[str, str]:
    """Sec 5: the authoritative contracts exist, are non-empty and sit in the sandbox's protected set."""
    from creator import sandbox as S
    docs = ["BLUEPRINT.md", "SELF_LEARNING_CONTRACT.md", "RESEARCH_BRAIN_CONTRACT.md", "PREDICTION_ERROR_ADDITION.md", "BIBLE.md",
            "TEN_HOUR_EXECUTION_CHECKLIST.md", "ULTIMATE_MASTER_PROMPT.md", "MASTER_EXECUTION_PROMPT.md", "CREATOR_MASTER_PROMPT.md"]
    absent = [d for d in docs if not (root / d).is_file() or (root / d).stat().st_size == 0]
    if not (root / "canon" / "CANON.md").is_file():
        absent.append("canon/CANON.md")
    unprotected = [d for d in docs if d != "BLUEPRINT.md" and not S.is_protected(d)]
    if absent or unprotected:
        return FAIL, f"missing or empty: {absent}; not protected from the Creator: {unprotected}"
    return PASS, f"{len(docs)} contract documents + canon/CANON.md present; all contracts are sandbox-protected"


def check_subsections(root: Path, *sections: str) -> tuple[str, str]:
    """A preamble section holds exactly when the sub-sections it introduces hold (worst result wins)."""
    res = [r for r in evaluate(root, only=sections)]
    worst = _worst([r.status for r in res])
    bad = [r.rule.rule_id for r in res if r.status != PASS]
    return worst, f"{len(res)} rules in {', '.join(sections)}; not PASS: {bad}" if bad else f"all {len(res)} rules in {', '.join(sections)} PASS"


LOOP_SYMBOLS = (  # sec 65: stage -> where it is implemented
    ("RECEIVE/UNDERSTAND OBJECTIVE", "creator/objective.py::parse_objective"), ("COMPILE REQUIREMENTS", "creator/objective.py::compile_capabilities"),
    ("INSPECT CURRENT SYSTEM", "creator/selfmodel.py::build"), ("COMPARE REQUIREMENTS TO CAPABILITIES", "creator/gaps.py::assess"),
    ("IDENTIFY/CLASSIFY GAPS", "creator/gaps.py::sync"), ("RESEARCH UNKNOWN INFORMATION", "creator/research.py::generate_questions"),
    ("GENERATE DESIGN OPTIONS", "creator/design.py::generate_options"), ("SELECT STRATEGY", "creator/design.py::select"),
    ("DECOMPOSE/PLAN", "creator/planner.py::plan_next"), ("ALLOCATE RESOURCES", "creator/schedule.py::schedule"),
    ("IMPLEMENT", "creator/kernel.py::execute"), ("BUILD", "creator/build.py::run_build"), ("GENERATE TESTS", "creator/testgen.py::generate"),
    ("RUN TESTS", "creator/testrun.py::run_pytest"), ("EVALUATE AGAINST OBJECTIVE", "creator/evaluate.py::compare"),
    ("DIAGNOSE ROOT CAUSES", "creator/debug.py::diagnose"), ("DESIGN REPAIRS", "creator/debug.py::propose_repairs"),
    ("REGRESSION TEST", "creator/testrun.py::compare_runs"), ("MEASURE IMPROVEMENT", "creator/model.py::improvement_verdict"),
    ("ADOPT OR ROLLBACK", "creator/sandbox.py::Sandbox"), ("RECORD DEVELOPMENT KNOWLEDGE", "creator/curriculum.py::LessonLog"),
    ("EVALUATE/IMPROVE STRATEGY", "creator/recursion.py::step"), ("ONE CYCLE OF THE LOOP", "creator/kernel.py::cycle"))


def check_loop_stages(root: Path) -> tuple[str, str]:
    """Sec 65: every stage of the loop has an implementation at the named place."""
    missing = [f"{stage} -> {ref}" for stage, ref in LOOP_SYMBOLS if not _ref_exists(root, ref)]
    if missing:
        return FAIL, f"stages without an implementation: {missing}"
    return PASS, f"all {len(LOOP_SYMBOLS)} loop stages resolve to existing code"


CHECKS: dict[str, Callable[..., tuple[str, str]]] = {
    f.__name__[len("check_"):]: f for f in (
        check_status_states, check_execution_order, check_limitation_recordable, check_package_fields, check_package_size,
        check_self_target, check_work_traces_to_requirements, check_complexity_vs_capability, check_depth_is_behaviour,
        check_meaningful_ignores_padding, check_capability_tested, check_claude_share, check_recursion_demonstrated,
        check_goals_pipeline, check_authoritative_documents, check_subsections, check_loop_stages)}


# ---------------------------------------------------------------------------------------------------- references
def _ref_exists(root: Path, ref: str) -> bool:
    path, _, sym = ref.partition("::")
    f = root / path
    if not f.is_file():
        return False
    if not sym:
        return True
    text = f.read_text(encoding="utf-8", errors="replace")
    return re.search(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(sym)}\b|^{re.escape(sym)}\s*[:=]", text, re.M) is not None


def _junit(root: Path, test_file: str, memo: dict[str, Optional[dict[str, bool]]]) -> Optional[dict[str, bool]]:
    if test_file not in memo:
        p = root / JUNIT / (test_file.replace("/", "__") + ".xml")
        cases: Optional[dict[str, bool]] = None
        if p.is_file():
            try:
                cases = {}
                for tc in ET.parse(p).getroot().iter("testcase"):
                    bad = tc.find("failure") is not None or tc.find("error") is not None
                    skipped = tc.find("skipped") is not None
                    name = tc.get("name", "")
                    cases[name] = cases.get(name, True) and not bad and not skipped
            except ET.ParseError:
                cases = None
        memo[test_file] = cases
    return memo[test_file]


def _resolve_ref(root: Path, ref: str, memo: dict[str, Optional[dict[str, bool]]]) -> tuple[str, str]:
    if not _ref_exists(root, ref):
        return FAIL, f"{ref} does not exist"
    path, _, sym = ref.partition("::")
    if not path.startswith("tests/") or not sym:
        return PASS, ref
    cases = _junit(root, path, memo)
    if cases is None:
        return UNPROVEN, f"{ref}: no junit evidence for {path}"
    hits = {n: ok for n, ok in cases.items() if n == sym or n.startswith(sym + "[")}
    if not hits:
        return UNPROVEN, f"{ref}: not in the junit result"
    if not all(hits.values()):
        return FAIL, f"{ref}: failing or skipped in the junit result"
    return PASS, ref


def _worst(statuses: list[str]) -> str:
    for s in (FAIL, UNPROVEN, PENDING, NOT_CHECKABLE):
        if s in statuses:
            return s
    return PASS


def evaluate_rule(root: Path, rule: Rule, memo: Optional[dict[str, Optional[dict[str, bool]]]] = None) -> Result:
    memo = {} if memo is None else memo
    if rule.kind == INTENT:
        return Result(rule, NOT_CHECKABLE, rule.note)
    parts: list[tuple[str, str]] = []
    if rule.kind == CHECK:
        fn = CHECKS.get(rule.check)
        if fn is None:
            return Result(rule, FAIL, f"no such check: {rule.check}")
        try:
            parts.append(fn(root, *rule.args))
        except Exception as exc:                                    # a crashing check is a failure, never a pass
            parts.append((FAIL, f"check {rule.check} crashed: {type(exc).__name__}: {exc}"))
    parts += [_resolve_ref(root, ref, memo) for ref in rule.refs]
    status = _worst([s for s, _ in parts])
    if status == PASS:
        head = parts[0][1] + "; " if rule.kind == CHECK else ""
        return Result(rule, status, head + f"{len(rule.refs)} cited reference(s) exist and every cited test passes in the junit evidence")
    return Result(rule, status, "; ".join(w for s, w in parts if s != PASS))


def evaluate(root: Path = ROOT, only: Optional[tuple[str, ...]] = None) -> list[Result]:
    memo: dict[str, Optional[dict[str, bool]]] = {}
    return [evaluate_rule(root, r, memo) for r in RULES if only is None or r.section in only]


def roll_up(section: str, results: list[Result]) -> tuple[str, str]:
    """(checklist status, note) for one doctrine section from its rules' results; never VALIDATED."""
    mine = [r for r in results if r.rule.section == section]
    if not mine:
        return "NOT_STARTED", "no rules mapped to this section"
    counts: dict[str, int] = {}
    for r in mine:
        counts[r.status] = counts.get(r.status, 0) + 1
    worst = _worst([r.status for r in mine])
    status = {FAIL: "FAILED", UNPROVEN: "IN_PROGRESS", PENDING: "IMPLEMENTED", NOT_CHECKABLE: "IMPLEMENTED", PASS: "TESTING"}[worst]
    flagged = [f"{r.rule.rule_id} {r.status}" for r in mine if r.status != PASS][:8]
    note = f"{len(mine)} rules: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    if flagged:
        note += "; " + ", ".join(flagged)
    return status, note + "; state/build/DOCTRINE_MAP.md"


def render_map(results: list[Result], titles: dict[str, str]) -> str:
    out = ["# DOCTRINE MAP", "",
           "Generated by `python scripts/creator_checklist_status.py` from `creator/doctrine.py`. Every doctrine/process section of",
           "CREATOR_MASTER_PROMPT.md is split into its rules; each rule is MAPPED to existing code and passing tests, a computed CHECK,",
           "or an INTENT statement that is honestly NOT_CHECKABLE. Nothing here is VALIDATED. Do not edit by hand.", "",
           "Result vocabulary: PASS (computed/mapped and passing), UNPROVEN (cited test has no passing junit result), PENDING (an",
           "outcome the rule asks for has not happened), FAIL (violated), NOT_CHECKABLE (pure intent).", ""]
    for sec in dict.fromkeys(r.rule.section for r in results):
        status, note = roll_up(sec, results)
        out += [f"## {sec} - {titles.get(sec, '')}", "", f"Section status: **{status}** ({note})", "",
                "| rule | kind | result | rule text | check / evidence |", "|---|---|---|---|---|"]
        for r in (x for x in results if x.rule.section == sec):
            where = r.rule.check + ("(" + ",".join(map(str, r.rule.args)) + ")" if r.rule.args else "") if r.rule.kind == CHECK else ""
            refs = "; ".join(r.rule.refs)
            cell = " ".join(p for p in (where and f"check `{where}`", refs and f"refs `{refs}`", r.evidence) if p)
            out.append(f"| {r.rule.rule_id} | {r.rule.kind} | {r.status} | {r.rule.text} | {cell.replace('|', '/')} |")
        out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------- the rules
_T = "tests/"
_LED, _AUD, _KER, _CUR, _SBX = (_T + "test_creator_ledger.py", _T + "test_creator_audit.py", _T + "test_creator_kernel.py",
                                _T + "test_creator_curriculum.py", _T + "test_creator_sandbox.py")
_INT, _REC, _EFF, _PLN, _GAP = (_T + "test_creator_integration.py", _T + "test_creator_recursion.py", _T + "test_creator_efficiency.py",
                                _T + "test_creator_planner.py", _T + "test_creator_gaps.py")
_DOC = _T + "test_creator_doctrine.py"
_ENGINE = ("K02", "K03", "K04", "K07", "K08", "K09", "K10", "K11", "K12", "K13", "K14", "K24")      # the hierarchy of sec 0

_rules: list[Rule] = []


def _add(section: str, text: str, kind: str, refs: tuple[str, ...] = (), check: str = "", args: tuple[Any, ...] = (), note: str = "") -> None:
    n = sum(1 for r in _rules if r.section == section) + 1
    _rules.append(Rule(section, f"{section}.{n}", text, kind, refs, check, args, note))


def _check(section: str, text: str, check: str, args: tuple[Any, ...] = (), refs: tuple[str, ...] = ()) -> None:
    _add(section, text, CHECK, refs + (_DOC + f"::test_{check}",) if not any(r.startswith(_DOC) for r in refs) else refs, check, args)


def _mapped(section: str, text: str, *refs: str) -> None:
    _add(section, text, MAPPED, refs)


def _intent(section: str, text: str, why: str) -> None:
    _add(section, text, INTENT, note=why)


TITLES = {
    "CR209": "0. BUILD THE SYSTEM THAT CAN BUILD ITSELF", "CR210": "THE ONLY MAJOR THING YOU ARE BUILDING NOW IS THE CREATOR",
    "CR211": "THE CENTRAL RULE", "CR212": "1. THE MOST IMPORTANT DISTINCTION", "CR215": "2. DO NOT MAKE WEEKLY7 THE FIRST AUTONOMOUS TARGET",
    "CR216": "3. CLAUDE'S ROLE", "CR217": "Initially", "CR218": "During autonomous-development construction", "CR219": "Eventually",
    "CR220": "4. THE PRIMARY DEVELOPMENT PRINCIPLE", "CR221": "5. EXISTING PROJECT MATERIAL IS AUTHORITATIVE FOUNDATION",
    "CR222": "7. STATUS STATES ARE SACRED", "CR223": "8. THE EXECUTION ORDER", "CR278": "62. LINE-DEPTH REQUIREMENTS",
    "CR279": "MEANINGFUL LINES ONLY", "CR280": "63. WORK PACKAGE GRANULARITY", "CR281": "65. REQUIRED AUTONOMOUS LOOP",
    "CR283": "67. DO NOT MANUALLY SOLVE WHAT THE AUTONOMOUS DEVELOPER SHOULD SOLVE", "CR285": "69. DO NOT CONFUSE COMPLEXITY WITH CAPABILITY",
    "CR286": "70. FAILURE IS NOT A REASON TO LOWER THE STANDARD", "CR296": "81. WHEN THE CHECKLIST RUNS OUT",
    "CR297": "82. CURRENT DOWNSTREAM SYSTEMS ARE NOT THE FINISH LINE", "CR299": "84. FINAL OPERATING DIRECTIVE", "CR300": "DO",
    "CR301": "DO NOT", "CR302": "85. THE FINAL MISSION", "CR303": "86. NON-STOP EXECUTION RULE", "CR304": "87. ABSOLUTE FINAL RULE"}
SECTIONS = tuple(TITLES)

# --- 0 / the creator / the central rule / distinction / weekly7 not first target
s = "CR209"
_check(s, "Not asked to finish Weekly7, the stock learner, C62/C66/C67/C68 or to perfect the research system by hand", "self_target")
_check(s, "The product is an engine that understands itself, finds what it needs, researches, designs, decomposes, implements, builds, "
          "tests, evaluates, finds failures, diagnoses, repairs, retests, measures improvement, learns, improves its process, repeats",
       "capability_tested", _ENGINE, (_INT + "::test_the_whole_pipeline_from_objective_to_resolved_lesson",))
_check(s, "Each stage of the hierarchy exists as code", "loop_stages")
_intent(s, "CLAUDE builds a rough but sound foundation from which the engine grows", "a statement of who builds what; the later rules of "
        "section 3 (CR216-CR219) carry the checkable part")
s = "CR210"
_check(s, "The only major thing being built is the Creator (no recorded change targets the downstream engine)", "self_target")
_mapped(s, "The Creator's registered objective is the development of the Creator itself", "creator/objective.py::SELF_STATEMENT",
        _T + "test_creator_objective.py::test_self_objective_and_ladder_are_idempotent")
_check(s, "The Creator has every verb of the objective (understand, identify, research, design, decompose, implement, test, diagnose, "
          "repair, measure, learn, improve process, recurse)", "capability_tested", _ENGINE)
s = "CR211"
_check(s, "Build the seed and the machinery that lets it understand what it must become (self-model, objective, gap analysis)",
       "capability_tested", ("K02", "K03", "K04"))
_mapped(s, "The machinery grows itself: a measured improving change is adopted and used by the next iteration",
        _REC + "::test_an_improving_change_is_adopted_and_used_by_the_next_iteration",
        _INT + "::test_the_whole_pipeline_from_objective_to_resolved_lesson")
_check(s, "The job is not to grow the tree manually: the teacher's share of adopted work must fall", "claude_share")
s = "CR212"
_mapped(s, "A rough downstream foundation is acceptable only if structurally sound: stubs, untested modules and unreached modules are "
           "computed limitations, and a change must build and pass its tests to be adopted", "creator/selfmodel.py::Limitation",
        _T + "test_creator_selfmodel.py::test_components_interfaces_and_stubs", _KER + "::test_an_untested_change_is_never_adopted")
_mapped(s, "A shallow autonomous-development capability is not acceptable: a component counts only with its own tests passing and no stubs",
        "creator/objective.py::_no_stubs", _T + "test_creator_selfmodel.py::test_capability_states_are_computed_from_evidence")
_check(s, "Development effort moves toward the autonomous developer (recorded cycles change the Creator, not the downstream system)", "self_target")
_intent(s, "The foundation may be incomplete, imperfect, partially validated, rough, over-specialized, under-integrated, missing "
           "capabilities", "a permission, not a requirement; nothing to compute")
s = "CR215"
_check(s, "Do not tell the finished developer 'now finish Weekly7': no recorded cycle changes the downstream engine", "self_target")
_check(s, "The first target is the system itself: it can inspect, understand, plan, modify, test, evaluate, repair and improve itself",
       "capability_tested", _ENGINE, (_INT + "::test_the_whole_pipeline_from_objective_to_resolved_lesson",))
_mapped(s, "Do not become a manually managed completion cycle: the kernel decides adoption by measurement, whoever the worker is",
        _KER + "::test_the_handoff_worker_waits_for_the_session_and_the_kernel_still_decides", "creator/kernel.py::execute")
_intent(s, "Only after self-development exists is the Creator ready for larger external objectives", "an ordering of future work; "
        "readiness for an external objective has no computable definition yet")
# --- 3 Claude's role
s = "CR216"
_check(s, "Claude's share of adopted work is computed and reported", "claude_share", (), (_CUR + "::test_scores_and_claude_share",))
_mapped(s, "Claude's role shifts over time: students try each package first and take over a kind of task once they have proven it",
        _CUR + "::test_student_tried_before_claude_and_claude_skipped_when_it_claims",
        _CUR + "::test_routing_hands_over_only_after_threshold_and_never_drops")
s = "CR217"
_intent(s, "Initially: architect, foundation builder, infrastructure builder, integrator, tester, auditor", "a description of the "
        "builder's own role; no artifact of it can be computed")
s = "CR218"
_mapped(s, "Evaluator / adversarial reviewer: an adversary attacks the Creator's own defences and every attack must be caught for the "
           "right reason", _AUD + "::test_every_attack_is_caught_for_the_right_reason", "creator/audit/checks.py::adversary")
_mapped(s, "Independent verifier: VALIDATED needs an independent role and evidence, never the builder", _LED + "::test_validated_needs_an_independent_role_and_evidence")
_mapped(s, "Safety authority / supervisor: protected paths are never merged, a red audit stops development, one kernel runs at a time",
        _KER + "::test_a_protected_path_is_never_merged", _KER + "::test_a_red_audit_stops_development", _KER + "::test_one_kernel_at_a_time")
s = "CR219"
_check(s, "The autonomous system becomes the primary developer: its share of adopted work exceeds the teacher's", "claude_share")
_mapped(s, "Claude is not the hidden intelligence: every teacher solution is captured as a lesson with its reasoning and a student can "
           "replay it", _CUR + "::test_lesson_captured_with_reasoning_from_done_file_and_adopted_filled", "creator/replay_student.py::ReplayStudent")
_mapped(s, "Build the capability instead of doing the work: failures of a student become lessons and deferred packages, not hand work",
        _CUR + "::test_claude_gets_it_when_all_students_fail_and_failure_is_a_lesson")
# --- 4 primary development principle
s = "CR220"
_check(s, "A task that is infrastructure for autonomous self-development is built: every recorded cycle works a requirement of the objective",
       "work_traces_to_requirements")
_mapped(s, "Work that is not infrastructure is not manually perfected: oversight flags adoptions that do not trace to the owner's objective",
        "creator/oversight.py::goal_drift", _T + "test_creator_oversight.py::test_an_adoption_under_a_non_owner_objective_is_drift")
_mapped(s, "Otherwise only the minimum sound foundation is built: planning takes the top unblocked gap, one package at a time",
        _PLN + "::test_plan_takes_the_top_unblocked_gap_and_writes_the_whole_chain", _GAP + "::test_ranking_puts_unblocked_important_gaps_first")
# --- 5 existing material
s = "CR221"
_check(s, "The authoritative contracts (C62-C68, canon, the blueprint) are preserved and out of the Creator's reach", "authoritative_documents",
       (), (_SBX + "::test_protected_paths_are_refused",))
_mapped(s, "Do not weaken a requirement or delete a difficult test: test weakening, deleted tests and 'or True' asserts are detected",
        _AUD + "::test_test_shape_and_weakening", _AUD + "::test_harness_files_cannot_silence_tests", _AUD + "::test_vacuous_or_assert_is_loose",
        _AUD + "::test_every_attack_is_caught_for_the_right_reason")
_mapped(s, "Do not lower a threshold or redefine a failure as success: thresholds and the verdict rule are protected paths and the "
           "ledger recomputes every verdict", _SBX + "::test_protected_paths_are_refused", _LED + "::test_improvement_claim_verdict_is_recomputed")
_mapped(s, "Anti-cheating and provenance requirements stay: evidence must hash to what is cited; answers are never hard-coded",
        _LED + "::test_evidence_must_hash_to_what_is_cited", _AUD + "::test_the_real_creator_has_no_hardcoded_answers_and_the_suite_is_sealed")
_intent(s, "Preserve C64's separation between what the running system can know and what hidden infrastructure may use for memory "
           "organisation", "enforced inside the downstream engine (engine/learning blind gates, tests/test_blind_gates.py), outside "
           "the Creator's junit evidence base")
# --- 7 status states
s = "CR222"
_check(s, "Use the eleven explicit states and never collapse them (IMPLEMENTED is not TESTED, TESTED is not VALIDATED)", "status_states",
       (), (_LED + "::test_status_machine_from_the_actual_state",))
_mapped(s, "TESTED needs a passing test run", _LED + "::test_tested_needs_a_passing_test_run")
_mapped(s, "VALIDATED is not PERFECT and needs an independent role with evidence", _LED + "::test_validated_needs_an_independent_role_and_evidence")
_mapped(s, "Code exists is not capability works: a capability state is computed from test evidence and goes stale when code is edited",
        _T + "test_creator_selfmodel.py::test_capability_states_are_computed_from_evidence",
        _T + "test_creator_selfmodel.py::test_an_edit_makes_old_test_evidence_stale")
_mapped(s, "A stale or unevidenced 'done' state is found by the audit", "creator/audit/checks.py::check_stale_done",
        _AUD + "::test_clean_ledger_audits_clean_and_drift_is_found")
# --- 8 execution order
s = "CR223"
_check(s, "The progression has no skipped step: the legal path to VALIDATED is the full chain", "execution_order")
_mapped(s, "Do not skip from implementation to completion: an untested change is never adopted and TESTED needs a passing run",
        _KER + "::test_an_untested_change_is_never_adopted", _LED + "::test_tested_needs_a_passing_test_run")
_mapped(s, "Operate, find failures, diagnose, fix, retest, regression-test, measure: one cycle goes through all of them on a real repository",
        _INT + "::test_the_whole_pipeline_from_objective_to_resolved_lesson", _KER + "::test_a_failing_candidate_is_diagnosed_and_the_next_attempt_sees_the_root_cause")
_mapped(s, "Adversarial testing and independent validation come before acceptance", _AUD + "::test_every_attack_is_caught_for_the_right_reason",
        _LED + "::test_validated_needs_an_independent_role_and_evidence")
# --- 62 / 63
s = "CR278"
_check(s, "Line floors are dropped (owner's ruling of 1 Oct 2026, 'capability, not lines'): depth is behaviour proven by each "
          "component's tests", "depth_is_behaviour")
_mapped(s, "A subsystem below its depth is not complete: the superseded 'depth' step is replaced by tested behaviour",
        "creator/objective.py::SUPERSEDED_STEPS", _T + "test_creator_objective.py::test_only_computed_checks_are_accepted")
_mapped(s, "Size is pushed down, not up: a real shrink is measured and adopted, and a gamed one is rejected",
        _EFF + "::test_a_real_shrink_is_measured_and_adopted", _EFF + "::test_gamed_shrinks_are_rejected")
_intent(s, "A subsystem above its minimum is not automatically good", "a caution, not a requirement; quality is judged by the other rules")
s = "CR279"
_check(s, "Blank lines, comments and docstrings do not count", "meaningful_ignores_padding", (), (_EFF + "::test_size_ignores_formatting_comments_and_docstrings",))
_mapped(s, "No padding by meaningless wrappers, duplicates, dead code or fake abstractions: gamed shrinks are rejected and coverage cannot be "
           "faked by a name in a comment", _EFF + "::test_gamed_shrinks_are_rejected",
        _EFF + "::test_a_name_in_a_comment_or_string_does_not_cover_it")
_mapped(s, "A stub is not code: stubs are found by the self-model", _T + "test_creator_selfmodel.py::test_components_interfaces_and_stubs")
s = "CR280"
_check(s, "Many small work packages, not broad phases: every recorded cycle changes few files and few lines", "package_size")
_check(s, "Every package carries the sixteen listed fields and a status", "package_fields")
_mapped(s, "No package is complete without its contract: completion needs a passing run and an adopt decision from a measured improvement",
        _LED + "::test_adopt_without_an_improvement_is_refused", _KER + "::test_an_untested_change_is_never_adopted")
# --- 65
s = "CR281"
_check(s, "Every stage of the autonomous loop is implemented", "loop_stages")
_mapped(s, "The loop runs end to end: objective, gaps, schedule, plan, sandbox, evaluation, verdict, merge, audit, lesson, reproduction",
        _INT + "::test_the_whole_pipeline_from_objective_to_resolved_lesson")
_mapped(s, "If the objective is not satisfied return to gap analysis: success leaves the gap for the next sync to prove or reopen",
        _PLN + "::test_success_leaves_the_gap_for_sync_to_prove", _GAP + "::test_fixing_closes_the_gap_and_breaking_marks_a_regression")
_mapped(s, "Adopt or roll back", _KER + "::test_a_change_that_fails_on_main_is_rolled_back", _INT + "::test_a_post_merge_rollback_is_recorded_diagnosed_and_not_left_unresolved")
_mapped(s, "Evaluate and improve the development strategy: the process itself is a measured, adoptable change", _REC + "::test_an_improving_change_is_adopted_and_used_by_the_next_iteration",
        _T + "test_creator_meta.py::test_propose_improvements")
_mapped(s, "When satisfied, independent validation follows", _LED + "::test_validated_needs_an_independent_role_and_evidence")
# --- 67
s = "CR283"
_mapped(s, "Students try before the teacher, and a student that has shown a kind of task takes it over for good",
        _CUR + "::test_student_tried_before_claude_and_claude_skipped_when_it_claims", _CUR + "::test_routing_hands_over_only_after_threshold_and_never_drops")
_mapped(s, "Every solved package becomes a lesson the system can learn from (the capability to discover the solution itself)",
        _CUR + "::test_lesson_captured_with_reasoning_from_done_file_and_adopted_filled", "creator/curriculum.py::Curriculum")
_mapped(s, "Whoever solves it, only a measured change is adopted", _KER + "::test_the_handoff_worker_waits_for_the_session_and_the_kernel_still_decides",
        _KER + "::test_an_unanswered_handoff_is_not_adopted")
_check(s, "The teacher's share is computed and must fall", "claude_share", (), (_CUR + "::test_scores_and_claude_share",))
_intent(s, "Ask 'what capability must I build so the developer can solve this itself?' instead of solving it by hand (exceptions "
           "only for establishing the foundation)", "a question the builder puts to itself; the effect is measured by teacher_share above")
# --- 69
s = "CR285"
_check(s, "Large code is not intelligent: an adopted change needs a measured gain, whatever its size", "complexity_vs_capability",
       (), (_LED + "::test_adopt_without_an_improvement_is_refused",))
_mapped(s, "Many agents are not autonomous: adoption of an agent's work needs the same measured claim, and a fake adoption is found",
        "creator/audit/checks.py::check_fake_adoption", _AUD + "::test_adoption_must_cite_a_real_merge")
_mapped(s, "Many tests are not validation: VALIDATED needs independent evidence", _LED + "::test_validated_needs_an_independent_role_and_evidence")
_mapped(s, "Many papers are not knowledge: research confidence is computed from evidence and contradictions, and unexplained failures "
           "become knowledge gaps", _T + "test_creator_research.py::test_confidence", _T + "test_creator_research.py::test_contradictions",
        _T + "test_creator_oversight.py::test_unexplained_failures_become_knowledge_gaps_once")
_mapped(s, "Many self-modifications are not improvement: a non-improving change is rejected", _REC + "::test_a_non_improving_change_is_rejected",
        _LED + "::test_adopt_without_an_improvement_is_refused")
_mapped(s, "A sophisticated architecture that cannot demonstrate its behaviour is incomplete: every claim is recomputed and an "
           "unsupported one is flagged", _T + "test_creator_claims.py::test_a_fabricated_claim_is_flagged_and_an_honest_one_is_not",
        _AUD + "::test_claims_are_recomputed_with_the_current_rule")
# --- 70
s = "CR286"
_mapped(s, "Do not delete the test or weaken it: both are detected and attacks on them are caught", _AUD + "::test_test_shape_and_weakening",
        _AUD + "::test_every_attack_is_caught_for_the_right_reason")
_mapped(s, "Do not weaken the threshold or change the definition: thresholds, the verdict rule, the model and the ledger are protected",
        _SBX + "::test_protected_paths_are_refused")
_mapped(s, "Do not hide the failure: a critical failure needs a diagnosis and a fix, or the owner's acceptance",
        "creator/audit/checks.py::check_unresolved_failures", _T + "test_creator_audit_recursion_failures.py::test_an_unresolved_rollback_is_critical",
        _T + "test_creator_audit_recursion_failures.py::test_only_the_owner_can_accept_a_critical_failure")
_mapped(s, "Do not mark it 'mostly complete': a state is only reached by its evidence and the verdict is recomputed",
        _LED + "::test_tested_needs_a_passing_test_run", _LED + "::test_improvement_claim_verdict_is_recomputed")
_mapped(s, "Do not remove the difficult case or substitute a weaker benchmark: the sealed suite is pinned and resealing is caught",
        _AUD + "::test_the_pinned_manifest_digest_matches_the_sealed_suite", _T + "test_creator_devbench.py::test_the_real_suite_is_sealed_intact")
_mapped(s, "Find, understand, fix, test, repeat: a failing candidate is diagnosed and its root cause reaches the next attempt",
        _KER + "::test_a_failing_candidate_is_diagnosed_and_the_next_attempt_sees_the_root_cause", _PLN + "::test_failures_replan_with_reasons_then_block")
_check(s, "A genuinely impossible requirement is recorded as a limitation, not passed", "limitation_recordable")
# --- 81
s = "CR296"
_check(s, "Do not stop when the written checklist ends: the objective still yields the next work (goal proposals)", "goals_pipeline")
_mapped(s, "Ask what the objective requires and whether the system possesses it: requirements are compiled and compared to computed capabilities",
        _GAP + "::test_assessment_is_computed_from_evidence", _T + "test_creator_objective.py::test_dependencies_follow_ladder_and_build_order")
_mapped(s, "Is it integrated, exercised, failed under adversarial conditions, repaired, measured: the ladder's steps, the adversary and the "
           "improvement verdict", "creator/objective.py::_integrated", _AUD + "::test_every_attack_is_caught_for_the_right_reason",
        _LED + "::test_improvement_claim_verdict_is_recomputed")
_mapped(s, "Can the system improve the capability and the process that improves it: recursion over the development process",
        _REC + "::test_an_improving_change_is_adopted_and_used_by_the_next_iteration")
_intent(s, "Has it generalized (question 8)", "generalization to unseen tasks has a holdout in the verdict, but the question as asked "
        "(across objectives) has nothing computable until a second objective exists")
# --- 82
s = "CR297"
_check(s, "Do not spend the majority of the project perfecting the downstream systems: no recorded change touches them", "self_target")
_mapped(s, "Downstream systems are test environments and sources of lessons: the Creator may only propose changes outside its own "
           "package through the measured loop", "creator/sandbox.py::PROTECTED", _KER + "::test_a_protected_path_is_never_merged")
_intent(s, "Treat the stock learner, research, patterns, movers and prediction-error systems as foundation, examples and development "
           "challenges", "a classification of existing material")
# --- 84 / DO / DO NOT
s = "CR299"
_check(s, "From this point forward every DO and DO NOT rule holds", "subsections", ("CR300", "CR301"))
s = "CR300"
_check(s, "Build the foundation and the autonomous-development machinery (ledger, sandbox, kernel)", "capability_tested", ("K01", "K06", "K14"))
_check(s, "Make it observable and measurable (kernel log, checkpoints, improvement measurement)", "capability_tested", ("K14", "K10"))
_check(s, "Make it testable and safe to modify (sandbox, regression selection, rollback)", "capability_tested", ("K06",))
_check(s, "Make it inspect itself and identify its own gaps", "capability_tested", ("K02", "K04"))
_check(s, "Make it research and design", "capability_tested", ("K07", "K08"))
_check(s, "Make it implement", "capability_tested", ("K05", "K09"))
_check(s, "Make it test (generate and run tests)", "capability_tested", ("K06", "K25"))
_check(s, "Make it debug and repair", "capability_tested", ("K11",))
_check(s, "Make it measure improvement", "capability_tested", ("K10",))
_check(s, "Make it learn from development and improve its development strategy", "capability_tested", ("K12", "K13"))
_check(s, "Make it recursively improve itself", "capability_tested", ("K24",))
_check(s, "Prove every major capability (independent audit, adversary, reproduction)", "capability_tested", ("K15", "K16", "K26"))
s = "CR301"
_check(s, "Do not manually finish the final system, make Weekly7 the target or turn the project into a stock-predictor completion project",
       "self_target")
_mapped(s, "Do not hard-code autonomous demonstrations or give the developer its answers: answer literals are scanned for and the suite is sealed",
        _AUD + "::test_the_real_creator_has_no_hardcoded_answers_and_the_suite_is_sealed", "creator/audit/checks.py::check_hardcoded_answers")
_mapped(s, "Do not fake self-improvement: a process change needs a recorded weakness, a measured claim, an adopt decision and an unbroken lineage",
        "creator/audit/checks.py::check_recursion", _T + "test_creator_audit_recursion_failures.py::test_an_adopted_step_without_decision_or_claim_is_critical")
_mapped(s, "Do not lower standards", _AUD + "::test_test_shape_and_weakening", _SBX + "::test_protected_paths_are_refused")
_mapped(s, "Do not delete failures or hide failed tests: failures are listed until resolved and test files cannot be silenced",
        _T + "test_creator_audit_recursion_failures.py::test_an_unresolved_rollback_is_critical", _AUD + "::test_harness_files_cannot_silence_tests")
_mapped(s, "Do not confuse code existence with capability or a passing test with validation", _T + "test_creator_selfmodel.py::test_an_edit_makes_old_test_evidence_stale",
        _LED + "::test_validated_needs_an_independent_role_and_evidence")
_mapped(s, "Do not confuse self-modification with self-improvement", _LED + "::test_adopt_without_an_improvement_is_refused",
        _REC + "::test_a_non_improving_change_is_rejected")
_mapped(s, "Do not tune against the final holdout: learning stores must not contain it and a holdout needs a frozen configuration",
        _AUD + "::test_learning_stores_must_not_contain_the_holdout", _T + "test_creator_devbench.py::test_holdout_needs_a_frozen_configuration")
_mapped(s, "Do not manually perform the work the developer must learn: students first, teacher share computed",
        _CUR + "::test_student_tried_before_claude_and_claude_skipped_when_it_claims", _CUR + "::test_scores_and_claude_share")
# --- 85
s = "CR302"
_check(s, "Understand itself, and what it needs to become", "capability_tested", ("K02", "K03", "K04"))
_check(s, "Research what it does not know and design what it needs", "capability_tested", ("K07", "K08"))
_check(s, "Build what it designs and test what it builds", "capability_tested", ("K09", "K06", "K25"))
_check(s, "Find and fix its own failures", "capability_tested", ("K11", "K16"))
_check(s, "Prove whether its fixes work", "capability_tested", ("K10", "K26"))
_check(s, "Learn which development methods work and improve its development process", "capability_tested", ("K12", "K13", "K24"))
_check(s, "Use the improved process to improve itself again, then repeat: a recursion step is in the ledger", "recursion_demonstrated")
_check(s, "Do not grow the tree yourself: the teacher's share falls", "claude_share")
# --- 86
s = "CR303"
_mapped(s, "Work continuously: one supervisor, restarted after a reboot, never two at once, with the owner's off switch honoured",
        _T + "test_nupen_service.py::test_the_supervisor_stops_on_the_stop_file_and_never_runs_twice",
        _T + "test_nupen_service.py::test_a_pidfile_from_before_the_boot_does_not_block_the_supervisor",
        _T + "test_nupen_service.py::test_the_pidfile_is_claimed_exclusively")
_mapped(s, "A convenient milestone is not a stop: the next package is planned from the open gaps of the objective",
        _PLN + "::test_plan_takes_the_top_unblocked_gap_and_writes_the_whole_chain", _PLN + "::test_planned_gaps_are_not_planned_twice_and_ids_are_sequential")
_mapped(s, "A failure becomes the next work item: a failing gap is replanned with the reasons, and an interrupted package is retried, not dropped",
        _PLN + "::test_failures_replan_with_reasons_then_block", _PLN + "::test_interruptions_are_not_attempts_so_a_gap_is_never_blocked_by_stops")
_mapped(s, "An evidence gap becomes the next work item: unexplained failures become knowledge gaps", _T + "test_creator_oversight.py::test_unexplained_failures_become_knowledge_gaps_once")
_mapped(s, "A depth gap becomes the next work item: efficiency work is planned from measured size, coverage and activation targets",
        _EFF + "::test_gap_work_comes_first_in_auto_mode", _PLN + "::test_the_coverage_planner_picks_the_module_with_the_biggest_test_gap")
_mapped(s, "Evidence is not assumed: stale evidence reopens a requirement", _GAP + "::test_stale_evidence_reopens_a_tested_requirement_without_calling_it_failed")
_check(s, "Continue until the checklist's own end is not a stop: goal proposals continue the work", "goals_pipeline")
_check(s, "Continue until recursive improvement has been demonstrated", "recursion_demonstrated")
_intent(s, "If Claude catches itself manually solving something the developer should solve: stop, reassess, build the general capability",
        "an instruction to the builder; its measurable effect is the teacher_share rules above")
# --- 87
s = "CR304"
_check(s, "The project is about building the system that can build the final system: the recorded changes are all to the Creator", "self_target")
_check(s, "The Creator must eventually create and improve itself: a recursion step is in the ledger", "recursion_demonstrated")
_check(s, "The Creator is the product built now: all its components exist and are tested", "capability_tested", _ENGINE)
_intent(s, "The final system is a future output of the Creator", "a statement about a future that has not happened")
_intent(s, "Build the creator, prove the creator, let the creator create", "a closing exhortation; its parts are the rules above")

RULES: tuple[Rule, ...] = tuple(_rules)
RULES_BY_SECTION = {sec: tuple(r for r in RULES if r.section == sec) for sec in SECTIONS}

