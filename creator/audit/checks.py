"""Creator K16 - the auditor and the adversary (C77 secs 33-35, 47, 48, 50, 68; package CR16) - IMPLEMENTED, NOT VALIDATED.

Lives under creator/audit/, a PROTECTED path: the Creator can never weaken the checks that judge it.

AUDIT (`audit()`): computed checks over the development ledger, the repository and the Creator's state. Each returns findings
with a severity; CRITICAL means a claim the system makes about itself is not backed by evidence.

    ledger_integrity     the hash chain verifies from disk
    evidence_drift       every cited evidence file still exists with the cited sha256
    stale_done           a requirement TESTED in the ledger whose computed check fails on the current source
    fake_adoption        an ADOPT decision without an IMPROVEMENT claim, a merge commit in git, or measurement evidence (sec 68)
    claim_recompute      every ImprovementClaim's verdict recomputed with the CURRENT rule (a rule change cannot silently bless
                         an old claim)
    sealed_suite         the devbench manifest and every task still hash to their sealed values
    hardcoded_answers    production code containing devbench task ids or answer-key literals (sec 50: no demonstrations)
    budget_anomalies     agent calls booked as free although they ran
    test_weakening       (on a diff) fewer tests/asserts, new skip/xfail, loosened comparisons - usable by the sandbox too

ADVERSARY (`adversary()`): attacks the integrity rules in throw-away directories and reports which ones were CAUGHT. A rule that is
not caught is a CRITICAL finding: the rule exists only on paper (memory: 'wired means reachable')."""
from __future__ import annotations

import ast
import dataclasses
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from creator import model as M
from creator.ledger import REPO_ROOT, Ledger, LedgerError

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")


@dataclasses.dataclass(frozen=True)
class AuditFinding:
    check: str
    severity: str
    subject: str
    detail: str


@dataclasses.dataclass(frozen=True)
class AuditReport:
    findings: tuple[AuditFinding, ...]
    checks_run: tuple[str, ...]
    errors: Mapping[str, str]                                           # a check that crashed is itself reported, never skipped

    @property
    def clean(self) -> bool:
        return not self.findings and not self.errors

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    def to_dict(self) -> dict[str, Any]:
        return {"clean": self.clean, "checks_run": list(self.checks_run), "errors": dict(self.errors),
                "counts": {s: self.count(s) for s in SEVERITIES}, "findings": [dataclasses.asdict(f) for f in self.findings]}


# ------------------------------------------------------------------------------------------------ ledger checks

def check_ledger_integrity(led: Ledger, **_: Any) -> list[AuditFinding]:
    try:
        led.verify()
        return []
    except LedgerError as e:
        return [AuditFinding("ledger_integrity", "CRITICAL", str(led.path), str(e))]


def check_evidence_drift(led: Ledger, **_: Any) -> list[AuditFinding]:
    out = []
    for e in led.view.entries:
        refs = list(e.record.evidence)
        if isinstance(e.record, M.Transition):
            refs += list(e.record.evidence)
        for ref in dict.fromkeys(refs):
            prob = ref.problem(led.evidence_root)
            if prob:
                sev = "CRITICAL" if e.rtype in ("Measurement", "TestRun", "ImprovementClaim", "Decision") else "HIGH"
                out.append(AuditFinding("evidence_drift", sev, e.id, prob))
    return out


def check_stale_done(led: Ledger, model: Any = None, **_: Any) -> list[AuditFinding]:
    if model is None:
        return []
    from creator import gaps as G
    out = []
    for a in G.assess(led, model):
        if a.status in M.DONE_STATES and not a.met:
            out.append(AuditFinding("stale_done", "CRITICAL", a.requirement_id,
                                    f"{a.key} is {a.status.value} in the ledger but its check fails now: {a.detail}"))
    return out


def _git(repo: Path, *args: str) -> tuple[int, str]:
    try:
        p = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=60)
        return p.returncode, p.stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def check_fake_adoption(led: Ledger, repo: Path = REPO_ROOT, **_: Any) -> list[AuditFinding]:
    out = []
    for e in led.of_type("Decision"):
        d = e.record
        if getattr(d, "verdict") is not M.DecisionVerdict.ADOPT:
            continue
        cid = getattr(d, "claim_id")
        claim = led.view.by_id.get(cid) if cid else None
        if claim is None or getattr(claim.record, "verdict") is not M.Verdict.IMPROVEMENT:
            out.append(AuditFinding("fake_adoption", "CRITICAL", e.id, "ADOPT without an IMPROVEMENT claim"))
            continue
        ms = list(getattr(claim.record, "baseline_ids")) + list(getattr(claim.record, "candidate_ids"))
        if not all(led.view.by_id[m].record.evidence for m in ms):
            out.append(AuditFinding("fake_adoption", "CRITICAL", e.id, "claim measurements without evidence"))
        merges = [ref for ref in d.evidence if ref.kind == "merge_commit"]
        for ref in merges:
            rc, _out = _git(repo, "cat-file", "-e", Path(ref.path).name)
            if rc != 0:
                out.append(AuditFinding("fake_adoption", "CRITICAL", e.id, f"merge commit {Path(ref.path).name} not in git"))
        if not merges:
            out.append(AuditFinding("fake_adoption", "HIGH", e.id, "ADOPT decision does not cite the merge commit"))
    return out


def check_claim_recompute(led: Ledger, **_: Any) -> list[AuditFinding]:
    out = []
    for e in led.of_type("ImprovementClaim"):
        c = e.record
        g = led.view.by_id

        def ms(ids: Iterable[str]) -> list[M.Measurement]:
            return [g[i].record for i in ids]                           # type: ignore[misc]
        hold = None
        if getattr(c, "holdout_baseline_id") and getattr(c, "holdout_candidate_id"):
            hold = (g[getattr(c, "holdout_baseline_id")].record, g[getattr(c, "holdout_candidate_id")].record)
        v, _d = M.improvement_verdict(ms(getattr(c, "baseline_ids")), ms(getattr(c, "candidate_ids")),
                                      list(zip(ms(getattr(c, "regression_baseline_ids")), ms(getattr(c, "regression_candidate_ids")))),
                                      hold, getattr(c, "min_effect"), getattr(c, "z"))  # type: ignore[arg-type]
        if v is not getattr(c, "verdict"):
            sev = "CRITICAL" if getattr(c, "verdict") is M.Verdict.IMPROVEMENT else "MEDIUM"
            out.append(AuditFinding("claim_recompute", sev, e.id,
                                    f"recorded {getattr(c, 'verdict').value}, current rule gives {v.value}"))
    return out


# ------------------------------------------------------------------------------------------------ repository checks

def check_sealed_suite(led: Optional[Ledger] = None, **_: Any) -> list[AuditFinding]:
    from creator import devbench as D
    try:
        m = D.load_manifest()
    except D.DevbenchError as e:
        return [AuditFinding("sealed_suite", "CRITICAL", "MANIFEST", str(e))]
    out = []
    for t in D.load_tasks():
        try:
            D.check_sealed(t, m)
        except D.DevbenchError as e:
            out.append(AuditFinding("sealed_suite", "CRITICAL", t.id, str(e)))
    return out


def answer_literals(sealed: Path) -> set[str]:
    """Distinctive literals from the sealed answer keys: long numbers and quoted strings that a hard-coded answer would contain."""
    lits: set[str] = set()
    for p in sealed.rglob("*.py"):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and _distinctive(n.value):
                lits.add(str(n.value))
    return lits


def _distinctive(v: object) -> bool:
    """Only literals unlikely to occur by coincidence (1 Oct: '2000' and 'no:cacheprovider' were flagged in build/testrun):
    integers of 5+ digits that are not round thousands; strings of 6+ characters that contain a digit or an inner hyphen and are
    not options, module paths or identifiers. Known blind spot, recorded: small answers (2.5, 'hi') cannot be told from
    ordinary code by text alone - the sealed holdout and the hidden tests remain the real defence."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return abs(v) >= 10000 and v % 1000 != 0
    if isinstance(v, str):
        t = v.strip()
        if any(ch in t for ch in "[]()*+?^$|{}"):                      # a regex or format pattern, not an answer
            return False
        if len(t) < 6 or t.startswith(("-", "_", ".")) or ":" in t or "/" in t or "\\" in t or t.isidentifier():
            return False
        if re.fullmatch(r"[\w.]+", t) and "." in t:                      # dotted module path / version
            return False
        return bool(re.search(r"\d", t) or re.search(r"[a-z]-[a-z]", t))
    return False


def check_hardcoded_answers(led: Optional[Ledger] = None, repo: Path = REPO_ROOT, scan: Sequence[str] = ("creator",),
                            allow: Sequence[str] = ("creator/devbench/", "creator/devbench.py", "creator/audit/"),
                            sealed_root: Optional[Path] = None, **_: Any) -> list[AuditFinding]:
    """`repo` is the tree scanned; `sealed_root` the repository whose sealed answer keys are compared (a worker's sandbox has
    them hidden, so its keys come from the main repository)."""
    src = sealed_root or repo
    sealed = src / "creator" / "devbench" / "sealed"
    lits = answer_literals(sealed) if sealed.is_dir() else set()
    task_ids = {p.name for p in (src / "creator" / "devbench" / "tasks").glob("*") if p.is_dir()}
    id_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, task_ids))) + r")\b") if task_ids else None
    out = []
    for base in scan:
        for p in sorted((repo / base).rglob("*.py")):
            rel = p.relative_to(repo).as_posix()
            if any(rel.startswith(a) for a in allow):
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            hits = sorted(x for x in lits if x in text)
            if hits:
                out.append(AuditFinding("hardcoded_answers", "CRITICAL", rel, f"answer-key literal(s): {hits[:5]}"))
            if id_re is not None:
                ids = sorted(set(id_re.findall(text)) - {"K01"})
                ids = [i for i in ids if re.search(rf"['\"]{i}['\"]", text)]
                if ids:
                    out.append(AuditFinding("hardcoded_answers", "HIGH", rel, f"devbench task id literal(s): {ids[:5]}"))
    return out


def check_budget_anomalies(led: Optional[Ledger] = None, budget_file: Optional[Path] = None, **_: Any) -> list[AuditFinding]:
    path = budget_file or (REPO_ROOT / "state" / "creator" / "agent_budget.json")
    if not path.is_file():
        return []
    out = []
    for c in json.loads(path.read_text(encoding="utf-8")).get("calls", []):
        if float(c.get("usd", 0)) == 0.0 and c.get("outcome") in ("OK", "CONTAMINATED", "TIMEOUT"):
            out.append(AuditFinding("budget_anomalies", "MEDIUM", str(c.get("run")), f"{c.get('outcome')} call booked at $0"))
    return out


# ------------------------------------------------------------------------------------------------ test weakening (diff check)

@dataclasses.dataclass(frozen=True)
class TestShape:
    tests: int
    asserts: int
    skips: int
    approx_loose: int


def test_shape(source: str) -> TestShape:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return TestShape(0, 0, 0, 0)
    tests = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"))
    asserts = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert))
    asserts += sum(1 for n in ast.walk(tree) if isinstance(n, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and getattr(i.context_expr.func, "attr", "") == "raises" for i in n.items))
    skips = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ("skip", "xfail", "skipif"))
    loose = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) in ("approx", "isclose"):
            for kw in n.keywords:
                if kw.arg in ("rel", "abs", "rel_tol", "abs_tol") and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, (int, float)) and kw.value.value >= 0.1:
                    loose += 1
        if isinstance(n, ast.Assert) and isinstance(n.test, ast.Constant) and n.test.value:
            loose += 1                                                  # `assert True`
    return TestShape(tests, asserts, skips, loose)


def check_test_weakening(before: Mapping[str, str], after: Mapping[str, str]) -> list[AuditFinding]:
    """before/after: test-file path -> source. Deleted test files, fewer tests or asserts, added skips, looser tolerances."""
    out = []
    for path in sorted(set(before) | set(after)):
        if not Path(path).name.startswith("test_"):
            continue
        if path in before and path not in after:
            out.append(AuditFinding("test_weakening", "CRITICAL", path, "test file deleted"))
            continue
        if path not in before:
            continue
        b, a = test_shape(before[path]), test_shape(after[path])
        if a.tests < b.tests:
            out.append(AuditFinding("test_weakening", "CRITICAL", path, f"tests {b.tests} -> {a.tests}"))
        if a.asserts < b.asserts:
            out.append(AuditFinding("test_weakening", "HIGH", path, f"assertions {b.asserts} -> {a.asserts}"))
        if a.skips > b.skips:
            out.append(AuditFinding("test_weakening", "HIGH", path, f"skip/xfail markers {b.skips} -> {a.skips}"))
        if a.approx_loose > b.approx_loose:
            out.append(AuditFinding("test_weakening", "HIGH", path, "looser tolerance or vacuous assert added"))
    return out


# ------------------------------------------------------------------------------------------------ the audit

CHECKS: dict[str, Callable[..., list[AuditFinding]]] = {
    "ledger_integrity": check_ledger_integrity, "evidence_drift": check_evidence_drift, "stale_done": check_stale_done,
    "fake_adoption": check_fake_adoption, "claim_recompute": check_claim_recompute, "sealed_suite": check_sealed_suite,
    "hardcoded_answers": check_hardcoded_answers, "budget_anomalies": check_budget_anomalies,
}


def audit(led: Ledger, model: Any = None, only: Optional[Sequence[str]] = None, **kw: Any) -> AuditReport:
    findings: list[AuditFinding] = []
    errors: dict[str, str] = {}
    names = list(only or CHECKS)
    for name in names:
        try:
            findings += CHECKS[name](led=led, model=model, **kw)
        except Exception as e:                                          # noqa: BLE001 - a broken check is reported, not skipped
            errors[name] = f"{type(e).__name__}: {e}"
    order = {s: i for i, s in enumerate(SEVERITIES)}
    return AuditReport(tuple(sorted(findings, key=lambda f: (order[f.severity], f.check, f.subject))), tuple(names), errors)


# ------------------------------------------------------------------------------------------------ the adversary

@dataclasses.dataclass(frozen=True)
class Attack:
    name: str
    rule: str
    caught: bool
    detail: str


def _attack(name: str, rule: str, fn: Callable[[Path], str], expect: str) -> Attack:
    """CAUGHT only when the refusal names the rule under attack: an attack that dies of an unrelated error proves nothing and
    is reported as NOT CAUGHT (a check that passes for the wrong reason is not a check)."""
    with tempfile.TemporaryDirectory(prefix="creator-adv-") as d:
        try:
            detail = fn(Path(d))
            return Attack(name, rule, False, f"NOT CAUGHT: {detail}")
        except _Caught as c:
            msg = str(c)
        except Exception as e:                                          # noqa: BLE001
            msg = f"refused: {type(e).__name__}: {e}"
        if re.search(expect, msg):
            return Attack(name, rule, True, msg)
        return Attack(name, rule, False, f"NOT CAUGHT (failed for another reason): {msg}")


class _Caught(Exception):
    pass


def _objective(led: Ledger) -> str:
    return led.append(M.Objective(created_by=M.Role.OWNER, statement="adversary world", acceptance_criteria=("x",)))


def _a_forge_tested(d: Path) -> str:
    led = Ledger(d / "l.jsonl", evidence_root=d)
    o = _objective(led)
    g = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(o,), kind=M.GapKind.TESTING, description="g", importance=0.5))
    led.transition(g, M.Status.IN_PROGRESS, "start", M.Role.KERNEL)
    led.transition(g, M.Status.IMPLEMENTED, "built", M.Role.KERNEL)
    led.transition(g, M.Status.TESTED, "trust me", M.Role.KERNEL)
    return "TESTED without a passing TestRun was accepted"


def _a_self_validate(d: Path) -> str:
    led = Ledger(d / "l.jsonl", evidence_root=d)
    o = _objective(led)
    c = led.append(M.Capability(created_by=M.Role.KERNEL, name="c", component="K99", description="d"))
    (d / "ev.txt").write_text("x", encoding="utf-8")
    ev = M.EvidenceRef.of(d / "ev.txt", d)
    tr = led.append(M.TestRun(created_by=M.Role.KERNEL, command="t", passed=1, failed=0, errors=0, skipped=0, duration_s=0.1,
                              evidence=(ev,)))
    for s in (M.Status.IN_PROGRESS, M.Status.IMPLEMENTED):
        led.transition(c, s, "x", M.Role.KERNEL)
    led.transition(c, M.Status.TESTED, "x", M.Role.KERNEL, justification_ids=(tr,))
    led.transition(c, M.Status.INTENDED_BEHAVIOR_VERIFIED, "x", M.Role.KERNEL, justification_ids=(tr,))
    led.transition(c, M.Status.VALIDATED, "I validate myself", M.Role.IMPLEMENTER, justification_ids=(tr,), evidence=(ev,))
    return f"VALIDATED by the implementer was accepted ({o})"


def _a_adopt_without_claim(d: Path) -> str:
    led = Ledger(d / "l.jsonl", evidence_root=d)
    o = _objective(led)
    g = led.append(M.Gap(created_by=M.Role.KERNEL, parents=(o,), kind=M.GapKind.TESTING, description="g", importance=0.5))
    ex = led.append(M.Experiment(created_by=M.Role.KERNEL, parents=(g,), hypothesis="h", design="d", metrics=("m",), seed=0,
                                 baseline_ref="b", candidate_ref="c"))
    led.append(M.Decision(created_by=M.Role.KERNEL, subject_id=ex, verdict=M.DecisionVerdict.ADOPT, reason="looks better"))
    return "ADOPT without a claim was accepted"


def _a_tamper_ledger(d: Path) -> str:
    led = Ledger(d / "l.jsonl", evidence_root=d)
    _objective(led)
    led.append(M.Objective(created_by=M.Role.OWNER, statement="second", acceptance_criteria=("y",)))
    lines = (d / "l.jsonl").read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("adversary world", "rewritten history")
    (d / "l.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Ledger(d / "l.jsonl", evidence_root=d)
    return "an edited ledger line loaded without complaint"


def _a_evidence_swap(d: Path) -> str:
    led = Ledger(d / "l.jsonl", evidence_root=d)
    (d / "log.txt").write_text("1 passed", encoding="utf-8")
    led.append(M.TestRun(created_by=M.Role.KERNEL, command="t", passed=1, failed=0, errors=0, skipped=0, duration_s=0.1,
                         evidence=(M.EvidenceRef.of(d / "log.txt", d),)))
    (d / "log.txt").write_text("1 failed", encoding="utf-8")
    if check_evidence_drift(led):
        raise _Caught("evidence_drift reported the swapped log")
    return "a rewritten test log went unnoticed"


def _a_weaken_tests(d: Path) -> str:
    before = {"tests/test_x.py": "def test_a():\n    assert f(1) == 2\n    assert f(2) == 4\n\ndef test_b():\n    assert g()\n"}
    after = {"tests/test_x.py": "import pytest\n\n@pytest.mark.skip\ndef test_a():\n    assert True\n"}
    if check_test_weakening(before, after):
        raise _Caught("test_weakening flagged it")
    return "a gutted test file was not flagged"


def _a_protected_write(d: Path) -> str:
    from creator import sandbox as S
    for p in ("creator/devbench.py", "creator/audit/checks.py", "creator/capabilities.json", "creator/model.py"):
        if not S.is_protected(p):
            return f"{p} is writable by the Creator"
    raise _Caught("every measuring-stick path is protected")


def _a_hardcode_answer(d: Path) -> str:
    (d / "creator").mkdir()
    (d / "creator" / "devbench" / "sealed" / "X1" / "hidden").mkdir(parents=True)
    (d / "creator" / "devbench" / "tasks" / "X1").mkdir(parents=True)
    (d / "creator" / "devbench" / "sealed" / "X1" / "hidden" / "test_h.py").write_text(
        "def test_h():\n    assert solve() == 299993\n", encoding="utf-8")
    (d / "creator" / "planner.py").write_text("def solve_task(t):\n    if t == 'X1':\n        return 299993\n", encoding="utf-8")
    if check_hardcoded_answers(repo=d):
        raise _Caught("hardcoded_answers flagged the planted answer")
    return "a hard-coded answer went unnoticed"


ATTACKS: tuple[tuple[str, str, Callable[[Path], str], str], ...] = (
    ("forge_tested", "TESTED needs a passing TestRun (sec 7)", _a_forge_tested, r"TESTED needs"),
    ("self_validate", "only an independent role may VALIDATE (sec 48)", _a_self_validate, r"may not validate"),
    ("adopt_without_claim", "ADOPT needs an IMPROVEMENT claim (sec 30)", _a_adopt_without_claim, r"ADOPT needs"),
    ("tamper_ledger", "the ledger is tamper-evident (sec 59)", _a_tamper_ledger, r"hash mismatch"),
    ("evidence_swap", "cited evidence cannot change silently (sec 35)", _a_evidence_swap, r"evidence_drift"),
    ("weaken_tests", "tests are never weakened to pass (sec 68)", _a_weaken_tests, r"test_weakening"),
    ("protected_write", "measuring sticks are protected (secs 33-34)", _a_protected_write, r"protected"),
    ("hardcode_answer", "no hard-coded demonstrations (sec 50)", _a_hardcode_answer, r"hardcoded_answers"),
)


def adversary(attacks: Sequence[tuple[str, str, Callable[[Path], str], str]] = ATTACKS) -> list[Attack]:
    return [_attack(n, r, f, x) for n, r, f, x in attacks]
