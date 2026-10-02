"""Reconcile the ledger's Capability records with reality, in the INDEPENDENT VALIDATOR's role (C77 secs 7, 48).

Found 1 Oct 2026: all 26 Capability records were still NOT_STARTED although the code exists and passes its tests, and the
independent validators' verdicts (state/validation/VALIDATION_REPORT.json) were never recorded - so every '<K>.validated' gap
stayed open and Nupen had no work but shrinking. This script is run by the validator (an agent that did not build the components),
while no kernel/swarm holds the ledger.

For each declared component it advances the Capability only as far as computed evidence justifies:
  - IMPLEMENTED   when the 'exists' and 'no_stubs' checks are met on the current tree;
  - TESTED        when its declared test files pass NOW (a fresh run, recorded as a TestRun with the log as evidence);
  - VALIDATED     only when, in addition, the 'integrated' check is met, the independent report says VALIDATED, and none of the
                  component's modules or tests changed since the report commit that first gave that verdict (a stale verdict is
                  not a verdict). It goes through INTENDED_BEHAVIOR_VERIFIED, both steps carrying the report + log as evidence.
Nothing is ever moved backwards here and nothing is invented: a failing test stops the component where it is.

    python scripts/record_validation.py [--dry-run] [--only K01,K02]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import model as M  # noqa: E402
from creator import objective as O  # noqa: E402
from creator import selfmodel as SM  # noqa: E402
from creator.ledger import Ledger  # noqa: E402

REPORT = "state/validation/VALIDATION_REPORT.json"
EVID = "state/creator/evidence/validation"
ORDER = (M.Status.IMPLEMENTED, M.Status.TESTED, M.Status.INTENDED_BEHAVIOR_VERIFIED, M.Status.VALIDATED)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def first_validated_commit(cid: str) -> Optional[str]:
    """The oldest commit of the report in which this component's verdict is VALIDATED (what the validator actually saw)."""
    for sha in reversed(git("log", "--format=%H", "--", REPORT).stdout.split()):
        show = git("show", f"{sha}:{REPORT}")
        try:
            v = json.loads(show.stdout)["components"].get(cid, {}).get("verdict")
        except (ValueError, KeyError, AttributeError):
            continue
        if v == "VALIDATED":
            return sha
    return None


def unchanged_since(sha: str, paths: list[str]) -> bool:
    return git("diff", "--quiet", sha, "HEAD", "--", *paths).returncode == 0 and \
        git("diff", "--quiet", "--", *paths).returncode == 0


def run_tests(cid: str, tests: list[str]) -> tuple[dict[str, int], float, Path]:
    out = ROOT / EVID / cid
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"pytest_{time.strftime('%Y%m%dT%H%M%S')}.log"
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rN", *tests], cwd=ROOT,
                       capture_output=True, text=True)
    dur = time.monotonic() - t0
    log.write_text(p.stdout + p.stderr, encoding="utf-8")
    counts = {k: 0 for k in ("passed", "failed", "errors", "skipped")}
    tail = (p.stdout.strip().splitlines() or [""])[-1]
    for n, word in __import__("re").findall(r"(\d+) (passed|failed|errors?|skipped)", tail):
        counts["errors" if word.startswith("error") else word] += int(n)
    if p.returncode != 0 and not counts["failed"] and not counts["errors"]:
        counts["errors"] = 1                                               # pytest itself failed: never count as a pass
    return counts, dur, log


def ref(path: Path) -> M.EvidenceRef:
    return M.EvidenceRef(path.relative_to(ROOT).as_posix(), M.sha256_file(path), "validation")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    specs = SM.load_capabilities()
    only = {x for x in a.only.split(",") if x}
    report = json.loads((ROOT / REPORT).read_text(encoding="utf-8"))["components"]
    led = Ledger(ROOT / "state" / "creator" / "ledger.jsonl", evidence_root=ROOT)
    model = SM.build(ROOT, capabilities=specs)
    summary: list[dict[str, Any]] = []
    for spec in specs:
        cid = spec.id
        if only and cid not in only:
            continue
        cap = O._find_capability(led, cid)
        row: dict[str, Any] = {"component": cid, "capability": cap}
        if cap is None:
            summary.append({**row, "result": "no Capability record"})
            continue
        cur = led.view.status[cap]
        c = model.capability(cid)
        checks = {k: O.CHECKS[k](model, c, led, cap)[0] for k in ("exists", "no_stubs", "integrated")}
        target = M.Status.IMPLEMENTED if checks["exists"] and checks["no_stubs"] else None
        counts, dur, log = ({}, 0.0, None) if (a.dry_run or target is None) else run_tests(cid, list(spec.tests))
        tests_pass = bool(counts) and counts["passed"] > 0 and not counts["failed"] and not counts["errors"]
        if target and tests_pass:
            target = M.Status.TESTED
        vsha = first_validated_commit(cid) if report.get(cid, {}).get("verdict") == "VALIDATED" else None
        fresh = bool(vsha) and unchanged_since(vsha, [*spec.modules, *spec.tests])
        if target is M.Status.TESTED and checks["integrated"] and fresh:
            target = M.Status.VALIDATED
        row.update(checks=checks, tests=counts, verdict=report.get(cid, {}).get("verdict"), verdict_commit=vsha, fresh=fresh,
                   current=cur.value, target=target.value if target else None)
        if a.dry_run or target is None or ORDER.index(target) <= (ORDER.index(cur) if cur in ORDER else -1):
            summary.append({**row, "result": "dry run" if a.dry_run else "nothing to record"})
            continue
        assert log is not None
        evidence = [ref(log), ref(ROOT / REPORT)]
        tr = led.append(M.TestRun(created_by=M.Role.VALIDATOR, command=" ".join(["pytest", *spec.tests]),
                                  passed=counts["passed"], failed=counts["failed"], errors=counts["errors"],
                                  skipped=counts["skipped"], duration_s=round(dur, 2), subject_ids=(cap,), evidence=tuple(evidence)))
        why = (f"validator reconciliation: checks {checks}, fresh test run {counts}; independent verdict "
               f"{row['verdict']} at {vsha[:9] if vsha else '-'}{' (code unchanged since)' if fresh else ''}")
        for step in M.reachable(cur, target)[1:]:
            led.transition(cap, step, reason=why[:900], created_by=M.Role.VALIDATOR, justification_ids=(tr,),
                           evidence=tuple(evidence))
        summary.append({**row, "result": f"{cur.value} -> {target.value}"})
    for r in summary:
        print(json.dumps(r, default=str))
    if not a.dry_run:
        print("ledger verifies:", led.verify())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
