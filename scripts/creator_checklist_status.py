"""Compute the C77 checklist (state/build/CREATOR_MASTER_CHECKLIST.json) statuses FROM EVIDENCE.

Sources: state/creator/STATUS.json (capability states from fresh test evidence), state/creator/AUDIT.json (audit + adversary),
state/creator/kernel_log.jsonl (real self-development cycles), state/build/CREATOR_PHASE0_LEDGER.json (reconnaissance).
Rules: a box backed by a module whose declared tests pass on the current source -> TESTING; code without passing tests ->
IMPLEMENTED; a box that needs a demonstrated autonomous result -> IN_PROGRESS while cycles ran without that result; nothing is ever
set VALIDATED here (only an independent validator may). Every changed box gets its evidence paths and a note saying why."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "state" / "build" / "CREATOR_MASTER_CHECKLIST.json"
STATUS = ROOT / "state" / "creator" / "STATUS.json"
AUDIT = ROOT / "state" / "creator" / "AUDIT.json"
KLOG = ROOT / "state" / "creator" / "kernel_log.jsonl"
PHASE0 = ROOT / "state" / "build" / "CREATOR_PHASE0_LEDGER.json"

# box -> capability ids whose state decides it (all must be TESTED for TESTING)
CAP = {
    **{f"CR{n:03d}": ("K01",) for n in (37, 38)}, "CR039": ("K06",), "CR040": ("K15",), "CR041": ("K14",),
    **{f"CR{n:03d}": ("K02",) for n in range(42, 51)},
    **{f"CR{n:03d}": ("K03",) for n in (51, 52, 53, 54, 55, 56)},
    **{f"CR{n:03d}": ("K04",) for n in (58, 60, 61, 62, 63)},
    **{f"CR{n:03d}": ("K07",) for n in range(64, 72)},
    **{f"CR{n:03d}": ("K08",) for n in range(72, 79)},
    **{f"CR{n:03d}": ("K09",) for n in (79, 80, 85)},
    "CR081": ("K14",), "CR084": ("K05",),
    "CR086": ("K05", "K14"), "CR087": ("K06",), "CR089": ("K14",), "CR090": ("K06",), "CR091": ("K06",), "CR092": ("K06",),
    "CR094": ("K06",), "CR095": ("K06",), "CR097": ("K16",), "CR099": ("K06",), "CR100": ("K15",),
    **{f"CR{n:03d}": ("K11",) for n in range(101, 110)},
    **{f"CR{n:03d}": ("K10",) for n in range(110, 118)},
    **{f"CR{n:03d}": ("K12",) for n in range(118, 125)},
    **{f"CR{n:03d}": ("K13",) for n in range(125, 133)},
    "CR133": ("K02",), "CR134": ("K04",), "CR144": ("K16",), "CR145": ("K16",), "CR147": ("K16",), "CR148": ("K16", "K06"),
    "CR149": ("K15",), "CR150": ("K16",),
    "CR159": ("K03",), "CR160": ("K03",), "CR161": ("K02",), "CR162": ("K04",), "CR163": ("K04", "K07"), "CR164": ("K07",),
    "CR165": ("K08",), "CR166": ("K09",), "CR167": ("K09",), "CR168": ("K05",), "CR171": ("K06",), "CR172": ("K11",),
    "CR173": ("K11",), "CR174": ("K06", "K14"), "CR175": ("K10",), "CR176": ("K06", "K14"), "CR177": ("K12",), "CR178": ("K13",),
    # added 1 Oct 2026 for components built after the first mapping (CAP is consulted before DEMO/RECURSIVE)
    **{f"CR{n:03d}": ("K01", "K02", "K06", "K14", "K16") for n in range(152, 159)},          # the foundation exists/builds/...
    "CR093": ("K25",), "CR135": ("K07",), "CR136": ("K08",), "CR138": ("K06", "K14"), "CR139": ("K10", "K02"),
    "CR140": ("K11",), "CR088": ("K17",), "CR187": ("K11",), "CR182": ("K07",),
    # 1 Oct 2026 evidence pass: each id below maps to the component whose OWN tests exercise it (citations in CITE)
    "CR057": ("K23",), "CR059": ("K23",), "CR083": ("K23",), "CR082": ("K22",), "CR096": ("K25",), "CR207": ("K26",),
    **{f"CR{n:03d}": ("K24",) for n in range(191, 197)},   # K24 recursion: weakness/design/implement/test/measure/use (mechanism)
    # 2 Oct 2026 audit-completion pass: the four boxes that had no code or no check now have both, with tests
    "CR098": ("K14", "K06"), "CR151": ("K16", "K24"), "CR204": ("K16",), "CR208": ("K26", "K16"),
}
# code exists and its component's tests pass, but the box asks for something those tests do not demonstrate (a real run, a
# whole-system property): at most IMPLEMENTED, whatever the test state
CODE_ONLY = {"CR142": ("K24",), "CR143": ("K24",), "CR197": ("K24",), "CR198": ("K24",)}
CITE = {
    "CR057": "creator/oversight.py:60 goal_drift; tests/test_creator_oversight.py::test_an_adoption_under_a_non_owner_objective_is_drift, test_concentrated_work_with_a_flat_metric_level_is_flagged_as_stalled",
    "CR059": "creator/oversight.py:97 knowledge_gaps; tests/test_creator_oversight.py::test_unexplained_failures_become_knowledge_gaps_once",
    "CR083": "creator/oversight.py:139 critical_path; tests/test_creator_oversight.py::test_critical_path_puts_the_most_blocking_gap_first",
    "CR082": "creator/swarm.py parallel workers; tests/test_creator_swarm.py::test_parallel_workers_on_disjoint_modules_all_get_adopted",
    "CR093": "creator/testgen.py:273 generate; tests/test_creator_testgen.py::test_generated_tests_pass_on_the_code_they_came_from, test_a_behaviour_change_makes_a_generated_test_fail",
    "CR096": "creator/testgen.py:103 _edge_values; tests/test_creator_testgen.py::test_candidates_cover_zero_negative_empty_none_and_defaults, test_bool_edge_values_are_not_merged_with_ints",
    "CR207": "creator/reproduce.py:187 reproduce, scripts/reproduce.py; tests/test_creator_reproduce.py::test_clean_ledger_reproduces, test_tampered_evidence_is_not_reproduced; real ledger (no --rerun): 10/10 claims REPRODUCED, chain ok over 1397 records, 713 evidence files unchanged - state/build/evidence/REPRODUCE_20261001.json",
    **{f"CR{n}": "creator/recursion.py (find_weaknesses:119 design_change:161 ab_test:220 step:276); tests/test_creator_recursion.py::"
       "test_weakness_is_found_from_planted_evidence, test_an_improving_change_is_adopted_and_used_by_the_next_iteration; "
       "mechanism on a stand-in workload, no real recursion step is in the ledger" for n in range(191, 197)},
    "CR142": "creator/recursion.py + scripts/creator_recurse.py exist and K24 tests pass, but no RECURSION_STEP finding is in state/creator/ledger.jsonl",
    "CR143": "same as CR142: the mechanism is tested, no real recursive improvement has run",
    "CR197": "needs a real adopted recursion step with a measured improvement; none is in the ledger",
    "CR198": "needs a second real recursion step under different conditions; none is in the ledger",
    "CR098": "tests/test_creator_integration.py drives objective compile -> gap sync -> schedule -> planner -> kernel.execute (sandbox, evaluate, "
             "improvement verdict, merge) -> post-merge audit -> curriculum lesson -> reproduce on a temporary git repo, plus a rejection path "
             "(regressing change: nothing merges, ledger verifies), a post-merge rollback path and a tampered-ledger stop; worker is the only fake",
    "CR151": "creator/audit/checks.py check_recursion (adopted process change needs a recorded weakness, a disjoint confirmation arm, an "
             "IMPROVEMENT claim + ADOPT decision, an unbroken lineage; process.json == last adopted process) and adversary attack "
             "unevidenced_process_change; tests/test_creator_audit_recursion_failures.py. Audits the mechanism: no real recursion step exists yet",
    "CR204": "creator/audit/checks.py check_unresolved_failures + enumerate_failures/failure_report (STATUS 'failures'): a CRITICAL failure "
             "(post-merge rollback, ledger/evidence/audit breach) needs Diagnosis + fix or the owner's ACCEPTED finding; adversary attack "
             "hidden_critical_failure; kernel now records Failure+Diagnosis+Repair on a rollback; tests/test_creator_audit_recursion_failures.py",
    "CR208": "creator/claims.py + scripts/claims_register.py -> state/build/CLAIMS.json: every ImprovementClaim, adoption and headline number with its "
             "evidence refs and a recomputation (creator/reproduce.py); a claim without recomputable evidence is UNSUPPORTED; "
             "tests/test_creator_claims.py::test_a_fabricated_claim_is_flagged_and_an_honest_one_is_not. Register lists its own UNSUPPORTED claims",
}
# boxes that need a demonstrated autonomous development result (an ADOPTED real cycle), not just code
DEMO = {"CR137", "CR141", "CR169", "CR170", "CR179", "CR180", "CR181", "CR183", "CR184", "CR185", "CR186", "CR188", "CR189",
        "CR190"}
RECURSIVE = {f"CR{n:03d}" for n in range(191, 199)} | {"CR142", "CR143"}
AUDIT_BOXES = {"CR146": "hardcoded_answers", "CR199": "ledger_integrity", "CR200": "hardcoded_answers",
               "CR202": "fake_adoption", "CR203": "claim_recompute", "CR205": "sealed_suite", "CR206": "*",
               "CR201": "memorization"}      # creator/audit/checks.py:248 check_memorization; test_creator_audit.py holdout/literal tests


ORDER = ["FAILED", "BLOCKED", "SCIENTIFIC_LIMITATION", "NOT_STARTED", "IN_PROGRESS", "IMPLEMENTED", "TESTING", "VALIDATED"]
# SECTION REQUIREMENT roll-up: section number -> the checklist groups whose items ARE that section's measurable members.
# Sections absent here (doctrine, roles, status states, line depth ...) have no measurable members and stay NOT_STARTED.
SECTION_GROUPS: dict[int, tuple[str, ...]] = {
    9: ("64 FOUNDATION", "79 FOUNDATION"), 10: ("64 FOUNDATION",), 11: ("64 SELF-MODEL",), 12: ("64 OBJECTIVE ENGINE",),
    13: ("64 OBJECTIVE ENGINE",), 14: ("64 GAP ANALYSIS",), 15: ("64 GAP ANALYSIS",), 16: ("64 RESEARCH",), 17: ("64 DESIGN",),
    18: ("64 PLANNING",), 19: ("64 PLANNING",), 20: ("64 PLANNING",), 21: ("64 PLANNING",), 22: ("64 IMPLEMENTATION",),
    23: ("64 IMPLEMENTATION",), 24: ("64 TESTING",), 25: ("64 TESTING",), 26: ("64 DEBUGGING",), 27: ("64 DEBUGGING",),
    28: ("64 DEBUGGING",), 29: ("64 IMPROVEMENT",), 30: ("64 IMPROVEMENT",), 31: ("64 IMPROVEMENT",), 32: ("64 IMPROVEMENT",),
    33: ("64 AUDIT",), 34: ("64 AUDIT",), 35: ("64 AUDIT",), 36: ("64 AUDIT",), 37: ("64 MEMORY",), 38: ("64 MEMORY",),
    39: ("64 META-LEARNING",), 40: ("64 META-LEARNING",), 41: ("64 SELF-MODEL",), 42: ("64 SELF-MODEL",),
    43: ("64 RECURSION",), 44: ("64 RECURSION",), 45: ("64 RECURSION",), 46: ("64 RECURSION",), 47: ("64 AUDIT",),
    48: ("64 AUDIT",), 49: ("79 TRUE AUTONOMY",), 50: ("79 TRUE AUTONOMY",), 51: ("79 TRUE AUTONOMY",),
    52: ("79 TRUE AUTONOMY",), 53: ("79 RECURSIVE IMPROVEMENT",), 54: ("79 RECURSIVE IMPROVEMENT",),
    55: ("79 RECURSIVE IMPROVEMENT",), 56: ("79 VALIDATION",), 57: ("79 VALIDATION",), 58: ("64 FOUNDATION",),
    59: ("64 FOUNDATION",), 60: ("64 FOUNDATION",), 61: ("64 AUDIT",), 66: ("79 RECURSIVE IMPROVEMENT",),
    68: ("79 VALIDATION",), 71: ("64 GAP ANALYSIS",), 72: ("64 META-LEARNING",), 73: ("64 META-LEARNING",),
    74: ("64 META-LEARNING",), 75: ("64 RECURSION",), 76: ("79 VALIDATION",), 77: ("79 TRUE AUTONOMY",),
    78: ("79 VALIDATION",), 80: ("79 VALIDATION",), 83: ("79 TRUE AUTONOMY", "79 RECURSIVE IMPROVEMENT", "79 VALIDATION"),
}
SECTION_EXTRA: dict[str, tuple[str, ...]] = {"A. THE FOUNDATION": ("64 FOUNDATION", "79 FOUNDATION"),
                                             "B. THE AUTONOMOUS DEVELOPER": ("79 AUTONOMOUS DEVELOPER",),
                                             "SELF-MODIFICATION IS NOT SELF-IMPROVEMENT.": ("64 IMPROVEMENT",)}
ROLLUP_CAP = "IMPLEMENTED"      # the members cover a section's bullets only loosely: never above IMPLEMENTED, never VALIDATED here


def _gkey(group: str) -> str:
    for sep in (" THE MASTER CHECKLIST / ", " COMPLETION GATE / "):
        if sep in group:
            head, _, tail = group.partition(sep)
            return f"{head.split('.')[0]} {tail}"
    return group


def rollup(items: list[dict[str, Any]]) -> int:
    """SECTION REQUIREMENT boxes take the status of the WEAKEST member of the groups mapped to their section (capped)."""
    members: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        if not it["description"].startswith("SECTION REQUIREMENT"):
            members.setdefault(_gkey(it["group"]), []).append(it)
    changed = 0
    for it in items:
        if not it["description"].startswith("SECTION REQUIREMENT"):
            continue
        name = it["group"].split("SECTION / ", 1)[1]
        num = name.split(".")[0]
        gs = SECTION_EXTRA.get(name) or (SECTION_GROUPS.get(int(num), ()) if num.isdigit() else ())
        mem = [m for g in gs for m in members.get(g, [])]
        if not mem:
            new = ("NOT_STARTED", "no measurable member items map to this section (doctrine/role/process rule)")
        else:
            worst = min((m["status"] for m in mem), key=ORDER.index)
            weak = sorted(m["id"] for m in mem if m["status"] == worst)[:6]
            st = worst if ORDER.index(worst) <= ORDER.index(ROLLUP_CAP) else ROLLUP_CAP
            new = (st, f"weakest of {len(mem)} members in {', '.join(gs)} is {worst} ({', '.join(weak)}); capped at {ROLLUP_CAP}: "
                       "the section's own bullets are not individually verified")
        if (it["status"], it["notes"]) != new:
            it["status"], it["notes"] = new
            it["evidence"] = ["state/build/CREATOR_MASTER_CHECKLIST.json"] if mem else []
            changed += 1
    return changed


def main() -> int:
    data = json.loads(CHECK.read_text(encoding="utf-8"))
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    caps = {c["id"]: c for c in st["capabilities"]}
    audit = json.loads(AUDIT.read_text(encoding="utf-8")) if AUDIT.is_file() else {}
    findings = audit.get("audit", {}).get("findings", [])
    adversary = audit.get("adversary", [])
    cycles = [json.loads(ln) for ln in KLOG.read_text(encoding="utf-8").splitlines() if ln.strip()] if KLOG.is_file() else []
    adopted = [c for c in cycles if c.get("outcome") == "ADOPTED"]
    p0 = json.loads(PHASE0.read_text(encoding="utf-8")) if PHASE0.is_file() else {}
    changed = 0
    for it in data["items"]:
        iid = it["id"]
        new: tuple[str, list[str], str] | None = None
        n = int(iid[2:]) if iid[2:].isdigit() else 0
        if 1 <= n <= 31:
            key = next((k for k in p0 if k.startswith(f"CR{n:03d}") or (k.startswith("CR001-004") and n <= 4)), None)
            if key:
                new = ("IMPLEMENTED", ["state/build/CREATOR_PHASE0_LEDGER.json", "scripts/phase0_recon.py"],
                       f"computed reconnaissance section {key}")
        elif 32 <= n <= 36:
            new = ("IMPLEMENTED", ["state/build/CREATOR_PHASE0_LEDGER.json", "creator/ARCHITECTURE.md"],
                   "mapped by the computed Phase-0 ledger and the architecture")
        elif iid in CAP:
            ks = CAP[iid]
            states = [caps.get(k, {}).get("state", "NOT_STARTED") for k in ks]
            if all(s == "TESTED" for s in states):
                status = "TESTING"
            elif any(s in ("TESTED", "IMPLEMENTED", "FAILED") for s in states):
                status = "IMPLEMENTED"
            else:
                status = "NOT_STARTED"
            if status != "NOT_STARTED":
                new = (status, ["state/creator/STATUS.json"] + [m for k in ks for m in ("creator/capabilities.json",)][:1],
                       f"capabilities {', '.join(f'{k}={s}' for k, s in zip(ks, states))} (computed from test evidence)"
                       + (f"; {CITE[iid]}" if iid in CITE else ""))
        elif iid in CODE_ONLY:
            ks = CODE_ONLY[iid]
            states = [caps.get(k, {}).get("state", "NOT_STARTED") for k in ks]
            if any(x in ("TESTED", "IMPLEMENTED", "FAILED") for x in states):
                new = ("IMPLEMENTED", ["state/creator/STATUS.json", "creator/capabilities.json"],
                       f"capabilities {', '.join(f'{k}={x}' for k, x in zip(ks, states))}; capped at IMPLEMENTED: {CITE[iid]}")
        elif iid in DEMO:
            if adopted:
                new = ("TESTING", ["state/creator/kernel_log.jsonl"], f"{len(adopted)} real cycle(s) ADOPTED by evidence")
            elif cycles:
                new = ("IN_PROGRESS", ["state/creator/kernel_log.jsonl"],
                       f"{len(cycles)} real cycle(s) ran, none adopted yet: {[c.get('outcome') for c in cycles]}")
        elif iid in RECURSIVE:
            pass                                                        # no recursive cycle has run: stays NOT_STARTED
        elif iid in AUDIT_BOXES:
            chk = AUDIT_BOXES[iid]
            bad = [f for f in findings if chk == "*" or f.get("check") == chk]
            caught = all(a.get("caught") for a in adversary) if adversary else False
            if audit:
                new = ("FAILED" if bad else "TESTING", ["state/creator/AUDIT.json"],
                       f"audit check {chk}: {len(bad)} finding(s); adversary {'all caught' if caught else 'NOT all caught'}"
                       + ("; creator/audit/checks.py:248 check_memorization, adversary 'holdout_in_memory'" if iid == "CR201" else ""))
        if new and (it["status"], it["notes"]) != (new[0], new[2]):
            it["status"], it["evidence"], it["notes"] = new[0], new[1], new[2]
            changed += 1
    changed += rollup(data["items"])
    CHECK.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print("changed", changed, dict(Counter(i["status"] for i in data["items"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
