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
    "CR093": ("K18",), "CR135": ("K07",), "CR136": ("K08",), "CR138": ("K06", "K14"), "CR139": ("K10", "K02"),
    "CR140": ("K11",), "CR088": ("K17",), "CR187": ("K11",), "CR182": ("K07",),
    **{f"CR{n:03d}": ("K19",) for n in (192, 193, 194, 195)},          # autotune designs/implements/tests/measures a change
}
# boxes that need a demonstrated autonomous development result (an ADOPTED real cycle), not just code
DEMO = {"CR137", "CR141", "CR169", "CR170", "CR179", "CR180", "CR181", "CR183", "CR184", "CR185", "CR186", "CR188", "CR189",
        "CR190"}
RECURSIVE = {f"CR{n:03d}" for n in range(191, 199)} | {"CR142", "CR143", "CR151"}
AUDIT_BOXES = {"CR146": "hardcoded_answers", "CR199": "ledger_integrity", "CR200": "hardcoded_answers",
               "CR202": "fake_adoption", "CR203": "claim_recompute", "CR205": "sealed_suite", "CR206": "*"}


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
                       f"capabilities {', '.join(f'{k}={s}' for k, s in zip(ks, states))} (computed from test evidence)")
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
                       f"audit check {chk}: {len(bad)} finding(s); adversary {'all caught' if caught else 'NOT all caught'}")
        if new and (it["status"], it["notes"]) != (new[0], new[2]):
            it["status"], it["evidence"], it["notes"] = new[0], new[1], new[2]
            changed += 1
    CHECK.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print("changed", changed, dict(Counter(i["status"] for i in data["items"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
