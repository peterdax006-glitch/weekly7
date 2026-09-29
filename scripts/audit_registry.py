"""Audit state/experiments.jsonl against the Phase 0.2 field list (read-only) and write
state/research/registry_audit.json.  Usage: python scripts/audit_registry.py [path-to-log]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import config as K  # noqa: E402
from engine.registry import REQUIRED, Registry  # noqa: E402


def main(path=None, out=None):
    reg = Registry(path)
    a = reg.audit()
    n = max(1, a["n_records"])
    by_event = {}
    for r in reg.records:
        by_event.setdefault(r.get("event", "?"), []).append(r)
    from engine.registry import _empty, _get
    per_event = {e: {"records": len(rs), "fully_complete": sum(all(not _empty(_get(r, f)) for f in REQUIRED) for r in rs),
                     "missing_fields": {f: sum(_empty(_get(r, f)) for r in rs) for f in REQUIRED
                                        if any(_empty(_get(r, f)) for r in rs)}} for e, rs in sorted(by_event.items())}
    report = {"log": str(reg.path), "n_records": a["n_records"], "ok": a["ok"], "bad_lines": a["bad_lines"],
              "field_completeness": {f: round(c, 4) for f, c in a["completeness"].items()},
              "missing_counts": {f: len(v) for f, v in a["missing_fields"].items()},
              "records_fully_complete": sum(all(not _empty(_get(r, f)) for f in REQUIRED) for r in reg.records),
              "duplicate_ids": a["duplicate_ids"], "orphans_no_outcome": len(a["orphans_no_outcome"]),
              "unknown_outcomes": a["unknown_outcomes"], "reproducibility_conflicts": a["reproducibility_conflicts"],
              "by_event": per_event, "summary": reg.summary(),
              "records_with_provenance_stamp": sum(1 for r in reg.records if r.get("git_commit")) / n}
    dest = Path(out) if out else K.STATE / "research" / "registry_audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = main(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps({k: r[k] for k in ("n_records", "ok", "records_fully_complete", "orphans_no_outcome", "missing_counts",
                                        "duplicate_ids", "records_with_provenance_stamp")}, indent=1, default=str))
