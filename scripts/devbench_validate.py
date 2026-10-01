"""Calibrate the sealed devbench (reference / null / liar / cheat on every task) and save the evidence. Exit 0 only if calibrated."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import devbench as D  # noqa: E402
from creator.ledger import tree_hash  # noqa: E402

OUT = ROOT / "state" / "creator" / "devbench_validation.json"


def main() -> int:
    t0 = time.time()
    m = D.load_manifest()
    v = D.validate_suite(manifest=m)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"ok": v.ok, "problems": list(v.problems), "per_task": v.per_task, "manifest_digest": m["digest"],
                               "creator_tree": tree_hash(ROOT / "creator"), "seconds": round(time.time() - t0),
                               "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1), encoding="utf-8")
    for tid, row in v.per_task.items():
        print(tid, row, flush=True)
    print("CALIBRATED" if v.ok else f"NOT CALIBRATED: {v.problems}", round(time.time() - t0), "s", flush=True)
    return 0 if v.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
