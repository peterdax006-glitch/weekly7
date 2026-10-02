"""Nupen constraint loop CLI: `measure [--window-h H] [--state DIR] [--no-act]` measures, snapshots and acts; `history` lists snapshots."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import constraints as CON  # noqa: E402


def _arg(argv: list[str], flag: str, default: str) -> str:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else default


def main(argv: list[str]) -> int:
    state = Path(_arg(argv, "--state", str(ROOT / "state" / "creator")))
    cmd = argv[0] if argv else "measure"
    if cmd == "history":
        for r in CON.history(state):
            if r.get("event") == "snapshot":
                print(r["at"], "top", r["top"], [(x["name"], x["score"]) for x in r["ranked"][:3]])
            elif r.get("event") in ("note", "act"):
                print("  ", r["event"], r.get("text") or r.get("constraint"), r.get("proposal", ""))
        return 0
    rep = CON.run(state, window_h=float(_arg(argv, "--window-h", "24")), do_act="--no-act" not in argv)
    print(CON.format_report(rep))
    if "--json" in argv:
        print(json.dumps(rep, indent=1, default=str))
    print("ACT", json.dumps(rep["act"], default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
