"""Nupen goal proposals: `propose` (computes and prints), `list`, `approve ID --by owner|teacher`, `reject ID --reason TEXT`.

State defaults to <repo>/state/creator; override with --state/--ledger/--repo (tests and read-only experiments on copies)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import goals as GO  # noqa: E402
from creator.ledger import Ledger  # noqa: E402


def _show(p: dict) -> None:
    print(f"{p['id']} [{p.get('status', 'NEW')}] ({p['source']}) value {p['value']} cost {p['cost']['packages']} pkgs: {p['title']}")
    print(f"    why: {p['rationale']}")
    print(f"    evidence: {', '.join(p['evidence'][:6])}{' ...' if len(p['evidence']) > 6 else ''}")
    print(f"    spec: modules {p['spec']['modules']} tests {p['spec']['tests']}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["propose", "list", "approve", "reject"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("--by", choices=["owner", "teacher"])
    ap.add_argument("--reason", default="")
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--state", type=Path)
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    state = a.state or a.repo / "state" / "creator"
    led = Ledger(a.ledger or state / "ledger.jsonl", evidence_root=a.repo)
    try:
        if a.cmd == "propose":
            new = GO.propose(led, state, a.repo)
            print(f"{len(new)} new proposal(s), {len(GO.pending(state))} pending")
            for p in new:
                _show({**p.to_dict(), "status": "NEW"})
        elif a.cmd == "list":
            rows = GO.listing(state)
            if a.json:
                print(json.dumps(rows, indent=1))
            for p in rows:
                _show(p)
        elif a.cmd == "approve":
            if not a.id or not a.by:
                ap.error("approve needs ID and --by owner|teacher")
            print("approved as", GO.approve(state, a.repo, led, a.id, a.by))
        else:
            if not a.id:
                ap.error("reject needs ID and --reason")
            GO.reject(state, a.id, a.reason)
            print("rejected (permanent)", a.id)
    except GO.GoalError as e:
        print("ERROR:", e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
