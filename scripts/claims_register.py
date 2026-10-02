"""Generate the claims register state/build/CLAIMS.json (CR208): every major claim with its evidence and a recomputation.

    python scripts/claims_register.py [--repo .] [--ledger state/creator/ledger.jsonl] [--extra claims_extra.json] [--live-adversary]

A claim without recomputable evidence is listed as UNSUPPORTED. Exit code 0 always when the register was written (the register IS the
report); `--strict` exits 1 if anything is UNSUPPORTED. Read-only on the ledger and the repo."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import claims as C  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--extra", type=Path, default=None, help="JSON list of hand-written claims")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rerun", action="store_true", help="also re-run each adoption's touched tests (slow)")
    ap.add_argument("--live-adversary", action="store_true", help="re-run every adversary attack now")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    ledger = a.ledger or repo / "state" / "creator" / "ledger.jsonl"
    extra = json.loads(a.extra.read_text(encoding="utf-8")) if a.extra and a.extra.is_file() else []
    reg = C.build_register(repo, ledger, extra, rerun=a.rerun, live_adversary=a.live_adversary)
    out = a.out or repo / "state" / "build" / "CLAIMS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reg, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"counts": reg["counts"], "unsupported": reg["unsupported"][:12]}))
    return 1 if a.strict and reg["unsupported"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
