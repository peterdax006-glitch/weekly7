"""Independent reproduction of the Creator's critical results (CR207/CR208). Read-only on the repo and ledger.

    python scripts/reproduce.py --repo <repo> --ledger <repo>/state/creator/ledger.jsonl --out report.json [--rerun]

Exit 0 when every claim is REPRODUCED and the chain is intact, 1 otherwise."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from creator.reproduce import REPRODUCED, reproduce  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", help="write the JSON report here (default: stdout)")
    ap.add_argument("--rerun", action="store_true", help="re-run each ADOPT's touched tests in a temporary detached worktree")
    ap.add_argument("--rerun-timeout", type=int, default=600)
    a = ap.parse_args(argv)
    rep = reproduce(a.repo, a.ledger, rerun=a.rerun, rerun_timeout=a.rerun_timeout)
    text = json.dumps(rep, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(json.dumps({"counts": rep["counts"], "chain": rep["chain"], "evidence": rep["evidence"], "overall": rep["overall"]}))
    else:
        print(text)
    return 0 if rep["overall"] == REPRODUCED else 1


if __name__ == "__main__":
    raise SystemExit(main())
