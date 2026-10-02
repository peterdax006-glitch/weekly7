"""C77 sec 6 - Phase 0 reconnaissance CLI (checklist CR001-CR031). The computation lives in creator/recon.py (tested by
tests/test_creator_recon.py). Writes state/build/CREATOR_PHASE0_LEDGER.json (+ .md). Reading-only except `--collect`
(pytest --collect-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import recon  # noqa: E402


def main(argv: list[str]) -> int:
    ledger = recon.compute(ROOT, masterstock=Path.home() / "Masterstock" / "MASTERSTOCK.md", collect="--collect" in argv)
    out = ROOT / "state" / "build" / "CREATOR_PHASE0_LEDGER.json"
    out.write_text(json.dumps(ledger, indent=1, default=str), encoding="utf-8")
    md = ["# C77 Phase 0 - computed reconnaissance ledger", "",
          f"Generated {ledger['generated']} by scripts/phase0_recon.py (creator/recon.py; reading only). Tree {ledger['tree_hash'][:16]}, "
          f"self-model digest {ledger['selfmodel_digest']}.", ""]
    for k, v in ledger.items():
        if k.startswith("CR"):
            body = json.dumps(v, indent=1, default=str)
            md += [f"## {k}", "", "```json", body[:4000] + ("\n... (truncated; full in the .json)" if len(body) > 4000 else ""),
                   "```", ""]
    (ROOT / "state" / "build" / "CREATOR_PHASE0_LEDGER.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, (dict, list)) else (len(v) if isinstance(v, list) else list(v)[:6]))
                      for k, v in ledger.items()}, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
