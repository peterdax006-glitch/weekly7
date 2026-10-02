"""Read-only: for each pending diff / unmeasured lesson, would it re-apply to the current tree exactly, at function level, or not at all?

    python scripts/pending_report.py [--state state/creator] [--tree .]

Nothing is written (git apply --check only; function-level runs in memory)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import pending as P  # noqa: E402


def report(state: Path, tree: Path) -> list[tuple[str, str, str]]:
    rows = []
    for it in P.load_items(state / "pending", state / "lessons.jsonl"):
        r = P.try_apply(it, tree, write=False, repos=[tree])
        detail = f"{len(r.applied)} applied, {len(r.skipped)} skipped" if r.ok else r.why[:90]
        rows.append((it.item_id, r.mode if r.ok else "none", detail))
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(ROOT / "state" / "creator"))
    ap.add_argument("--tree", default=str(ROOT))
    a = ap.parse_args(argv)
    rows = report(Path(a.state), Path(a.tree))
    for r in rows:
        print(f"{r[1]:9} {r[0]:60} {r[2]}")
    counts = {m: sum(1 for r in rows if r[1] == m) for m in ("exact", "function", "none")}
    print(f"total {len(rows)}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
