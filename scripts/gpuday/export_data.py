"""GPU-day training data export (creator/gpuday.py does the work). Run at IDLE priority:

  ~/weekly7/.venv/Scripts/python.exe ~/weekly7/scripts/lowprio.py --idle ~/weekly7/.venv/Scripts/python.exe scripts/gpuday/export_data.py
      [--state ~/weekly7/state/creator] [--repo ~/weekly7] [--out ~/creator_runtime/gpuday/export] [--strict] [--bank FILE] [--no-commits]
      [--rl-tasks]          also (re)build rl_tasks.jsonl from the repo's pure functions (scripts/gpuday/rl_grpo.py make-tasks)
Prints the manifest (counts, drops, splits). The output directory is refused inside the repository; `audit` re-reads it."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from creator import gpuday as GD  # noqa: E402


def main(argv: list[str]) -> int:
    home = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(home / "weekly7" / "state" / "creator"))
    ap.add_argument("--repo", default=str(home / "weekly7"))
    ap.add_argument("--out", default=str(GD.default_out()))
    ap.add_argument("--strict", action="store_true", help="also exclude the frozen planning items' candidate packages")
    ap.add_argument("--bank", default="")
    ap.add_argument("--no-commits", action="store_true")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--rl-tasks", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    man = GD.export(Path(a.state), Path(a.repo), out, strict=a.strict, bank=Path(a.bank) if a.bank else None, eval_frac=a.eval_frac,
                    with_commits=not a.no_commits)
    if a.rl_tasks:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gpuday" / "rl_grpo.py"), "make-tasks", "--repo", a.repo, "--out",
                            str(out / "rl_tasks.jsonl")], capture_output=True, text=True, timeout=3600)
        man["rl_tasks"] = r.stdout.strip()[-400:]
    bad = GD.audit_export(out, GD.Frozen.load(Path(a.state) / "thinkbench" / "items.json", strict=a.strict)
                          if (Path(a.state) / "thinkbench" / "items.json").is_file() else GD.Frozen.empty())
    man["audit"] = bad or "clean"
    print(json.dumps({k: v for k, v in man.items() if k != "format"}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
