"""GPU-day CODER HARNESS - replay held-out handoffs through a served model and judge every answer with the home kernel's sandbox
gates, evaluation only (nothing is ever merged). The process lowers itself to IDLE priority; generations run ahead on the GPU while
sandbox evaluations run one at a time on this PC.

  python scripts/gpuday/coder_harness.py --url http://127.0.0.1:18140 [--n 8] [--eval FILE ...] [--repo DIR] [--out FILE]
     --url     the llama-server (through the pulse tunnel: creator.gpupulse.endpoints()['models'][m]['urls'][0] without '/v1')
     --repo    a SEPARATE clone (default ~/creator_runtime/gpuday/harness_repo, made with `git clone ~/weekly7` on first use) so the
               sandboxes' branches never touch the main repository's refs
     --eval    held-out files of an export (default: commit_eval.jsonl and handoff_eval.jsonl)"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from creator import gpuday as GD  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--repo", default=str(GD.default_out().parent / "harness_repo"))
    ap.add_argument("--source", default=str(Path.home() / "weekly7"))
    ap.add_argument("--out", default=str(GD.default_out().parent / "harness_result.json"))
    ap.add_argument("--gen-threads", type=int, default=4)
    ap.add_argument("--think", action="store_true")
    a = ap.parse_args(argv)
    GD._idle_priority()
    ex = GD.default_out()
    files = [Path(f) for f in a.eval] or [ex / "commit_eval.jsonl", ex / "handoff_eval.jsonl"]
    repo = GD.harness_clone(Path(a.repo).expanduser(), Path(a.source).expanduser())
    rep = GD.run_harness(repo, files, a.url, n=a.n, scratch=Path(a.repo).expanduser().parent / "harness_sandboxes", out=Path(a.out).expanduser(),
                         gen_threads=a.gen_threads, think=a.think)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
