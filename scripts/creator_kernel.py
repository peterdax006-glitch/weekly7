"""Run the Creator's self-development kernel on this repository.

    python scripts/creator_kernel.py --cycles 1 --allow-agent-calls [--call-usd 1.5] [--daily-usd 5]

Without --allow-agent-calls the worker's budget refuses before any spend (the cycle reports BUDGET). The kernel never pushes; review
`git log` and state/creator/cycles/<package>/ before pushing an adopted change."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import agents as AG  # noqa: E402
from creator import kernel as K  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--allow-agent-calls", action="store_true")
    ap.add_argument("--call-usd", type=float, default=0.75)
    ap.add_argument("--daily-usd", type=float, default=5.0)
    ap.add_argument("--daily-calls", type=int, default=12)
    a = ap.parse_args(argv)
    if a.allow_agent_calls:
        os.environ["CREATOR_AGENT_CALLS"] = "1"
    os.environ["CREATOR_AGENT_CALL_USD"] = str(a.call_usd)
    os.environ["CREATOR_AGENT_DAILY_USD"] = str(a.daily_usd)
    os.environ["CREATOR_AGENT_DAILY_CALLS"] = str(a.daily_calls)
    os.environ["CREATOR_AGENT_JOB_CALLS"] = "1"
    budget = AG.Budget()
    spec = dataclasses.replace(AG.DEVELOPER, role="implementer", timeout_s=1800)
    worker = K.AgentWorker(budget, spec)
    cfg = K.KernelConfig(repo=ROOT, state=ROOT / "state" / "creator")
    print(f"budget: {budget.policy} today={budget.today()}", flush=True)
    reps = K.run(cfg, worker, a.cycles, on_cycle=lambda r: print(json.dumps(dataclasses.asdict(r), default=str)[:3000],
                                                                   flush=True))
    print("SUMMARY", json.dumps(K.summary(reps)), "budget today", budget.today(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
