"""GPU-day job list for the pulse runner (creator/gpuday.py: DAY, day_jobs).

  python scripts/gpuday/jobs.py plan                         the 24 h blocks: hours, $ at $0.343/h, what comes home, readiness
  python scripts/gpuday/jobs.py write --out list.json        the runner's job list (needs a finished export; embeds the audited upload)
  python scripts/gpuday/jobs.py check                        validate the job list against creator.gpupulse (no network, no pod)
The runner can also take the list directly:  python scripts/gpu_pulse.py run --jobs-from creator.gpuday:day_jobs"""
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
    ap.add_argument("cmd", choices=["plan", "write", "check"])
    ap.add_argument("--out", default="")
    ap.add_argument("--export", default=str(GD.default_out()))
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        p = GD.day_plan()
        for b in p["blocks"]:
            print(f"{b['start_h']:5.1f}-{b['end_h']:4.1f} h  ${b['usd']:5.2f}  [{b['ready']}]  {b['name']}\n      home: {b['comes_home']}")
        print(f"total {p['total_hours']} h  ${p['total_usd']}")
        return 0
    from creator import gpupulse as GP
    cfg = GP.load_config()
    jobs = GD.day_jobs(cfg, Path(a.export))
    for j in jobs:
        GP.ext_job(j) if isinstance(j, dict) else GP.parse_job(j)
    if a.cmd == "write":
        out = GD.outside_repo(Path(a.out or GD.default_out().parent / "day_jobs.json"))
        out.write_text(json.dumps(jobs), encoding="utf-8")
        print(f"{len(jobs)} jobs -> {out} ({out.stat().st_size // 1024} KB)")
    else:
        print(f"{len(jobs)} jobs valid: " + ", ".join(j["name"] if isinstance(j, dict) else j.split(':')[0] + ':' + j.split(':')[1][:24] for j in jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
