"""Nupen's GPU PULSE RUNNER - front door (creator/gpupulse.py does the work). Rent a GPU for minutes, spend them on thinking, destroy it.
The plan it implements: docs/GPU_PULSE_BLUEPRINT.md.

  python scripts/gpu_pulse.py checklist              the owner's step-by-step (key, template, config, commands)
  python scripts/gpu_pulse.py init                   write a config skeleton to ~/creator_runtime/gpu/pulse.json (never in the repo)
  python scripts/gpu_pulse.py plan [--pulse N]       dry run: minutes and dollars per job (nothing is contacted)
  python scripts/gpu_pulse.py prepare [--pulse N]    BEFORE renting: stage and check everything that needs no GPU (no paid minute for it)
  python scripts/gpu_pulse.py setup [--serve M]      bootstrap the pod (idempotent): llama.cpp, models (sha256), servers, dead-man switch
  python scripts/gpu_pulse.py tunnel [--serve M]     forward the pod's servers to 127.0.0.1 (writes ~/creator_runtime/gpu/tunnel.json)
  python scripts/gpu_pulse.py bench                  one short timed request per served model through the tunnel (tok/s)
  python scripts/gpu_pulse.py run [--pulse N | JOB ...]  run jobs; results go to Nupen's state with 'gpu_pulse'; GPU busy % + tok/s every minute
  python scripts/gpu_pulse.py status                 ledger: spent / remaining of the budget
  python scripts/gpu_pulse.py teardown [--destroy]   stop servers + tunnel, close the pulse, print (or with --destroy request) the destroy
  python scripts/gpu_pulse.py manifest               exactly what leaves this machine
JOB = <smoke|probe|thinkbench|judgment|drills|traces>:<model>[:<n>[:<max minutes>]]; other modules' jobs: --jobs-file / --jobs-from
(external-job dicts: creator.gpupulse.ext_job). Scripts reach the served models through creator.gpupulse.endpoints()."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creator import gpupulse as GP  # noqa: E402

DEFAULT_RESERVE = int(GP.DEFAULTS["reserve_minutes"])
CHECKLIST = f"""OWNER CHECKLIST - GPU pulses (Vast.ai offer #44585860, 1x RTX 4090 24 GB, $0.343/h, verified, reliability 98.4%)

BEFORE RENTING (free):
1. Make a dedicated SSH key, once (PowerShell):
     ssh-keygen -t ed25519 -f $env:USERPROFILE\\.ssh\\nupen_vast -C nupen-gpu-pulse
   (cmd.exe: ssh-keygen -t ed25519 -f %USERPROFILE%\\.ssh\\nupen_vast -C nupen-gpu-pulse ; an empty passphrase is fine for this
   throw-away key - the private half never leaves this PC.)
2. cloud.vast.ai -> Account (left menu: 'Keys') -> SSH Keys -> '+ New': paste the ONE line of %USERPROFILE%\\.ssh\\nupen_vast.pub
   (type %USERPROFILE%\\.ssh\\nupen_vast.pub to show it). Do it BEFORE renting: keys are copied into an instance when it is created.
3. python scripts/gpu_pulse.py init        writes %USERPROFILE%\\creator_runtime\\gpu\\pulse.json (outside the repository)
4. python scripts/gpu_pulse.py prepare     stages and checks everything that needs no GPU; must say "ready": true
   python scripts/gpu_pulse.py plan        minutes and dollars per job of pulse 1

RENT (the meter starts here):
5. Search -> offer #44585860 (or any verified RTX 4090 near $0.35/h, reliability > 98%). Template 'Llama.cpp' (image vastai/llama-cpp);
   click the template's edit (pencil) before Rent:
     - Image tag: a  ...-cuda-12.9  tag (NOT -cuda-13.2: the host driver is CUDA 13.1 and a 13.2 build can silently run on the CPU).
     - Disk space: 80 GB.
     - Launch mode: SSH on (Jupyter may stay).
     - Environment: LLAMA_MODEL EMPTY (then the template starts no server of its own; setup downloads and sha256-checks the models).
       Leave LLAMA_ARGS / PROVISIONING_SCRIPT unset.
   Rent. Note the time (or put it in the config as "rented_at", epoch seconds, so the ledger counts every billed minute).
6. When the instance shows 'Running': its key/'Connect' button -> 'Direct ssh connect' shows e.g.
     ssh -p 41234 root@185.x.y.z -L 8080:localhost:8080
   Put into %USERPROFILE%\\creator_runtime\\gpu\\pulse.json:
     "host": "185.x.y.z", "port": 41234, "instance_id": "<the instance number shown on the card>"
   ("key_path" is already "~/.ssh/nupen_vast"; no password, no API key, nothing of this goes into the repository.)
7. python scripts/gpu_pulse.py setup       (~6 min: probe, models ~18 GB from Hugging Face, sha256, first server, dead-man switch)
8. python scripts/gpu_pulse.py run --pulse 1
   (logs GPU busy % and tok/s every minute; a job that keeps the GPU under 70% busy for 8 minutes is stopped with the reason;
    the run stops by itself {DEFAULT_RESERVE} min before the budget cap - default $8 across all pulses, ledger %USERPROFILE%\\creator_runtime\\gpu\\ledger.json)
9. python scripts/gpu_pulse.py teardown --destroy
   then check the Vast console shows NO instance (a merely stopped instance still bills storage).
   If the PC cannot reach the pod: vastai destroy instance <id>, or Destroy in the console.

AFTER PULSE 1: set "best_model" in pulse.json to the winner the thinkbench comparisons show; pulses 2-4 use it:
   python scripts/gpu_pulse.py plan --pulse 2   ...   run --pulse 2   (judgment rounds; 3 = reasoning epochs; 4 = worked-example bank)
Safety nets: budget cap on this PC + a dead-man timer on the pod that runs 'vastai stop instance' on itself when the budget's time is up
(config "deadman": "destroy" to destroy instead, "off" to disable)."""

SKELETON = {"host": "", "port": 22, "user": "root", "key_path": "~/.ssh/nupen_vast", "instance_id": "", "rented_at": None, "usd_per_hr": 0.343,
            "bandwidth_usd_per_tb": 0.0, "budget_usd": 8.0, "models": list(GP.DEFAULTS["models"]), "best_model": GP.M8, "slots": "auto",
            "ctx_per_slot": 8192, "deadman": "stop"}


def _say(s: str) -> None:
    print(s, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("checklist", "init", "plan", "prepare", "setup", "tunnel", "bench", "run", "status", "teardown", "manifest"))
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--pulse", type=int, default=1, help="the blueprint's pulse number (1-4) when no JOB is given")
    ap.add_argument("--jobs-file", default="", help="a JSON list of jobs (strings and/or external-job dicts, see creator.gpupulse.ext_job)")
    ap.add_argument("--jobs-from", default="", help="'package.module:function' returning such a list (e.g. another module's GPU-day list)")
    ap.add_argument("--serve", action="append", default=None, help="model(s) to serve (default: the first configured)")
    ap.add_argument("--destroy", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="requests in flight (default: the served model's slots)")
    ap.add_argument("--state", default=str(ROOT / "state" / "creator"))
    ap.add_argument("--owner-dir", default=str(Path.home() / "Masterstock"))
    a = ap.parse_args(argv)
    if a.cmd == "checklist":
        print(CHECKLIST)
        return 0
    if a.cmd == "init":
        p = GP.config_path()
        if p.exists():
            print(f"{p} exists; not overwritten")
            return 0
        GP._write_json(p, SKELETON)
        print(f"wrote {p}: fill in host, port, instance_id after renting")
        return 0
    try:
        cfg = GP.load_config()
        jobs = GP.job_list(cfg, a.jobs, a.pulse, a.jobs_file, a.jobs_from)
        if a.cmd == "plan":
            print(f"pulse {a.pulse if not a.jobs else '(custom)'}: {len(jobs)} jobs")
            print(GP.format_plan(GP.plan(cfg, jobs, Path(a.state))))
            return 0
        if a.cmd == "prepare":
            r = GP.prepare(cfg, jobs, Path(a.state), ROOT)
            print(json.dumps({k: r[k] for k in ("ready", "blockers", "owner_todo", "checks")}, indent=1))
            print(GP.format_plan(r["plan"]))
            print(f"written: {GP.gpu_dir() / 'prepared.json'}")
            return 0 if r["ready"] else 4
        if a.cmd == "manifest":
            print(json.dumps(GP.manifest(cfg), indent=1))
            return 0
        if a.cmd == "status":
            b = GP.budget_for(cfg)
            print(json.dumps({"spent_usd": b.spent(), "remaining_usd": b.remaining(), "cap_usd": b.cap_usd,
                              "minutes_left_at_rate": round(b.seconds_left(float(cfg["usd_per_hr"])) / 60, 1), "open_pulse": b.open_pulse()}, indent=1))
            return 0
        if a.cmd == "setup":
            r = GP.setup(cfg, serve_models=a.serve, say=_say)
            print(json.dumps({k: r[k] for k in ("fetched", "cached", "plan", "seconds")}))
        elif a.cmd == "tunnel":
            plan = GP.serve(cfg, a.serve or list(cfg["models"])[:1], say=_say)
            pulse = GP.budget_for(cfg).begin(float(cfg["usd_per_hr"]), str(cfg.get("instance_id", "")))
            print(f"tunnel file: {GP.open_tunnel(cfg, plan, pulse['id'])}  (jobs use it with env NUPEN_GPU_PULSE=<that file>)")
        elif a.cmd == "bench":
            t = json.loads(GP.tunnel_path().read_text(encoding="utf-8"))
            for m, ports in t["models"].items():
                body = json.dumps({"messages": [{"role": "user", "content": "Count from 1 to 200 separated by spaces. /no_think"}],
                                   "max_tokens": 400, "temperature": 0}).encode()
                t0 = time.monotonic()
                req = urllib.request.Request(f"http://127.0.0.1:{ports[0]}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.loads(r.read())
                dt_s = time.monotonic() - t0
                n = int(d.get("usage", {}).get("completion_tokens", 0))
                tm = d.get("timings", {})
                print(f"{m}: {n} tokens in {dt_s:.1f} s wall = {n / dt_s:.0f} tok/s incl. tunnel; server says {tm.get('predicted_per_second', 0):.0f} tok/s")
        elif a.cmd == "run":
            res = GP.run_jobs(cfg, jobs, Path(a.state), ROOT, Path(a.owner_dir), workers=a.workers, say=_say)
            print(json.dumps(res, indent=1)[:4000])
            print("Done. Next: python scripts/gpu_pulse.py teardown --destroy")
        elif a.cmd == "teardown":
            GP.teardown(cfg, destroy=a.destroy, say=_say)
    except GP.PulseError as e:
        print(f"STOPPED: {e}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
