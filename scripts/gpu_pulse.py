"""Nupen's GPU PULSE RUNNER - front door (creator/gpupulse.py does the work). Rent a GPU for minutes, spend them on thinking, destroy it.

  python scripts/gpu_pulse.py checklist            the owner's step-by-step (key, template, config, commands)
  python scripts/gpu_pulse.py init                 write a config skeleton to ~/creator_runtime/gpu/pulse.json (never in the repo)
  python scripts/gpu_pulse.py setup [--serve M]    bootstrap the pod (idempotent): llama.cpp, models (sha256), servers, dead-man switch
  python scripts/gpu_pulse.py tunnel [--serve M]   forward the pod's servers to 127.0.0.1 (writes ~/creator_runtime/gpu/tunnel.json)
  python scripts/gpu_pulse.py bench                one short timed request per served model through the tunnel (tok/s)
  python scripts/gpu_pulse.py run [JOB ...]        run jobs (default: the planned first pulse); results go to Nupen's state with 'gpu_pulse'
  python scripts/gpu_pulse.py status               ledger: spent / remaining of the budget
  python scripts/gpu_pulse.py teardown [--destroy] stop servers + tunnel, close the pulse, print (or with --destroy request) the destroy
  python scripts/gpu_pulse.py manifest             exactly what leaves this machine
JOB = thinkbench:<model> | judgment:<model>:<batches>[:<max minutes>] | drills:<model>:<batches>[:<max minutes>]"""
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

M4, M8, M14 = "Qwen3-4B-Q4_K_M.gguf", "Qwen3-8B-Q4_K_M.gguf", "Qwen3-14B-Q4_K_M.gguf"
FIRST_PULSE = [f"thinkbench:{M4}", f"thinkbench:{M8}", f"judgment:{M8}:60:12", f"drills:{M8}:40:8",
               f"thinkbench:{M14}", f"judgment:{M14}:40:10"]

CHECKLIST = f"""OWNER CHECKLIST - first GPU pulse (Vast.ai offer #44585860, RTX 4090 24 GB, $0.343/hr)

1. Make a dedicated SSH key (PowerShell or Git Bash, once):
     ssh-keygen -t ed25519 -f %USERPROFILE%\\.ssh\\nupen_vast -C nupen-gpu-pulse
   (empty passphrase is fine for this throw-away key; the private key never leaves this PC.)
2. Vast.ai console -> Account (or Keys) -> SSH Keys -> Add: paste the ONE line of  %USERPROFILE%\\.ssh\\nupen_vast.pub
   (do this BEFORE renting; keys are copied into new instances at creation).
3. Rent: search offer #44585860 -> template 'Llama.cpp' (image vastai/llama-cpp). Edit the template before Rent:
     - image tag: prefer the newest  ...-cuda-12.9  tag (runs natively on the host's 13.1 driver). The default ...-cuda-13.2 tag
       relies on CUDA minor-version compatibility on a 13.1 driver; setup checks the CUDA device and falls back automatically.
     - Disk: 80 GB.   Launch mode: SSH (keep SSH on; Jupyter may stay).   Env LLAMA_MODEL: leave EMPTY (setup fetches and verifies
       the models itself and would otherwise stop the template's server on port 18000 anyway).   Leave LLAMA_ARGS / PROVISIONING_SCRIPT unset.
4. When the instance is 'Running': click the key icon / 'Connect' and copy the 'Direct ssh connect' line, e.g.
     ssh -p 41234 root@185.x.y.z -L 8080:localhost:8080
5. python scripts/gpu_pulse.py init     then edit %USERPROFILE%\\creator_runtime\\gpu\\pulse.json:
     "host": "185.x.y.z", "port": 41234, "key_path": "~/.ssh/nupen_vast", "instance_id": "<the instance number>",
     "rented_at": <optional: epoch seconds when you clicked Rent, so the ledger counts every billed minute>
   (outside the repository; never commit it. No password and no account API key are needed.)
6. python scripts/gpu_pulse.py setup          (~5 min: probe, models 18 GB from Hugging Face at ~100 MB/s, sha256, first server)
7. python scripts/gpu_pulse.py run            (the planned first pulse; stops by itself at the budget cap)
8. python scripts/gpu_pulse.py teardown --destroy     then check the Vast console shows NO instance (storage bills while stopped).
   Without --destroy it prints:  vastai destroy instance <id>  (or use Destroy in the console).
Safety nets: the run stops {DEFAULT_RESERVE} min before the cap (default $8 across all pulses, ledger ~/creator_runtime/gpu/ledger.json),
and the pod carries a dead-man timer that runs 'vastai stop instance' on itself when the budget's time is up (config "deadman": "destroy" to
destroy instead, "off" to disable)."""
DEFAULT_RESERVE = int(GP.DEFAULTS["reserve_minutes"])

SKELETON = {"host": "", "port": 22, "user": "root", "key_path": "~/.ssh/nupen_vast", "instance_id": "", "usd_per_hr": 0.343,
            "bandwidth_usd_per_tb": 0.0, "budget_usd": 8.0, "models": [M4, M8, M14], "slots": 4, "ctx_per_slot": 8192, "deadman": "stop"}


def _say(s: str) -> None:
    print(s, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("checklist", "init", "setup", "tunnel", "bench", "run", "status", "teardown", "manifest", "plan"))
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--serve", action="append", default=None, help="model(s) to serve (default: the first configured)")
    ap.add_argument("--destroy", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="parallel requests (default: the pod's slots)")
    ap.add_argument("--state", default=str(ROOT / "state" / "creator"))
    ap.add_argument("--owner-dir", default=str(Path.home() / "Masterstock"))
    a = ap.parse_args()
    if a.cmd == "checklist":
        print(CHECKLIST)
        return 0
    if a.cmd == "plan":
        print("\n".join(FIRST_PULSE))
        return 0
    if a.cmd == "init":
        p = GP.config_path()
        if p.exists():
            print(f"{p} exists; not overwritten")
            return 0
        GP._write_json(p, SKELETON)
        print(f"wrote {p}: fill in host, port, instance_id")
        return 0
    cfg = GP.load_config()
    if a.cmd == "manifest":
        print(json.dumps(GP.manifest(cfg), indent=1))
        return 0
    if a.cmd == "status":
        b = GP.budget_for(cfg)
        print(json.dumps({"spent_usd": b.spent(), "remaining_usd": b.remaining(), "cap_usd": b.cap_usd,
                          "minutes_left_at_rate": round(b.seconds_left(float(cfg["usd_per_hr"])) / 60, 1), "open_pulse": b.open_pulse()}, indent=1))
        return 0
    try:
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
            jobs = a.jobs or FIRST_PULSE
            res = GP.run_jobs(cfg, jobs, Path(a.state), ROOT, Path(a.owner_dir), workers=a.workers or int(cfg.get("slots", 4)), say=_say)
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
