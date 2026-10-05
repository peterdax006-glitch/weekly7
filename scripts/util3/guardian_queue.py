"""h64 GPU guardian work queue, PC side (owner 3 Oct: "when fine tuning run a separate task that still needs to get done then and enhance tasks
based on available data and priority"; priorities 1 efficiency, 2 coding, 3 language).

Writes the pod's queue.txt (guard.sh serves the best item that fits the free VRAM; traces are the last fallback, driven on the pod by feed.py)
and drives the queued JOBS against whatever filler port serves their model, through one SSH tunnel:
  efficiency - the effladder variants the night's runner skipped (creator.effladder.run, same items, configs and sample sizes as
               jobs_tiers.json; restart-safe: recorded (item, config) pairs are skipped; results -> state/creator/thinking/effladder.jsonl;
               the coding tests run here on the PC at idle priority inside effladder). Note: these variants share the GPU with other fillers,
               so their tok/s rows are contended; accuracy is unaffected.
A job's line is removed from queue.txt once a pass finds nothing left to ask; guard.sh then stops its instance and serves the next item.
    python scripts/util3/guardian_queue.py --host H --sshport P [--relay host:port]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

RT = Path.home() / ("creator_runtime/util3/guardian")
PORTS = (18350, 18351, 18352, 18353)
LADDER_JOBS = Path.home() / ("creator_runtime/gpu/ladder/jobs_tiers.json")
EFFICIENCY = ["Qwen3-1.7B-Q3_K_M.gguf", "Qwen3-4B-Q3_K_M.gguf", "Qwen3-0.6B-Q5_K_M.gguf", "Qwen3-1.7B-Q5_K_M.gguf", "Qwen3-4B-Q5_K_M.gguf"]
TRACES = [(8, "Qwen3-4B-Q4_K_M.gguf"), (9, "Qwen3-1.7B-Q4_K_M.gguf")]
CODING = [(5, "Qwen3-4B-Q4_K_M.gguf")]               # pod-driven role trajectories (coder.py); below efficiency, above traces


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=str(Path.home() / "weekly7"))
    ap.add_argument("--host", required=True)
    ap.add_argument("--sshport", required=True)
    ap.add_argument("--relay", default="")
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "nupen_vast"))
    a = ap.parse_args()
    main_dir = Path(a.main)
    sys.path.insert(0, str(main_dir))
    sys.path.insert(0, str(main_dir / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.chdir(main_dir)
    from lowprio import lower_own_priority
    lower_own_priority()                                                  # BELOW_NORMAL: it feeds the paid GPU
    from creator import effladder as E
    from creator import gpupulse as GP
    import guardian_sync as GS
    state = main_dir / "state" / "creator"
    routes = [(a.host, a.sshport)] + ([tuple(a.relay.split(":"))] if a.relay else [])

    def ssh(cmd: str) -> str:
        return GS.run_ssh(a, cmd, timeout=120).decode(errors="replace")

    spec = {j["model"]: j["args"] for j in json.loads(LADDER_JOBS.read_text()) if j.get("call") == "creator.effladder:ladder_job"}
    body = E.load_items(Path(spec[EFFICIENCY[0]]["items"]))
    lines = []
    for m in EFFICIENCY:
        e = E.LADDER[m]
        lines.append(f"1 {m} effladder {e['bytes']} {e['sha256']} {GP.HF_BASE}/{e['repo']}/resolve/main/{m}")
    lines += [f"{p} {m} coding" for p, m in CODING] + [f"{p} {m} traces" for p, m in TRACES]
    GS.run_ssh(a, "cat > /root/guardian/queue.txt.new && mv -f /root/guardian/queue.txt.new /root/guardian/queue.txt",
               stdin=("\n".join(lines) + "\n").encode())
    log(f"queue written: {len(EFFICIENCY)} efficiency jobs, traces fallback")
    remaining = set(EFFICIENCY)
    running: dict[int, threading.Thread] = {}

    def drive(port: int, model: str) -> None:
        args = spec.get(model) or spec[EFFICIENCY[0]]
        ep = E.Endpoint(port, model)
        errs = {"n": 0}

        def say(s: str) -> None:
            if s.startswith("ladder "):
                log(f"{port} {s[:300]}")
        try:
            res = E.run(body, ep, state, suites=list(args.get("suites") or ["coding"]), configs=args.get("configs"), n=args.get("n"),
                        workers=48, run_id=f"guardian-ladder-{model}", pulse="guardian", say=say)
        except Exception as e:                                           # noqa: BLE001
            log(f"{port} {model}: {e!r}")
            return
        if res.get("calls", 0) == 0 and res.get("errors", 0) == 0:       # nothing left to ask: the job is done
            ssh(f"sed -i '/ {model} /d' /root/guardian/queue.txt")
            remaining.discard(model)
            log(f"job done: {model} (removed from the queue)")

    fw = [x for p in PORTS for x in ("-L", f"{p}:127.0.0.1:{p}")]
    ri = 0

    def tunnel(r):
        return subprocess.Popen(["ssh", "-i", a.key, "-p", r[1], "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-o", "ExitOnForwardFailure=yes",
                                 "-o", "ServerAliveInterval=15", "-N", *fw, f"root@{r[0]}"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tun = tunnel(routes[ri])
    from util3_traces import served_model                                 # noqa: E402 - same /v1/models probe
    beat = 0.0
    while not (RT / "STOP_QUEUE").exists() and remaining:
        if time.monotonic() - beat > 60:                                  # the pod serves queued jobs only while this driver is alive
            try:
                ssh("touch /root/guardian/pc_heartbeat")
                beat = time.monotonic()
            except Exception as e:                                       # noqa: BLE001
                log(f"heartbeat failed: {e!r}"[:200])
        if tun.poll() is not None:
            time.sleep(3)
            ri = (ri + 1) % len(routes)
            tun = tunnel(routes[ri])
        for p in PORTS:
            t = running.get(p)
            if t is not None and t.is_alive():
                continue
            m = served_model(p)
            if m in remaining:
                running[p] = threading.Thread(target=drive, args=(p, m), daemon=True)
                running[p].start()
                log(f"driving {m} on {p}")
        time.sleep(10)
    log("efficiency jobs finished" if not remaining else "stopped")
    tun.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
