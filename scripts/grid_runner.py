"""Run many experiments at once (owner request), safely: a queue of (variant x window) jobs, launched in
parallel while free memory stays above a floor (the system reaped jobs before when memory ran out).

usage: grid_runner.py <queue.json> [max_parallel] [min_free_gb]
queue.json: [{"tag": "_u", "variant": {...}}, ...]  -> runs scripts/movers.py run mXX <variant> for m01..mNN"""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
queue = json.loads(Path(sys.argv[1]).read_text())
MAXP = int(sys.argv[2]) if len(sys.argv) > 2 else 5
FLOOR = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
WINDOWS = [f"m{i:02d}" for i in range(1, 14)]


def free_gb():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB"], capture_output=True, text=True)
    try:
        return float(out.stdout.strip().replace(",", "."))
    except ValueError:
        return 0.0


jobs = [(q["tag"], w, {**q["variant"], "tag": q["tag"]}) for q in queue for w in WINDOWS
        if not (ROOT / "state" / "movers" / f"{w}{q['tag']}" / "result.json").exists()]
running, done, t0 = [], 0, time.time()
print(f"{len(jobs)} jobs, up to {MAXP} at once, memory floor {FLOOR} GB", flush=True)
while jobs or running:
    running = [(p, j) for p, j in running if p.poll() is None]
    done_now = 0
    while jobs and len(running) < MAXP and free_gb() > FLOOR:
        tag, w, v = jobs.pop(0)
        p = subprocess.Popen([PY, "-u", str(ROOT / "scripts" / "movers.py"), "run", w, json.dumps(v)],
                             cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        running.append((p, (tag, w)))
        time.sleep(3)
    time.sleep(5)
    print(f"[{time.time() - t0:5.0f}s] running {len(running)}, queued {len(jobs)}, free {free_gb():.1f} GB", flush=True)
print("GRID DONE", flush=True)
