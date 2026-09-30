"""Run many experiments at once (owner request), safely: a queue of (variant x window) jobs, launched in
parallel while free memory stays above a floor (the system reaped jobs before when memory ran out).

B12 (INTEGRATION): before anything launches, every job goes through engine.experiment_memory's pre-launch check against
state/research/tried/movers_grid.jsonl: a job whose (window, variant) already FINISHED under any tag is an exact repeat and
is skipped; a job is registered as tried only after its process exits cleanly with a result.json (a crash can be retried).

usage: grid_runner.py <queue.json> [max_parallel] [min_free_gb]
queue.json: [{"tag": "_u", "variant": {...}}, ...]  -> runs scripts/movers.py run mXX <variant> for m01..mNN"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import experiment_memory as EM                                         # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
WINDOWS = [f"m{i:02d}" for i in range(1, 14)]
RUNNER = "movers_grid"


def free_gb():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB"], capture_output=True, text=True)
    try:
        return float(out.stdout.strip().replace(",", "."))
    except ValueError:
        return 0.0


def job_cfg(window, variant):
    """What makes two jobs the same experiment: the window and the variant, never the output tag."""
    return {"runner": "movers", "window": window, "variant": {k: v for k, v in variant.items() if k != "tag"}}


def result_path(root, window, tag):
    return Path(root) / "state" / "movers" / f"{window}{tag}" / "result.json"


def plan_jobs(queue, index, root=ROOT, windows=WINDOWS, log=print):
    """(jobs, skipped). A job already finished under its own tag (result.json) is done; the experiment memory then drops
    exact repeats of anything finished under another tag, and repeats inside this queue."""
    cands = []
    for q in queue:
        for w in windows:
            if result_path(root, w, q["tag"]).exists():
                continue
            cands.append((f"movers:{w}{q['tag']}", job_cfg(w, q["variant"])))
    go, skipped = EM.filter_batch(index, cands, log=log)
    by_id = {eid: cfg for eid, cfg in go}
    jobs = [(q["tag"], w, {**q["variant"], "tag": q["tag"]}) for q in queue for w in windows
            if f"movers:{w}{q['tag']}" in by_id]
    return jobs, skipped


def finish(index, tag, window, variant, returncode, root=ROOT, log=print):
    """Register a job as tried only after a clean exit that left its result behind."""
    rp = result_path(root, window, tag)
    if returncode == 0 and rp.exists():
        EM.record_launch(index, f"movers:{window}{tag}", job_cfg(window, variant), reason=str(rp))
        return True
    log(f"  {window}{tag} exited {returncode} without a result: not registered, it can be retried")
    return False


def launch(tag, window, variant, root=ROOT):
    return subprocess.Popen([PY, "-u", str(Path(root) / "scripts" / "movers.py"), "run", window, json.dumps(variant)],
                            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def main(argv=None, launcher=launch, free=free_gb, index=None, root=ROOT, poll_s=5.0):
    argv = sys.argv[1:] if argv is None else argv
    queue = json.loads(Path(argv[0]).read_text())
    maxp = int(argv[1]) if len(argv) > 1 else 5
    floor = float(argv[2]) if len(argv) > 2 else 2.5
    index = index or EM.launch_index(RUNNER)
    jobs, skipped = plan_jobs(queue, index, root)
    running, t0 = [], time.time()
    print(f"{len(jobs)} jobs ({len(skipped)} skipped by experiment memory), up to {maxp} at once, memory floor {floor} GB",
          flush=True)
    while jobs or running:
        still = []
        for p, j in running:
            if p.poll() is None:
                still.append((p, j))
            else:
                finish(index, *j, p.returncode, root)
        running = still
        while jobs and len(running) < maxp and free() > floor:
            tag, w, v = jobs.pop(0)
            running.append((launcher(tag, w, v, root), (tag, w, v)))
            time.sleep(min(3.0, poll_s))
        if jobs or running:
            time.sleep(poll_s)
            print(f"[{time.time() - t0:5.0f}s] running {len(running)}, queued {len(jobs)}, free {free():.1f} GB", flush=True)
    print("GRID DONE", flush=True)
    return {"skipped": skipped}


if __name__ == "__main__":
    main()
