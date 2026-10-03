"""The swarm's self-benchmark filler (moved out of creator.swarm on 3 Oct 2026 so the swarm's eager start load stays under its guard):
benchmark the system's OWN search worker on dev tasks it has not measured for the current code (holdout never touched)."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable, Optional


def self_bench_filler(store: Any, tasks: Optional[list[Any]] = None) -> Callable[[], Optional[Callable[[], None]]]:
    """Filler work for leftover memory: benchmark the system's OWN search worker on dev tasks it has not measured for the current
    code (holdout never touched). Results append to `store` - real data on what the system can already do by itself."""
    from creator import devbench as D
    from creator import generator as G
    store = Path(store)
    code = hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest()[:12]
    done = set()
    if store.is_file():
        for ln in store.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(ln)
                if row.get("code") == code:
                    done.add(row["task"])
            except ValueError:
                continue
    todo = [t for t in (tasks or D.load_tasks()) if t.split == "dev" and t.id not in done]
    manifest = D.load_manifest()
    write_lock = threading.Lock()

    def next_job() -> Optional[Callable[[], None]]:
        if not todo:
            return None
        task = todo.pop(0)

        def job() -> None:
            try:
                s = D.run_task(task, G.SearchSolver(), manifest, solver_name="self-search")
                row = {"task": task.id, "category": task.category, "outcome": s.outcome, "seconds": s.seconds, "code": code}
            except Exception as e:                                      # noqa: BLE001 - a filler never stops the swarm
                row = {"task": task.id, "outcome": "ERROR", "error": f"{type(e).__name__}: {e}", "code": code}
            with write_lock:
                store.parent.mkdir(parents=True, exist_ok=True)
                with store.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
        return job
    return next_job
