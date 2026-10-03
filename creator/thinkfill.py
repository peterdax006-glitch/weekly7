"""The thinking-focus filler (moved out of scripts/creator_swarm.py on 3 Oct 2026 so it is loaded on demand, not at swarm start: the
merged thinking work had pushed the swarm's eager start load past its 85,000-node guard).

In THINKING focus (state/creator/focus.json; owner, 2 Oct 2026: 'have Nupen work souly on thinking until its opinion is trust worthy') the
swarm's leftover capacity runs: the thinking drills over all data (own repo, other projects, public repos - fetched and acquired), live
predictions every think_every seconds, a blueprint every blueprint_every, the learning loop every learn_every, and local-model batches
(judgment, reasoning drills) every third call - independent jobs the Governor admits until CPU or RAM is full.

Everything the swarm script owns (its registry handle, state dir, timestamps, cadences, the live-prediction job) is passed in, so a test that
patches the script's objects patches what runs here.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional


def make(REG: Any, state: Path, root: Path, last: dict[str, float], think_every: float, blueprint_every: float, learn_every: float,
         drill_live: Callable[[], Any]) -> Callable[[], Optional[Callable[[], Any]]]:
    drills = REG.get("drillsources").drill_filler(state, root, Path.home() / "Masterstock" / "JOURNAL.md",
                                                  root / "state" / "research", processes=True,   # every core, not one (GIL)
                                                  public=True)   # fetch public upstreams, acquire more when dry
    models = []                                   # local-model batches: judgment, and (3 Oct 2026) the reasoning drills
    for name, make_fn in (("judgment", "judgment_filler"), ("reasondrills", "reasoning_filler")):
        try:
            models.append(getattr(REG.get(name), make_fn)(state, root, max_servers=int(REG.get("device").settings().get("llama_servers", 1))))
        except Exception as e:                    # noqa: BLE001 - optional; the drills still run
            print(f"{name.upper()} filler unavailable: {type(e).__name__}: {e}", flush=True)

    def next_job() -> Optional[Callable[[], Any]]:
        now = time.time()
        if now - last["run"] >= think_every:
            last["run"] = now
            return lambda: (drill_live(), REG.get("thinking").run(state))   # live drill predictions first: trust.json counts them
        if now - last["blueprint"] >= blueprint_every:
            last["blueprint"] = now
            return lambda: subprocess.run([sys.executable, str(root / "scripts" / "nupen_blueprint.py"), "--state", str(state)],
                                          capture_output=True, timeout=1800, cwd=root)
        learn = REG.optional("learnloop")
        if learn is not None and now - last.get("learn", 0.0) >= learn_every:
            last["learn"] = now                   # test on the data, diagnose, improve, re-measure (creator.learnloop)
            return lambda: learn.run(state, root)
        n = int(last.get("n", 0)) + 1
        last["n"] = n
        k = n // 3 % max(1, len(models))          # every third call a model batch first (RAM-heavy; admission decides), in turns
        for f in ((models[k:] + models[:k]) if n % 3 == 0 else []) + [drills] + models:
            j = f()
            if j is not None:
                return j
        return None
    return next_job
