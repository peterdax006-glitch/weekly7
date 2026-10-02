"""Creator K20 - the kernel's worker made of the Creator's OWN parts: the local model develops the Creator (owner, 1 Oct 2026: "it can
start working on building itself and then you just direct") - IMPLEMENTED, NOT VALIDATED.

For each work package the kernel hands over: show the local model the package (creator.kernel.render_package) and the files it
names; it answers with search/replace edits (files are too large to rewrite whole); the edits are applied (an edit whose SEARCH
text is not found exactly once is refused, never guessed), the tests that reach the changed files run, and their failures go
back to the model for another attempt. The kernel then measures, decides, merges or rejects exactly as for any worker - this
worker's claim is never trusted. The model server is started only when a package arrives and stopped after it (owner: never run
more code than necessary)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from creator import generator as G
from creator import kernel as K
from creator import model as M
from creator import planner as P

SYSTEM = ("You are developing a Python project called the Creator. You receive a work package and the current code. Make the "
          "smallest correct change that meets the package. Reply ONLY with edits in exactly this format, one block per edit:\n"
          "FILE: <path>\n<<<<<<< SEARCH\n<exact existing lines>\n=======\n<new lines>\n>>>>>>> REPLACE\n"
          "To create a new file, leave the SEARCH part empty. Never weaken or delete tests.")
MAX_FILE_CHARS = 24000


def relevant_files(package: M.WorkPackage) -> list[str]:
    """The files the package names (outputs, inputs) that exist or must be created, test files included."""
    names = [x for x in (*package.outputs, *package.inputs) if x.endswith(".py")]
    return list(dict.fromkeys(names))


def tests_for(paths: Sequence[str], workdir: Path) -> list[str]:
    """Test files to run after an edit: the named test files plus every test file that imports a changed module."""
    from creator import testrun as T
    sel = T.select_tests(T.ImportGraph.build(workdir), list(paths))
    tests = [p for p in paths if Path(p).name.startswith("test_") and (workdir / p).is_file()]
    tests += [t for t in sel.tests if (workdir / t).is_file()]
    return sorted(set(tests))


def run_tests(workdir: Path, tests: Sequence[str], timeout: float = 900.0) -> tuple[bool, str]:
    if not tests:
        return False, "no test reaches the change - add a test for it"
    from creator import devbench as D
    D.purge_bytecode(workdir)                    # rewrites of the same size in the same second must not run stale bytecode
    try:
        p = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests], cwd=workdir,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"the tests timed out after {timeout:.0f}s - make the change cheaper or fix a hang"
    return p.returncode == 0, (p.stdout + p.stderr)[-3000:]


class LocalWorker:
    """The Creator's own worker for kernel packages. `llm_factory` starts the local model (lazily, per package)."""
    name = "creator-local-v1"

    def __init__(self, llm_factory: Callable[[], Any] = lambda: G.LocalModel(ctx=16384), attempts: int = 3,
                 temperature: float = 0.2, max_tokens: int = 2500) -> None:
        self.llm_factory, self.attempts, self.temperature, self.max_tokens = llm_factory, attempts, temperature, max_tokens

    def _prompt(self, plan: P.Plan, package: M.WorkPackage, workdir: Path) -> str:
        parts = [K.render_package(plan, package), "", "Current files:"]
        for rel in relevant_files(package):
            p = workdir / rel
            body = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else "(does not exist yet)"
            parts.append(f"FILE: {rel}\n```python\n{body[:MAX_FILE_CHARS]}\n```")
        return "\n".join(parts)

    def __call__(self, plan: P.Plan, package: M.WorkPackage, workdir: Path) -> K.WorkResult:
        t0 = time.monotonic()
        notes: list[str] = []
        with self.llm_factory() as llm:
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": self._prompt(plan, package, workdir)}]
            changed: list[str] = []
            for attempt in range(self.attempts):
                reply = llm.chat(messages, max_tokens=self.max_tokens, temperature=self.temperature, seed=attempt)
                edits = G.parse_edits(reply)
                applied, refused = G.apply_edits(workdir, edits)
                changed += applied
                notes.append(f"attempt {attempt + 1}: {len(edits)} edits, {len(applied)} applied, {len(refused)} refused")
                if not edits:
                    messages += [{"role": "assistant", "content": reply},
                                 {"role": "user", "content": "Reply with edits in the FILE / SEARCH / REPLACE format only."}]
                    continue
                ok, out = run_tests(workdir, tests_for(changed, workdir))
                if ok and not refused:
                    return K.WorkResult(True, f"{'; '.join(notes)}; tests pass ({time.monotonic() - t0:.0f}s, {llm.calls} calls)")
                feedback = ("Some edits were refused: " + "; ".join(refused) + "\n" if refused else "") + \
                    ("The tests fail:\n" + out if not ok else "")
                messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": feedback + "\nFix it."}]
        return K.WorkResult(False, f"{'; '.join(notes)} ({time.monotonic() - t0:.0f}s)")
