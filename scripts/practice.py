"""Offline practice rounds: many MEASURED student-candidate outcomes per hour, instead of a handful of kernel cycles (2 Oct 2026).

For a snapshot of the repo (git archive of a revision into a scratch directory - read-only for the repo, state/research omitted)
every editable Creator module is a target; every valid action creator.action_student enumerates is applied, measured in-process
(AST-size delta, kernel-start activation delta), pre-screened (module imports, directly affected tests pass) and appended as one row
to state/creator/practice_rows.jsonl:  {source: 'prescreen', path, src_hash, metric, objective, action, size_delta, act_delta,
static_delta, tests_pass, tests, good, seconds}.  good = the targeted delta is negative AND no direct test broke.

These are real measurements, NOT kernel adoptions: they are a separate, weighted training source for creator.chooser
(chooser.practice_rows) and are never read by the shadow switch rule (creator.shadow reads lessons.jsonl only).

    python scripts/lowprio.py --idle python scripts/practice.py --minutes 20 [--rev HEAD] [--no-tests] [--report]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from creator import action_student as A      # noqa: E402
from creator import chooser as CH            # noqa: E402
from creator import efficiency as E          # noqa: E402
from creator.generator import MachineLock    # noqa: E402
from creator import prescreen as PS          # noqa: E402
from creator import testrun as TR            # noqa: E402

OBJECTIVES = {"size": "shrink {rel} without losing capability", "activation": "load less code at start: {rel}"}


def snapshot(repo: Path, rev: str, dest: Path) -> Path:
    """git archive of `rev` without state/research into dest (read-only for the repo)."""
    dest.mkdir(parents=True, exist_ok=True)
    arc = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", rev, "--", ".", ":(exclude)state/research"],
                         capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arc.stdout, check=True)
    return dest


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def done_groups(out: Path) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    try:
        for ln in out.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
                seen.add((r["path"], r["src_hash"]))
            except (ValueError, KeyError):
                continue
    except OSError:
        pass
    return seen


def practice(root: Path, out: Path, src_dir: Path, minutes: float = 20.0, run_tests: bool = True, max_targets: int = 0,
             python: str = sys.executable, files: Optional[list[str]] = None,
             test_timeout: float = 45.0, lock_wait_s: float = 0.0) -> dict[str, Any]:
    """One practice round over the tree at `root` (a scratch snapshot: files are rewritten and restored in place). Only one round
    writes `out` at a time (an OS lock beside it, dropped if the holder dies); a second one raises TimeoutError after lock_wait_s."""
    lock = MachineLock(out.with_name(out.name + ".lock"), wait_s=lock_wait_s)    # two runs interleaving rows in one jsonl (2 Oct)
    lock.acquire()                                                              # TimeoutError: another practice run holds it
    try:
        return _practice(root, out, src_dir, minutes, run_tests, max_targets, python, files, test_timeout)
    finally:
        lock.release()


def _practice(root: Path, out: Path, src_dir: Path, minutes: float, run_tests: bool, max_targets: int, python: str,
              files: Optional[list[str]], test_timeout: float) -> dict[str, Any]:
    t0 = time.monotonic()
    foot = PS.Footing(root)
    graph = TR.ImportGraph.build(root) if run_tests else None
    for f in files or ():
        foot.load(f)
    texts = foot.texts
    targets = [f for f in (files or sorted(texts, key=lambda f: -foot.sizes[f])) if E.editable(f) and f in texts]
    if max_targets:
        targets = targets[:max_targets]
    seen = done_groups(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    st = {"targets": 0, "candidates": 0, "rows": 0, "tested": 0, "good": 0, "metric_rejected": 0, "test_failed": 0, "import_failed": 0}
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        for rel in targets:
            if time.monotonic() - t0 > minutes * 60:
                break
            src = texts[rel]
            h = sha(src)
            if (rel, h) in seen:
                continue
            others = [t for r, t in texts.items() if r != rel]
            cands = A.enumerate_actions(src, rel, A._elsewhere(others))
            if not cands:
                continue
            (src_dir / f"{h}.py").write_text(src, encoding="utf-8", newline="\n")
            st["targets"] += 1
            for a in cands:
                if time.monotonic() - t0 > minutes * 60:
                    break
                new = A.apply_action(src, a, A._elsewhere(others))
                if new is None:
                    continue
                st["candidates"] += 1
                sd, ad, sl = foot.deltas(rel, new)
                imp = {m: PS.improves(sd, ad, m) for m in ("size", "activation")}
                v = PS.Verdict(True, "", "metric")
                if any(imp.values()):
                    st["tested"] += 1
                    v = PS.check_in_tree(root, rel, new, graph, run_tests, python, test_timeout)
                    st["import_failed"] += v.stage == "import"
                    st["test_failed"] += v.stage == "tests"
                else:
                    st["metric_rejected"] += 1
                for m in ("size", "activation"):
                    if m == "activation" and ad is None:
                        continue
                    good = bool(imp[m] and v.ok and v.tests_pass is not False)
                    st["good"] += good
                    row = {"source": CH.PRACTICE_SOURCE, "path": rel, "src_hash": h, "metric": m,
                           "objective": OBJECTIVES[m].format(rel=rel), "action": a.to_dict(), "size_delta": sd, "act_delta": ad,
                           "static_delta": sl, "tests_pass": v.tests_pass if imp[m] else None, "tests": list(v.tests),
                           "stage": v.stage if imp[m] else "metric", "reason": v.reason if imp[m] else "cannot improve",
                           "inconclusive": v.reason.startswith("inconclusive") or not run_tests, "good": good, "seconds": round(v.seconds, 3), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    st["rows"] += 1
                fh.flush()
    el = time.monotonic() - t0
    return {**st, "seconds": round(el, 1), "rows_per_hour": round(st["rows"] / el * 3600) if el else 0,
            "candidates_per_hour": round(st["candidates"] / el * 3600) if el else 0,
            "tested_per_hour": round(st["tested"] / el * 3600) if el else 0}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--scratch", default="")
    ap.add_argument("--out", default=str(ROOT / "state/creator/practice_rows.jsonl"))
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--max-targets", type=int, default=0)
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    scratch = Path(a.scratch) if a.scratch else Path(tempfile.mkdtemp(prefix="practice_"))
    snap = snapshot(Path(a.repo), a.rev, scratch / "snap")
    out = Path(a.out)
    try:
        res = practice(snap, out, out.parent / "practice_src", a.minutes, not a.no_tests, a.max_targets)
    except TimeoutError as e:
        print(f"another practice run is writing {out.name}: {e}", file=sys.stderr)
        return 3
    finally:
        if not a.keep:
            shutil.rmtree(scratch, ignore_errors=True)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
