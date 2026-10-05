"""Orchestration for the pinpoint data pipeline (loaded on demand): survey -> plan -> dataset -> mbfl -> eval.

  python -m creator.tools.pinpoint_cli survey  --root . --out survey.json       run every test file once, record seconds/coverage
  python -m creator.tools.pinpoint_cli plan    --root . --survey survey.json --out plan.json   module -> fast covering test files
  python -m creator.tools.pinpoint_cli dataset --root . --plan plan.json --out rows.jsonl      planted bugs + candidate features
  python -m creator.tools.pinpoint_cli mbfl    --root . --rows rows.jsonl --out mbfl.jsonl     mutation-based features on a sample
  python -m creator.tools.pinpoint_cli eval    --rows rows.jsonl [--mbfl mbfl.jsonl]           held-out-by-module numbers
Run heavy steps through scripts/lowprio.py --idle. Everything happens in scratch copies; generated data stays outside the repo.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from creator.tools import pinpoint as P
from creator.tools import pinpoint_feat as PF

CUTOFF = "2026-10-01T21:58:30+00:00"        # trust-gate exclusion: no rows from code last changed at/after this instant


def survey(root: str | Path, out_json: str | Path, workers: int = 12, max_seconds: float = 40.0) -> dict[str, Any]:
    """Run every test file once (warm workers, scratch copies); record seconds, outcome and covered-line counts per module."""
    from creator.tools.pinpoint_worker import WarmPool
    rootp = Path(root)
    tests = sorted(p.relative_to(rootp).as_posix() for p in (rootp / "tests").rglob("test_*.py"))
    base = tempfile.mkdtemp(prefix="pp_survey_")
    pool = WarmPool(root, workers, base)
    out: dict[str, Any] = {}

    def one(t: str) -> None:
        w = pool.acquire()
        t0 = time.monotonic()
        try:
            res = w.run([t], timeout=max_seconds)
            cov: dict[str, int] = {}
            for r in res.values():
                for f, lns in r["lines"].items():
                    cov[f] = max(cov.get(f, 0), len(lns))
            ok = bool(res) and all(r["outcome"] != "failed" for r in res.values())
            out[t] = {"s": round(time.monotonic() - t0, 2), "ok": ok, "n": len(res), "cov": cov}
        except RuntimeError:
            out[t] = {"s": round(time.monotonic() - t0, 2), "ok": False, "n": 0, "cov": {}}
        finally:
            pool.release(w)

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, tests))
    finally:
        pool.close()
        shutil.rmtree(base, ignore_errors=True)
    Path(out_json).write_text(json.dumps(out), encoding="utf-8")
    return out


def plan_from_survey(survey_json: str | Path, root: str | Path, *, max_test_s: float = 6.0, min_lines: int = 25,
                     max_files: int = 3) -> dict[str, list[str]]:
    sv = json.loads(Path(survey_json).read_text(encoding="utf-8"))
    mods: dict[str, list[tuple[float, str, int]]] = {}
    for t, d in sv.items():
        if not d["ok"] or d["s"] > max_test_s:
            continue
        for f, n in d["cov"].items():
            if n >= 8 and not f.startswith("scripts/") and "pinpoint" not in f:
                mods.setdefault(f, []).append((d["s"], t, n))
    plan: dict[str, list[str]] = {}
    for f, lst in sorted(mods.items()):
        if max(n for _s, _t, n in lst) < min_lines:
            continue
        try:
            cd = subprocess.run(["git", "log", "-1", "--format=%cI", "--", f], cwd=str(root), capture_output=True, text=True,
                                timeout=30).stdout.strip()
        except Exception:                                                  # noqa: BLE001
            cd = ""
        if not cd:
            continue
        from datetime import datetime
        if datetime.fromisoformat(cd) >= datetime.fromisoformat(CUTOFF):
            continue                      # changed at/after the trust-gate cutoff
        plan[f] = [t for _s, t, _n in sorted(lst, key=lambda x: -x[2])[:max_files]]
    return plan


class BuggyView:
    """Worker wrapper for PF.mbfl(): every run() also carries the buggy source of the planted file unless the caller overrides it."""

    def __init__(self, w: Any, rel: str, buggy: str) -> None:
        self.w, self.rel, self.buggy, self.tree = w, rel, buggy, w.tree
        self.cpu = 0.0                  # CPU seconds of every run through this view

    @property
    def last_cpu(self) -> float:
        return float(self.w.last_cpu)

    def run(self, tests: Sequence[str], sources: Mapping[str, str] | None = None, timeout: float = 150.0, trace: bool = True) -> Any:
        src = {self.rel: self.buggy}
        src.update(sources or {})
        try:
            return self.w.run(tests, src, timeout=timeout, trace=trace)
        finally:
            self.cpu += float(self.w.last_cpu)


def run_mbfl(a: Any) -> None:
    """MBFL phase: for a sample of bugs spread over modules, re-create the buggy run, rank by the a-model (trained on the train
    modules only), mutate the top-N lines, write one JSON line per bug."""
    import random

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from creator.tools.pinpoint_worker import WarmPool
    rows = PF.load_rows(a.rows)
    train_m, _test_m = PF.split_modules([r["file"] for r in rows])
    x_tr, y_tr, _g = PF._matrix([r for r in rows if r["file"] in train_m], PF.A_FEATS, None)
    sc = StandardScaler().fit(x_tr)
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(sc.transform(x_tr), y_tr)
    rnd = random.Random(3)
    by_mod: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_mod.setdefault(r["file"], []).append(r)
    sample: list[dict[str, Any]] = []
    per = max(1, a.n_bugs // max(1, len(by_mod)))
    for _f, lst in sorted(by_mod.items()):
        rnd.shuffle(lst)
        sample += lst[:per]
    base = tempfile.mkdtemp(prefix="pp_mbfl_")
    pool = WarmPool(a.root, a.workers, base)
    lock = threading.Lock()
    secs: list[float] = []

    def one(r: dict[str, Any]) -> None:
        w = pool.acquire()
        t0 = time.monotonic()
        try:
            orig = (w.tree / r["file"]).read_text(encoding="utf-8").encode("utf-8")
            m = r["mutant"]
            buggy = (orig[:m["start"]] + m["new"].encode("utf-8") + orig[m["end"]:]).decode("utf-8")
            runs = w.run(r["tests"], {r["file"]: buggy}, timeout=60.0)
            failing = PF.failing_ids(runs)
            if not failing:
                return
            keys, xs = PF.line_features(runs, w.tree, sources={r["file"]: buggy})
            pseudo = {"file": r["file"], "true_line": r["true_line"], "cand": {"cols": PF.BASE_FEATS, "keys": keys, "x": xs}}
            sc_ = model.decision_function(sc.transform(PF._matrix([pseudo], PF.A_FEATS, None)[0]))
            order = list(np.argsort(-sc_, kind="stable"))[:a.topn]
            lines = [(keys[i][0], keys[i][1]) for i in order]
            view = BuggyView(w, r["file"], buggy)
            res = PF.mbfl(view, failing, runs, {r["file"]: buggy}, lines, per_line=a.per_line,
                          budget_s=a.budget)
            mm = {f"{f}:{ln}": v for (f, ln), v in res.items()}
            with lock:
                with open(a.out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"id": PF.bug_id(r), "m": mm, "s": round(time.monotonic() - t0, 2), "cpu": round(view.cpu, 2)}) + "\n")
                secs.append(time.monotonic() - t0)
        except RuntimeError:
            pass
        finally:
            pool.release(w)

    t0 = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(one, sample))
    finally:
        pool.close()
        shutil.rmtree(base, ignore_errors=True)
    print(json.dumps({"bugs": len(secs), "mean_s_per_bug": round(sum(secs) / max(1, len(secs)), 2),
                      "wall_s": round(time.monotonic() - t0, 1)}))


def main(argv: Sequence[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="pinpoint_cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("survey")
    s1.add_argument("--root", default=".")
    s1.add_argument("--out", required=True)
    s1.add_argument("--workers", type=int, default=12)
    s2 = sub.add_parser("plan")
    s2.add_argument("--root", default=".")
    s2.add_argument("--survey", required=True)
    s2.add_argument("--out", required=True)
    s3 = sub.add_parser("dataset")
    s3.add_argument("--root", default=".")
    s3.add_argument("--plan", required=True)
    s3.add_argument("--out", required=True)
    s3.add_argument("--rows-per-module", type=int, default=150)
    s3.add_argument("--workers", type=int, default=12)
    s3.add_argument("--per-function", type=int, default=4)
    s3.add_argument("--only", action="append")
    s3.add_argument("--seed", type=int, default=0)
    s3.add_argument("--slow", action="store_true", help="single traced pass per mutant (default: two-phase, see pinpoint._run_mutant)")
    s4 = sub.add_parser("mbfl")
    s4.add_argument("--root", default=".")
    s4.add_argument("--rows", required=True)
    s4.add_argument("--out", required=True)
    s4.add_argument("--n-bugs", type=int, default=600)
    s4.add_argument("--topn", type=int, default=15)
    s4.add_argument("--per-line", type=int, default=3)
    s4.add_argument("--budget", type=float, default=30.0, help="CPU seconds of mutant reruns per bug")
    s4.add_argument("--workers", type=int, default=12)
    s5 = sub.add_parser("eval")
    s5.add_argument("--rows", required=True)
    s5.add_argument("--mbfl")
    a = ap.parse_args(list(argv))
    if a.cmd == "survey":
        r = survey(a.root, a.out, a.workers)
        print(json.dumps({"files": len(r), "ok": sum(1 for v in r.values() if v["ok"])}))
    elif a.cmd == "plan":
        plan = plan_from_survey(a.survey, a.root)
        Path(a.out).write_text(json.dumps(plan, indent=1), encoding="utf-8")
        print(json.dumps({"modules": len(plan)}))
    elif a.cmd == "dataset":
        from creator.tools.pinpoint_worker import WarmPool
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
        targets = [t for t in plan if not a.only or t in a.only]
        base = tempfile.mkdtemp(prefix="pp_ds_")
        pool = WarmPool(a.root, a.workers, base)
        kept = tried = 0
        t0 = time.monotonic()
        try:
            for t in targets:
                st = P.generate(a.root, [t], a.out, tests={t: plan[t]}, max_rows=a.rows_per_module, per_function=a.per_function,
                                pool=pool, evaluate=False, timeout=150.0, seed=a.seed, fast=not a.slow)
                kept += st.kept
                tried += st.tried
                print(json.dumps({"module": t, "kept": st.kept, "tried": st.tried, "s": round(st.seconds, 1)}), flush=True)
        finally:
            pool.close()
            shutil.rmtree(base, ignore_errors=True)
        wall = time.monotonic() - t0
        print(json.dumps({"total_rows": kept, "tried": tried, "wall_s": round(wall, 1), "rows_per_hour": round(kept / wall * 3600)}))
    elif a.cmd == "mbfl":
        run_mbfl(a)
    else:
        rows = PF.load_rows(a.rows)
        mb = None
        if a.mbfl:
            mb = {}
            for ln in Path(a.mbfl).read_text(encoding="utf-8").splitlines():
                d = json.loads(ln)
                mb[d["id"]] = d["m"]
        res = PF.evaluate_all(rows, mb)
        res["leave_modules_out_cv"] = PF.cv_report(rows, mb)
        print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
