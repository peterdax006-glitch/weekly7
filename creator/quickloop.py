"""The fast edit loop: an incrementally refreshed import graph and the one-call edit -> static checks -> affected tests cycle.
Split out of creator.testrun so the swarm entry point does not load it at start (creator.testrun re-exports both names lazily)."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from creator import testrun as T


_GRAPH_AT: dict[str, float] = {}
_GRAPH_CACHE: dict[str, tuple["T.ImportGraph", dict[str, tuple[int, int]]]] = {}


def cached_graph(root: str | Path, touched: Iterable[str] = (), max_age_s: float = 0.0) -> "T.ImportGraph":
    """ImportGraph of `root`, refreshed incrementally: one stat walk, only files whose (mtime, size) changed are re-parsed
    (the per-file parse is also content-cached), deleted files are dropped. Same result as ImportGraph.build."""
    rootp = Path(root)
    key = str(rootp.resolve())
    fast = _GRAPH_CACHE.get(key)
    if fast is not None and max_age_s > 0 and time.monotonic() - _GRAPH_AT.get(key, 0.0) < max_age_s:
        g0, st0 = fast                      # fast path: trust the last full walk, re-read only the files the caller says it touched
        for rel in touched:
            rel = rel.replace("\\", "/")
            if rel.endswith(".py") and (rootp / rel).is_file():
                g0.unparsable.pop(rel, None)
                g0._add_file(rel, rootp / rel)
                st = (rootp / rel).stat()
                st0[rel] = (st.st_mtime_ns, st.st_size)
        return g0
    _GRAPH_AT[key] = time.monotonic()
    stats: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(rootp):
        at_root = Path(dirpath) == rootp
        dirnames[:] = sorted(d for d in dirnames if d not in T.SKIP_DIRS and not d.startswith(".") and not (at_root and d in T.ROOT_SKIP_DIRS))
        for fn in filenames:
            if fn.endswith(".py"):
                full = Path(dirpath) / fn
                st = full.stat()
                stats[full.relative_to(rootp).as_posix()] = (st.st_mtime_ns, st.st_size)
    hit = _GRAPH_CACHE.get(key)
    if hit is None:
        g = T.ImportGraph.build(rootp)
        _GRAPH_CACHE[key] = (g, stats)
        return g
    g, old = hit
    for rel in set(old) - set(stats):
        name = T.module_name_for(rel)
        g.modules.pop(name or "", None)
        g.imports.pop(name or "", None)
        g.nonmodule_tests.pop(rel, None)
        g.unparsable.pop(rel, None)
    for rel, sig in stats.items():
        if old.get(rel) != sig:
            g.unparsable.pop(rel, None)
            g._add_file(rel, rootp / rel)
    _GRAPH_CACHE[key] = (g, stats)
    return g


def quick_cycle(files: Any, root: str | Path, patch: str, junit_path: str | Path, *, default_path: str = "", smoke: Iterable[str] = (),
                warm: bool = True, mypy: bool = False, max_tests: int = 0, timeout: float = 600.0, impact_dir: str = "") -> dict[str, Any]:
    """edit -> static checks -> affected tests in one call. `files` is a creator.tools.files.FileTools. The patch is applied atomically; when
    the static checks fail the tests are not run (and the undo record is returned so the caller can revert). Returns timings per stage."""
    import time
    from creator import staticcheck as SC
    t0 = time.monotonic()
    ap = files.apply_patch(patch, default_path)
    out: dict[str, Any] = {"apply": {k: v for k, v in ap.items() if k != "undo"}, "undo": ap.get("undo")}
    t1 = time.monotonic()
    if not ap.get("ok"):
        out["ok"] = False
        return out
    rootp = Path(root)
    rels = sorted({Path(p).resolve().relative_to(rootp.resolve()).as_posix() for p in ap["files"]})
    rep = SC.check_files(rootp, rels, mypy=mypy)
    out["static"] = SC.report_dict(rep)
    t2 = time.monotonic()
    out["stage_s"] = {"apply": round(t1 - t0, 3), "static": round(t2 - t1, 3)}
    if not rep.ok:
        out["ok"] = False
        out["why"] = "static checks failed"
        return out
    sel = T.select_tests(cached_graph(rootp, rels, max_age_s=60.0), rels, smoke)
    tests = list(sel.tests[:max_tests] if max_tests else sel.tests)
    cfg = T.PytestConfig(warm=warm, timeout=timeout)
    imp = None
    if impact_dir and warm:
        from creator import impactmap as IM
        imp = IM.ImpactMap(impact_dir, rootp)
        narrowable = [t for t in tests if sel.reasons.get(t, "").startswith("imports a changed module")]
        got = imp.narrow(rels, {t: [] for t in narrowable})
        tests = [t for t in tests if t not in got] + [x for t, v in got.items() for x in ([t] if v is None else v)]
        out["impact"] = {"files_before": len(sel.tests), "targets": len(tests), "whole_files": sum(1 for v in got.values() if v is None)}
        cfg.impact_out = str(Path(junit_path).with_suffix(".impact.json"))
    t3 = time.monotonic()
    run = T.run_pytest(rootp, tests, junit_path, label="candidate", config=cfg)
    if imp is not None and run.status in (T.RunStatus.PASSED, T.RunStatus.FAILED, T.RunStatus.NO_TESTS) and (run.status is not T.RunStatus.NO_TESTS or not tests):
        imp.update(cfg.impact_out if Path(cfg.impact_out).is_file() else "", commit=rels)
    out.update({"selection": sel.to_dict(), "run": {"status": run.status.value, "counts": run.counts(), "seconds": round(run.seconds, 3),
                                                    "problems": list(run.problems)}})
    out["stage_s"].update({"select": round(t3 - t2, 3), "tests": round(time.monotonic() - t3, 3)})
    out["total_s"] = round(time.monotonic() - t0, 3)
    out["ok"] = run.status is T.RunStatus.PASSED or (imp is not None and not tests)
    return out


def run_warm(argv: Sequence[str], root: str | Path, sel: Sequence[str], jp: Path, cfg: T.PytestConfig) -> T.ProcResult:
    """The same pytest invocation, served by the warm worker. ProcResult-shaped so the verdict logic below is untouched."""
    from creator import warmworker as W
    r = W.get_worker(cfg.python).run(root, list(sel), jp, timeout=cfg.timeout, extra=cfg.extra_args, env=T.clean_env(root, cfg.env),
                                           impact=cfg.impact_out)
    return T.ProcResult(tuple(argv), str(root), r["rc"], r.get("out", ""), "", float(r["seconds"]), timed_out=bool(r["timed_out"]),
                      launch_error=str(r.get("launch_error", "")))
