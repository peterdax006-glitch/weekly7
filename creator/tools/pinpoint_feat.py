"""Features, mutation-based localization (MBFL) and the tiny learned ranker for creator.tools.pinpoint (loaded on demand).

line_features(runs, root)        -> candidate table for one failing run: Ochiai / DStar2 / Barinel, failing-test counts, distance
                                    from the innermost traceback frame and from the end of the failing execution, structure flags
mbfl(worker, ...)                -> Metallaxis-style features: mutate the top suspicious lines of the BUGGY tree, re-run only the
                                    failing tests plus a sample of passing ones, see which mutants flip failing tests to passing
train / evaluate                 -> logistic and gradient-boosted rankers (scikit-learn, CPU), train/held-out split BY MODULE
Pure stdlib at import time; numpy / scikit-learn are imported inside the functions that need them.
"""
from __future__ import annotations

import ast
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from creator.tools import pinpoint as P

K_CAND = 100
BASE_FEATS = ["ef", "ep", "F", "P", "fr", "och", "dstar2", "barinel", "prior", "in_inner", "in_other", "frame_delta", "same_func",
              "tail_pos", "tail_depth", "recency", "is_def", "has_cmp", "has_bin", "has_ret", "has_if", "has_const", "has_name",
              "func_len", "ncand_file"]
SPEC_FEATS = ["ef", "ep", "F", "P", "fr", "och", "dstar2", "barinel"]
DIST_FEATS = ["in_inner", "in_other", "frame_delta", "same_func", "tail_pos", "tail_depth", "recency"]
STRUCT_FEATS = ["prior", "is_def", "has_cmp", "has_bin", "has_ret", "has_if", "has_const", "has_name", "func_len", "ncand_file"]
REL_SRC = ["och", "dstar2", "barinel", "tail_pos", "recency", "prior", "frame_delta", "ef"]
REL_FEATS = [f"{c}_{k}" for c in REL_SRC for k in ("gap", "pct", "ties")]      # within-bug, relative to the other candidates
MBFL_FEATS = ["met_n", "met_och", "met_fix", "met_pbreak", "met_any_fix"]
A_FEATS = SPEC_FEATS + DIST_FEATS + STRUCT_FEATS + REL_FEATS


# ------------------------------------------------------------------------------------------------------------ source info
_INFO: dict[tuple[str, int], dict[str, Any]] = {}


def file_info(src: str) -> dict[str, Any]:
    key = (str(hash(src)), len(src))
    hit = _INFO.get(key)
    if hit is not None:
        return hit
    flags: dict[int, list[float]] = {}                 # line -> [prior, is_def, cmp, bin, ret, if, const, name]
    funcs: list[tuple[int, int]] = []
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        tree = None
    if tree is not None:
        for n in ast.walk(tree):
            ln = getattr(n, "lineno", None)
            if ln is None:
                continue
            f = flags.setdefault(ln, [0.0] * 8)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                f[1] = 1.0
                f[0] = min(f[0], -0.5)
                if not isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef)):
                    funcs.append((n.lineno, getattr(n, "end_lineno", n.lineno)))
                for d in getattr(n, "decorator_list", []):
                    flags.setdefault(d.lineno, [0.0] * 8)[0] = -0.5
            elif isinstance(n, (ast.Compare, ast.BoolOp)):
                f[2] = 1.0
                f[0] = max(f[0], 1.0)
            elif isinstance(n, ast.BinOp):
                f[3] = 1.0
                f[0] = max(f[0], 1.0)
            elif isinstance(n, ast.Return):
                f[4] = 1.0
                f[0] = max(f[0], 0.6)
            elif isinstance(n, (ast.If, ast.While)):
                f[5] = 1.0
                f[0] = max(f[0], 0.6)
            elif isinstance(n, (ast.Assign, ast.AugAssign)):
                f[0] = max(f[0], 0.6)
            elif isinstance(n, ast.Constant) and type(n.value) is int:
                f[6] = 1.0
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                f[7] = 1.0
    out = {"flags": flags, "funcs": sorted(funcs)}
    if len(_INFO) > 400:
        _INFO.clear()
    _INFO[key] = out
    return out


def _func_of(info: Mapping[str, Any], line: int) -> tuple[int, int] | None:
    best = None
    for s, e in info["funcs"]:
        if s <= line <= e and (best is None or e - s < best[1] - best[0]):
            best = (s, e)
    return best


# ------------------------------------------------------------------------------------------------------------- features
def failing_ids(runs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted(k for k, r in runs.items() if r["outcome"] == "failed")


def line_features(runs: Mapping[str, Mapping[str, Any]], root: str | Path, *, sources: Mapping[str, str] | None = None,
                  k: int = K_CAND) -> tuple[list[tuple[str, int]], list[list[float]]]:
    """Candidate lines (covered by >= 1 failing test) with BASE_FEATS, the best `k` by a cheap pre-score. Sources override disk."""
    failing = failing_ids(runs)
    passing = [x for x, r in runs.items() if r["outcome"] == "passed"]
    nf, npass = len(failing), len(passing)
    if not nf:
        return [], []
    ef: dict[tuple[str, int], int] = {}
    ep: dict[tuple[str, int], int] = {}
    for t in failing:
        for f, lns in runs[t]["lines"].items():
            for ln in lns:
                ef[(f, ln)] = ef.get((f, ln), 0) + 1
    for t in passing:
        for f, lns in runs[t]["lines"].items():
            for ln in lns:
                if (f, ln) in ef:
                    ep[(f, ln)] = ep.get((f, ln), 0) + 1
    inner: set[tuple[str, int]] = set()
    other: set[tuple[str, int]] = set()
    inner_frames: list[tuple[str, int]] = []
    for t in failing:
        tb = P.parse_traceback(runs[t].get("text", ""), root)
        rf = tb.repo_frames()
        for i, fr in enumerate(rf):
            (inner if i == 0 else other).add((fr.file, fr.line))
        if rf:
            inner_frames.append((rf[0].file, rf[0].line))
    tail_pos: dict[tuple[str, int], int] = {}
    tail_depth: dict[tuple[str, int], int] = {}
    recency: dict[tuple[str, int], float] = {}
    for t in failing:
        tail = runs[t].get("tail") or []
        n = len(tail)
        for i, ent in enumerate(tail):
            key = (ent[0], ent[1])
            pos = n - 1 - i
            if pos <= tail_pos.get(key, 999):
                tail_pos[key] = pos
                tail_depth[key] = ent[2] if len(ent) > 2 else 0
            recency[key] = max(recency.get(key, 0.0), (i + 1) / max(1, n))
    infos: dict[str, dict[str, Any]] = {}
    per_file: dict[str, int] = {}
    for (f, _l) in ef:
        per_file[f] = per_file.get(f, 0) + 1
    rootp = Path(root)

    def info(f: str) -> dict[str, Any]:
        if f not in infos:
            src = (sources or {}).get(f)
            if src is None:
                try:
                    src = (rootp / f).read_text(encoding="utf-8")
                except OSError:
                    src = ""
            infos[f] = file_info(src)
        return infos[f]

    inner_funcs = [(f, _func_of(info(f), l)) for f, l in inner_frames]
    rows: list[tuple[float, tuple[str, int], list[float]]] = []
    for key, e in ef.items():
        f, ln = key
        p = ep.get(key, 0)
        och = P.ochiai(e, p, nf)
        dstar = min(1000.0, e * e / (p + (nf - e) + 1e-9))
        barinel = 1.0 - p / (p + e)
        fl = info(f)["flags"].get(ln, [0.0] * 8)
        fn = _func_of(info(f), ln)
        fd = min([abs(ln - l2) for f2, l2 in inner_frames if f2 == f] or [99])
        same = 1.0 if any(f2 == f and fu is not None and fn == fu for (f2, fu) in inner_funcs) else 0.0
        rec = recency.get(key, 0.0)
        pre = och + 0.2 * (key in inner) + 0.1 * (key in other) + 0.01 * rec + 0.02 * fl[0]
        row = [e, p, nf, npass, e / nf, och, math.log1p(dstar), barinel, fl[0], float(key in inner), float(key in other),
               float(min(fd, 99)), same, float(tail_pos.get(key, 60)), float(tail_depth.get(key, -1)), rec, fl[1], fl[2], fl[3],
               fl[4], fl[5], fl[6], fl[7], float((fn[1] - fn[0] + 1) if fn else 0), float(per_file[f])]
        rows.append((pre, key, row))
    rows.sort(key=lambda t: (-t[0], t[1]))
    rows = rows[:k]
    return [r[1] for r in rows], [[round(v, 4) for v in r[2]] for r in rows]


# -------------------------------------------------------------------------------------------------------------- MBFL
def mbfl(worker: Any, tests_failing: Sequence[str], runs: Mapping[str, Mapping[str, Any]], buggy: Mapping[str, str],
         lines: Sequence[tuple[str, int]], *, per_line: int = 3, sample_pass: int = 3, budget_s: float = 30.0,
         run_timeout: float = 150.0, max_failing: int = 4) -> dict[tuple[str, int], list[float]]:
    """Metallaxis-style features per line (MBFL_FEATS order). `buggy`: file -> source of the BUGGY tree for the files involved
    (others are read from the worker's tree). Mutants of a line are re-run on the failing tests + <= sample_pass passing tests that
    cover the line; a mutant 'kills' a test when its outcome differs from the buggy run. Lines left over when the time budget is
    spent get zeros (met_n == 0)."""
    spent = 0.0                                  # CPU seconds of the reruns (worker-reported; wall time under load is not a budget)
    out: dict[tuple[str, int], list[float]] = {}
    tests_failing = list(tests_failing)[:max_failing]        # a few failing tests are enough to tell whether a mutant repairs the bug
    F = len(tests_failing)
    passing = sorted(k for k, r in runs.items() if r["outcome"] == "passed")
    for (f, ln) in lines:
        if spent > budget_s:
            break
        src = buggy.get(f)
        if src is None:
            src = (worker.tree / f).read_text(encoding="utf-8")
        muts = P.enumerate_mutants(src, {ln})
        if not muts:
            out[(f, ln)] = [0.0] * len(MBFL_FEATS)
            continue
        by_kind: dict[str, list[Any]] = {}
        for m in muts:
            by_kind.setdefault(m.kind, []).append(m)
        chosen: list[Any] = []
        for grp in zip_longest_kinds(by_kind):
            chosen.append(grp)
            if len(chosen) >= per_line:
                break
        cover = [t for t in passing if ln in runs[t]["lines"].get(f, ())][:sample_pass]
        tests = sorted(set(tests_failing) | set(cover))
        files = sorted({t.split("::", 1)[0] for t in tests})
        base = {t: runs[t]["outcome"] for t in tests}
        och_best = fix_best = pb_sum = 0.0
        any_fix = 0.0
        n = 0
        for m in chosen:
            if spent > budget_s:
                break
            try:
                res = worker.run(tests, {f: P.apply_mutant(src, m)}, timeout=run_timeout, trace=False)
            except RuntimeError:
                spent += run_timeout
                continue
            spent += float(getattr(worker, "last_cpu", 0.0) or 0.0) or 1.0
            n += 1
            fixed = sum(1 for t in tests_failing if res.get(t, {}).get("outcome") == "passed")
            broke = sum(1 for t in cover if res.get(t, {}).get("outcome") == "failed")
            och_best = max(och_best, fixed / math.sqrt(F * (fixed + broke)) if fixed else 0.0)
            fix_best = max(fix_best, fixed / F)
            pb_sum += broke / max(1, len(cover))
            any_fix = max(any_fix, 1.0 if fixed else 0.0)
        out[(f, ln)] = [float(n), round(och_best, 4), round(fix_best, 4), round(pb_sum / max(1, n), 4), any_fix]
    return out


def mbfl_fast(worker: Any, failing: Sequence[str], base: Mapping[str, Mapping[str, Any]], file: str, buggy: str,
              lines: Sequence[tuple[str, int]], *, per_line: int = 3, n_fail: int = 2, n_pass: int = 0, budget_cpu: float = 1.5,
              stop_on_fix: bool = True, run_timeout: float = 60.0) -> tuple[dict[tuple[str, int], list[float]], float, int]:
    """Cheap MBFL (same MBFL_FEATS as mbfl()): `lines` are visited in the ranker's order (best first); each gets <= per_line single-site
    mutants of the buggy source; each mutant re-runs only the n_fail CHEAPEST failing tests (+ n_pass passing tests of the line) untraced.
    Early stop: the first line whose mutant makes every re-run failing test pass (a candidate FIX) ends the search; so does the CPU budget
    (worker-reported seconds). Lines never reached keep zeros. Returns (features, cpu spent, runs)."""
    spent, runs_n = 0.0, 0
    out: dict[tuple[str, int], list[float]] = {}
    fail = sorted(failing, key=lambda k: float(base.get(k, {}).get("cpu", 0.1)))[:max(1, n_fail)]
    F = len(fail)
    for (f, ln) in lines:
        if spent >= budget_cpu:
            break
        if f != file:
            out[(f, ln)] = [0.0] * len(MBFL_FEATS)
            continue
        muts = P.enumerate_mutants(buggy, {ln})
        if not muts:
            out[(f, ln)] = [0.0] * len(MBFL_FEATS)
            continue
        by_kind: dict[str, list[Any]] = {}
        for m in muts:
            by_kind.setdefault(m.kind, []).append(m)
        chosen: list[Any] = []
        for m in zip_longest_kinds(by_kind):
            chosen.append(m)
            if len(chosen) >= per_line:
                break
        cover = [t for t in sorted(base) if base[t]["outcome"] == "passed" and ln in base[t]["lines"].get(f, ())
                 and t not in fail][:max(0, n_pass)]
        tests = sorted(set(fail) | set(cover))
        och_best = fix_best = pb_sum = any_fix = 0.0
        n = 0
        for m in chosen:
            if spent >= budget_cpu:
                break
            try:
                res = worker.run(tests, {file: P.apply_mutant(buggy, m)}, timeout=run_timeout, trace=False)
            except RuntimeError:                                  # timeout / crash (the worker is replaced): the run is lost, charge a nominal second
                spent += 1.0
                continue
            spent += float(getattr(worker, "last_cpu", 0.0) or 0.0) or 0.15
            runs_n += 1
            n += 1
            fixed = sum(1 for t in fail if res.get(t, {}).get("outcome") == "passed")
            broke = sum(1 for t in cover if res.get(t, {}).get("outcome") == "failed")
            och_best = max(och_best, fixed / math.sqrt(F * (fixed + broke)) if fixed else 0.0)
            fix_best = max(fix_best, fixed / F)
            pb_sum += broke / max(1, len(cover))
            any_fix = max(any_fix, 1.0 if fixed else 0.0)
        out[(f, ln)] = [float(n), round(och_best, 4), round(fix_best, 4), round(pb_sum / max(1, n), 4), any_fix]
        if stop_on_fix and fix_best >= 1.0:
            break
    return out, spent, runs_n


def zip_longest_kinds(by_kind: Mapping[str, list[Any]]) -> Any:
    """Round-robin over mutation kinds (diverse per-line sample)."""
    from itertools import zip_longest
    for grp in zip_longest(*by_kind.values()):
        for m in grp:
            if m is not None:
                yield m


# ------------------------------------------------------------------------------------------------------- dataset / eval
def load_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def bug_id(row: Mapping[str, Any]) -> str:
    m = row["mutant"]
    return f'{row["file"]}:{row["true_line"]}:{m["kind"]}:{m["start"]}:{m["new"]}'


def split_modules(files: Sequence[str], test_frac: float = 0.34, seed: int = 7) -> tuple[set[str], set[str]]:
    """Deterministic module-level split: no module appears on both sides."""
    import random
    fs = sorted(set(files))
    random.Random(seed).shuffle(fs)
    n_test = max(1, round(len(fs) * test_frac))
    return set(fs[n_test:]), set(fs[:n_test])


def _rel_block(x: Any, cols: Sequence[str]) -> Any:
    """Within-bug relative features per REL_SRC column: gap to the best candidate, percentile rank, share of candidates tied."""
    import numpy as np
    out = []
    for c in REL_SRC:
        v = x[:, cols.index(c)]
        if c in ("tail_pos", "frame_delta"):
            v = -v                                       # smaller distance = more suspicious
        n = len(v)
        order = v.argsort(kind="stable")
        pct = np.empty(n)
        pct[order] = np.arange(n) / max(1, n - 1)
        ties = np.array([(np.abs(v - a) < 1e-9).sum() for a in v]) / n
        out += [v - v.max(), pct, ties]
    return np.stack(out, axis=1)


def _matrix(rows: Sequence[Mapping[str, Any]], feats: Sequence[str], mbfl_map: Mapping[str, Mapping[str, list[float]]] | None):
    import numpy as np
    xs, ys, gid = [], [], []
    use_rel = any(c in feats for c in REL_FEATS)
    use_mb = any(c in feats for c in MBFL_FEATS)
    for gi, r in enumerate(rows):
        cols = r["cand"]["cols"]
        idx = [cols.index(c) for c in BASE_FEATS if c in feats]
        mb = (mbfl_map or {}).get(bug_id(r), {}) if use_mb else {}
        base = np.array(r["cand"]["x"], dtype=float)
        parts = [base[:, idx]]
        if use_rel:
            parts.append(_rel_block(base, cols))
        if use_mb:
            parts.append(np.array([mb.get(f"{k[0]}:{k[1]}", [0.0] * len(MBFL_FEATS)) for k in r["cand"]["keys"]], dtype=float))
        xs.append(np.concatenate(parts, axis=1))
        ys += [1 if (k[0] == r["file"] and k[1] == r["true_line"]) else 0 for k in r["cand"]["keys"]]
        gid += [gi] * len(r["cand"]["keys"])
    return np.concatenate(xs, axis=0), np.array(ys), np.array(gid)


def feature_names(feats: Sequence[str]) -> list[str]:
    return [c for c in BASE_FEATS if c in feats] + [c for c in MBFL_FEATS if c in feats]


def topk_from_scores(scores: Any, ys: Any, gid: Any, ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    import numpy as np
    hits = {k: 0 for k in ks}
    n = int(gid.max()) + 1 if len(gid) else 0
    order = np.argsort(gid, kind="stable")
    bounds = np.searchsorted(gid[order], np.arange(n + 1))
    for g in range(n):
        sl = order[bounds[g]:bounds[g + 1]]
        s, y = scores[sl], ys[sl]
        rank = np.argsort(-s, kind="stable")
        pos = np.where(y[rank] == 1)[0]
        if len(pos):
            for k in ks:
                hits[k] += int(pos[0] < k)
    return {f"top{k}": round(hits[k] / max(1, n), 4) for k in ks} | {"n": n}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 4), round((c + h) / d, 4)


def topk_ci(scores: Any, ys: Any, gid: Any, ks: Sequence[int] = (1, 3, 5)) -> dict[str, Any]:
    """topk_from_scores + the 95% Wilson interval of every top-k rate (bugs are the unit)."""
    r = topk_from_scores(scores, ys, gid, ks)
    n = int(r["n"])
    for k in ks:
        r[f"top{k}_ci95"] = list(wilson(round(r[f"top{k}"] * n), n))
    return r


def fit_score(kind: str, x_tr: Any, y_tr: Any, x_te: Any, seed: int = 0) -> Any:
    import numpy as np
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(x_tr)
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(sc.transform(x_tr), y_tr)
        return m.decision_function(sc.transform(x_te))
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4, min_samples_leaf=20, random_state=seed)
    m.fit(x_tr, y_tr)
    return np.asarray(m.predict_proba(x_te)[:, 1])


def heuristic_scores(rows: Sequence[Mapping[str, Any]]) -> Any:
    """The P0.5 baseline formula (Ochiai + frame boosts + tiny tie-breakers), recomputed from the stored features."""
    import numpy as np
    out = []
    for r in rows:
        c = r["cand"]["cols"]
        ix = {n: c.index(n) for n in ("och", "in_inner", "in_other", "recency", "prior")}
        for x in r["cand"]["x"]:
            out.append(x[ix["och"]] + 0.2 * x[ix["in_inner"]] + 0.1 * x[ix["in_other"]] + 0.01 * x[ix["recency"]] + 0.02 * x[ix["prior"]])
    return np.array(out)


def by_kind_report(scores: Any, ys: Any, gid: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    out = {}
    for kind in sorted({r["mutation"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["mutation"] == kind]
        mask = _mask(gid, sel)
        g2 = _renumber(gid[mask])
        out[kind] = topk_from_scores(scores[mask], ys[mask], g2)
    return out


def _mask(gid: Any, sel: Sequence[int]) -> Any:
    import numpy as np
    return np.isin(gid, np.array(sel))


def _renumber(gid: Any) -> Any:
    import numpy as np
    _u, inv = np.unique(gid, return_inverse=True)
    return inv


def evaluate_all(rows: Sequence[dict[str, Any]], mbfl_map: Mapping[str, Mapping[str, list[float]]] | None = None,
                 test_frac: float = 0.34, seed: int = 7) -> dict[str, Any]:
    """Held-out-by-module numbers for: baseline heuristic, a (spectrum+distance+structure features), b (a + MBFL), c (boosted)."""
    train_m, test_m = split_modules([r["file"] for r in rows], test_frac, seed)
    tr = [r for r in rows if r["file"] in train_m]
    te = [r for r in rows if r["file"] in test_m]
    res: dict[str, Any] = {"train_modules": sorted(train_m), "test_modules": sorted(test_m), "n_train": len(tr), "n_test": len(te)}
    x_te, y_te, g_te = _matrix(te, A_FEATS, None)
    recall = sum(1 for r in te if r["label"] >= 0) / max(1, len(te))
    res["candidate_recall_test"] = round(recall, 4)
    h = heuristic_scores(te)
    res["baseline"] = topk_from_scores(h, y_te, g_te)
    res["baseline_by_kind"] = by_kind_report(h, y_te, g_te, te)
    ssp = SPEC_FEATS
    x_tr, y_tr, g_tr = _matrix(tr, ssp, None)
    xs_te, _y, _g = _matrix(te, ssp, None)
    res["a1_spectrum_only_logistic"] = topk_from_scores(fit_score("logistic", x_tr, y_tr, xs_te), y_te, g_te)
    x_tr, y_tr, g_tr = _matrix(tr, A_FEATS, None)
    s_a = fit_score("logistic", x_tr, y_tr, x_te)
    res["a_all_signals_logistic"] = topk_from_scores(s_a, y_te, g_te)
    res["a_by_kind"] = by_kind_report(s_a, y_te, g_te, te)
    s_ag = fit_score("gbm", x_tr, y_tr, x_te)
    res["a_all_signals_gbm"] = topk_from_scores(s_ag, y_te, g_te)
    if mbfl_map:
        have = [r for r in rows if bug_id(r) in mbfl_map]
        tr_b = [r for r in have if r["file"] in train_m]
        te_b = [r for r in have if r["file"] in test_m]
        res["n_mbfl_train"], res["n_mbfl_test"] = len(tr_b), len(te_b)
        if tr_b and te_b:
            xa_tr, ya_tr, _ = _matrix(tr_b, A_FEATS, None)
            xa_te, ya_te, ga_te = _matrix(te_b, A_FEATS, None)
            res["subset_a_logistic"] = topk_from_scores(fit_score("logistic", x_tr, y_tr, xa_te), ya_te, ga_te)
            res["subset_baseline"] = topk_from_scores(heuristic_scores(te_b), ya_te, ga_te)
            allf = A_FEATS + MBFL_FEATS
            xb_tr, yb_tr, _ = _matrix(tr_b, allf, mbfl_map)
            xb_te, yb_te, gb_te = _matrix(te_b, allf, mbfl_map)
            s_b = fit_score("logistic", xb_tr, yb_tr, xb_te)
            res["b_plus_mbfl_logistic"] = topk_from_scores(s_b, yb_te, gb_te)
            res["b_by_kind"] = by_kind_report(s_b, yb_te, gb_te, te_b)
            s_c = fit_score("gbm", xb_tr, yb_tr, xb_te)
            res["c_gbm_all_features"] = topk_from_scores(s_c, yb_te, gb_te)
            res["c_by_kind"] = by_kind_report(s_c, yb_te, gb_te, te_b)
            xg_tr, yg_tr, _ = _matrix(tr_b, A_FEATS, None)
            res["c_gbm_a_features_same_subset"] = topk_from_scores(fit_score("gbm", xg_tr, yg_tr, xa_te), ya_te, ga_te)
    return res


def cv_scores(rows: Sequence[dict[str, Any]], feats: Sequence[str], kind: str, mbfl_map: Mapping[str, Any] | None = None,
              folds: int = 5, seed: int = 7) -> tuple[Any, Any, Any]:
    """Leave-modules-out cross-validation: modules are dealt into `folds` groups; every bug is scored by a model trained on the
    OTHER groups only, so no module is ever on both sides. Returns (scores, labels, group ids) aligned with _matrix(rows)."""
    import numpy as np
    mods = sorted({r["file"] for r in rows})
    import random
    random.Random(seed).shuffle(mods)
    fold_of = {m: i % folds for i, m in enumerate(mods)}
    x, y, g = _matrix(rows, feats, mbfl_map)
    row_fold = np.array([fold_of[rows[i]["file"]] for i in g])
    scores = np.zeros(len(y))
    for f in range(folds):
        te = row_fold == f
        tr = ~te
        if te.any() and tr.any() and y[tr].sum() > 0:
            scores[te] = fit_score(kind, x[tr], y[tr], x[te])
    return scores, y, g


def cv_report(rows: Sequence[dict[str, Any]], mbfl_map: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    x, y, g = _matrix(rows, A_FEATS, None)
    h = heuristic_scores(rows)
    out["baseline"] = topk_from_scores(h, y, g)
    out["baseline_by_kind"] = by_kind_report(h, y, g, rows)
    s, y, g = cv_scores(rows, SPEC_FEATS, "logistic")
    out["a1_spectrum_dstar_barinel_logistic"] = topk_from_scores(s, y, g)
    s, y, g = cv_scores(rows, A_FEATS, "logistic")
    out["a_all_signals_logistic"] = topk_from_scores(s, y, g)
    out["a_logistic_by_kind"] = by_kind_report(s, y, g, rows)
    s, y, g = cv_scores(rows, A_FEATS, "gbm")
    out["a_all_signals_gbm"] = topk_from_scores(s, y, g)
    out["a_gbm_by_kind"] = by_kind_report(s, y, g, rows)
    if mbfl_map:
        have = [r for r in rows if bug_id(r) in mbfl_map]
        if len(have) > 30:
            out["n_mbfl_bugs"] = len(have)
            x, y, g = _matrix(have, A_FEATS, None)
            out["subset_baseline"] = topk_from_scores(heuristic_scores(have), y, g)
            s, y, g = cv_scores(have, A_FEATS, "logistic")
            out["subset_a_logistic"] = topk_from_scores(s, y, g)
            allf = A_FEATS + MBFL_FEATS
            s, y, g = cv_scores(have, allf, "logistic", mbfl_map)
            out["b_plus_mbfl_logistic"] = topk_from_scores(s, y, g)
            out["b_by_kind"] = by_kind_report(s, y, g, have)
            s, y, g = cv_scores(have, allf, "gbm", mbfl_map)
            out["c_gbm_all_features"] = topk_from_scores(s, y, g)
            out["c_by_kind"] = by_kind_report(s, y, g, have)
    return out


def cv_ci_report(rows: Sequence[dict[str, Any]], mbfl_map: Mapping[str, Any]) -> dict[str, Any]:
    """Leave-modules-out CV on the bugs that have MBFL features, top-1/3/5 with 95% Wilson intervals: the P0.5 heuristic, the a-model
    (spectrum + distance + structure), and a-model + MBFL (logistic and boosted), overall and per mutation kind."""
    have = [r for r in rows if bug_id(r) in mbfl_map]
    out: dict[str, Any] = {"n_bugs": len(have), "n_modules": len({r["file"] for r in have})}
    if len(have) < 30:
        return out
    x, y, g = _matrix(have, A_FEATS, None)
    out["baseline"] = topk_ci(heuristic_scores(have), y, g)
    s_a, y, g = cv_scores(have, A_FEATS, "logistic")
    out["a_logistic"] = topk_ci(s_a, y, g)
    out["a_logistic_by_kind"] = by_kind_report(s_a, y, g, have)
    allf = A_FEATS + MBFL_FEATS
    s_b, y, g = cv_scores(have, allf, "logistic", mbfl_map)
    out["a_plus_mbfl_logistic"] = topk_ci(s_b, y, g)
    out["b_by_kind"] = by_kind_report(s_b, y, g, have)
    s_c, y, g = cv_scores(have, allf, "gbm", mbfl_map)
    out["a_plus_mbfl_gbm"] = topk_ci(s_c, y, g)
    out["gbm_by_kind"] = by_kind_report(s_c, y, g, have)
    kinds = sorted({r["mutation"] for r in have})
    out["kind_share"] = {k: round(sum(1 for r in have if r["mutation"] == k) / len(have), 3) for k in kinds}
    bal: dict[str, Any] = {}
    for name, sc in (("a_logistic", s_a), ("a_plus_mbfl_logistic", s_b), ("a_plus_mbfl_gbm", s_c)):
        per = by_kind_report(sc, y, g, have)
        bal[name] = {f"top{k}": round(sum(per[kd][f"top{k}"] for kd in kinds) / len(kinds), 4) for k in (1, 3, 5)}
    out["kind_balanced_macro"] = bal
    return out
