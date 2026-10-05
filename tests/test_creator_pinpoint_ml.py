"""Warm worker, candidate features, MBFL and the module-level split/metrics of the pinpoint pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from creator.tools import pinpoint_feat as PF
from creator.tools.pinpoint_worker import Warm

LIB = ("def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n\n\n"
       "def total(xs):\n    s = 0\n    for v in xs:\n        s = s + v\n    return s\n")
TESTS = ("from creator import lib\n\n\n"
         "def test_clamp():\n    assert lib.clamp(5, 0, 3) == 3\n    assert lib.clamp(-1, 0, 3) == 0\n    assert lib.clamp(2, 0, 3) == 2\n\n\n"
         "def test_total():\n    assert lib.total([1, 2, 3]) == 6\n")
COPY = ("creator/__init__.py", "creator/tools/__init__.py", "creator/tools/pinpoint_plugin.py", "creator/tools/pinpoint_worker.py",
        "creator/tools/pinpoint_feat.py", "creator/tools/pinpoint.py", "creator/build.py")


def _mini_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "creator" / "tools").mkdir(parents=True)
    (root / "tests").mkdir()
    real = Path(__file__).resolve().parents[1]
    for rel in COPY:
        (root / rel).write_text((real / rel).read_text(encoding="utf-8"), encoding="utf-8")
    (root / "creator" / "lib.py").write_text(LIB, encoding="utf-8")
    (root / "tests" / "test_lib.py").write_text(TESTS, encoding="utf-8")
    return root


def test_warm_worker_matches_fresh_run_and_restores_files(tmp_path):
    root = _mini_project(tmp_path)
    w = Warm(root)
    try:
        bug = LIB.replace("s = s + v", "s = s - v")
        for _ in range(2):                                              # the second call must re-import the changed module
            warm = w.run(["tests/test_lib.py"], {"creator/lib.py": bug})
            assert [k for k, r in warm.items() if r["outcome"] == "failed"] == ["tests/test_lib.py::test_total"]
            assert (root / "creator" / "lib.py").read_text(encoding="utf-8") == LIB
        clean = w.run(["tests/test_lib.py"])
        assert {r["outcome"] for r in clean.values()} == {"passed"}      # the purge really dropped the buggy import
    finally:
        w.stop()


def test_line_features_and_mbfl_find_the_planted_line(tmp_path):
    root = _mini_project(tmp_path)
    bug = LIB.replace("s = s + v", "s = s - v")
    w = Warm(root)
    try:
        runs = w.run(["tests/test_lib.py"], {"creator/lib.py": bug})
        keys, xs = PF.line_features(runs, root, sources={"creator/lib.py": bug})
        assert len(keys) == len(xs) and ("creator/lib.py", 12) in keys
        row = dict(zip(PF.BASE_FEATS, xs[keys.index(("creator/lib.py", 12))]))
        assert row["ef"] == 1 and row["F"] == 1 and row["och"] > 0.5 and row["dstar2"] > 0 and row["tail_pos"] < 60

        class View:
            tree = root

            def run(self, tests, sources=None, timeout=20.0, trace=True):
                return w.run(tests, {"creator/lib.py": bug, **(sources or {})}, timeout, trace=trace)

        got = PF.mbfl(View(), PF.failing_ids(runs), runs, {"creator/lib.py": bug}, [("creator/lib.py", 12)], per_line=4, budget_s=60)
        n, och, fix, _pb, anyfix = got[("creator/lib.py", 12)]
        assert n >= 1 and fix == 1.0 and anyfix == 1.0 and och > 0      # flipping the operator back repairs the failing test
    finally:
        w.stop()


def test_module_split_and_topk_metrics():
    files = ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"]
    tr, te = PF.split_modules(files)
    assert tr and te and not (tr & te) and tr | te == set(files)
    scores = np.array([0.1, 0.9, 0.5, 0.2, 0.8, 0.3])
    ys = np.array([0, 1, 0, 0, 0, 1])
    gid = np.array([0, 0, 0, 1, 1, 1])
    m = PF.topk_from_scores(scores, ys, gid, ks=(1, 2))
    assert m["top1"] == 0.5 and m["top2"] == 1.0 and m["n"] == 2


def test_evaluate_all_on_synthetic_rows_never_mixes_modules():
    rng = np.random.default_rng(0)
    rows = []
    for mod in ("a.py", "b.py", "c.py", "d.py", "e.py", "f.py"):
        for i in range(12):
            xs = rng.random((6, len(PF.BASE_FEATS))).round(3).tolist()
            t = int(rng.integers(0, 6))
            xs[t][PF.BASE_FEATS.index("och")] = 5.0                      # the true line is the one with the high Ochiai
            keys = [[mod, 10 + j] for j in range(6)]
            rows.append({"file": mod, "true_line": 10 + t, "mutation": "cmp_flip" if i % 2 else "swap_var", "mutant": {"kind": "k", "start": i, "new": "x"},
                         "label": t, "cand": {"cols": PF.BASE_FEATS, "keys": keys, "x": xs}})
    res = PF.evaluate_all(rows)
    assert not set(res["train_modules"]) & set(res["test_modules"])
    assert res["a_all_signals_logistic"]["top1"] > 0.9 and res["a_all_signals_gbm"]["top1"] > 0.9
    assert set(res["a_by_kind"]) == {"cmp_flip", "swap_var"}
