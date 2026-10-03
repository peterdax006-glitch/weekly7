"""Feature predictors: no future information, walk-forward split, calibration, learner sanity."""
from __future__ import annotations

import copy
import math
import random
from typing import Any

from creator import predictors as P
from creator import thinking as T


def _commits(n: int, seed: int = 1) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        fs = {f"creator/m{rnd.randrange(8)}.py", f"tests/test_m{rnd.randrange(8)}.py"} if rnd.random() < 0.5 else {f"creator/m{rnd.randrange(8)}.py"}
        out.append({"h": f"{i:040d}", "t": 1.7e9 + i * 3600.0, "s": rnd.choice(["feat: x", "fix: bug in y", "test: z", "wip"]), "files": fs,
                    "lines": rnd.randrange(1, 300), "add": rnd.randrange(1, 200), "del": rnd.randrange(0, 100), "au": "a", "body": "Co-Authored-By: Claude" if i % 3 == 0 else ""})
    return out


def test_git_features_ignore_later_commits() -> None:
    cs = _commits(120)
    base = P.git_rows(cs, 10, "fixed")
    cut = 60
    alt = copy.deepcopy(cs)
    rnd = random.Random(9)
    for c in alt[cut:]:                                         # rewrite every record after the cut
        c["files"] = {f"other/z{rnd.randrange(50)}.py"}
        c["s"] = "fix: totally different"
        c["lines"], c["add"], c["del"], c["body"] = 5000, 4000, 1000, ""
        c["t"] += 7 * 3600.0
    new = P.git_rows(alt, 10, "fixed")
    for i in range(cut):
        assert base[i].x == new[i].x and base[i].subject == new[i].subject    # features of earlier commits never saw the rewrite


def test_pkg_features_use_only_earlier_resolutions() -> None:
    items = [T.Item(f"P{i}", 1000.0 + i * 100, req="K1.exists" if i % 2 else "K2.size", spec_len=500 + i, n_files=1 + i % 3, resolved=1000.0 + i * 100 + 150,
                    outcome="ADOPTED" if i % 3 else "REJECTED", seconds=900.0 + 40 * i) for i in range(40)]
    base = P.pkg_rows(items, "verdict")
    alt = copy.deepcopy(items)
    for it in alt[25:]:
        it.outcome, it.seconds, it.spec_len = "ERROR", 99999.0, 9999999
    new = P.pkg_rows(alt, "verdict")
    for a, b in zip(base[:20], new[:20]):                      # rows whose creation precedes the rewritten items' resolution
        assert a.x == b.x


def _separable(n: int, seed: int = 3) -> list[P.Row]:
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rnd.gauss(0, 1), rnd.gauss(0, 1)
        y = int(rnd.random() < 1 / (1 + math.exp(-(1.8 * a - 0.5))))
        rows.append(P.Row(f"s{i}", float(i * 10), float(i * 10 + 5), y, [a, b]))
    return rows


def test_walk_forward_trains_only_on_resolved_before_and_beats_base_rate() -> None:
    rows = _separable(500)
    ps = P.walk_forward_model(rows, "t", "logit", refit_every=20)
    sc = T.score(ps[100:])
    assert sc["brier"] < sc["brier_base_rate"] - 0.03 and sc["gain_ci95"][0] > 0 and sc["ece"] < 0.1
    # a future label flip must not change any earlier prediction
    alt = [P.Row(r.subject, r.created, r.resolved, 1 - r.y if i >= 300 else r.y, r.x) for i, r in enumerate(rows)]
    ps2 = P.walk_forward_model(alt, "t", "logit", refit_every=20)
    for a, b in zip(ps[:300], ps2[:300]):
        assert abs(a.p - b.p) < 1e-12


def test_gbm_learner_and_select_report_split() -> None:
    rows = _separable(400)
    r = P.select_and_report(rows, "t", candidates=({"learner": "logit", "lam": 5.0}, {"learner": "gbm"}), refit_every=50)
    n = len(r["preds"])
    assert r["select"]["n"] == int(n * P.SELECT_SPLIT) and r["heldout"]["n"] == n - int(n * P.SELECT_SPLIT)
    assert r["heldout"]["brier"] < r["heldout"]["brier_base_rate"]


def test_platt_calibration_repairs_overconfidence() -> None:
    rnd = random.Random(5)
    raw = [rnd.choice([0.05, 0.95]) for _ in range(2000)]
    y = [int(rnd.random() < (0.3 if r > 0.5 else 0.1)) for r in raw]      # claims 0.95 but is right 30% of the time
    a, b = P.platt(raw, y)
    p = 1 / (1 + math.exp(-(a * math.log(0.95 / 0.05) + b)))
    assert abs(p - 0.3) < 0.08
