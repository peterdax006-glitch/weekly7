"""FEATURE PREDICTORS for the thinking drills (owner, 2 Oct 2026: "predictions must become trustworthy").

The drills in creator.thinking / creator.drillsources predict from decayed frequencies over thin keys. This module builds real FEATURES from setup-time
information only and learns from them with regularised models, under the same honest protocol:
  * WALK-FORWARD: every prediction is made at its creation time by a model fitted only on items RESOLVED strictly before that time; the model is
    refitted every `refit_every` new resolutions (never on anything later), so each new outcome changes later predictions.
  * CALIBRATED: a Platt map (logistic on the logit of the model's own earlier out-of-sample predictions) is fitted only on predictions whose outcome
    had already resolved, then applied; reliability is the gate's ECE.
  * NO FUTURE: git features of commit i read commits 0..i only (tests rewrite later records and check nothing changes); junit/duration features read
    only items resolved before the prediction. state/livesim and secret-looking paths are never read (drillsources.excluded).
  * HONEST SELECT/REPORT: `select_and_report` picks the learner on the first SELECT_SPLIT of the time-ordered predictions, reports the rest.
Learners: 'logit' (L2 logistic, numpy) and 'gbm' (lightgbm, small trees, imported on demand). Loaded on demand (registry 'predictors')."""
from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from creator import thinking as T

SELECT_SPLIT = 0.6
MIN_TRAIN = 40
CAP = 0.02
TEST_PATH = re.compile(r"(^|/)(tests?/|test_)")
AGENT_TRAILER = re.compile(r"co-authored-by:\s*claude", re.I)
CLASSES_MSG = ("feat", "fix", "test", "docs", "refactor", "merge", "wip", "chore", "nupen", "creator", "other")


@dataclass
class Row:
    """One drill event with setup-time features x (never the outcome)."""
    subject: str
    created: float
    resolved: Optional[float]
    y: int
    x: list[float] = field(default_factory=list)


def _l2(n: int) -> float:
    return math.log2(n + 1)


# ------------------------------------------------------------------------------------------------ git features (time-ordered, past only)
GIT_FEATURES = ("files", "add", "del", "lines", "test_share", "dirs", "msg_len", "is_fixword", "agent", "hour", "dow", "gap_log", "burst24",
                "area_n", "area_fix_rate", "area_recent50", "area_age_log") + tuple(f"msg_{c}" for c in CLASSES_MSG)


def git_rows(commits: Sequence[dict[str, Any]], window: int, mode: str) -> list[Row]:
    """Rows for the git drills. Features of commit i use ONLY commits 0..i (its own diff, and the earlier history of the same files). The label
    and resolution time are those of drillsources._git_events (the window of the next commits)."""
    from creator import drillsources as D                      # on demand: shared label/exclusion definitions
    last_touch: dict[str, int] = {}                            # file -> index of its latest earlier touch
    touch_n: dict[str, int] = {}
    touch_fix: dict[str, int] = {}
    touches: dict[str, list[int]] = {}
    times = [c["t"] for c in commits]
    out: list[Row] = []
    for i, c in enumerate(commits):
        fs = sorted(c["files"])
        subj = c["s"]
        mc = D._msg_class(subj)
        mc = mc if mc in CLASSES_MSG else "other"
        nt = sum(1 for f in fs if TEST_PATH.search(f))
        isfix = bool(D.FIX_WORDS.search(subj))
        an = [touch_n.get(f, 0) for f in fs]
        af = [touch_fix.get(f, 0) for f in fs]
        rec = sum(len(touches.get(f, [])) - bisect.bisect_left(touches.get(f, []), i - 50) for f in fs)
        age = min((i - last_touch[f] for f in fs if f in last_touch), default=1000)
        lo = bisect.bisect_left(times, c["t"] - 86400.0, 0, i)                      # commits in the prior 24h (only earlier indices)
        d = __import__("datetime").datetime.fromtimestamp(c["t"], __import__("datetime").timezone.utc)
        x = [_l2(len(fs)), _l2(c.get("add", 0)), _l2(c.get("del", 0)), _l2(c["lines"]), nt / len(fs) if fs else 0.0, float(len({f.split("/")[0] for f in fs})),
             float(len(subj)) / 20.0, float(isfix), float(bool(AGENT_TRAILER.search(c.get("body", "")) or "claude" in c.get("au", "").lower())),
             math.sin(2 * math.pi * d.hour / 24), float(d.weekday() >= 5), _l2(int(min(c["t"] - times[i - 1], 1e7)) if i else 0), _l2(i - lo),
             _l2(sum(an)), (sum(af) + 0.5) / (sum(an) + 2.0), _l2(rec), _l2(age)] + [float(mc == k) for k in CLASSES_MSG]
        # resolution of the label (window of next commits) exactly as the baseline drills define it
        if i + window >= len(commits):
            out.append(Row(c["h"][:10], c["t"], None, 0, x))
        else:
            nxt = commits[i + 1:i + 1 + window]
            if mode == "fixed":
                y = int(any(D.FIX_WORDS.search(n["s"]) and n["files"] & c["files"] for n in nxt))
            else:
                y = int(any(n["files"] & c["files"] for n in nxt))
            out.append(Row(c["h"][:10], c["t"], max(c["t"], nxt[-1]["t"]), y, x))
        for f in fs:                                              # update history AFTER the features of commit i were taken
            last_touch[f] = i
            touch_n[f] = touch_n.get(f, 0) + 1
            touch_fix[f] = touch_fix.get(f, 0) + int(isfix)
            touches.setdefault(f, []).append(i)
    return out


# ------------------------------------------------------------------------------------------------ package features (verdict / duration / cost)
PKG_FEATURES = ("spec_log", "files", "hour_sin", "hour_cos", "prior_same_req", "prior_same_adopt", "recent3_adopt", "recent10_adopt", "inflight",
                "gap_log", "kind_adopt_rate", "prior_n_log") + tuple(f"kind_{k}" for k in ("exists", "tested", "size", "other"))


def pkg_rows(items: Sequence[T.Item], topic: str) -> list[Row]:
    """Rows for verdict / duration / cost from the packages. Features of package i use its spec and the items RESOLVED before it was created
    (their outcomes), plus how many packages were in flight (created, not yet resolved) at that moment."""
    its = sorted(items, key=lambda i: i.created)
    res = sorted((i for i in its if i.resolved is not None), key=lambda i: i.resolved or 0.0)
    rtimes = [i.resolved or 0.0 for i in res]
    out: list[Row] = []
    for k, it in enumerate(its):
        if it.resolved is None or (topic in ("duration", "cost") and it.seconds <= 0):
            continue
        before = res[:bisect.bisect_left(rtimes, it.created)]
        same = [b for b in before if b.req == it.req]
        samek = [b for b in before if b.kind == it.kind]
        ad = lambda xs: (sum(1 for b in xs if b.outcome == "ADOPTED") + 0.5) / (len(xs) + 1.0)           # noqa: E731
        inflight = sum(1 for o in its[:k] if o.resolved is None or o.resolved >= it.created)
        d = __import__("datetime").datetime.fromtimestamp(it.created, __import__("datetime").timezone.utc)
        prev = its[k - 1].created if k else it.created
        kd = it.kind if it.kind in ("exists", "tested", "size") else "other"
        if topic == "verdict":
            hist = [ad(samek), len(samek)]
        else:
            ls = [math.log(max(b.seconds, 1.0)) for b in samek if b.seconds > 0]
            hist = [(sum(ls) / len(ls) if ls else math.log(T.SLOW_S)) / 7.0, len(ls)]
        x = [math.log(max(it.spec_len, 1)), float(it.n_files), math.sin(2 * math.pi * d.hour / 24), math.cos(2 * math.pi * d.hour / 24),
             _l2(len(same)), ad(same), ad(before[-3:]), ad(before[-10:]), float(inflight), _l2(int(min(it.created - prev, 1e7))), hist[0], _l2(int(hist[1]))] + \
            [float(kd == v) for v in ("exists", "tested", "size", "other")]
        y = int(it.outcome == "ADOPTED") if topic == "verdict" else int(it.seconds > T.SLOW_S)
        out.append(Row(it.pkg, it.created, it.resolved, y, x))
    return out


# ------------------------------------------------------------------------------------------------ learners
def _np() -> Any:
    import numpy as np
    return np


def fit_logit(X: Any, y: Any, lam: float = 5.0) -> Callable[[Any], Any]:
    """L2-regularised logistic regression on standardised features (Newton / IRLS, <=30 iterations); returns predict_proba."""
    np = _np()
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    reg = np.eye(Z.shape[1]) * lam
    reg[-1, -1] = 0.0
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + reg @ w
        H = (Z.T * (p * (1 - p) + 1e-6)) @ Z + reg
        step = np.linalg.solve(H, g)
        w -= step
        if float(np.abs(step).max()) < 1e-6:
            break
    return lambda Xn: 1 / (1 + np.exp(-np.clip(np.hstack([(Xn - mu) / sd, np.ones((len(Xn), 1))]) @ w, -30, 30)))


def fit_gbm(X: Any, y: Any, trees: int = 60) -> Callable[[Any], Any]:
    """Small, strongly regularised gradient-boosted trees (lightgbm, imported on demand, single thread so it never competes with Nupen's work)."""
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=trees, learning_rate=0.05, num_leaves=5, min_child_samples=25, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=5.0, n_jobs=1, verbose=-1, random_state=0)
    m.fit(X, y)
    return lambda Xn: m.predict_proba(Xn)[:, 1]


LEARNERS: dict[str, Callable[..., Callable[[Any], Any]]] = {"logit": fit_logit, "gbm": fit_gbm}


def platt(raw: Sequence[float], y: Sequence[int]) -> tuple[float, float]:
    """Fit p = sigmoid(a*logit(raw)+b) by Newton on earlier out-of-sample predictions, lightly ridge-pulled toward the identity (a=1, b=0)."""
    np = _np()
    z = np.log(np.clip(raw, 1e-4, 1 - 1e-4) / (1 - np.clip(raw, 1e-4, 1 - 1e-4)))
    Z = np.stack([z, np.ones_like(z)], 1)
    yy = np.asarray(y, dtype=float)
    w = np.array([1.0, 0.0])
    prior = np.array([1.0, 0.0])
    for _ in range(60):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - yy) + 2.0 * (w - prior)
        H = (Z.T * (p * (1 - p) + 1e-6)) @ Z + 2.0 * np.eye(2)
        w = w - np.clip(np.linalg.solve(H, g), -1.0, 1.0)       # damped: a pure Newton step overshoots from a confident start
    return float(w[0]), float(w[1])


def walk_forward_model(rows: Sequence[Row], topic: str, learner: str = "logit", refit_every: int = 50, calibrate: bool = True, lam: float = 5.0,
                       cols: Optional[Sequence[int]] = None) -> list[T.Pred]:
    """Predict each resolved row at its creation time from a model fitted on rows resolved strictly before it (refit every `refit_every` new
    resolutions); then Platt-calibrate on earlier out-of-sample predictions whose outcome was already known. Baselines as in the other drills."""
    np = _np()
    rs = [r for r in rows if r.resolved is not None]
    ev = sorted([(r.created, 1, i) for i, r in enumerate(rs)] + [(r.resolved or 0.0, 0, i) for i, r in enumerate(rs)])
    X = np.array([[r.x[c] for c in (cols if cols is not None else range(len(r.x)))] for r in rs], dtype=float) if rs else np.zeros((0, 0))
    Y = np.array([r.y for r in rs], dtype=float)
    done: list[int] = []                                       # row indices resolved so far
    hist: list[int] = []
    model: Optional[Callable[[Any], Any]] = None
    fitted_at = -10 ** 9
    pend: dict[int, float] = {}                                # row -> its raw out-of-sample prediction, until it resolves
    cal_raw: list[float] = []
    cal_y: list[int] = []
    ab = (1.0, 0.0)
    cal_at = -10 ** 9
    out: list[T.Pred] = []
    for _t, kind, i in ev:
        if kind == 0:
            done.append(i)
            hist.append(rs[i].y)
            if i in pend:
                cal_raw.append(pend.pop(i))
                cal_y.append(rs[i].y)
        else:
            base, last = T._laplace(hist), T._last(hist)
            if len(done) >= MIN_TRAIN and len(set(Y[done])) == 2:
                if len(done) - fitted_at >= refit_every:
                    kw = {"lam": lam} if learner == "logit" else {}
                    model = LEARNERS[learner](X[done], Y[done], **kw)
                    fitted_at = len(done)
                if calibrate and len(cal_raw) >= 80 and len(cal_raw) - cal_at >= refit_every and len(set(cal_y)) == 2:
                    ab = platt(cal_raw, cal_y)
                    cal_at = len(cal_raw)
            raw = float(model(X[i:i + 1])[0]) if model is not None else base
            pend[i] = raw
            p = raw
            if calibrate and model is not None and cal_at > 0:
                z = math.log(min(1 - 1e-4, max(1e-4, raw)) / (1 - min(1 - 1e-4, max(1e-4, raw))))
                p = 1 / (1 + math.exp(-(ab[0] * z + ab[1])))
            out.append(T.Pred(topic, rs[i].subject, rs[i].created, min(1 - CAP, max(CAP, p)), base, last, rs[i].y, extra={"learner": learner}))
    return out


# ------------------------------------------------------------------------------------------------ honest selection
CANDIDATES = ({"learner": "logit", "lam": 5.0}, {"learner": "logit", "lam": 30.0}, {"learner": "gbm"})


def select_and_report(rows: Sequence[Row], topic: str, candidates: Sequence[dict[str, Any]] = CANDIDATES, refit_every: int = 50) -> dict[str, Any]:
    """Run each candidate walk-forward, choose the one with the lowest Brier on the FIRST SELECT_SPLIT of the time-ordered predictions and report
    its score on the held-out tail (plus the frequency model's score on the same tail for the before/after)."""
    best: Optional[tuple[float, dict[str, Any], list[T.Pred]]] = None
    tried = []
    for c in candidates:
        ps = walk_forward_model(rows, topic, refit_every=refit_every, **c)
        cut = int(len(ps) * SELECT_SPLIT)
        sc = T.score(ps[:cut])
        tried.append({"candidate": c, "select_brier": sc.get("brier")})
        if sc.get("n") and (best is None or sc["brier"] < best[0]):
            best = (sc["brier"], c, ps)
    if best is None:
        return {"n": 0}
    ps = best[2]
    cut = int(len(ps) * SELECT_SPLIT)
    return {"chosen": best[1], "tried": tried, "heldout": T.score(ps[cut:]), "select": T.score(ps[:cut]), "preds": ps}
