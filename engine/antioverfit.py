"""Bible Phase 26 (anti-overfitting battery); serves the no-look-ahead and no-fake-signal canons.

An *evaluator* is any callable  evaluator(X, y, now) -> dict  where X/y follow the panel convention (MultiIndex
(date, ticker)); the dict must hold "metric" (float, higher is better, e.g. out-of-sample rank IC) and may hold
"t" (evidence statistic), "importance" ({feature: weight}), "predictions" (Series on the panel index), "ic_series",
"splits". The battery never looks inside the evaluator: it destroys or perturbs the INPUT in a way whose effect is
known in advance and returns a verdict comparing the perturbed runs with the real run:

  A future_scramble      rows after `now` scrambled -> nothing dated <= now may change
  B label_permutation    labels permuted inside each date -> metric must collapse to ~0
  C ticker_permutation   tickers relabelled (invariance) and ticker y-histories swapped (collapse)
  D date_disguise        all dates shifted by whole weeks -> metric must not change
  E randomized_outcomes  labels redrawn from the pooled marginal -> false-discovery rate must stay near alpha
  F feature_shuffle      the most important features shuffled -> signal must fall
  G dead_feature_injection   random / ticker-constant / random-walk columns -> no stable gain, no importance share
  H duplicate_feature_injection   copies of real features -> evidence must not double
  I regime_split         metric evaluated inside each regime separately -> reports concentration or sign flips
  J walk_forward         splits audited: no training row within `horizon` of, or after, a test row

Verdict status: "pass", "fail" (the pipeline manufactures or hides signal), "not_significant" (pipeline clean, but the
real result is not distinguishable from the null), "flag" (worth a human look), "inconclusive" (the real run has no
signal for the test to bite on) or "error" (a test crashed - treated as a failure, fail closed).
Everything is deterministic in `seed`."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TOL = {"invariance": 1e-8, "dup_gain_rel": 0.10, "dup_gain_abs": 0.002, "dup_evidence_ratio": 1.15,
       "dead_gain_abs": 0.003, "collapse_z": 4.0, "collapse_abs": 0.005, "real_z": 2.0, "alpha": 0.05,
       "retain_max": 0.5, "concentration": 0.8}
PASSING = ("pass", "skipped", "inconclusive", "flag", "not_significant")
ALL_TESTS = ("future_scramble", "label_permutation", "ticker_permutation", "date_disguise", "randomized_outcomes",
             "feature_shuffle", "dead_feature_injection", "duplicate_feature_injection", "regime_split",
             "walk_forward")


# ------------------------------------------------------------------ panel helpers
def check_panel(X, y):
    """Validate the panel convention and return (X, y) sorted by (date, ticker). Raises ValueError with the reason."""
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise ValueError("X must be a DataFrame and y a Series")
    if len(X) == 0:
        raise ValueError("empty panel")
    if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels != 2:
        raise ValueError("X must be indexed by MultiIndex (date, ticker)")
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    if X.index.has_duplicates:
        raise ValueError("duplicate (date, ticker) rows")
    if X.shape[1] == 0:
        raise ValueError("no feature columns")
    return X.sort_index(), y.sort_index()


def feature_cols(X):
    return [c for c in X.columns if not str(c).startswith("m_")]


def _bounds(dates):
    """Start/stop row positions of each date block in a date-sorted index."""
    d = np.asarray(dates)
    cut = np.flatnonzero(d[1:] != d[:-1]) + 1
    return np.concatenate([[0], cut]), np.concatenate([cut, [len(d)]])


def permute_within_date(values, dates, rng):
    """Permute `values` inside every date block (dates must be sorted)."""
    out = np.array(values, copy=True)
    for a, b in zip(*_bounds(dates)):
        out[a:b] = out[a:b][rng.permutation(b - a)]
    return out


def _rank_center(X):
    return (X.groupby(level=0).rank(pct=True) - 0.5).fillna(0.0)


def _rankdata(a):
    return stats.rankdata(a, method="average")


# ------------------------------------------------------------------ default evaluator
def walk_forward_splits(dates, n_folds=4, horizon=5, min_train_dates=20):
    """Expanding-window splits over sorted unique `dates`. A training block ends `horizon` sessions before the test
    block starts, so a label that looks `horizon` sessions ahead cannot reach into the test period."""
    ud = np.sort(pd.DatetimeIndex(pd.unique(np.asarray(dates))).values)
    chunks = np.array_split(np.arange(len(ud)), n_folds + 1)
    out = []
    for k in range(1, n_folds + 1):
        if len(chunks[k]) == 0:
            continue
        t0 = int(chunks[k][0])
        tr_end = t0 - horizon
        if tr_end < min_train_dates:
            continue
        out.append({"train_start": pd.Timestamp(ud[0]), "train_end": pd.Timestamp(ud[tr_end - 1]),
                    "test_start": pd.Timestamp(ud[t0]), "test_end": pd.Timestamp(ud[int(chunks[k][-1])]),
                    "n_train_dates": int(tr_end), "n_test_dates": int(len(chunks[k])), "horizon": int(horizon)})
    return out


def verify_walk_forward(splits, dates, horizon=None):
    """Phase 26 J. Return a list of violations (empty = clean): training must end at least `horizon` sessions before
    the test starts, folds must move forward, and no test block may precede its own training block."""
    ud = np.sort(pd.DatetimeIndex(pd.unique(np.asarray(dates))).values)
    bad = []
    if not splits:
        return ["no splits"]
    prev_test = None
    for i, s in enumerate(splits):
        h = int(s.get("horizon", 0) if horizon is None else horizon)
        te, ts = pd.Timestamp(s["train_end"]), pd.Timestamp(s["test_start"])
        if te >= ts:
            bad.append(f"fold {i}: training ends {te.date()} not before test start {ts.date()}")
            continue
        gap = int(np.searchsorted(ud, np.datetime64(ts))) - int(np.searchsorted(ud, np.datetime64(te))) - 1
        if gap < h:
            bad.append(f"fold {i}: only {gap} sessions between train end and test start, horizon is {h}")
        if pd.Timestamp(s["test_end"]) < ts:
            bad.append(f"fold {i}: test ends before it starts")
        if prev_test is not None and ts < prev_test:
            bad.append(f"fold {i}: test start {ts.date()} earlier than the previous fold's")
        prev_test = ts
    return bad


def wf_evaluate(X, y, now, n_folds=4, horizon=5, lam=1.0, min_train_dates=20):
    """Reference evaluator: expanding-window ridge on cross-sectionally ranked features, scored by mean daily rank IC
    on the held-out blocks. Only rows dated <= `now` are read. Pure numpy/scipy, deterministic."""
    now = pd.Timestamp(now)
    dates = X.index.get_level_values(0)
    keep = (dates <= now) & y.notna().values
    X, y = X[keep], y[keep]
    empty = {"metric": float("nan"), "t": float("nan"), "importance": {}, "predictions": pd.Series(dtype=float),
             "ic_series": pd.Series(dtype=float), "splits": [], "note": "insufficient dates"}
    if len(X) == 0:
        return empty
    cols = feature_cols(X)
    if not cols:
        return {**empty, "note": "no non-context features"}
    R = _rank_center(X[cols]).values
    yd = (y - y.groupby(level=0).transform("mean")).values
    d = X.index.get_level_values(0)
    splits = walk_forward_splits(d, n_folds, horizon, min_train_dates)
    if not splits:
        return empty
    preds, ics, imp = [], {}, np.zeros(len(cols))
    for s in splits:
        tr = np.asarray(d <= s["train_end"])
        te = np.asarray((d >= s["test_start"]) & (d <= s["test_end"]))
        A, b = R[tr], yd[tr]
        beta = np.linalg.solve(A.T @ A / len(A) + lam * np.eye(len(cols)) * 0.01, A.T @ b / len(A))
        imp += np.abs(beta)
        p = R[te] @ beta
        preds.append(pd.Series(p, index=X.index[te]))
        dte, yte = d[te], yd[te]
        for a, z in zip(*_bounds(dte)):
            if z - a < 5:
                continue
            rp, ry = _rankdata(p[a:z]), _rankdata(yte[a:z])
            if rp.std() > 0 and ry.std() > 0:
                ics[dte[a]] = float(np.corrcoef(rp, ry)[0, 1])
    ic = pd.Series(ics).sort_index()
    if len(ic) < 3:
        return {**empty, "splits": splits}
    n_eff = max(len(ic) / max(horizon, 1), 1.0)
    sd = float(ic.std(ddof=1))
    t = float(ic.mean() / (sd / np.sqrt(n_eff))) if sd > 0 else 0.0
    tot = imp.sum() or 1.0
    return {"metric": float(ic.mean()), "t": t, "se": float(sd / np.sqrt(n_eff)), "importance": dict(zip(cols, (imp / tot).tolist())),
            "predictions": pd.concat(preds), "ic_series": ic, "splits": splits, "horizon": horizon}


def miner_evaluator(params=None):
    """Adapter that lets the battery drive engine.patterns.PatternMiner. metric = the miner's out-of-sample gate
    correlation on its confirmation window; importance = share of active patterns that use each feature."""
    from engine.patterns import PatternMiner

    def run(X, y, now):
        m = PatternMiner(params).fit(X, y, now)
        act = m.patterns[m.patterns["status"].isin(["active", "rescoped"])] if len(m.patterns) else m.patterns
        imp = {f: 0.0 for f in m.feats}
        for k in (act["key"] if len(act) else []):
            for j in range(1, len(k), 2):
                if isinstance(k[j], (int, np.integer)) and k[j] < len(m.feats):
                    imp[m.feats[k[j]]] += 1.0
        tot = sum(imp.values()) or 1.0
        n_act = int(len(act))
        return {"metric": float(getattr(m, "gate_corr", 0.0)), "t": float(np.sqrt(n_act)),
                "importance": {f: v / tot for f, v in imp.items()}, "evidence": float(n_act)}
    return run


# ------------------------------------------------------------------ verdict plumbing
def _verdict(name, status, reason, **kw):
    return {"name": name, "status": status, "reason": reason, **kw}


def _null_summary(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "q95": float("nan")}
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "q95": float(np.quantile(v, 0.95))}


def _emp_p(real, null):
    v = np.asarray([x for x in null if np.isfinite(x)])
    return float((1 + (v >= real).sum()) / (len(v) + 1))


def _se_ref(real):
    """Standard error of ONE run's metric, from the real run: what a null run's metric should scatter by."""
    se = real.get("se")
    if se is None and real.get("t") and np.isfinite(real["t"]) and real["t"] != 0:
        se = abs(real["metric"] / real["t"])
    return float(se) if se is not None and np.isfinite(se) else 0.0


def _collapsed(ns, se_ref=0.0):
    """True when the null metric's mean is statistically and practically indistinguishable from zero. The scatter of a
    single null run is taken as the larger of the observed spread and 0.6x the real run's standard error, so a handful of
    repetitions cannot under-estimate it."""
    if ns["n"] < 2:
        return False
    se = max(ns["sd"], 0.6 * se_ref, 1e-12) / np.sqrt(ns["n"])
    return abs(ns["mean"]) <= max(TOL["collapse_z"] * se, TOL["collapse_abs"])


def _run(ev, X, y, now):
    r = ev(X, y, now)
    if "metric" not in r:
        raise ValueError("evaluator result has no 'metric'")
    return r


def _real_has_signal(real):
    return np.isfinite(real["metric"]) and real["metric"] > 0 and real.get("t", 0.0) >= TOL["real_z"]


def _null_test(name, ev, X, y, now, real, make_null, n_rep, rng, what):
    """Shared body for label-destroying nulls (B, E, C-collapse): run the evaluator on n_rep destroyed copies."""
    vals = [_run(ev, X, make_null(rng), now)["metric"] for _ in range(n_rep)]
    ns = _null_summary(vals)
    p = _emp_p(real["metric"], vals)
    if not _collapsed(ns, _se_ref(real)):
        return _verdict(name, "fail", f"{what}: metric stayed at {ns['mean']:.4f} (should be ~0) - the pipeline "
                        "manufactures signal from noise", real=real["metric"], null=ns, p_value=p)
    if real["metric"] > ns["q95"] and p <= max(0.1, 1.5 / (ns["n"] + 1)):      # p cannot go below 1/(n+1)
        return _verdict(name, "pass", f"{what}: null collapsed to {ns['mean']:.4f}; real {real['metric']:.4f} beats it",
                        real=real["metric"], null=ns, p_value=p)
    return _verdict(name, "not_significant", f"{what}: null collapsed, but real {real['metric']:.4f} does not clear the "
                    f"null 95th percentile {ns['q95']:.4f}", real=real["metric"], null=ns, p_value=p)


# ------------------------------------------------------------------ A. future scramble
def future_scramble(ev, X, y, now, real, rng, cutoff=None, **_):
    """Rows after `cutoff` (default: the date 60% through the panel) get scrambled, rescaled features and fresh wide labels. The evaluator
    is called with now=cutoff; predictions and metric for rows dated <= cutoff must be bit-identical."""
    d = X.index.get_level_values(0)
    ud = np.sort(d.unique())
    cutoff = pd.Timestamp(ud[int(len(ud) * 0.6)] if cutoff is None else cutoff)
    fut = np.asarray(d > cutoff)
    if fut.sum() < 2 or (~fut).sum() < 2:
        return _verdict("future_scramble", "inconclusive", "no rows after the cutoff to scramble")
    base = _run(ev, X, y, cutoff)
    Xs, ys = X.copy(), y.astype(float)
    idx = np.flatnonzero(fut)
    # permuting alone keeps every pooled statistic (mean, quantiles) of the future intact and would hide a pipeline that
    # reads them, so the scrambled rows are also rescaled and given fresh labels from a much wider distribution
    Xs.iloc[idx] = X.iloc[idx[rng.permutation(len(idx))]].values * 3.0
    ys.iloc[idx] = rng.standard_normal(len(idx)) * 5.0 * float(y.std())
    alt = _run(ev, Xs, ys, cutoff)
    diffs = []
    if not np.allclose([base["metric"]], [alt["metric"]], rtol=0, atol=TOL["invariance"], equal_nan=True):
        diffs.append(f"metric {base['metric']!r} -> {alt['metric']!r}")
    bp, ap = base.get("predictions"), alt.get("predictions")
    if bp is not None and ap is not None and len(bp):
        b0 = bp[bp.index.get_level_values(0) <= cutoff]
        a0 = ap.reindex(b0.index)
        n_bad = int((~np.isclose(b0.values, a0.values, rtol=0, atol=TOL["invariance"], equal_nan=True)).sum())
        if n_bad:
            diffs.append(f"{n_bad} of {len(b0)} predictions dated <= cutoff changed")
        late = ap[ap.index.get_level_values(0) > cutoff]
        if len(late):
            diffs.append(f"{len(late)} predictions issued for dates after `now`")
    if diffs:
        return _verdict("future_scramble", "fail", "future data affected earlier decisions: " + "; ".join(diffs),
                        cutoff=str(cutoff.date()))
    return _verdict("future_scramble", "pass", "scrambling everything after the cutoff changed nothing",
                    cutoff=str(cutoff.date()), n_scrambled=int(fut.sum()))


# ------------------------------------------------------------------ B. label permutation
def label_permutation(ev, X, y, now, real, rng, n_rep=12, **_):
    d = y.index.get_level_values(0)
    return _null_test("label_permutation", ev, X, y, now, real,
                      lambda g: pd.Series(permute_within_date(y.values, d, g), index=y.index), n_rep, rng,
                      "labels permuted within date")


# ------------------------------------------------------------------ C. ticker permutation
def _relabel_tickers(X, y, rng):
    tk = X.index.get_level_values(1)
    uniq = pd.unique(tk)
    new = dict(zip(uniq, rng.permutation(uniq)))
    idx = pd.MultiIndex.from_arrays([X.index.get_level_values(0), tk.map(new)], names=X.index.names)
    X2, y2 = X.copy(), y.copy()
    X2.index, y2.index = idx, idx
    return X2.sort_index(), y2.sort_index()


def _swap_ticker_histories(y, rng):
    """Every ticker receives another ticker's whole label history (same dates)."""
    wide = y.unstack()
    n = wide.shape[1]
    if n < 3:
        return None
    shift = int(rng.integers(1, n))
    perm = (np.arange(n) + shift) % n
    sw = wide.values[:, perm]
    di = wide.index.get_indexer(y.index.get_level_values(0))
    ti = wide.columns.get_indexer(y.index.get_level_values(1))
    return pd.Series(sw[di, ti], index=y.index)


def ticker_permutation(ev, X, y, now, real, rng, n_rep=8, **_):
    """(1) Relabelling tickers by a random bijection must not change the metric (no dependence on identity or on
    ticker sort order). (2) Handing each ticker another ticker's label history must collapse the metric."""
    X2, y2 = _relabel_tickers(X, y, rng)
    alt = _run(ev, X2, y2, now)
    delta = abs(alt["metric"] - real["metric"])
    if not np.isfinite(delta) or delta > TOL["invariance"]:
        return _verdict("ticker_permutation", "fail", f"relabelling tickers moved the metric by {delta:.3g} - the "
                        "result depends on ticker identity or ordering", real=real["metric"], relabelled=alt["metric"])
    if y.unstack().shape[1] < 3:
        return _verdict("ticker_permutation", "inconclusive", "fewer than 3 tickers, history swap impossible")
    v = _null_test("ticker_permutation", ev, X, y, now, real,
                   lambda g: _swap_ticker_histories(y, g), n_rep, rng, "ticker label histories swapped")
    v["reason"] = "relabel invariant; " + v["reason"]
    return v


# ------------------------------------------------------------------ D. date disguise
def date_disguise(ev, X, y, now, real, rng, **_):
    """Shift every date (and `now`) by a random whole number of weeks. Structure is intact, absolute dates are not."""
    delta = pd.Timedelta(days=7 * int(rng.integers(52, 52 * 12)))
    d = X.index.get_level_values(0) + delta
    idx = pd.MultiIndex.from_arrays([d, X.index.get_level_values(1)], names=X.index.names)
    X2, y2 = X.copy(), y.copy()
    X2.index, y2.index = idx, idx
    alt = _run(ev, X2, y2, pd.Timestamp(now) + delta)
    diff = abs(alt["metric"] - real["metric"])
    if not np.isfinite(diff) or diff > TOL["invariance"]:
        return _verdict("date_disguise", "fail", f"shifting dates by {delta.days} days moved the metric by {diff:.3g} "
                        "- behaviour is tied to absolute calendar dates", real=real["metric"], shifted=alt["metric"],
                        shift_days=int(delta.days))
    return _verdict("date_disguise", "pass", f"metric unchanged after a {delta.days}-day shift",
                    real=real["metric"], shifted=alt["metric"], shift_days=int(delta.days))


# ------------------------------------------------------------------ E. randomized outcomes
def randomized_outcomes(ev, X, y, now, real, rng, n_rep=20, alpha=None, **_):
    """Labels redrawn i.i.d. from the pooled empirical marginal. Discovery must then fire at about the nominal rate:
    the count of runs with a 'significant' positive t must sit under the 99% binomial bound."""
    alpha = TOL["alpha"] if alpha is None else alpha
    pool = y.dropna().values
    vals, ts = [], []
    for _ in range(n_rep):
        r = _run(ev, X, pd.Series(rng.choice(pool, len(y)), index=y.index), now)
        vals.append(r["metric"]); ts.append(r.get("t", np.nan))
    ns = _null_summary(vals)
    z = stats.norm.ppf(1 - alpha)
    hits = int(np.nansum(np.asarray(ts, dtype=float) > z))
    bound = int(stats.binom.ppf(0.99, n_rep, alpha))
    if hits > bound or not _collapsed(ns, _se_ref(real)):
        return _verdict("randomized_outcomes", "fail", f"{hits}/{n_rep} random-outcome runs looked significant "
                        f"(bound {bound}); null mean {ns['mean']:.4f} - discovery does not collapse toward chance",
                        null=ns, false_discoveries=hits, bound=bound)
    return _verdict("randomized_outcomes", "pass", f"{hits}/{n_rep} false discoveries at alpha={alpha} (bound {bound})",
                    null=ns, false_discoveries=hits, bound=bound)


# ------------------------------------------------------------------ F. feature shuffle
def _shuffle_cols(X, cols, rng):
    Xs = X.copy()
    d = X.index.get_level_values(0)
    for c in cols:
        Xs[c] = permute_within_date(X[c].values, d, rng)
    return Xs


def feature_shuffle(ev, X, y, now, real, rng, n_rep=6, top_k=3, **_):
    imp = real.get("importance") or {}
    if not imp:
        return _verdict("feature_shuffle", "inconclusive", "evaluator reports no importance")
    if not _real_has_signal(real):
        return _verdict("feature_shuffle", "inconclusive", "real run has no significant positive signal to destroy",
                        real=real["metric"])
    top = [f for f, _ in sorted(imp.items(), key=lambda kv: -kv[1])[:top_k] if f in X.columns]
    per = {}
    for f in top:
        v = [_run(ev, _shuffle_cols(X, [f], rng), y, now)["metric"] for _ in range(n_rep)]
        ns = _null_summary(v)
        per[f] = {"shuffled_mean": ns["mean"], "drop": real["metric"] - ns["mean"],
                  "z": (real["metric"] - ns["mean"]) / max(ns["sd"], 1e-9)}
    joint = _null_summary([_run(ev, _shuffle_cols(X, top, rng), y, now)["metric"] for _ in range(n_rep)])
    retain = joint["mean"] / real["metric"]
    if retain > TOL["retain_max"]:
        return _verdict("feature_shuffle", "fail", f"shuffling the top {len(top)} features kept {retain:.0%} of the "
                        "signal - it is carried by something the importance table does not show", per_feature=per,
                        joint=joint, retained=retain)
    dull = [f for f, r in per.items() if r["z"] < TOL["real_z"]]
    st = "flag" if dull else "pass"
    why = f"joint shuffle kept {retain:.0%}" + (f"; individually not load-bearing: {dull}" if dull else "")
    return _verdict("feature_shuffle", st, why, per_feature=per, joint=joint, retained=retain)


# ------------------------------------------------------------------ G. dead features
def _dead_columns(X, n, rng):
    """Three kinds of features that carry no information: iid noise, per-ticker constants, per-ticker random walks."""
    d, tk = X.index.get_level_values(0), X.index.get_level_values(1)
    out = {}
    ut, ud = pd.unique(tk), np.sort(pd.unique(d))
    for i in range(n):
        kind = i % 3
        if kind == 0:
            out[f"dead_noise_{i}"] = rng.standard_normal(len(X))
        elif kind == 1:
            out[f"dead_const_{i}"] = pd.Series(tk).map(dict(zip(ut, rng.standard_normal(len(ut))))).values
        else:
            walk = np.cumsum(rng.standard_normal((len(ud), len(ut))), axis=0)
            out[f"dead_walk_{i}"] = walk[np.searchsorted(ud, d.values), pd.Index(ut).get_indexer(tk)]
    return pd.DataFrame(out, index=X.index)


def dead_feature_injection(ev, X, y, now, real, rng, n_rep=6, n_dead=9, **_):
    gains, shares, only = [], [], []
    for _i in range(n_rep):
        D = _dead_columns(X, n_dead, rng)
        r = _run(ev, pd.concat([X, D], axis=1), y, now)
        gains.append(r["metric"] - real["metric"])
        imp = r.get("importance") or {}
        if imp:
            shares.append(sum(v for k, v in imp.items() if str(k).startswith("dead_")))
        o = _run(ev, D.join(X[[c for c in X.columns if str(c).startswith("m_")]]), y, now)
        only.append((o["metric"], o.get("t", np.nan)))
    g = _null_summary(gains)
    se = max(g["sd"], 1e-12) / np.sqrt(max(g["n"], 1))
    om = _null_summary([m for m, _ in only])
    hits = int(np.nansum(np.asarray([t for _, t in only], dtype=float) > stats.norm.ppf(1 - TOL["alpha"])))
    bound = int(stats.binom.ppf(0.99, n_rep, TOL["alpha"]))
    nfeat = len(feature_cols(X))
    expected = n_dead / (n_dead + nfeat)
    share = float(np.mean(shares)) if shares else None
    problems = []
    if g["mean"] > max(TOL["dead_gain_abs"], TOL["real_z"] * se):
        problems.append(f"adding dead features raised the metric by {g['mean']:.4f}")
    if hits > bound or not _collapsed(om, _se_ref(real)):
        problems.append(f"dead features alone looked predictive ({hits}/{n_rep} significant, mean {om['mean']:.4f})")
    if share is not None and share > 2 * expected and share > 0.25:
        problems.append(f"dead features took {share:.0%} of importance (chance {expected:.0%})")
    kw = dict(gain=g, dead_only=om, importance_share=share, expected_share=expected)
    if problems:
        return _verdict("dead_feature_injection", "fail", "; ".join(problems), **kw)
    return _verdict("dead_feature_injection", "pass", f"{n_dead} dead columns: gain {g['mean']:+.4f}, alone "
                    f"{om['mean']:+.4f}", **kw)


# ------------------------------------------------------------------ H. duplicate features
def duplicate_feature_injection(ev, X, y, now, real, rng, copies=3, top_k=3, **_):
    """Add `copies` exact duplicates of the top features. Evidence ('evidence' or 't') and metric may not inflate."""
    imp = real.get("importance") or {}
    cols = feature_cols(X)
    order = [f for f, _ in sorted(imp.items(), key=lambda kv: -kv[1]) if f in X.columns] or cols
    pick = order[:top_k]
    D = pd.concat({f"dup{k}_{f}": X[f] for f in pick for k in range(copies)}, axis=1)
    D.columns = list(D.columns.get_level_values(0)) if isinstance(D.columns, pd.MultiIndex) else D.columns
    r = _run(ev, pd.concat([X, D], axis=1), y, now)
    dm = r["metric"] - real["metric"]
    ek = "evidence" if "evidence" in real else "t"
    e0, e1 = real.get(ek), r.get(ek)
    problems = []
    if np.isfinite(dm) and dm > TOL["dup_gain_abs"] + TOL["dup_gain_rel"] * abs(real["metric"]) + 0.5 * _se_ref(real):
        problems.append(f"metric rose {dm:+.4f} from copies alone")
    ratio = None
    if e0 is not None and e1 is not None and np.isfinite(e0) and e0 > 0:
        ratio = float(e1 / e0)
        if ratio > TOL["dup_evidence_ratio"] and e0 >= TOL["real_z"]:       # a ratio on near-zero evidence is noise
            problems.append(f"{ek} inflated x{ratio:.2f} - duplicates double-count evidence")
    kw = dict(metric_change=float(dm), evidence_key=ek, evidence_ratio=ratio, duplicated=pick)
    if problems:
        return _verdict("duplicate_feature_injection", "fail", "; ".join(problems), **kw)
    return _verdict("duplicate_feature_injection", "pass", f"{copies} copies of {len(pick)} features: metric "
                    f"{dm:+.4f}, {ek} ratio {ratio if ratio is None else round(ratio, 2)}", **kw)


# ------------------------------------------------------------------ I. regimes
def regime_labels(X, col="m_vix", n=3):
    """Date -> regime label by quantile of a date-level context column; None when the column is missing."""
    if col not in X.columns:
        return None
    dv = X[col].groupby(level=0).mean()
    if dv.nunique() < n:
        return None
    return pd.qcut(dv.rank(method="first"), n, labels=[f"{col}_q{i + 1}" for i in range(n)])


def regime_split(ev, X, y, now, real, rng, regimes=None, regime_col="m_vix", n_regimes=3, **_):
    lab = regimes if regimes is not None else regime_labels(X, regime_col, n_regimes)
    if lab is None:
        return _verdict("regime_split", "inconclusive", f"no regime definition ({regime_col} missing or constant)")
    per = {}
    ic = real.get("ic_series")
    h = max(int(real.get("horizon", 1) or 1), 1)
    if ic is not None and len(ic):
        # score the ONE model the real run produced inside each regime (not a model re-fitted per regime)
        for r in sorted(pd.unique(lab.dropna()).tolist(), key=str):
            v = ic[ic.index.isin(lab.index[lab == r])]
            if len(v) < 8:
                per[str(r)] = {"metric": float("nan"), "t": float("nan"), "n_dates": int(len(v)), "note": "too few dates"}
                continue
            sd = float(v.std(ddof=1))
            per[str(r)] = {"metric": float(v.mean()), "n_dates": int(len(v)),
                           "t": float(v.mean() / (sd / np.sqrt(max(len(v) / h, 1.0)))) if sd > 0 else 0.0}
    else:
        d = X.index.get_level_values(0)
        for r in sorted(pd.unique(lab.dropna()).tolist(), key=str):
            m = np.asarray(d.isin(lab.index[lab == r]))
            if m.sum() < 50:
                per[str(r)] = {"metric": float("nan"), "t": float("nan"), "n_rows": int(m.sum()), "note": "too few rows"}
                continue
            o = _run(ev, X[m], y[m], now)
            per[str(r)] = {"metric": o["metric"], "t": o.get("t", float("nan")), "n_rows": int(m.sum())}
    ok = {k: v for k, v in per.items() if np.isfinite(v["metric"])}
    if len(ok) < 2:
        return _verdict("regime_split", "inconclusive", "fewer than two regimes had enough data", regimes=per)
    neg = [k for k, v in ok.items() if v["t"] <= -TOL["real_z"]]
    pos_sum = sum(max(v["metric"], 0) for v in ok.values())
    top = max(ok.items(), key=lambda kv: kv[1]["metric"])
    conc = top[1]["metric"] / pos_sum if pos_sum > 0 else 0.0
    if neg:
        return _verdict("regime_split", "fail", f"significantly negative in {neg}", regimes=per)
    if len(ok) > 2 and conc > TOL["concentration"] and pos_sum > 0:
        return _verdict("regime_split", "flag", f"{conc:.0%} of the positive signal comes from {top[0]}", regimes=per,
                        concentration=conc)
    npos = sum(v["metric"] > 0 for v in ok.values())
    st = "pass" if npos >= 0.66 * len(ok) else "flag"
    return _verdict("regime_split", st, f"{npos}/{len(ok)} regimes positive", regimes=per, concentration=conc)


# ------------------------------------------------------------------ J. walk-forward
def walk_forward(ev, X, y, now, real, rng, horizon=None, **_):
    splits = real.get("splits")
    if not splits:
        return _verdict("walk_forward", "fail", "evaluator exposes no splits, so the absence of look-ahead in "
                        "training cannot be audited")
    dates = X.index.get_level_values(0)
    bad = verify_walk_forward(splits, dates, horizon if horizon is not None else real.get("horizon"))
    over = [i for i, s in enumerate(splits) if pd.Timestamp(s["test_end"]) > pd.Timestamp(now)]
    bad += [f"fold {i}: test ends after now" for i in over]
    if bad:
        return _verdict("walk_forward", "fail", "; ".join(bad[:5]), violations=bad)
    return _verdict("walk_forward", "pass", f"{len(splits)} folds, training never within the label horizon of a test row",
                    n_folds=len(splits))


_TESTS = {"future_scramble": future_scramble, "label_permutation": label_permutation,
          "ticker_permutation": ticker_permutation, "date_disguise": date_disguise,
          "randomized_outcomes": randomized_outcomes, "feature_shuffle": feature_shuffle,
          "dead_feature_injection": dead_feature_injection, "duplicate_feature_injection": duplicate_feature_injection,
          "regime_split": regime_split, "walk_forward": walk_forward}


# ------------------------------------------------------------------ the battery
def run_battery(evaluator, X, y, now, seed=7, tests=ALL_TESTS, cfg=None, **kw):
    """Run the real evaluation, then every named test. Extra keyword arguments (n_rep, top_k, regime_col, ...) are
    passed to each test, which ignores those it does not use. A crashing test is recorded as status 'error'."""
    X, y = check_panel(X, y)
    unknown = [t for t in tests if t not in _TESTS]
    if unknown:
        raise ValueError(f"unknown tests: {unknown}")
    now = pd.Timestamp(now)
    real = _run(evaluator, X, y, now)
    seeds = np.random.SeedSequence(seed).spawn(len(_TESTS))
    report = {"seed": seed, "now": str(now.date()), "n_rows": int(len(X)), "n_features": int(X.shape[1]),
              "real": {"metric": real["metric"], "t": real.get("t")}, "verdicts": []}
    try:
        from engine.provenance import stamp
        report["stamp"] = stamp(cfg if cfg is not None else {"tests": list(tests), **{k: v for k, v in kw.items()
                                if isinstance(v, (int, float, str))}}, seed)
    except Exception as e:                                  # provenance must never hide the verdicts
        report["stamp"] = {"error": repr(e)}
    if not np.isfinite(real["metric"]):
        report["overall"] = "fail"
        report["reason"] = "the real run produced no metric (" + str(real.get("note", "unknown")) + ")"
        return report
    for name, ss in zip(_TESTS, seeds):
        if name not in tests:
            continue
        try:
            v = _TESTS[name](evaluator, X, y, now, real, np.random.default_rng(ss), **kw)
        except Exception as e:
            v = _verdict(name, "error", f"{type(e).__name__}: {e}")
        report["verdicts"].append(v)
    counts = pd.Series([v["status"] for v in report["verdicts"]]).value_counts().to_dict()
    report["counts"] = {k: int(n) for k, n in counts.items()}
    bad = [v["name"] for v in report["verdicts"] if v["status"] not in PASSING]
    report["failed"] = bad
    report["overall"] = "fail" if bad else "pass"
    return report


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(o)
    return o


def write_report(report, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(report), indent=1), encoding="utf-8")
    return path


def render_text(report):
    """Plain-English summary, one line per test."""
    lines = [f"Anti-overfitting battery: {report['overall'].upper()} (real metric {report['real']['metric']:.4f})"]
    if "reason" in report:
        lines.append("  " + report["reason"])
    for v in report["verdicts"]:
        lines.append(f"  [{v['status']:>15}] {v['name']}: {v['reason']}")
    return "\n".join(lines)
