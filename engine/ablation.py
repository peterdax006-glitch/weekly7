"""Bible Phase 34 (required ablation framework); serves the incremental-out-of-sample-value canon.

Two layers.
  1. `ablate`: generic remove-one / add-one ablation of named components. The caller supplies
     evaluate(active_components) -> per-period out-of-sample score (pd.Series indexed by period). Every comparison is
     PAIRED on the periods both runs scored and carries a moving-block bootstrap confidence interval, because periods
     are autocorrelated (overlapping labels) and an i.i.d. interval would be too narrow.
  2. `feature_ablation`: the six arms the Bible demands for a new Algorithm feature -
        baseline | baseline + feature | feature alone | baseline + randomized feature |
        baseline + shuffled feature | baseline + future-scrambled feature
     - and a deployment verdict: the feature earns consideration only if baseline+feature beats baseline with a CI above
     zero AND that gain beats what a random column and a shuffled copy of the feature achieve, AND (where applicable)
     scrambling the future does not move earlier decisions and does not keep the gain.
Deterministic in `seed`. The default evaluator is engine.antioverfit.wf_evaluate (walk-forward, so scores are OOS)."""
import numpy as np
import pandas as pd

from engine import antioverfit as ao


# ------------------------------------------------------------------ statistics
def default_block(n):
    return max(1, int(round(n ** (1 / 3))))


def paired_ci(a, b, rng, n_boot=2000, block=None, alpha=0.05, min_periods=10):
    """Paired difference a-b over shared periods with a circular moving-block bootstrap CI.
    Returns dict(mean, lo, hi, p_le0, n, block); `insufficient` is True (and stats NaN) below `min_periods`."""
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    d = (a - b).dropna()
    n = len(d)
    nan = float("nan")
    if n < min_periods:
        return {"mean": nan, "lo": nan, "hi": nan, "p_le0": nan, "n": int(n), "block": 0, "insufficient": True}
    v = d.to_numpy()
    blk = min(default_block(n) if block is None else int(block), n)
    nb = int(np.ceil(n / blk))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(blk)[None, None, :]) % n
    means = v[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "p_le0": float((means <= 0).mean()),
            "n": int(n), "block": int(blk), "insufficient": False}


def _series(res):
    """Accept a Series, or an evaluator dict carrying 'ic_series' (or 'series')."""
    if isinstance(res, pd.Series):
        return res.dropna()
    if isinstance(res, dict):
        for k in ("ic_series", "series"):
            if k in res:
                return pd.Series(res[k]).dropna()
    raise ValueError("evaluator must return a per-period pd.Series or a dict with 'ic_series'")


def _call(evaluate, *a):
    s = _series(evaluate(*a))
    if not s.index.is_unique:
        raise ValueError("per-period scores must have a unique index")
    return s


def _judge(ci, min_gain=0.0):
    """earns_keep / harmful / no_effect / insufficient from a paired CI of (with - without)."""
    if ci["insufficient"]:
        return "insufficient"
    if ci["lo"] > min_gain:
        return "earns_keep"
    if ci["hi"] < 0:
        return "harmful"
    return "no_effect"


# ------------------------------------------------------------------ layer 1: generic component ablation
def ablate(evaluate, components, seed=7, mode="remove", baseline=(), n_boot=2000, block=None, min_gain=0.0):
    """Ablate each component.
    mode='remove': full set vs full set minus X - a component 'earns_keep' if removing it lowers the score (CI > 0).
    mode='add'   : `baseline` vs baseline + X, for components not already in the baseline.
    `evaluate(active: tuple)` must be deterministic; results are cached per active set."""
    comps = list(components)
    if not comps:
        raise ValueError("no components to ablate")
    if len(set(comps)) != len(comps):
        raise ValueError("duplicate component names")
    cache = {}

    def score(active):
        key = frozenset(active)
        if key not in cache:
            cache[key] = _call(evaluate, tuple(c for c in comps if c in key) if mode == "remove" else tuple(sorted(key, key=str)))
        return cache[key]

    base_set = tuple(comps) if mode == "remove" else tuple(baseline)
    if mode not in ("remove", "add"):
        raise ValueError("mode must be 'remove' or 'add'")
    ref = score(base_set)
    ss = np.random.SeedSequence(seed).spawn(len(comps))
    rows = []
    for c, s in zip(comps, ss):
        if mode == "remove":
            with_, without = ref, score(tuple(x for x in comps if x != c))
        else:
            if c in base_set:
                continue
            with_, without = score(tuple(base_set) + (c,)), ref
        ci = paired_ci(with_, without, np.random.default_rng(s), n_boot=n_boot, block=block)
        rows.append({"component": c, "verdict": _judge(ci, min_gain), "delta": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
                     "p_le0": ci["p_le0"], "n_periods": ci["n"], "score_with": float(with_.mean()),
                     "score_without": float(without.mean())})
    table = pd.DataFrame(rows)
    return {"mode": mode, "seed": seed, "reference_score": float(ref.mean()), "n_ref_periods": int(len(ref)),
            "table": table, "keep": [r["component"] for r in rows if r["verdict"] == "earns_keep"],
            "drop": [r["component"] for r in rows if r["verdict"] in ("no_effect", "harmful", "insufficient")]}


# ------------------------------------------------------------------ layer 2: the six-arm feature ablation
def _with_column(X, name, values):
    Z = X.copy()
    Z[name] = values
    return Z


def _random_col(X, rng):
    return rng.standard_normal(len(X))


def feature_ablation(X, y, now, feature, baseline_cols=None, evaluate=None, seed=7, n_rep=5, n_boot=2000, block=None,
                     min_gain=0.0, future_applicable=True, cut_frac=0.6):
    """Bible Phase 34. `feature` is a column of X (or a Series on X's index, then it is added under name 'candidate').
    `baseline_cols` defaults to every other non-context column; context (m_*) columns are always kept.
    `evaluate(X, y, now)` returns a per-period OOS score (Series or dict with ic_series); default = walk-forward ridge IC."""
    X, y = ao.check_panel(X, y)
    if isinstance(feature, pd.Series):
        if not feature.index.equals(X.index) and not feature.sort_index().index.equals(X.index):
            raise ValueError("feature Series must share X's index")
        X = _with_column(X, "candidate", feature.reindex(X.index).values)
        feature = "candidate"
    if feature not in X.columns:
        raise ValueError(f"feature {feature!r} not in X")
    ctx = [c for c in X.columns if str(c).startswith("m_")]
    base = [c for c in (baseline_cols if baseline_cols is not None else ao.feature_cols(X)) if c != feature]
    if not base:
        raise ValueError("baseline needs at least one column besides the feature")
    ev = evaluate or (lambda A, b, t: ao.wf_evaluate(A, b, t))
    now = pd.Timestamp(now)
    root = np.random.SeedSequence(seed)
    s_ci, s_rand, s_shuf, s_fut, s_leak = [np.random.default_rng(s) for s in root.spawn(5)]
    dates = X.index.get_level_values(0)
    Xb, Xf = X[base + ctx], X[[feature] + ctx]
    arms = {"baseline": _call(ev, Xb, y, now),
            "baseline_plus_feature": _call(ev, X[base + [feature] + ctx], y, now),
            "feature_alone": _call(ev, Xf, y, now)}

    def mean_arm(make):
        ser = [_call(ev, make(), y, now) for _ in range(n_rep)]
        return pd.concat(ser, axis=1).mean(axis=1)                       # per-period mean over reps

    arms["baseline_plus_random"] = mean_arm(lambda: _with_column(Xb, "random_col", _random_col(X, s_rand)))
    arms["baseline_plus_shuffled"] = mean_arm(lambda: _with_column(
        Xb, feature, ao.permute_within_date(X[feature].values, dates, s_shuf)))
    if future_applicable:
        ud = np.sort(dates.unique())
        cut = pd.Timestamp(ud[int(len(ud) * cut_frac)])
        late = np.flatnonzero(np.asarray(dates > cut))

        def fut():
            v = X[feature].values.copy()
            v[late] = v[late[s_fut.permutation(len(late))]]
            return _with_column(Xb, feature, v)
        arms["baseline_plus_future_scrambled"] = mean_arm(fut)
    else:
        cut = None
    b, f = arms["baseline"], arms["baseline_plus_feature"]
    mk = lambda: np.random.default_rng(s_ci.integers(0, 2 ** 32))
    inc = paired_ci(f, b, mk(), n_boot, block)
    vs_rand = paired_ci(f, arms["baseline_plus_random"], mk(), n_boot, block)
    vs_shuf = paired_ci(f, arms["baseline_plus_shuffled"], mk(), n_boot, block)
    checks = {"incremental": inc, "vs_random": vs_rand, "vs_shuffled": vs_shuf}
    reasons = []
    ok = True
    if inc["insufficient"]:
        ok = False; reasons.append("too few scored periods for a confidence interval")
    elif inc["lo"] <= min_gain:
        ok = False; reasons.append(f"baseline+feature does not beat baseline (gain {inc['mean']:+.4f}, CI lower {inc['lo']:+.4f})")
    for nm, ci, arm in (("random column", vs_rand, "baseline_plus_random"), ("shuffled feature", vs_shuf, "baseline_plus_shuffled")):
        if not ci["insufficient"] and ci["lo"] <= 0:
            ok = False; reasons.append(f"gain is not distinguishable from a {nm} (CI lower {ci['lo']:+.4f})")
    if future_applicable:
        fs = arms["baseline_plus_future_scrambled"]
        vs_fut = paired_ci(f.loc[f.index > cut], fs.loc[fs.index > cut], mk(), n_boot, block)
        checks["vs_future_scrambled_zone"] = vs_fut
        if not vs_fut["insufficient"] and vs_fut["lo"] <= 0:
            ok = False; reasons.append("the real feature does not beat its own future-scrambled copy in the scrambled zone")
        leak_ev = _leak_evaluator(ev, base + [feature] + ctx)
        leak = ao.future_scramble(leak_ev, X, y, now, {"metric": 0.0}, s_leak, cutoff=cut)
        checks["future_leak_check"] = {"status": leak["status"], "reason": leak["reason"]}
        if leak["status"] == "fail":
            ok = False; reasons.append("look-ahead: " + leak["reason"])
    alone = arms["feature_alone"]
    notes = []
    if len(alone) and alone.mean() <= 0 and inc["mean"] > 0:
        notes.append("feature alone has no OOS signal; any gain is interaction with the baseline")
    return {"seed": seed, "feature": feature, "baseline_cols": base, "n_rep": n_rep,
            "arm_means": {k: float(v.mean()) if len(v) else float("nan") for k, v in arms.items()},
            "arm_periods": {k: int(len(v)) for k, v in arms.items()}, "checks": checks,
            "deploy": bool(ok), "verdict": "earns_deployment_consideration" if ok else "rejected",
            "reasons": reasons, "notes": notes}


def _leak_evaluator(ev, cols):
    """Wrap a per-period evaluator into the battery's contract so ao.future_scramble can audit it: metric = mean score,
    predictions = the per-period scores themselves (dated), compared exactly on dates <= cutoff."""
    def run(X, y, now):
        s = _call(ev, X[cols], y, now)
        idx = pd.MultiIndex.from_arrays([s.index, ["_"] * len(s)])
        return {"metric": float(s.mean()) if len(s) else float("nan"), "predictions": pd.Series(s.values, index=idx)}
    return run


def render_text(rep):
    lines = [f"Feature ablation '{rep['feature']}': {rep['verdict'].upper()}"]
    for k, v in rep["arm_means"].items():
        lines.append(f"  {k:<34}{v:+.4f}  ({rep['arm_periods'][k]} periods)")
    for r in rep["reasons"]:
        lines.append("  - " + r)
    for n in rep["notes"]:
        lines.append("  * " + n)
    return "\n".join(lines)
