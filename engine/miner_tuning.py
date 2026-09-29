"""A7 (canon C34-C37, blueprint: the Algorithm tunes itself): walk-forward self-tuning of the pattern miner's own
settings. The system - not a person - decides how fast memory fades (half-life), how strongly market similarity
weights the past (context bandwidth), how strict discovery is (FDR q, minimum support) and how hard effects shrink.

Honesty rules (anti-overfitting, Bible Phase 26/34):
  * rolling-origin walk-forward: every fold fits on the past and is scored on the next, never-seen block;
  * the last `holdout_frac` of history is never used while tuning; the chosen settings are judged there once, against
    the defaults, and the verdict is reported even when it is bad;
  * a change is adopted only if its gain over the incumbent is positive in >= `min_fold_share` of folds AND the
    bootstrap lower bound of the per-day IC difference (resampled by week) is above zero - "never learns for no reason";
  * coordinate search, one knob at a time, from the defaults: small steps, each separately justified.
Deterministic given the seed."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .patterns import PatternMiner

SPACE = {
    "half_life_years": [1.0, 2.0, 4.0, 8.0, 16.0],
    "ctx_bandwidth": [0.75, 1.5, 3.0],
    "fdr_q": [0.02, 0.05, 0.10],
    "min_n": [200, 300, 600],
    "shrink_k": [100, 400, 1600],
}


@dataclass
class FoldScore:
    fold: int
    train_end: str
    test_start: str
    test_end: str
    ic_mean: float
    ic_days: int
    daily_ic: pd.Series = field(repr=False, default=None)


@dataclass
class TuningResult:
    defaults: dict
    chosen: dict
    steps: list                      # one record per knob tried: values, fold gains, CI, adopted?
    holdout: dict                    # {"default_ic", "chosen_ic", "diff_lo", "diff_hi", "verdict"}
    adopted_any: bool

    def to_record(self):
        return {"defaults": self.defaults, "chosen": self.chosen, "steps": self.steps, "holdout": self.holdout,
                "adopted_any": self.adopted_any}


def daily_ic(score: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation for one day (NaN when a side is constant)."""
    a, b = score.rank(), y.rank()
    if a.std() == 0 or b.std() == 0 or len(a) < 5:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def folds(dates, n_folds=4, holdout_frac=0.2, min_train_frac=0.4):
    """Rolling-origin folds over the tuning part of history (the holdout is excluded).
    Returns (list of (train_end, test_start, test_end), holdout_start)."""
    ud = np.sort(pd.DatetimeIndex(dates).unique())
    hold = ud[int(len(ud) * (1 - holdout_frac))]
    tune = ud[ud < hold]
    first = int(len(tune) * min_train_frac)
    edges = np.linspace(first, len(tune), n_folds + 1).astype(int)
    out = []
    for k in range(n_folds):
        a, b = edges[k], edges[k + 1]
        if b - a < 2:
            continue
        out.append((tune[a - 1], tune[a], tune[b - 1]))
    return out, hold


def evaluate(params, X, y, train_end, test_start, test_end, base=None):
    """Fit on rows <= train_end, score every day in [test_start, test_end]; returns the per-day IC series."""
    d = X.index.get_level_values(0)
    tr = d <= train_end
    M = PatternMiner({**(base or {}), **params}).fit(X[tr], y[tr], now=train_end)
    te_days = np.sort(pd.DatetimeIndex(d[(d >= test_start) & (d <= test_end)]).unique())
    ics = {}
    for day in te_days:
        Xd = X.xs(day, level=0)
        yd = y.xs(day, level=0).reindex(Xd.index)
        ok = yd.notna()
        if ok.sum() < 5:
            continue
        ics[day] = daily_ic(M.score(Xd[ok]), yd[ok])
    return pd.Series(ics, dtype=float).dropna()


def weekly_bootstrap_diff(a: pd.Series, b: pd.Series, reps=500, seed=0, q=(0.05, 0.95)):
    """CI of mean(a - b) over days, resampling whole ISO weeks (days in a week share shocks)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if j.empty:
        return np.nan, np.nan
    diff = j["a"] - j["b"]
    wk = pd.DatetimeIndex(diff.index).to_period("W")
    groups = [g.values for _, g in diff.groupby(wk)]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    return float(np.quantile(means, q[0])), float(np.quantile(means, q[1]))


def tune(X, y, space=None, base=None, n_folds=4, holdout_frac=0.2, min_fold_share=0.75, seed=0, knobs=None,
         log=lambda *a: None) -> TuningResult:
    """Coordinate walk-forward search from the defaults. Returns the chosen settings and the untouched-holdout verdict."""
    space = space or SPACE
    base = dict(base or {})
    from .patterns import MINER_DEFAULT
    defaults = {k: base.get(k, MINER_DEFAULT.get(k)) for k in space}
    F, hold = folds(X.index.get_level_values(0), n_folds, holdout_frac)
    if not F:
        raise ValueError("not enough history for walk-forward folds")
    cache = {}

    def run(params):
        key = tuple(sorted(params.items()))
        if key not in cache:
            cache[key] = [evaluate(params, X, y, a, b, c, base) for a, b, c in F]
        return cache[key]

    current = dict(defaults)
    steps = []
    for knob in (knobs or list(space)):
        inc = run(current)
        best_val, best_rec = current[knob], None
        for v in space[knob]:
            if v == current[knob]:
                continue
            trial = {**current, knob: v}
            res = run(trial)
            gains = []
            for r_, i_ in zip(res, inc):
                dd = (r_ - i_).dropna()                           # days scored under both settings
                gains.append(float(dd.mean()) if len(dd) else np.nan)
            share = float(np.mean([g > 0 for g in gains if np.isfinite(g)])) if gains else 0.0
            lo, hi = weekly_bootstrap_diff(pd.concat(res), pd.concat(inc), seed=seed)
            rec = {"knob": knob, "value": v, "from": current[knob], "fold_gains": gains, "fold_share": share,
                   "diff_lo": lo, "diff_hi": hi,
                   "passes": bool(share >= min_fold_share and np.isfinite(lo) and lo > 0)}
            steps.append(rec)
            log(f"  {knob}={v}: share {share:.2f}, CI [{lo:+.4f}, {hi:+.4f}]{' PASS' if rec['passes'] else ''}")
            if rec["passes"] and (best_rec is None or lo > best_rec["diff_lo"]):
                best_val, best_rec = v, rec
        if best_rec is not None:
            current[knob] = best_val
            best_rec["adopted"] = True
    # the holdout: judged once, chosen vs defaults, never used above
    d = X.index.get_level_values(0)
    last = pd.DatetimeIndex(d).max()
    pre = pd.DatetimeIndex(np.sort(d.unique()))
    train_end = pre[pre < hold][-1]
    ic_def = evaluate(defaults, X, y, train_end, hold, last, base)
    ic_ch = ic_def if current == defaults else evaluate(current, X, y, train_end, hold, last, base)
    lo, hi = weekly_bootstrap_diff(ic_ch, ic_def, seed=seed + 1)
    if current == defaults:
        verdict = "no change adopted"
    elif np.isfinite(lo) and lo > 0:
        verdict = "confirmed on holdout"
    elif np.isfinite(hi) and hi < 0:
        verdict = "WORSE on holdout - revert to defaults"
    else:
        verdict = "not confirmed on holdout (CI straddles 0) - keep defaults"
    holdout = {"start": str(pd.Timestamp(hold).date()), "default_ic": float(ic_def.mean()) if len(ic_def) else np.nan,
               "chosen_ic": float(ic_ch.mean()) if len(ic_ch) else np.nan, "diff_lo": lo, "diff_hi": hi,
               "verdict": verdict}
    chosen = current if verdict == "confirmed on holdout" else dict(defaults)
    return TuningResult(defaults, chosen, steps, holdout, adopted_any=chosen != defaults)
