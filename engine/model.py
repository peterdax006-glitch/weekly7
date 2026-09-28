"""Learning layer (Blueprint Part C): evidence composite + LightGBM ranker + triple-barrier
classifier, trained walk-forward with a purge/embargo gap, isotonic-calibrated.

Out-of-sample only: every prediction for year Y comes from models fit on data that ended
at least EMBARGO sessions before Y began."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from .config import STATE

EMBARGO = 10
TRAIN_YEARS = 6
REGIME_COLS_PREFIX = "m_"

# Evidence prior (Part B / C2 layer 4): sign and weight from the literature, never fitted.
EVIDENCE = {
    "ear": 1.0, "ins_buyers30": 0.8, "ins_officer30": 0.4, "ev_activist": 0.8,
    "dist_52wh": 0.7, "ind_mom60": 0.6, "frog": 0.4, "mom_12_1": 0.3,
    "r5_nonews": -0.5,          # vol_surge5 and overnight20 removed: failed pre-2017 sign check (registry)
    "max20": -0.8, "ev_offering": -1.0, "ev_shelf": -0.3, "ev_red_flag": -1.0, "skew60": -0.2,
}
MONOTONE = {"max20": -1, "ev_offering": -1, "ev_red_flag": -1, "ins_buyers30": 1, "ear": 1}


def normalise(X: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per date for stock features; regime features untouched."""
    stock_cols = [c for c in X.columns if not c.startswith(REGIME_COLS_PREFIX)]
    R = X[stock_cols].groupby(level=0).rank(pct=True).astype("float32") - 0.5
    for c in X.columns:
        if c.startswith(REGIME_COLS_PREFIX):
            R[c] = X[c]
    return R


def evidence_score(R: pd.DataFrame, weights=None) -> pd.Series:
    weights = weights or EVIDENCE
    s = sum(w * R[c].fillna(0) for c, w in weights.items() if c in R)
    return s.groupby(level=0).rank(pct=True).rename("evidence")


def _params(seed=7):
    return dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=400,
                subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                random_state=seed, verbose=-1, n_jobs=-1)


def _mono(cols):
    return [MONOTONE.get(c, 0) for c in cols]


def fit_models(R, y_bar, fwd):
    cols = list(R.columns)
    # the LambdaRank ranker scored backwards out of sample (IC -0.0095, t=-2.8): retired, see registry
    ranker = None
    clf = lgb.LGBMClassifier(objective="multiclass", monotone_constraints=_mono(cols), **_params(11))
    clf.fit(R, (y_bar + 1).astype(int))             # classes 0=stop, 1=neither, 2=target
    reg = lgb.LGBMRegressor(objective="huber", alpha=0.05, monotone_constraints=_mono(cols), **_params(13))
    ex = fwd - fwd.groupby(level=0).transform("mean")      # market-relative: that's how the score uses it
    reg.fit(R, ex.clip(-0.4, 0.4))
    return {"ranker": ranker, "clf": clf, "reg": reg, "cols": cols}


def predict(models, R):
    R = R[models["cols"]]
    out = pd.DataFrame(index=R.index)
    out["rank_raw"] = np.nan
    p = models["clf"].predict_proba(R)
    out["p_stop"], out["p_target_raw"] = p[:, 0], p[:, 2]
    out["mu_raw"] = models["reg"].predict(R)
    return out


def walk_forward(X, y_bar, fwd, first_year=2017, sample_every=2, log=print):
    """Yearly refits; returns OOS predictions for every row from first_year on."""
    R = normalise(X)
    dates = R.index.get_level_values(0)
    udates = pd.DatetimeIndex(sorted(dates.unique()))
    preds, calib_pool = [], []
    iso = None
    for year in range(first_year, udates[-1].year + 1):
        test_mask = dates.year == year
        if not test_mask.any():
            continue
        i0 = udates.searchsorted(pd.Timestamp(f"{year}-01-01"))
        cut = udates[max(0, i0 - EMBARGO)]
        lo = pd.Timestamp(f"{year - TRAIN_YEARS}-01-01")
        tr_dates = udates[(udates >= lo) & (udates < cut)][::sample_every]
        tr = dates.isin(tr_dates) & y_bar.notna().values & fwd.notna().values
        log(f"  fold {year}: train {tr.sum():,} rows, test {test_mask.sum():,}")
        m = fit_models(R[tr], y_bar[tr], fwd[tr])
        P = predict(m, R[test_mask])
        # calibrate with isotonic fitted on previous folds' OOS output only
        P["p_target"] = iso.predict(P["p_target_raw"]) if iso is not None else P["p_target_raw"]
        preds.append(P)
        hit = (y_bar[test_mask] == 1).astype(float)
        ok = hit.notna() & y_bar[test_mask].notna()
        calib_pool.append(pd.DataFrame({"p": P["p_target_raw"][ok], "y": hit[ok]}))
        pool = pd.concat(calib_pool)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pool["p"], pool["y"])
    P = pd.concat(preds)
    P["evidence"] = evidence_score(R.loc[P.index])
    return P, m, iso


def blend(P: pd.DataFrame, w_model=0.5) -> pd.Series:
    g = P.groupby(level=0)
    model_rank = (g["rank_raw"].rank(pct=True) + g["mu_raw"].rank(pct=True) + g["p_target"].rank(pct=True)) / 3
    return (w_model * model_rank + (1 - w_model) * P["evidence"]).rename("score")


def daily_ic(score: pd.Series, fwd: pd.Series) -> pd.Series:
    df = pd.DataFrame({"s": score, "f": fwd.reindex(score.index)}).dropna()
    return df.groupby(level=0).apply(lambda d: d["s"].rank().corr(d["f"].rank()))
