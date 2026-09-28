"""Production model training (Part C4) and the guarded weekly retrain (Part M step 6)."""
import json, pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config as K, data, features, model

MODELS = K.STATE / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def build_panel():
    stocks, market = data.load("stocks"), data.load("market")
    ev = pd.read_parquet(K.CACHE / "events.parquet")
    sic = pd.read_parquet(K.CACHE / "sic.parquet")
    ins = pd.read_parquet(K.CACHE / "insider.parquet") if (K.CACHE / "insider.parquet").exists() else None
    X, atr = features.build(stocks, market, ev, ins, sic)
    yb, fw = features.labels(stocks, atr)
    return X, yb.stack(future_stack=True).reindex(X.index), fw.stack(future_stack=True).reindex(X.index)


def _fit(X, y, f, end=None, sample_every=2):
    R = model.normalise(X)
    d = R.index.get_level_values(0)
    ud = np.array(sorted(d.unique()))
    end = end or ud[-1]
    lo = pd.Timestamp(end) - pd.DateOffset(years=model.TRAIN_YEARS)
    tr_dates = ud[(ud >= np.datetime64(lo)) & (ud <= np.datetime64(end))][::sample_every]
    ok = d.isin(tr_dates) & y.notna().values & f.notna().values
    return model.fit_models(R[ok], y[ok], f[ok]), R


def save(m, iso=None):
    for k in ("clf", "reg"):
        m[k].booster_.save_model(str(MODELS / f"{k}.txt"))
    (MODELS / "cols.json").write_text(json.dumps(m["cols"]))
    if iso is not None:
        (MODELS / "iso.pkl").write_bytes(pickle.dumps(iso))


def initial_train():
    """First production fit: all data; isotonic from the research run's OOS predictions."""
    X, y, f = build_panel()
    m, _ = _fit(X, y, f)
    iso = None
    oos = K.CACHE / "oos_preds.parquet"
    if oos.exists():
        P = pd.read_parquet(oos)
        yy = (y.reindex(P.index) == 1).astype(float)
        ok = y.reindex(P.index).notna()
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(P.loc[ok, "p_target_raw"], yy[ok])
    save(m, iso)
    meta_p = MODELS / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"version": "1.0", "w_model": 0.5, "w_options": 0.15}
    meta["trained_through"] = str(X.index.get_level_values(0).max().date())
    meta_p.write_text(json.dumps(meta, indent=1))


def retrain_guarded(holdout=60):
    """Fit a candidate on data ending before a recent holdout, compare its holdout IC with the
    live champion's; only if it holds up is the model refit on everything and swapped in."""
    import lightgbm as lgb
    data.update("stocks"); data.update("market")
    X, y, f = build_panel()
    ud = np.array(sorted(X.index.get_level_values(0).unique()))
    labelled = ud[: -6]
    ho = labelled[-holdout:]
    cut = labelled[-holdout - model.EMBARGO]
    cand, R = _fit(X, y, f, end=cut)
    mask = R.index.get_level_values(0).isin(ho) & f.notna().values
    Rh, fh = R[mask], f[mask]
    new_ic = model.daily_ic(pd.Series(cand["reg"].predict(Rh[cand["cols"]]), index=Rh.index), fh).mean()
    old = lgb.Booster(model_file=str(MODELS / "reg.txt"))
    cols = json.loads((MODELS / "cols.json").read_text())
    old_ic = model.daily_ic(pd.Series(old.predict(Rh.reindex(columns=cols)), index=Rh.index), fh).mean()
    from .improve import log_experiment
    rec = {"event": "retrain_check", "new_holdout_ic": float(new_ic), "old_holdout_ic": float(old_ic)}
    # the old model saw the holdout in training, so it is favoured; a small tolerance keeps this fair
    if new_ic >= old_ic - 0.01:
        m, _ = _fit(X, y, f)
        save(m)
        rec["decision"] = "swapped"
    else:
        rec["decision"] = "kept old"
    log_experiment(rec)
    return f"Retrain: new holdout IC {new_ic:+.4f} vs current {old_ic:+.4f} -> {rec['decision']}."


if __name__ == "__main__":
    initial_train()
