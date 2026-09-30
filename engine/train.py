"""Production model training (Part C4) and the guarded weekly retrain (Part M step 6).

Bible Phase 1.1 / 1.3 (INTEGRATION B01):
  * `_fit` takes its training rows from pit.purged_training_set: a row is used only when its forward label had CLOSED by the
    training cut (decide at close t, label closes at t + LABEL_HORIZON sessions). The old ad-hoc cut kept every row dated up
    to `end`, so the last LABEL_HORIZON sessions trained on returns realised after the cut (inside the holdout).
  * `future_scramble_gate` runs pit.future_scramble over the real LightGBM fit (engine.scramble_audit adds the PatternMiner,
    Memory and the adaptive Session, kept out of this trader-reachable module) and fails closed when anything they output at `as_of` changes when the data after `as_of`
    (including labels that close after it) is removed or scrambled. retrain_guarded runs it on a seeded ticker sample
    before it may swap a model in."""
import json, pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config as K, data, features, model, pit

MODELS = K.STATE / "models"
MODELS.mkdir(parents=True, exist_ok=True)
LABEL_HORIZON = 5            # features.labels(horizon=5): decide at close t, the label closes at close t+5


def build_panel():
    stocks, market = data.load("stocks"), data.load("market")
    ev = pd.read_parquet(K.CACHE / "events.parquet")
    sic = pd.read_parquet(K.CACHE / "sic.parquet")
    ins = pd.read_parquet(K.CACHE / "insider.parquet") if (K.CACHE / "insider.parquet").exists() else None
    X, atr = features.build(stocks, market, ev, ins, sic)
    yb, fw = features.labels(stocks, atr)
    return X, yb.stack(future_stack=True).reindex(X.index), fw.stack(future_stack=True).reindex(X.index)


def training_rows(R, y, f, end, sample_every=2, horizon=LABEL_HORIZON, calendar=None):
    """(R_train, y_train, f_train, info): the TRAIN_YEARS window ending at `end`, thinned to every `sample_every`-th date,
    then purged by pit so no row's label closes after `end` (pit verifies that before returning)."""
    d = R.index.get_level_values(0)
    ud = np.array(sorted(d.unique()))
    lo = pd.Timestamp(end) - pd.DateOffset(years=model.TRAIN_YEARS)
    tr_dates = ud[(ud >= np.datetime64(lo)) & (ud <= np.datetime64(pd.Timestamp(end)))][::sample_every]
    ok = np.asarray(d.isin(tr_dates) & y.notna().values & f.notna().values)
    cal = calendar or pit.Calendar(pd.DatetimeIndex(ud))
    Rp, yp = pit.purged_training_set(R[ok], y[ok], end, horizon, cal)
    info = {"window_rows": int(ok.sum()), "purged_rows": int(ok.sum() - len(Rp)), "train_rows": int(len(Rp)),
            "last_row_date": str(Rp.index.get_level_values(0).max().date()) if len(Rp) else None, "cut": str(pd.Timestamp(end).date())}
    return Rp, yp, f.reindex(Rp.index), info


def _fit(X, y, f, end=None, sample_every=2, fast=False, horizon=LABEL_HORIZON):
    R = model.normalise(X)
    ud = np.array(sorted(R.index.get_level_values(0).unique()))
    end = end if end is not None else ud[-1]
    Rt, yt, ft, info = training_rows(R, y, f, end, sample_every, horizon)
    _fit.last_rows = info
    return model.fit_models(Rt, yt, ft, fast=fast), R


# ==================================================================================================================
# Phase 1.3 future scramble over the real pipeline components
# ==================================================================================================================
def label_future_mask(calendar, horizon=LABEL_HORIZON):
    """mask_fn for a (date, ticker) label frame: a row is 'future' when its label closes after as_of."""
    def mask(df, as_of):
        lc = pit.label_close_dates(df.index.get_level_values(0), horizon, calendar)
        return np.asarray(lc > pd.Timestamp(as_of))
    return mask


def _model_outputs(fr, as_of, horizon):
    X, L = fr["X"], fr["L"].reindex(fr["X"].index)
    m, R = _fit(X, L["y"], L["f"], end=as_of, fast=True, horizon=horizon)
    Rd = R.xs(pd.Timestamp(as_of), level=0, drop_level=False)
    return {"pred": pd.Series(m["reg"].predict(Rd[m["cols"]]), index=Rd.index).round(12), "rows": _fit.last_rows["train_rows"]}


def scramble_pipeline(components=("model",), horizon=LABEL_HORIZON):
    """pipeline(frames, as_of) -> {"model": outputs at as_of}, for pit.future_scramble. frames: X ((date, ticker) features) and
    L (labels y, f on the same index). Only the model component lives here: this module is on the trader's import closure
    (improve -> retrain_guarded), so the PatternMiner / Memory / adaptive probes live in engine.scramble_audit (F06 found the
    miner import turned leak channel 8d to LEAK)."""
    bad = set(components) - {"model"}
    if bad or not components:
        raise ValueError(f"train.scramble_pipeline covers only the model; use engine.scramble_audit for {sorted(bad) or 'none'}")

    def pipeline(fr, as_of):
        return {"model": _model_outputs(fr, as_of, horizon)}
    return pipeline


def future_scramble_gate(frames, as_of, components=("model",), seed=0, horizon=LABEL_HORIZON, require=True, pipeline=None):
    """pit.future_scramble over the real components (the model here; engine.scramble_audit passes a pipeline covering the
    miner, Memory and the adaptive Session). The label frame's future is every row whose label closes after as_of (not only
    rows dated after it). Returns the ScrambleReport; with require=True a failure raises pit.FailClosed."""
    ud = pd.DatetimeIndex(sorted(frames["X"].index.get_level_values(0).unique()))
    masks = {"L": label_future_mask(pit.Calendar(ud), horizon)}
    pipe = pipeline if pipeline is not None else scramble_pipeline(components, horizon)
    rep = pit.future_scramble(pipe, frames, as_of, seed=seed, mask_fns=masks)
    if require:
        rep.require()
    return rep


def scramble_sample(X, y, f, as_of, n_tickers=60, years=3, seed=0):
    """A seeded ticker sample of the last `years` before as_of (plus the label tail after it) for the pre-swap gate:
    the whole panel is too large to fit three times (CONTEXT rule 10)."""
    tick = np.array(sorted(X.index.get_level_values(1).unique()))
    pick = np.random.default_rng(seed).choice(tick, min(n_tickers, len(tick)), replace=False)
    d, t = X.index.get_level_values(0), X.index.get_level_values(1)
    lo = pd.Timestamp(as_of) - pd.DateOffset(years=years)
    m = np.asarray(t.isin(pick) & (d >= lo))
    return {"X": X[m], "L": pd.DataFrame({"y": y[m], "f": f[m]})}


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
    rec = {"event": "retrain_check", "new_holdout_ic": float(new_ic), "old_holdout_ic": float(old_ic),
           "purge": dict(_fit.last_rows)}
    # B01: the candidate's fit must not see anything after its cut (fail closed: a failed or crashed gate keeps the old model)
    try:
        scr = future_scramble_gate(scramble_sample(X, y, f, cut), cut, components=("model",), require=False)
        scramble_ok, rec["future_scramble"] = scr.passed, scr.summary()
    except Exception as e:                                 # noqa: BLE001 - an unprovable fit is not swapped in
        scramble_ok, rec["future_scramble"] = False, f"error {type(e).__name__}: {e}"
    # the old model saw the holdout in training, so it is favoured; a small tolerance keeps this fair
    if new_ic >= old_ic - 0.01 and scramble_ok:
        m, _ = _fit(X, y, f)
        save(m)
        rec["decision"] = "swapped"
    else:
        rec["decision"] = "kept old"
    # Phase 0.2: the champion's training record carries its exact model parameters, seed, ranges and decision
    log_experiment(rec, cfg=model._params(), seed=7,
                   train_range=f"..{pd.Timestamp(cut).date()}", validation_range=f"{pd.Timestamp(ho[0]).date()}..{pd.Timestamp(ho[-1]).date()}",
                   test_range="live (forward)", window_ids=["live"],
                   metrics={"new_holdout_ic": float(new_ic), "old_holdout_ic": float(old_ic)},
                   gates={"holdout_ic_within_0.01": bool(new_ic >= old_ic - 0.01), "future_scramble": bool(scramble_ok)},
                   outcome="adopt" if rec["decision"] == "swapped" else "reject",
                   reason=f"holdout IC {new_ic:+.4f} vs champion {old_ic:+.4f} (tolerance 0.01; champion saw the holdout); "
                          f"future scramble {'PASS' if scramble_ok else 'FAILED - kept old'}")
    return f"Retrain: new holdout IC {new_ic:+.4f} vs current {old_ic:+.4f} -> {rec['decision']}."


if __name__ == "__main__":
    initial_train()
