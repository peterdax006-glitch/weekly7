"""Direction research on PREDICTED movers (Bible checklist V2/V5; canons C23, C24, C56).

Question: given a stock the system expects to move +-10% next week, is there ANY signal for which side, and how does
accuracy trade against coverage?  B07 scored realised movers with stand-in inputs; this module scores the picks the mover
model would really have made, with the literature-backed inputs the system actually computes:

  * pead          post-earnings-announcement drift (earnings reaction, volume surge, days since the release)
  * insider       opportunistic insider-buying clusters
  * reversal_mom  short-term reversal vs 12-1 momentum vs 52-week-high distance (news-conditioned reversal, Chan 2003)
  * event_regime  event type of the setup (earnings / filing / none) and market regime
  * miner         the pattern miner's signed score, trained past-only
  * combined / blend   all inputs at once (GBM) and an average of every model's logit

Point-in-time protocol (C56).  Test years Y are evaluated in walk-forward blocks:
    train  = rows dated before (start of Y-1) minus an embargo, labels closed before the calibration block begins;
    calib  = the picks of year Y-1 (Platt calibration AND the confidence thresholds are taken from here);
    test   = the picks of year Y, never seen by anything.
A bet at target coverage c is placed when the row's calibrated confidence |p-0.5| is at least the (1-c) quantile of the
CALIBRATION block, so no threshold is ever tuned on the block it is judged on.  Confidence intervals resample WEEKS
(rows of one week share the market).  Controls (random scores, a model fitted on shuffled labels) must read ~50%, a planted
signal must be found at the coverage where it lives, and a look-ahead canary must be caught by `audit_point_in_time`.

Honest limits are reported, not hidden: the price panel is survivor-only, the number of (model, segment, coverage) cells
tried is counted, and the 80% question is answered with a lower bound corrected for that count (`eighty_question`)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from .direction import brier, ece, ece_noise_floor, reliability_bins, wilson

MOVE, HORIZON = 0.10, 5
COVERAGES = (0.01, 0.05, 0.10, 0.25, 1.0)
GRID = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0)
LABEL_SPAN_DAYS = 9          # a label decided at a close is known 5 sessions later: at most 9 calendar days
CONTROLS = ("ctl_", "planted", "leak")               # random / shuffled controls, planted signal, look-ahead canary
NON_SIGNAL = CONTROLS + ("blend",)                    # none of these enters the blend

FAMILIES = {
    "pead": ["ear", "ear_recent", "ear_sign_recent", "ear_x_vol", "ear_volsurge", "days_since_earn", "days_to_earn",
             "earn_in_week", "gap_today", "overnight20"],
    "insider": ["ins_buyers30", "ins_value30", "ins_officer30", "ins_opportunistic30"],
    "reversal_mom": ["r1", "r5", "r20", "r60", "r120", "mom_12_1", "dist_52wh", "dist_ma50", "dist_ma200", "r5_nonews",
                     "r5_news", "rel_ind20", "rel_ind60", "frog", "close_loc", "intraday20", "rev5_news_adj"],
    "event_regime": ["news5", "ev_offering", "ev_shelf", "ev_activist", "ev_activist_amend", "ev_agreement", "ev_red_flag",
                     "evt_earn", "evt_filing", "m_spy_ma50", "m_spy_ma200", "m_spy_r5", "m_vix", "m_vix_chg5",
                     "m_vix_term", "m_breadth", "m_dispersion"],
}
FAMILIES["combined"] = sorted({c for v in FAMILIES.values() for c in v})
_EV = ["ev_offering", "ev_shelf", "ev_activist", "ev_activist_amend", "ev_agreement", "ev_red_flag"]


# ---- labels ------------------------------------------------------------------------------------------------------
def week_end_positions(sessions: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the last session of each ISO week: a decision is taken at that close. The final session of the data
    is never a week end (its week is incomplete)."""
    iso = sessions.isocalendar()
    key = iso["year"].to_numpy() * 100 + iso["week"].to_numpy()
    return np.flatnonzero(key[1:] != key[:-1])


def weekly_labels(O, H, L, C, dec, thr=MOVE, horizon=HORIZON) -> dict:
    """First-touch labels for decisions at the close of session `dec` (arrays sessions x tickers).
    Entry = the NEXT session's open (canon C33); the window is sessions dec+1..dec+horizon, nothing earlier or later.
    mover = a high reached entry*(1+thr) or a low reached entry*(1-thr); up_first = the up barrier came on a strictly
    earlier session than the down barrier (NaN for non-movers and for same-session double touches, counted as `amb`);
    fwd = close of the last session / entry - 1. A missing bar anywhere in the window invalidates the row."""
    O, H, L, C = (np.asarray(a, dtype=np.float64) for a in (O, H, L, C))
    dec = np.asarray(dec, dtype=int)
    dec = dec[(dec >= 0) & (dec + horizon < len(O))]
    entry = O[dec + 1]
    ok = np.isfinite(entry) & (entry > 0)
    up_day = np.full(entry.shape, np.inf)
    dn_day = np.full(entry.shape, np.inf)
    for d in range(horizon, 0, -1):                     # walk backwards so the earliest touching session wins
        h, l = H[dec + d], L[dec + d]
        ok &= np.isfinite(h) & np.isfinite(l)
        up_day = np.where(h >= entry * (1 + thr), d, up_day)
        dn_day = np.where(l <= entry * (1 - thr), d, dn_day)
    end = C[dec + horizon]
    ok &= np.isfinite(end)
    fwd = end / np.where(ok, entry, np.nan) - 1
    mover = ok & (np.isfinite(up_day) | np.isfinite(dn_day))
    amb = mover & (up_day == dn_day)
    up_first = np.where(mover & ~amb, (up_day < dn_day).astype(float), np.nan)
    return dict(dec=dec, ok=ok, entry=np.where(ok, entry, np.nan), fwd=np.where(ok, fwd, np.nan), mover=mover,
                amb=amb, up_first=up_first, up_sign=np.where(ok, (fwd > 0).astype(float), np.nan))


def labels_frame(sessions, tickers, lab: dict) -> pd.DataFrame:
    """Long (date, ticker) frame of the valid rows of `weekly_labels`, with the entry and window-end sessions attached
    so the point-in-time audit can check every label starts after its decision."""
    dec = lab["dec"]
    dates = sessions[dec]
    ent = sessions[dec + 1]
    end = sessions[np.minimum(dec + HORIZON, len(sessions) - 1)]
    keep = lab["ok"]
    ii, jj = np.nonzero(keep)
    idx = pd.MultiIndex.from_arrays([dates[ii], np.asarray(tickers)[jj]], names=["date", "ticker"])
    return pd.DataFrame({"entry_date": ent[ii], "end_date": end[ii], "fwd": lab["fwd"][keep].astype("float32"),
                         "mover": lab["mover"][keep], "amb": lab["amb"][keep], "up_first": lab["up_first"][keep],
                         "up_sign": lab["up_sign"][keep]}, index=idx)


# ---- features and segments ---------------------------------------------------------------------------------------
def event_type(X: pd.DataFrame) -> pd.Series:
    """Setup type known at the decision close: 'earn' = a release in the last 5 sessions or one expected inside the coming
    week (quarterly cadence); 'filing' = otherwise a material filing in the recent window; 'none'. Filings that will
    arrive DURING the week are unknowable and are not used."""
    earn = ((X["days_since_earn"] <= 5) | (X["earn_in_week"] > 0)).fillna(False).to_numpy()
    ev = X[[c for c in _EV if c in X]].max(axis=1).fillna(0) if any(c in X for c in _EV) else 0.0
    filing = (~earn) & (((X["news5"].fillna(0) > 0) | (ev > 0)).to_numpy())
    return pd.Series(np.where(earn, "earn", np.where(filing, "filing", "none")), index=X.index, name="seg")


def regime_type(X: pd.DataFrame) -> pd.Series:
    """Bull/bear from the market's position against its 200-day average (a value known at the close); 'na' if missing."""
    v = X["m_spy_ma200"]
    return pd.Series(np.where(v.isna(), "na", np.where(v > 0, "bull", "bear")), index=X.index, name="reg")


def add_derived(X: pd.DataFrame) -> pd.DataFrame:
    """Signed, interpretable versions of the literature inputs. The earnings reaction only matters while the release is
    recent (the panel forward-fills it 60 sessions), so `ear_recent` keeps it for 20 sessions and zeroes it after."""
    d = X.copy()
    recent = (X["days_since_earn"] <= 20) & X["ear"].notna()
    d["ear_recent"] = X["ear"].where(recent, 0.0)
    d["ear_sign_recent"] = np.sign(d["ear_recent"])
    d["ear_x_vol"] = d["ear_recent"] * np.log1p(X["ear_volsurge"].clip(0, 20).fillna(0))
    d["rev5_news_adj"] = -X["r5_nonews"].fillna(0) + X["r5_news"].fillna(0)      # Chan: fade no-news, follow news
    d["evt_earn"] = (event_type(X) == "earn").astype("float32").to_numpy()
    d["evt_filing"] = (event_type(X) == "filing").astype("float32").to_numpy()
    return d


def xs_rank(X: pd.DataFrame, cols) -> pd.DataFrame:
    """Cross-sectional percentile rank per date, centred on 0; market (m_) columns pass through raw."""
    keep = [c for c in cols if not c.startswith("m_")]
    R = X[keep].groupby(level=0).rank(pct=True).astype("float32") - 0.5
    for c in cols:
        if c.startswith("m_"):
            R[c] = X[c].astype("float32")
    return R[list(cols)]


# ---- movers: the walk-forward that decides which stocks are 'predicted movers' -----------------------------------
def mover_walk_forward(R: pd.DataFrame, y: pd.Series, years, embargo_days=14, seed=7, max_train=600_000, params=None):
    """Score every row of each year in `years` with a LightGBM fitted only on rows dated before that year's start minus
    the embargo (labels of those rows have closed). Returns (scores, log) where scores is NaN for other years."""
    import lightgbm as lgb
    dates = pd.DatetimeIndex(R.index.get_level_values(0))
    yy = y.reindex(R.index)
    out = pd.Series(np.nan, index=R.index, dtype="float32")
    log = []
    p = dict(n_estimators=250, num_leaves=31, learning_rate=0.05, min_child_samples=200, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.7, n_jobs=6, verbose=-1)
    p.update(params or {})
    rng = np.random.default_rng(seed)
    for Y in years:
        cut = pd.Timestamp(year=int(Y), month=1, day=1) - pd.Timedelta(days=embargo_days)
        tr = np.flatnonzero((dates < cut) & yy.notna().to_numpy())
        te = np.flatnonzero(dates.year == Y)
        if len(te) == 0 or len(tr) < 1000 or yy.iloc[tr].nunique() < 2:
            log.append(dict(year=int(Y), n_train=len(tr), n_test=len(te), fitted=False))
            continue
        if len(tr) > max_train:
            tr = np.sort(rng.choice(tr, max_train, replace=False))
        m = lgb.LGBMClassifier(random_state=seed, **p).fit(R.iloc[tr], yy.iloc[tr].astype(int))
        out.iloc[te] = m.predict_proba(R.iloc[te])[:, 1].astype("float32")
        log.append(dict(year=int(Y), n_train=len(tr), n_test=len(te), fitted=True, train_end=str(dates[tr].max().date()),
                        base_rate=float(yy.iloc[tr].mean())))
    return out, pd.DataFrame(log)


def select_picks(score: pd.Series, n_pick=30, n_pool=100) -> pd.DataFrame:
    """Per date, the n_pick highest scores are the week's predicted movers (`pick`); the n_pool highest form the training
    pool for direction models (same kind of stock, three times the rows). NaN scores are never chosen."""
    rk = score.groupby(level=0).rank(ascending=False, method="first")
    return pd.DataFrame({"pick": (rk <= n_pick).to_numpy(), "pool": (rk <= n_pool).to_numpy(), "rank": rk.to_numpy()},
                        index=score.index)


# ---- direction models --------------------------------------------------------------------------------------------
class DirModel:
    """One direction learner. kind='linear': winsorised z-scores + ridge logistic regression; kind='gbm': a small
    LightGBM (8 leaves, 120 trees, 100-row leaves) that cannot memorise a few thousand rows."""

    def __init__(self, kind="linear", seed=0):
        self.kind, self.seed = kind, seed

    def fit(self, F: np.ndarray, y: np.ndarray):
        if self.kind == "linear":
            self.lo, self.hi = np.nanquantile(F, 0.01, axis=0), np.nanquantile(F, 0.99, axis=0)
            Z = np.clip(F, self.lo, self.hi)
            self.mu, self.sd = np.nanmean(Z, axis=0), np.nanstd(Z, axis=0)
            self.sd[~(self.sd > 0)] = 1.0
            self.m = LogisticRegression(C=0.1, max_iter=300).fit(self._z(F), y)
        else:
            import lightgbm as lgb
            self.m = lgb.LGBMClassifier(n_estimators=120, num_leaves=8, learning_rate=0.05, min_child_samples=100,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=4, verbose=-1,
                                        random_state=self.seed).fit(F, y)
        return self

    def _z(self, F):
        return np.nan_to_num((np.clip(F, self.lo, self.hi) - self.mu) / self.sd, nan=0.0)

    def prob(self, F: np.ndarray) -> np.ndarray:
        return self.m.predict_proba(self._z(F) if self.kind == "linear" else F)[:, 1]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _platt(raw, y, is_prob=True):
    """Calibrator fitted on the calibration block only: logistic regression of the label on the (logit of the) raw score."""
    x = _logit(raw) if is_prob else np.asarray(raw, float)
    if len(np.unique(y)) < 2 or np.nanstd(x) == 0:
        base = float(np.mean(y)) if len(y) else 0.5
        return lambda r, ip=is_prob: np.full(len(r), base)
    mu, sd = float(np.mean(x)), float(np.std(x))
    m = LogisticRegression(C=1e3, max_iter=300).fit(((x - mu) / sd)[:, None], y)
    return lambda r, ip=is_prob: m.predict_proba((((_logit(r) if ip else np.asarray(r, float)) - mu) / sd)[:, None])[:, 1]


def _survival(calib_conf: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """q = share of calibration rows at least as confident as this row: bet at coverage c  <=>  q <= c."""
    if len(calib_conf) == 0:
        return np.ones(len(conf))
    s = np.sort(calib_conf)
    return 1.0 - np.searchsorted(s, conf, side="left") / len(s)


def walk_forward(pool: pd.DataFrame, F: dict, target: str, test_years, models=("linear", "gbm"), extra=None,
                 seg_cols=("seg", "reg"), embargo_days=14, min_train=2000, min_calib=200, seed=0, controls=True):
    """Yearly walk-forward. `pool` has columns date, year, pick, `target` (1 = up, 0 = down, NaN = not scored) and the
    segment columns; F maps a family name to a float feature matrix aligned row-for-row with `pool`. `extra` maps a name
    to fn(train, calib, test) -> (raw_calib, raw_test, is_prob) over positional index arrays (the miner uses it).
    Returns (pred, blocks): pred has one row per (test row, model) with calibrated p, q (calibration survival, overall)
    and q_<seg> (survival among calibration rows of the same segment); blocks documents every split for the audit."""
    rng = np.random.default_rng(seed)
    date = pd.DatetimeIndex(pool["date"]).to_numpy()
    year = pool["year"].to_numpy()
    yv = pool[target].to_numpy(dtype=float)
    pick = pool["pick"].to_numpy(bool)
    have = np.isfinite(yv)
    segs = {c: pool[c].to_numpy() for c in seg_cols if c in pool}
    recs, blocks = [], []
    Fm = {k: np.asarray(v, dtype=np.float32) for k, v in F.items()}
    for Y in test_years:
        ca_start = pd.Timestamp(year=int(Y) - 1, month=1, day=1)
        tr = np.flatnonzero(have & (date < (ca_start - pd.Timedelta(days=embargo_days)).to_datetime64()))
        ca_end = (pd.Timestamp(year=int(Y), month=1, day=1) - pd.Timedelta(days=LABEL_SPAN_DAYS)).to_datetime64()
        ca = np.flatnonzero(have & pick & (year == Y - 1) & (date < ca_end))       # last calibration label closes before test starts
        te = np.flatnonzero(have & pick & (year == Y))
        info = dict(year=int(Y), n_train=len(tr), n_calib=len(ca), n_test=len(te), fitted=False)
        if len(tr) < min_train or len(ca) < min_calib or len(te) == 0 or len(np.unique(yv[tr])) < 2:
            blocks.append(info)
            continue
        info.update(fitted=True, train_end=str(pd.Timestamp(date[tr].max()).date()), calib_start=str(pd.Timestamp(date[ca].min()).date()),
                    calib_end=str(pd.Timestamp(date[ca].max()).date()), test_start=str(pd.Timestamp(date[te].min()).date()),
                    base_rate=float(yv[tr].mean()), calib_base=float(yv[ca].mean()))
        blocks.append(info)
        base = float(yv[tr].mean())
        raw = {}
        for fam, M in Fm.items():
            for kind in models:
                mdl = DirModel(kind, seed).fit(M[tr], yv[tr])
                raw[f"{fam}:{kind}"] = (mdl.prob(M[ca]), mdl.prob(M[te]), True)
        if controls and "combined" in Fm:
            perm = rng.permutation(yv[tr])
            mdl = DirModel("gbm", seed).fit(Fm["combined"][tr], perm)
            raw["ctl_shuffled"] = (mdl.prob(Fm["combined"][ca]), mdl.prob(Fm["combined"][te]), True)
            raw["ctl_random"] = (rng.uniform(size=len(ca)), rng.uniform(size=len(te)), True)
        for name, fn in (extra or {}).items():
            rc, rt, ip = fn(tr, ca, te)
            raw[name] = (np.asarray(rc, float), np.asarray(rt, float), ip)
        cal_all = [k for k in raw if not k.startswith(NON_SIGNAL)]
        if len(cal_all) > 1:                                          # blend: mean z-scored logit of every real model
            def _zl(v, ip, ref):
                x = _logit(v) if ip else np.asarray(v, float)
                r = _logit(ref) if ip else np.asarray(ref, float)
                s = np.std(r)
                return (x - np.mean(r)) / (s if s > 0 else 1.0)
            bc = np.mean([_zl(raw[k][0], raw[k][2], raw[k][0]) for k in cal_all], axis=0)
            bt = np.mean([_zl(raw[k][1], raw[k][2], raw[k][0]) for k in cal_all], axis=0)
            raw["blend"] = (bc, bt, False)
        for name, (rc, rt, ip) in raw.items():
            cal = _platt(rc, yv[ca], ip)
            pc, pt = cal(rc), cal(rt)
            conf_c, conf_t = np.abs(pc - 0.5), np.abs(pt - 0.5)
            if np.ptp(conf_c) < 1e-9:                                  # no spread: cannot rank confidence, never bet below 100%
                q = np.ones(len(te))
            else:
                q = _survival(conf_c, conf_t)
            d = {"row": te, "date": date[te], "year": Y, "model": name, "p": pt, "up": yv[te], "q": q, "p0": base}
            for c, arr in segs.items():
                qs = np.ones(len(te))
                for s in np.unique(arr[te]):
                    mt, mc = arr[te] == s, arr[ca] == s
                    qs[mt] = _survival(conf_c[mc], conf_t[mt]) if mc.sum() >= 30 and np.ptp(conf_c[mc]) > 1e-9 else 1.0
                d[f"q_{c}"] = qs
                d[c] = arr[te]
            recs.append(pd.DataFrame(d))
    pred = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(
        columns=["row", "date", "year", "model", "p", "up", "q", "p0"])
    return pred, pd.DataFrame(blocks)


# ---- statistics: frontier, calibration, the 80% question ---------------------------------------------------------
def frontier(pred: pd.DataFrame, model: str, coverages=COVERAGES, segcol=None, seg=None, B=1000, seed=0, level=0.95,
             n_cells=1) -> pd.DataFrame:
    """Accuracy achievable at coverage c. Bets = rows whose calibration-survival q (or q_<segcol> inside one segment) is <= c.
    Intervals: percentile bootstrap over WEEKS; `lo_adj` subtracts z(1 - 0.05/n_cells) bootstrap standard errors, the
    price of having looked at n_cells (model, segment, coverage) cells. `wilson_lo` ignores week clustering (optimistic)."""
    cols = ["model", "segcol", "seg", "cov_target", "n_rows", "n", "cov_real", "acc", "lo", "hi", "lo_adj", "wilson_lo",
            "up_rate", "weeks"]
    d = pred[pred["model"] == model]
    qcol = "q"
    if segcol is not None:
        d = d[d[segcol] == seg]
        qcol = f"q_{segcol}"
    rows = []
    rng = np.random.default_rng(seed)
    if d.empty:
        return pd.DataFrame([[model, segcol, seg, c, 0, 0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0]
                             for c in coverages], columns=cols)
    wk, uniq = pd.factorize(d["date"])
    W = len(uniq)
    correct = ((d["p"].to_numpy() >= 0.5) == (d["up"].to_numpy() > 0.5)).astype(float)
    q = d[qcol].to_numpy()
    idx = rng.integers(0, W, size=(B, W))
    za = stats.norm.ppf(1 - 0.05 / max(n_cells, 1))
    for c in coverages:
        m = q <= c + 1e-12
        N = np.bincount(wk[m], minlength=W).astype(float)
        K = np.bincount(wk[m], weights=correct[m], minlength=W)
        n = int(N.sum())
        if n == 0:
            rows.append([model, segcol, seg, c, len(d), 0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0])
            continue
        acc = K.sum() / n
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = K[idx].sum(1) / N[idx].sum(1)
        bs = bs[np.isfinite(bs)]
        a = (1 - level) / 2
        lo, hi = (np.quantile(bs, [a, 1 - a]) if len(bs) > 20 else (np.nan, np.nan))
        lo_adj = acc - za * bs.std() if len(bs) > 20 else np.nan
        rows.append([model, segcol, seg, c, len(d), n, n / len(d), acc, lo, hi, lo_adj,
                     wilson(int(K.sum()), n, z=stats.norm.ppf(1 - (1 - level) / 2))[0], float(d["up"].to_numpy()[m].mean()),
                     int((N > 0).sum())])
    return pd.DataFrame(rows, columns=cols)


def frontier_table(pred: pd.DataFrame, coverages=COVERAGES, seg_cols=("seg", "reg"), B=1000, seed=0, n_cells=None) -> pd.DataFrame:
    """Every model overall and inside each segment value. n_cells defaults to the number of cells produced (the honest
    multiple-comparison count for `lo_adj`)."""
    if pred.empty:
        return frontier(pred, "none", coverages)
    models = sorted(pred["model"].unique())
    plan = [(m, None, None) for m in models]
    for sc in seg_cols:
        if sc in pred:
            plan += [(m, sc, s) for m in models for s in sorted(pred[sc].dropna().unique())]
    total = n_cells or len(plan) * len(coverages)
    return pd.concat([frontier(pred, m, coverages, sc, s, B=B, seed=seed, n_cells=total) for m, sc, s in plan],
                     ignore_index=True)


def calibration_table(pred: pd.DataFrame, coverages=(0.05, 0.25, 1.0)) -> pd.DataFrame:
    """Per model and coverage: Brier of the calibrated P(up) against the train-block base rate, skill = 1 - Brier/base,
    ECE with its sampling floor. Brier below base means the probabilities carry information; ECE well above the floor
    means they are not calibrated."""
    rows = []
    for m, d in pred.groupby("model"):
        for c in coverages:
            s = d[d["q"] <= c + 1e-12]
            if len(s) < 30:
                continue
            b, b0 = brier(s["p"], s["up"]), brier(s["p0"], s["up"])
            rows.append(dict(model=m, cov_target=c, n=len(s), brier=b, brier_base=b0, skill=1 - b / b0,
                             ece=ece(s["p"], s["up"]), ece_floor=ece_noise_floor(s["p"]), acc=float(((s["p"] >= 0.5) == (s["up"] > 0.5)).mean())))
    return pd.DataFrame(rows)


def reliability(pred: pd.DataFrame, model: str, n_bins=10) -> pd.DataFrame:
    d = pred[pred["model"] == model]
    return reliability_bins(d["p"].to_numpy(), d["up"].to_numpy(), n_bins, strategy="quantile")


def era_table(pred: pd.DataFrame, coverages=(0.10, 0.25, 1.0)) -> pd.DataFrame:
    """Accuracy per test year, model and coverage with a Wilson interval (no week bootstrap: one year is ~50 weeks)."""
    rows = []
    for (m, y), d in pred.groupby(["model", "year"]):
        ok = ((d["p"] >= 0.5) == (d["up"] > 0.5)).to_numpy()
        for c in coverages:
            s = (d["q"] <= c + 1e-12).to_numpy()
            n = int(s.sum())
            lo, hi = wilson(int(ok[s].sum()), n, z=1.96) if n else (np.nan, np.nan)
            rows.append(dict(model=m, year=int(y), cov_target=c, n=n, acc=float(ok[s].mean()) if n else np.nan, lo=lo, hi=hi,
                             up_rate=float(d["up"].to_numpy()[s].mean()) if n else np.nan))
    return pd.DataFrame(rows)


def eighty_question(ft: pd.DataFrame, gate=0.80, min_bets=30) -> dict:
    """At what coverage, if any, is 80% reached with the 95% week-bootstrap lower bound >= 0.80? Controls are excluded;
    they are reported so a 'hit' among them exposes a broken protocol. `lo_adj` applies the multiple-comparison price."""
    real = ft[~ft["model"].str.startswith(CONTROLS) & (ft["n"] >= min_bets)]
    hit = real[real["lo"] >= gate]
    hit_adj = real[real["lo_adj"] >= gate]
    ctl = ft[ft["model"].str.startswith(("ctl_",)) & (ft["n"] >= min_bets) & (ft["lo"] >= gate)]
    pos = ft[ft["model"].str.startswith(("planted", "leak")) & ft["segcol"].isna() & (ft["n"] >= min_bets)]
    best = real.sort_values("lo", ascending=False).head(1)
    return dict(cells=int(len(ft)), cells_with_min_bets=int(len(real)), reached_acc=int((real["acc"] >= gate).sum()),
                reached_lo=int(len(hit)), reached_lo_adj=int(len(hit_adj)), control_hits=int(len(ctl)),
                positive_controls_found=bool((pos["lo"] >= gate).any()) if len(pos) else False,
                hits=hit.sort_values("cov_real", ascending=False).head(10).to_dict("records"),
                best=best.to_dict("records")[0] if len(best) else None)


# ---- pattern miner as a direction model --------------------------------------------------------------------------
def miner_hook(Xm: pd.DataFrame, ydir: np.ndarray, dates: np.ndarray, miner_params=None):
    """Extra-model factory for `walk_forward`. Xm: feature frame aligned with the pool (MultiIndex date, ticker);
    ydir: +1 up / -1 down / NaN not scored, aligned. Each block fits a PatternMiner on the TRAIN rows only (`now` = the day
    before calibration starts) and scores the calibration and test rows one date at a time; scores are signed, unbounded
    (is_prob False) and are calibrated like every other model."""
    from .patterns import PatternMiner
    ydir = np.asarray(ydir, float)

    def fn(tr, ca, te):
        now = pd.Timestamp(dates[ca].min()) - pd.Timedelta(days=1)
        ok = tr[np.isfinite(ydir[tr])]
        yt = pd.Series(ydir[ok] - np.nanmean(ydir[ok]), index=Xm.index[ok])
        miner = PatternMiner(miner_params or {"max_rows": 200_000, "null_reps": 1, "horizon": HORIZON}).fit(Xm.iloc[ok], yt, now=now)

        def score(rows):
            out = np.zeros(len(rows))
            sub = Xm.iloc[rows]
            d = pd.DatetimeIndex(sub.index.get_level_values(0))
            for day in d.unique():
                m = np.flatnonzero(d == day)
                out[m] = miner.score(sub.iloc[m].droplevel(0)).to_numpy()
            return out
        st = miner.patterns["status"].value_counts().to_dict() if len(miner.patterns) else {}
        fn.info.append(dict(now=str(now.date()), train_rows=len(ok), patterns=int(len(miner.patterns)),
                            active=int(st.get("active", 0)), rescoped=int(st.get("rescoped", 0))))
        return score(ca), score(te), False
    fn.info = []                                  # one dict per block: how many patterns the miner kept
    return fn


# ---- controls, positive control and the point-in-time audit -------------------------------------------------------
def plant_signal(up: np.ndarray, mask: np.ndarray, acc=0.9, seed=0) -> np.ndarray:
    """A feature that states the true direction (+1/-1) with probability `acc` on the rows in `mask` and is 0 elsewhere.
    A working pipeline must find it at coverage ~ mask.mean() with accuracy ~ acc."""
    rng = np.random.default_rng(seed)
    sgn = np.where(np.nan_to_num(up, nan=0.5) > 0.5, 1.0, -1.0)
    flip = rng.uniform(size=len(up)) > acc
    return np.where(mask, np.where(flip, -sgn, sgn), 0.0).astype("float32")


def audit_point_in_time(blocks: pd.DataFrame, labels: pd.DataFrame = None, embargo_days=14) -> dict:
    """Fail-closed checks (C56). Returns {'ok': bool, 'violations': [...]}.
      * every fitted block trains on rows that ended before calibration began, calibrates before testing;
      * the embargo really separates train from calibration (label span included);
      * with a labels frame: every label's entry session is strictly after its decision date and ends within the horizon."""
    bad = []
    fitted = blocks[blocks["fitted"].astype(bool)] if len(blocks) and "fitted" in blocks else blocks.iloc[0:0]
    for b in fitted.itertuples():
        te, ce, ca, tr = (pd.Timestamp(x) for x in (b.test_start, b.calib_end, b.calib_start, b.train_end))
        if not tr + pd.Timedelta(days=LABEL_SPAN_DAYS) < ca:
            bad.append(f"{b.year}: train labels (end {tr.date()}) still open when calibration starts {ca.date()}")
        if not ce + pd.Timedelta(days=LABEL_SPAN_DAYS) < te:
            bad.append(f"{b.year}: calibration labels (end {ce.date()}) still open when the test starts {te.date()}")
        if (ca - tr).days < embargo_days:
            bad.append(f"{b.year}: embargo {(ca - tr).days}d < {embargo_days}d")
    if labels is not None and len(labels):
        d = pd.DatetimeIndex(labels.index.get_level_values(0))
        if not (pd.DatetimeIndex(labels["entry_date"]) > d).all():
            bad.append("label entry session is not strictly after its decision date")
        if not ((pd.DatetimeIndex(labels["end_date"]) - d).days <= LABEL_SPAN_DAYS + 4).all():
            bad.append("label window longer than the horizon allows")
    return dict(ok=not bad, violations=bad)
