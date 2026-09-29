"""Bible Phase 6 (pattern -> Find volatility integration) and Phase 34 (required ablation framework).

Turns a fitted engine.patterns.PatternMiner into inputs for the mover model and decides, against controls, whether
the patterns may be used at all:
  * Bank / score_panel     - a fitted miner's live patterns evaluated on any panel, vectorised, with the same
                              per-date quintiles and regime scoping the miner used;
  * movement + direction   - one miner mined on |excess return| (movement) and one on excess return (direction);
  * MovementCalibrator     - pattern scores -> a probability that a stock moves;
  * attribute              - per-pattern contribution and leave-one-out value;
  * interaction_table      - how the pattern score and the existing mover model agree / add;
  * run_ablation           - mover alone, pattern alone, mover+pattern, mover+random, mover+shuffled,
                              mover+future-scrambled (and mover+patterns mined from pure noise), all scored
                              out of sample on the SAME test dates by the same model class;
  * decide / rolling_ablation - "If real patterns do not beat the controls: do not deploy them."
No look-ahead: the miner sees only the first part of the training window, the mover models are fitted on the later
part (so pattern scores in the fit are out of sample for the miner), and nothing after `split` is read before scoring.
Deterministic: every draw is seeded.  Panel convention: X indexed (date, ticker), market columns start `m_`."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .patterns import CTX, PatternMiner

LIVE = ("active", "rescoped")

ABL_DEFAULT = {"mover_q": 0.80,        # a stock "moves" when |y| is in the top fifth of the training window
               "miner_frac": 0.60,     # share of the training dates the miner learns from
               "gap_days": 10,         # calendar days purged between windows (forward-return horizon)
               "model": "lgbm", "n_controls": 3, "noise_mined": True,
               "topk_frac": 0.05, "base_cols": None,   # columns the existing mover model sees (None = all of X)
                "boot": 400, "block": 5, "min_test_dates": 20,
               "min_gain": 0.002,      # smallest AUC gain over the mover model alone worth deploying
               "alone_margin": 0.005,  # pattern score alone must beat chance by this much
               "miner": {"max_pairs": 300, "max_unless": 60, "null_reps": 1, "min_n": 150, "half_life_years": 50}}


# ------------------------------------------------------------------ metrics
def auc(score, label):
    """Rank AUC with tie handling; nan when a class is missing."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1 = int(label.sum()); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _date_codes(index):
    codes, uniq = pd.factorize(index.get_level_values(0), sort=True)
    return codes, uniq


def daily_auc(score, label, dcode, min_rows=20):
    """Cross-sectional AUC per date (does the score rank tomorrow's movers above the rest?). Series by date code."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    order = np.argsort(dcode, kind="stable")
    bounds = np.flatnonzero(np.diff(dcode[order])) + 1
    out = {}
    for g in np.split(order, bounds):
        if len(g) >= min_rows:
            a = auc(score[g], label[g])
            if np.isfinite(a):
                out[int(dcode[g[0]])] = a
    return pd.Series(out, dtype=float)


def daily_spearman(score, target, dcode, min_rows=20):
    score, target = np.asarray(score, float), np.asarray(target, float)
    order = np.argsort(dcode, kind="stable")
    bounds = np.flatnonzero(np.diff(dcode[order])) + 1
    out = {}
    for g in np.split(order, bounds):
        if len(g) >= min_rows:
            a, b = rankdata(score[g]), rankdata(target[g])
            if a.std() > 0 and b.std() > 0:
                out[int(dcode[g[0]])] = float(np.corrcoef(a, b)[0, 1])
    return pd.Series(out, dtype=float)


def top_lift(proba, label, dcode, frac=0.05, min_rows=20):
    """Mover rate among each date's top `frac` by score, divided by the overall mover rate."""
    proba, label = np.asarray(proba, float), np.asarray(label, bool)
    order = np.argsort(dcode, kind="stable")
    bounds = np.flatnonzero(np.diff(dcode[order])) + 1
    hit = tot = 0
    for g in np.split(order, bounds):
        if len(g) < min_rows:
            continue
        k = max(1, int(round(len(g) * frac)))
        top = g[np.argsort(-proba[g], kind="stable")[:k]]
        hit += label[top].sum(); tot += k
    base = label.mean() if len(label) else np.nan
    return float((hit / tot) / base) if tot and base > 0 else np.nan


def block_bootstrap_mean(x, block, n, rng):
    """Moving-block bootstrap of the mean of a dependent series -> (mean, 5th pct, 95th pct, share of draws <= 0)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(x.mean()) if len(x) else np.nan, np.nan, np.nan, 1.0
    block = max(1, min(block, len(x)))
    nb = int(np.ceil(len(x) / block))
    starts = rng.integers(0, len(x) - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :len(x)]
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95)), float((m <= 0).mean())


# ------------------------------------------------------------------ pattern bank
def ctx_cols_of(X):
    """The regime columns the miner used when it was fitted on X (its scopes index into this list)."""
    return [c for c in CTX if c in X.columns]


@dataclass
class Bank:
    """The live patterns of a fitted miner, detached from it: keys index `feats`, scopes index `ctx_cols`."""
    feats: list
    ctx_cols: list
    keys: list = field(default_factory=list)
    names: list = field(default_factory=list)
    effects: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scopes: list = field(default_factory=list)

    def __len__(self):
        return len(self.keys)


def bank_from_miner(miner, ctx_cols):
    P = miner.patterns
    b = Bank(list(miner.feats), list(ctx_cols))
    if P is None or len(P) == 0:
        return b
    P = P[P["status"].isin(LIVE)]
    b.keys = list(P["key"])
    b.names = list(P["key_named"])
    b.effects = P["effect"].to_numpy(float)
    b.scopes = [s if isinstance(s, tuple) else None for s in P["scope"]]
    return b


def quintiles(X, feats):
    """Per-date quintile of every feature - the same construction the miner and PatternMiner.score use."""
    missing = [c for c in feats if c not in X.columns]
    if missing:
        raise ValueError(f"panel lacks miner features: {missing[:5]}")
    Q = np.empty((len(X), len(feats)), dtype=np.int8)
    for j, c in enumerate(feats):
        r = X[c].groupby(level=0).rank(pct=True).fillna(0.5).to_numpy()
        Q[:, j] = np.minimum((r * 5).astype(np.int8), 4)
    return Q


def key_mask(key, Q):
    if key[0] == "s":
        return Q[:, key[1]] == key[2]
    if key[0] == "p":
        return (Q[:, key[1]] == key[2]) & (Q[:, key[3]] == key[4])
    if key[0] == "u":
        return (Q[:, key[1]] == key[2]) & (Q[:, key[3]] == key[4]) & (Q[:, key[5]] != key[6])
    raise ValueError(f"unknown pattern kind {key[0]!r}")


def _scope_mask(scope, X, ctx_cols):
    """Rows where a rescoped pattern's regime holds; a scope we cannot evaluate never fires (fail closed)."""
    ci, lab, lo_, hi_ = scope
    if ci >= len(ctx_cols) or ctx_cols[ci] not in X.columns:
        return np.zeros(len(X), bool)
    v = X[ctx_cols[ci]].to_numpy(float)
    ok = np.isfinite(v)
    with np.errstate(invalid="ignore"):
        m = (v <= lo_) if lab == "low" else (v > hi_) if lab == "high" else ((v > lo_) & (v <= hi_))
    return m & ok


def fire_masks(bank, X, Q=None):
    """Yield (i, boolean row mask) for every pattern in the bank, scope applied."""
    if len(bank) == 0:
        return
    Q = quintiles(X, bank.feats) if Q is None else Q
    for i, key in enumerate(bank.keys):
        m = key_mask(key, Q)
        if bank.scopes[i] is not None:
            m = m & _scope_mask(bank.scopes[i], X, bank.ctx_cols)
        yield i, m


def score_panel(bank, X, matrix=False):
    """Pattern score per row (sum of effects of every pattern that fires) and the number that fire.
    matrix=True also returns the (rows x patterns) contribution matrix for attribution."""
    n = len(X)
    s = np.zeros(n); nf = np.zeros(n)
    C = np.zeros((n, len(bank)), dtype=np.float32) if matrix else None
    for i, m in fire_masks(bank, X):
        s += bank.effects[i] * m
        nf += m
        if matrix:
            C[:, i] = bank.effects[i] * m
    out = (pd.Series(s, X.index, name="score"), pd.Series(nf, X.index, name="n_fire"))
    return (*out, C) if matrix else out


def random_bank(bank, rng, avoid=()):
    """Control: same number of patterns, same kinds, same effect sizes and scopes, but random conditions.  No random
    pattern may use any (feature, quintile) literal the real banks use (direction AND movement) - otherwise a control that happens to redraw
    the real signal would 'beat' the real patterns for the wrong reason."""
    if len(bank) == 0:
        return Bank(bank.feats, bank.ctx_cols)
    F = len(bank.feats)
    used = {(k[i], k[i + 1]) for b in (bank, *avoid) for k in b.keys for i in range(1, len(k), 2)}
    used_f = {j for j, _ in used}
    keys = []
    for key in bank.keys:
        for attempt in range(80):
            k = 1 if key[0] == "s" else 2 if key[0] == "p" else 3
            js = rng.choice(F, min(k, F), replace=False)
            qs = rng.integers(0, 5, size=len(js))
            if key[0] == "s":
                new = ("s", int(js[0]), int(qs[0]))
            elif key[0] == "p" and len(js) == 2:
                new = ("p", int(js[0]), int(qs[0]), int(js[1]), int(qs[1]))
            elif key[0] == "u" and len(js) == 3:
                new = ("u", int(js[0]), int(qs[0]), int(js[1]), int(qs[1]), int(js[2]), int(rng.choice([0, 4])))
            else:
                new = key
            lits = {(new[i], new[i + 1]) for i in range(1, len(new), 2)}
            # first insist on features the real bank never touches (a negated literal such as "f0 != q0" still
            # fires on the real signal); if the feature space is too small, fall back to literal-level exclusion
            if new != key and not (lits & used) and (attempt >= 40 or not ({j for j, _ in lits} & used_f)):
                break
        keys.append(new)
    order = rng.permutation(len(bank))
    return Bank(bank.feats, bank.ctx_cols, keys, [f"random_{i}" for i in range(len(keys))],
                bank.effects[order].copy(), [bank.scopes[i] for i in order])


def shuffle_within_dates(values, dcode, rng):
    """Permute a column among the stocks of each date: keeps every date's distribution, breaks stock identity."""
    out = np.asarray(values, float).copy()
    order = np.argsort(dcode, kind="stable")
    for g in np.split(order, np.flatnonzero(np.diff(dcode[order])) + 1):
        out[g] = out[rng.permutation(g)]
    return out


def scramble_dates(values, dcode, rng):
    """Future-scramble: each date receives another date's score vector (cycled to fit), so cross-sectional shape and
    level are realistic but the timing is wrong - a pattern that only 'works' through timing luck survives shuffling
    within dates and dies here."""
    values = np.asarray(values, float)
    nd = int(dcode.max()) + 1
    groups = [np.flatnonzero(dcode == d) for d in range(nd)]
    perm = rng.permutation(nd)
    if nd > 1:                                          # no date may keep its own vector
        fixed = np.flatnonzero(perm == np.arange(nd))
        for f in fixed:
            j = (f + 1) % nd
            perm[f], perm[j] = perm[j], perm[f]
    out = np.zeros(len(values))
    for d in range(nd):
        src = values[groups[perm[d]]]
        if len(src):
            out[groups[d]] = np.resize(src, len(groups[d]))
    return out


# ------------------------------------------------------------------ mining for movement and direction
def demeaned_abs(y):
    a = y.abs()
    return a - a.groupby(level=0).transform("mean")


def demeaned(y):
    return y - y.groupby(level=0).transform("mean")


def mine_banks(X, y, now, params=None, seed=7, noise=False):
    """Direction miner (excess return) and movement miner (excess |return|) on the rows given.
    noise=True mines on outcomes shuffled within each date: the bank shows what the search invents from nothing."""
    p = {**ABL_DEFAULT["miner"], **(params or {}), "seed": seed}
    cc = ctx_cols_of(X)
    yd, ym = demeaned(y), demeaned_abs(y)
    if noise:
        dc, _ = _date_codes(y.index)
        rng = np.random.default_rng(seed + 991)
        yd = pd.Series(shuffle_within_dates(yd.values, dc, rng), y.index)
        ym = pd.Series(shuffle_within_dates(ym.values, dc, rng), y.index)
    out = []
    for target in (yd, ym):
        m = PatternMiner(p).fit(X, target, now)
        b = bank_from_miner(m, cc)
        b.report = dict(m.report)
        out.append(b)
    return out[0], out[1]


def pattern_features(dir_bank, mov_bank, X):
    """The columns Find volatility can consume: direction score, movement score, and how many patterns fire."""
    sd, nd = score_panel(dir_bank, X)
    sm, nm = score_panel(mov_bank, X)
    return pd.DataFrame({"pat_dir": sd, "pat_mov": sm, "pat_fire": nd + nm}, index=X.index)


# ------------------------------------------------------------------ mover model + calibration
def fit_logit(F, y, l2=1.0, iters=50):
    """Ridge logistic regression by Newton steps on standardised inputs -> (coef incl. intercept, mean, sd)."""
    F = np.nan_to_num(np.asarray(F, float)); y = np.asarray(y, float)
    mu, sd = F.mean(axis=0), F.std(axis=0); sd[sd < 1e-12] = 1.0
    Z = np.c_[np.ones(len(F)), (F - mu) / sd]
    b = np.zeros(Z.shape[1]); b[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    R = np.eye(Z.shape[1]) * l2; R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ b))
        g = Z.T @ (p - y) + R @ b
        H = (Z * (p * (1 - p))[:, None]).T @ Z + R + 1e-9 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return b, mu, sd


def predict_logit(model, F):
    b, mu, sd = model
    Z = np.c_[np.ones(len(F)), (np.nan_to_num(np.asarray(F, float)) - mu) / sd]
    return 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))


class MoverModel:
    """The mover classifier every ablation arm uses: same class, same settings, same rows - only the inputs differ."""

    def __init__(self, kind="lgbm", seed=7):
        self.kind, self.seed = kind, seed
        self.const = None

    def fit(self, F, label):
        label = np.asarray(label, bool)
        if label.all() or (~label).all() or len(label) < 50:
            self.const = float(label.mean()) if len(label) else 0.0
            return self
        F = F.astype(float)
        if self.kind == "lgbm":
            import lightgbm as lgb
            self.m = lgb.LGBMClassifier(n_estimators=80, learning_rate=0.06, num_leaves=8, min_child_samples=100,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=self.seed,
                                        n_jobs=2, deterministic=True, force_row_wise=True, verbose=-1)
            self.m.fit(F.to_numpy(), label)
        else:
            self.m = fit_logit(F.to_numpy(), label)
        return self

    def predict(self, F):
        if self.const is not None:
            return np.full(len(F), self.const)
        if self.kind == "lgbm":
            return self.m.predict_proba(F.astype(float).to_numpy())[:, 1]
        return predict_logit(self.m, F.to_numpy())


class MovementCalibrator:
    """Pattern movement score -> probability that the stock moves (Phase 6 'movement probability')."""

    def __init__(self, l2=1.0):
        self.l2 = l2

    def _design(self, pf):
        return np.c_[pf["pat_mov"].to_numpy(), np.abs(pf["pat_dir"].to_numpy()), pf["pat_fire"].to_numpy()]

    def fit(self, pf, label):
        label = np.asarray(label, bool)
        self.const = float(label.mean()) if (label.all() or (~label).all() or len(label) < 30) else None
        if self.const is None:
            self.m = fit_logit(self._design(pf), label, self.l2)
        return self

    def predict(self, pf):
        return np.full(len(pf), self.const) if self.const is not None else predict_logit(self.m, self._design(pf))


# ------------------------------------------------------------------ attribution and interaction
def attribute(bank, X, label, y_excess, max_loo=60):
    """Per-pattern contribution: fire rate, mover rate and excess return when it fires, share of the score's
    mass, and leave-one-out change in the daily rank correlation of the score with movement (positive = the
    pattern helps the score). A pattern can look strong alone and add nothing once the others are in."""
    if len(bank) == 0:
        return pd.DataFrame(columns=["name", "fire_rate", "mover_rate", "mover_lift", "mean_excess", "mass_share", "loo_ic_drop"])
    label = np.asarray(label, bool); ye = np.asarray(y_excess, float)
    dcode, _ = _date_codes(X.index)
    s, _, C = score_panel(bank, X, matrix=True)
    total = s.to_numpy()
    base_ic = daily_spearman(total, label.astype(float), dcode).mean()
    mass = np.abs(C).sum(axis=0)
    base_rate = label.mean()
    rows = []
    for i, nm in enumerate(bank.names):
        f = C[:, i] != 0
        loo = np.nan
        if i < max_loo:
            loo = base_ic - daily_spearman(total - C[:, i], label.astype(float), dcode).mean()
        rows.append({"name": nm, "effect": float(bank.effects[i]), "fire_rate": float(f.mean()),
                     "mover_rate": float(label[f].mean()) if f.any() else np.nan,
                     "mover_lift": float(label[f].mean() / base_rate) if f.any() and base_rate > 0 else np.nan,
                     "mean_excess": float(ye[f].mean()) if f.any() else np.nan,
                     "mass_share": float(mass[i] / mass.sum()) if mass.sum() > 0 else 0.0, "loo_ic_drop": float(loo)})
    return pd.DataFrame(rows).sort_values("mass_share", ascending=False, kind="mergesort").reset_index(drop=True)


def interaction_table(base_p, pat_score, label, dcode, bins=3):
    """Mover rate in each (mover-model tercile x pattern tercile) cell, ranked within each date, plus a synergy
    ratio for the top-top cell: observed lift over the product of the two marginal lifts (1.0 = independent
    information; >1 the two agree on the same stocks and reinforce; <1 they overlap)."""
    df = pd.DataFrame({"b": np.asarray(base_p, float), "p": np.asarray(pat_score, float),
                       "y": np.asarray(label, float), "d": dcode})
    df["rb"] = df.groupby("d")["b"].rank(pct=True)
    df["rp"] = df.groupby("d")["p"].rank(pct=True, method="first")
    df["cb"] = np.minimum((df["rb"] * bins).astype(int), bins - 1)
    df["cp"] = np.minimum((df["rp"] * bins).astype(int), bins - 1)
    tab = df.pivot_table(index="cb", columns="cp", values="y", aggfunc="mean")
    cnt = df.pivot_table(index="cb", columns="cp", values="y", aggfunc="size")
    base = df["y"].mean()
    top = bins - 1
    lift_b = df.loc[df["cb"] == top, "y"].mean() / base if base > 0 else np.nan
    lift_p = df.loc[df["cp"] == top, "y"].mean() / base if base > 0 else np.nan
    cell = df.loc[(df["cb"] == top) & (df["cp"] == top), "y"]
    synergy = (cell.mean() / base) / (lift_b * lift_p) if len(cell) >= 30 and base > 0 and lift_b * lift_p > 0 else np.nan
    corr = float(np.corrcoef(df["rb"], df["rp"])[0, 1]) if df["rb"].std() > 0 and df["rp"].std() > 0 else np.nan
    return {"rate": tab, "count": cnt, "base_rate": float(base), "synergy_top": float(synergy) if np.isfinite(synergy) else np.nan,
            "rank_corr": corr}


# ------------------------------------------------------------------ the ablation
def _split_dates(dates, split, gap_days, miner_frac):
    """Train (miner part, model part) and test date sets; every boundary is separated by the purge gap."""
    d = pd.DatetimeIndex(dates)
    split = pd.Timestamp(split)
    gap = pd.Timedelta(days=gap_days)
    train = d[d < split - gap]
    if len(train) < 10:
        return None
    k = int(len(train) * miner_frac)
    miner_d = train[:k]
    model_d = train[train > miner_d[-1] + gap] if k > 0 else train[:0]
    return miner_d, model_d, d[d >= split]


def _metrics(proba, label, y_abs, dcode, topk_frac):
    da = daily_auc(proba, label, dcode)
    ic = daily_spearman(proba, y_abs, dcode)
    return {"auc_daily": float(da.mean()) if len(da) else np.nan, "auc_pooled": auc(proba, label),
            "ic_abs": float(ic.mean()) if len(ic) else np.nan, "lift_topk": top_lift(proba, label, dcode, topk_frac),
            "n_dates": int(len(da))}, da


def _direction_accuracy(pat_dir, y_ex, label, frac=0.2):
    """Among realised movers where the pattern has a view, how often is the sign right? -> (accuracy, n, z vs 50%)."""
    pd_, ye = np.asarray(pat_dir, float), np.asarray(y_ex, float)
    m = np.asarray(label, bool) & (pd_ != 0) & (ye != 0)
    if m.sum() < 30:
        return np.nan, int(m.sum()), np.nan
    cut = np.quantile(np.abs(pd_[m]), 1 - frac)
    m &= np.abs(pd_) >= cut
    n = int(m.sum())
    acc = float((np.sign(pd_[m]) == np.sign(ye[m])).mean())
    return acc, n, float((acc - 0.5) / np.sqrt(0.25 / n)) if n else np.nan


def run_ablation(X, y, split, cfg=None, seed=7, test_end=None):
    """Phase 6 / 34: the six-arm out-of-sample comparison at one origin.  Returns per-arm metrics, paired daily-AUC
    differences and the deploy decision.  `split` is the first test date; nothing at or after it is used to learn."""
    c = {**ABL_DEFAULT, **(cfg or {})}
    if c.get("miner") is not None:
        c["miner"] = {**ABL_DEFAULT["miner"], **c["miner"]}
    sp = _split_dates(X.index.get_level_values(0).unique().sort_values(), split, c["gap_days"], c["miner_frac"])
    if sp is None or len(sp[1]) < 10:
        return {"ok": False, "reason": "not enough training dates", "deploy": False, "reasons": ["not enough training dates"]}
    miner_d, model_d, test_d = sp
    if test_end is not None:
        test_d = test_d[test_d < pd.Timestamp(test_end)]
    if len(test_d) < c["min_test_dates"]:
        return {"ok": False, "reason": "not enough test dates", "deploy": False, "reasons": ["not enough test dates"]}
    dts = X.index.get_level_values(0)
    ok = y.notna().to_numpy()
    take = lambda ds: np.isin(dts, ds) & ok
    Xm, ym = X[take(miner_d)], y[take(miner_d)]
    Xf, yf = X[take(model_d)], y[take(model_d)]
    Xt, yt = X[take(test_d)], y[take(test_d)]
    rng = np.random.default_rng(seed)

    train_abs = pd.concat([ym.abs(), yf.abs()]).to_numpy()
    thr = float(np.quantile(train_abs, c["mover_q"]))
    lab_f, lab_t = (yf.abs() >= thr).to_numpy(), (yt.abs() >= thr).to_numpy()
    dcf, _ = _date_codes(Xf.index); dct, _ = _date_codes(Xt.index)
    yt_abs = yt.abs().to_numpy()
    yt_ex = demeaned(yt).to_numpy()

    dir_b, mov_b = mine_banks(Xm, ym, miner_d[-1], c["miner"], seed)
    real_f = pattern_features(dir_b, mov_b, Xf); real_t = pattern_features(dir_b, mov_b, Xt)
    Xf_all, Xt_all = Xf, Xt

    def fit_eval(name, Ftr, Fte):
        mdl = MoverModel(c["model"], seed).fit(Ftr, lab_f)
        p = mdl.predict(Fte)
        met, da = _metrics(p, lab_t, yt_abs, dct, c["topk_frac"])
        return met, da, p

    arms, daily, probas = {}, {}, {}
    def add(name, Ftr, Fte):
        arms[name], daily[name], probas[name] = fit_eval(name, Ftr, Fte)

    if c["base_cols"] is not None:              # the mover model's own inputs; the miner still sees all of X
        Xf, Xt = Xf[list(c["base_cols"])], Xt[list(c["base_cols"])]
    add("base", Xf, Xt)
    add("pattern_only", real_f, real_t)
    add("base+pattern", pd.concat([Xf, real_f], axis=1), pd.concat([Xt, real_t], axis=1))

    fam = {"random": [], "shuffled": [], "scrambled": [], "noise_mined": []}
    for k in range(c["n_controls"]):
        r = np.random.default_rng([seed, k, 17])
        rd, rm = random_bank(dir_b, r, (mov_b,)), random_bank(mov_b, r, (dir_b,))
        fr, ft = pattern_features(rd, rm, Xf_all), pattern_features(rd, rm, Xt_all)
        nm = f"base+random#{k}"; add(nm, pd.concat([Xf, fr], axis=1), pd.concat([Xt, ft], axis=1)); fam["random"].append(nm)
        fs, ft2 = real_f.copy(), real_t.copy()
        for col in fs:
            fs[col] = shuffle_within_dates(fs[col].values, dcf, r)
            ft2[col] = shuffle_within_dates(ft2[col].values, dct, r)
        nm = f"base+shuffled#{k}"; add(nm, pd.concat([Xf, fs], axis=1), pd.concat([Xt, ft2], axis=1)); fam["shuffled"].append(nm)
        fs, ft2 = real_f.copy(), real_t.copy()
        for col in fs:
            fs[col] = scramble_dates(fs[col].values, dcf, r)
            ft2[col] = scramble_dates(ft2[col].values, dct, r)
        nm = f"base+scrambled#{k}"; add(nm, pd.concat([Xf, fs], axis=1), pd.concat([Xt, ft2], axis=1)); fam["scrambled"].append(nm)
    if c["noise_mined"]:
        nd_, nm_ = mine_banks(Xm, ym, miner_d[-1], c["miner"], seed, noise=True)
        add("base+noise_mined", pd.concat([Xf, pattern_features(nd_, nm_, Xf_all)], axis=1),
            pd.concat([Xt, pattern_features(nd_, nm_, Xt_all)], axis=1))
        fam["noise_mined"].append("base+noise_mined")

    gain = {k: arms[k]["auc_daily"] - arms["base"]["auc_daily"] for k in arms}
    diff = (daily["base+pattern"] - daily["base"]).dropna()
    boot = block_bootstrap_mean(diff.values, c["block"], c["boot"], np.random.default_rng([seed, 5]))
    acc, n_acc, z_acc = _direction_accuracy(real_t["pat_dir"].values, yt_ex, lab_t)
    cal = MovementCalibrator().fit(real_f, lab_f)
    cal_p = cal.predict(real_t)
    cal_auc = daily_auc(cal_p, lab_t, dct)
    inter = interaction_table(probas["base"], real_t["pat_mov"].values, lab_t, dct)
    res = {"ok": True, "split": str(pd.Timestamp(split).date()), "seed": seed, "mover_threshold": thr,
           "n_train_miner": int(len(ym)), "n_train_model": int(len(yf)), "n_test": int(len(yt)),
           "mover_rate_test": float(lab_t.mean()), "arms": arms, "gain_vs_base": gain, "families": fam,
           "real_vs_base": {"mean": boot[0], "lo5": boot[1], "hi95": boot[2], "p_le_0": boot[3]},
           "bank": {"direction": len(dir_b), "movement": len(mov_b),
                    "dir_report": getattr(dir_b, "report", {}), "mov_report": getattr(mov_b, "report", {})},
           "direction": {"accuracy": acc, "n": n_acc, "z": z_acc},
           "calibrated_movement_auc": float(cal_auc.mean()) if len(cal_auc) else np.nan,
           "interaction": {"synergy_top": inter["synergy_top"], "rank_corr": inter["rank_corr"], "base_rate": inter["base_rate"]},
           "attribution": attribute(mov_b, Xt_all, lab_t, yt_ex, max_loo=25) if len(mov_b) else pd.DataFrame(),
           "daily_diff_real_vs_base": diff}
    res.update(decide(res, c))
    return res


def decide(res, cfg=None):
    """Deploy only when the real patterns add out-of-sample value that no control reproduces.
    Every condition must hold; the reasons for a refusal are returned in words."""
    c = {**ABL_DEFAULT, **(cfg or {})}
    if not res.get("ok"):
        return {"deploy": False, "reasons": [res.get("reason", "ablation did not run")]}
    g = res["gain_vs_base"]; why = []
    if res["bank"]["direction"] + res["bank"]["movement"] == 0:
        why.append("miner found no live patterns")
    real = g["base+pattern"]
    if not real > c["min_gain"]:
        why.append(f"mover+pattern gain over mover alone {real:+.4f} <= {c['min_gain']}")
    if not (res["real_vs_base"]["lo5"] > 0):
        why.append(f"paired bootstrap lower bound {res['real_vs_base']['lo5']:+.4f} is not above 0")
    for fam, names in res["families"].items():
        if names:
            worst = max(g[n] for n in names)
            if not real > worst:
                why.append(f"real gain {real:+.4f} does not beat the {fam} control (best {worst:+.4f})")
    alone = res["arms"]["pattern_only"]["auc_daily"]
    if not (alone > 0.5 + c["alone_margin"]):
        why.append(f"pattern score alone AUC {alone:.4f} is not above chance by {c['alone_margin']}")
    return {"deploy": not why, "reasons": why or ["real patterns beat every control out of sample"]}


def rolling_ablation(X, y, origins, cfg=None, seed=7, win_share=0.6, min_origins=3):
    """Repeat the ablation at successive origins (each tests up to the next origin).  Deploy only when the patterns
    win at a majority of origins AND the pooled paired difference is positive - one lucky window is not evidence."""
    origins = [pd.Timestamp(o) for o in origins]
    per = []
    for i, o in enumerate(origins):
        end = origins[i + 1] if i + 1 < len(origins) else None
        r = run_ablation(X, y, o, cfg, seed + i, test_end=end)
        per.append(r)
    ran = [r for r in per if r.get("ok")]
    tbl = pd.DataFrame([{"split": r["split"], "base": r["arms"]["base"]["auc_daily"],
                         "base+pattern": r["arms"]["base+pattern"]["auc_daily"], "gain": r["gain_vs_base"]["base+pattern"],
                         "pattern_only": r["arms"]["pattern_only"]["auc_daily"], "deploy": r["deploy"]} for r in ran])
    out = {"per_origin": tbl, "results": per, "n_ran": len(ran)}
    if len(ran) < min_origins:
        out.update(deploy=False, reasons=[f"only {len(ran)} usable origins (<{min_origins})"])
        return out
    pooled = pd.concat([r["daily_diff_real_vs_base"] for r in ran]).values
    b = block_bootstrap_mean(pooled, (cfg or {}).get("block", ABL_DEFAULT["block"]), 400, np.random.default_rng([seed, 9]))
    share = float(tbl["deploy"].mean())
    why = []
    if share < win_share:
        why.append(f"real patterns passed at {share:.0%} of origins (<{win_share:.0%})")
    if not b[1] > 0:
        why.append(f"pooled paired difference lower bound {b[1]:+.4f} is not above 0")
    out.update(win_share=share, pooled={"mean": b[0], "lo5": b[1], "hi95": b[2]}, deploy=not why,
               reasons=why or ["passed at a majority of origins and pooled difference is positive"])
    return out


def format_ablation(res):
    """Plain-text table of the arms, gains and the verdict."""
    if not res.get("ok"):
        return f"ablation did not run: {res.get('reason')}"
    lines = [f"origin {res['split']}  movers {res['mover_rate_test']:.1%}  bank dir/mov {res['bank']['direction']}/{res['bank']['movement']}",
             f"{'arm':<22}{'AUC/day':>9}{'gain':>9}{'IC|y|':>9}{'lift@k':>8}"]
    for k, m in res["arms"].items():
        lines.append(f"{k:<22}{m['auc_daily']:>9.4f}{res['gain_vs_base'][k]:>+9.4f}{m['ic_abs']:>9.4f}{m['lift_topk']:>8.2f}")
    d = res["direction"]
    lines.append(f"direction accuracy on movers: {d['accuracy']:.3f} (n={d['n']}, z={d['z']:.2f})")
    lines.append("DEPLOY" if res["deploy"] else "DO NOT DEPLOY")
    lines += [f"  - {r}" for r in res["reasons"]]
    return "\n".join(lines)


# ------------------------------------------------------------------ calibration, attribution cuts, power
def calibration_table(p, label, bins=10):
    """Reliability of a movement probability: predicted vs realised mover rate per equal-count bin, the Brier score
    (and its skill against always predicting the base rate) and the expected calibration error."""
    p, label = np.asarray(p, float), np.asarray(label, float)
    if len(p) < bins * 5:
        return pd.DataFrame(columns=["bin", "n", "pred", "actual"]), {"brier": np.nan, "brier_skill": np.nan, "ece": np.nan}
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = [{"bin": int(i), "n": int((b == i).sum()), "pred": float(p[b == i].mean()), "actual": float(label[b == i].mean())}
            for i in range(len(edges) - 1) if (b == i).any()]
    t = pd.DataFrame(rows)
    brier = float(np.mean((p - label) ** 2))
    ref = float(np.mean((label.mean() - label) ** 2))
    return t, {"brier": brier, "brier_skill": 1 - brier / ref if ref > 0 else np.nan,
               "ece": float((t["n"] * (t["pred"] - t["actual"]).abs()).sum() / t["n"].sum())}


def kind_of(name):
    """single / pair / exception, from a pattern's readable name."""
    return "exception" if " unless " in name else "pair" if " & " in name else "single"


def attribute_by_kind(att):
    """Aggregate an attribute() table by pattern kind: how much of the score each kind carries and whether it helps."""
    if att is None or len(att) == 0:
        return pd.DataFrame(columns=["kind", "patterns", "mass_share", "mean_lift", "mean_loo_drop"])
    a = att.assign(kind=att["name"].map(kind_of))
    return a.groupby("kind").agg(patterns=("name", "size"), mass_share=("mass_share", "sum"), mean_lift=("mover_lift", "mean"),
                                 mean_loo_drop=("loo_ic_drop", "mean")).reset_index()


def attribute_by_feature(att):
    """Score mass per underlying feature (a feature that appears in many patterns carries more of the score)."""
    if att is None or len(att) == 0:
        return pd.DataFrame(columns=["feature", "mass"])
    mass = {}
    for nm, ms in zip(att["name"], att["mass_share"]):
        toks = [t.rsplit(" q", 1)[0] for t in nm.replace(" unless ", " & ").split(" & ")]
        for t in toks:
            mass[t] = mass.get(t, 0.0) + ms / len(toks)
    return pd.DataFrame(sorted(mass.items(), key=lambda kv: -kv[1]), columns=["feature", "mass"])


def detectable_gain(daily_diff, block=5, z=2.8):
    """Smallest true daily-AUC gain this test could have detected (80% power, 5% one-sided): z * block-adjusted
    standard error.  A 'no gain' verdict from a test whose detectable gain is larger than any gain worth deploying
    is not evidence of absence."""
    x = np.asarray(daily_diff, float); x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    e = x - x.mean()
    n = len(x)
    var = e @ e / n
    for l in range(1, min(block, n - 1) + 1):
        var += 2 * (1 - l / (block + 1)) * (e[l:] @ e[:-l]) / n
    return float(z * np.sqrt(max(var, 1e-18) / n))


def ablation_sweep(X, y, split, qs=(0.7, 0.8, 0.9), cfg=None, seed=7, test_end=None):
    """Repeat the ablation at several mover thresholds: a pattern family that helps only at one arbitrary cut-off is
    a threshold effect.  Returns one row per threshold."""
    rows = []
    for q in qs:
        r = run_ablation(X, y, split, {**(cfg or {}), "mover_q": q}, seed, test_end)
        if r.get("ok"):
            rows.append({"mover_q": q, "movers": r["mover_rate_test"], "base": r["arms"]["base"]["auc_daily"],
                         "gain": r["gain_vs_base"]["base+pattern"], "pattern_only": r["arms"]["pattern_only"]["auc_daily"],
                         "best_control_gain": max(v for k, v in r["gain_vs_base"].items() if "#" in k or "noise" in k),
                         "deploy": r["deploy"]})
        else:
            rows.append({"mover_q": q, "deploy": False})
    return pd.DataFrame(rows)


def bank_overlap(a, b):
    """Jaccard overlap of two banks' pattern names - how much of the pattern set survives a refit."""
    sa, sb = set(a.names), set(b.names)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else np.nan


# ------------------------------------------------------------------ the deployable object
class PatternMoverModel:
    """What Find volatility would hold: banks + calibrator + the deploy decision from the ablation.
    fit() learns from data ending strictly before `now`, runs the ablation on the LAST slice of that same data (never
    on anything later) and records whether the patterns earned deployment.  features() then refuses to emit anything
    but zeros unless they did - 'if real patterns do not beat the controls: do not deploy them' enforced in code."""

    def __init__(self, cfg=None, seed=7, holdout_frac=0.25):
        self.cfg, self.seed, self.holdout_frac = {**ABL_DEFAULT, **(cfg or {})}, seed, holdout_frac
        self.dir_bank = self.mov_bank = None
        self.calibrator = None
        self.decision = {"deploy": False, "reasons": ["not fitted"]}
        self.now = None

    def fit(self, X, y, now):
        now = pd.Timestamp(now)
        d = X.index.get_level_values(0)
        if len(d) and d.max() >= now:
            raise ValueError("training rows must end strictly before `now`")
        ud = d.unique().sort_values()
        if len(ud) < 60:
            self.decision = {"deploy": False, "reasons": ["fewer than 60 training dates"]}
            self.now = now
            return self
        hold_start = ud[int(len(ud) * (1 - self.holdout_frac))]
        self.ablation = run_ablation(X, y, hold_start, self.cfg, self.seed)
        self.decision = {"deploy": bool(self.ablation.get("deploy")), "reasons": list(self.ablation.get("reasons", []))}
        # the banks that would actually be used are refitted on ALL the data before `now`
        ok = y.notna().to_numpy()
        self.dir_bank, self.mov_bank = mine_banks(X[ok], y[ok], ud[-1], self.cfg["miner"], self.seed)
        pf = pattern_features(self.dir_bank, self.mov_bank, X[ok])
        self.threshold = float(np.quantile(y[ok].abs().to_numpy(), self.cfg["mover_q"]))
        self.calibrator = MovementCalibrator().fit(pf, (y[ok].abs() >= self.threshold).to_numpy())
        self.now = now
        return self

    def features(self, Xday, as_of):
        """Pattern columns for one decision day.  Zero-filled (and flagged) when patterns are not deployable."""
        if self.now is None:
            raise RuntimeError("fit() first")
        dts = Xday.index.get_level_values(0)
        if len(dts) and pd.Timestamp(dts.max()) > pd.Timestamp(as_of):
            raise ValueError("Xday holds rows after as_of")
        if pd.Timestamp(as_of) < self.now:
            raise ValueError("as_of precedes the training cut-off")
        out = pd.DataFrame(0.0, index=Xday.index, columns=["pat_dir", "pat_mov", "pat_fire", "p_move"])
        out["deployed"] = False
        if not self.decision["deploy"] or self.dir_bank is None:
            out["p_move"] = np.nan
            return out
        pf = pattern_features(self.dir_bank, self.mov_bank, Xday)
        out[["pat_dir", "pat_mov", "pat_fire"]] = pf
        out["p_move"] = self.calibrator.predict(pf)
        out["deployed"] = True
        return out
