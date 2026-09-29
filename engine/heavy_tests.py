"""Bible Phase 7 (heavy algorithm testing), with Phase 34 controls and the Phase 36 provenance block.

Runs the pattern miner through a rolling-origin walk-forward over the whole history and measures, on data the miner
never saw, everything Phase 7 lists: movement prediction (AUC of the pattern movement score), direction accuracy, IC,
rank IC, t-stat (naive and Newey-West), false discoveries (patterns that fail out of sample, plus what the search
invents from noise), pattern count, pattern survival across refits, turnover, costs, stability across seeds, decay with
model age, and regime sensitivity.  Every table is cut by era (pre-decimal, post-decimal, post-electronic, modern),
by market regime (bull, bear, high-vol, low-vol, high-dispersion) and over random hidden windows.
"Never optimize exclusively for average return": verdict() only passes when EVERY gate passes - mean IC alone can
never carry it - and a noise control (outcomes shuffled within each date, same pipeline) must be beaten.
No look-ahead: an origin's miner sees only dates at least `gap_days` before the origin; each test block is scored
with that fixed miner; dates are never reused to choose parameters.  Deterministic: seeds drive the miner, tie breaks
and the null shuffles.  Panel convention: X indexed (date, ticker), market columns start `m_`; y is forward return."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import pattern_movers as pm
from .patterns import PatternMiner

# era boundaries are calendar facts: NYSE decimalisation finished 9 Apr 2001; Reg NMS / electronic dominance ~2007
ERAS = {"pre_decimal": ("1900-01-01", "2001-04-08"), "post_decimal": ("2001-04-09", "2006-12-31"),
        "post_electronic": ("2007-01-01", "2015-12-31"), "modern": ("2016-01-01", "2200-01-01")}

HEAVY_DEFAULT = {"min_train_dates": 104, "step_dates": 26, "train_max_dates": None, "gap_days": 10,
                 "topk": 20, "cost_bps": 5.0, "nw_lag": 4, "movement": True, "mover_q": 0.80,
                 "min_rows_date": 30, "min_era_dates": 20, "cost_grid_bps": (0, 5, 15, 30),
                 "miner": {"max_pairs": 800, "max_unless": 120, "null_reps": 1, "min_n": 200}}

GATES_DEFAULT = {"min_ic": 0.005,              # mean rank IC over all out-of-sample dates
                 "min_t_nw": 2.0,              # Newey-West t of that mean
                 "min_pos_era_share": 0.6,     # share of eras (with enough dates) whose IC is positive
                 "min_worst_era_ic": -0.01,    # no era may be badly negative
                 "max_oos_false_rate": 0.6,    # live patterns that fail their own out-of-sample block
                 "min_net_spread": 0.0,        # top-k excess return net of costs
                 "min_seed_agree": 0.7,        # share of seeds with positive IC
                 "min_survival": 0.25,         # live patterns that are still live at the next refit
                 "max_regime_gap": 0.08}       # widest IC gap between regime groups (regime sensitivity)


# ------------------------------------------------------------------ input validation and tags
def validate_panel(X, y):
    """Hard errors stop the run; the diagnostics dict says what the harness will be working with."""
    if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels != 2:
        raise ValueError("X must be indexed by (date, ticker)")
    if not X.index.equals(y.index):
        raise ValueError("y must share X's index exactly")
    if X.index.duplicated().any():
        raise ValueError(f"{int(X.index.duplicated().sum())} duplicated (date, ticker) rows")
    dts = X.index.get_level_values(0)
    if not dts.is_monotonic_increasing:
        raise ValueError("rows must be sorted by date")
    num = X.select_dtypes("number")
    if np.isinf(num.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("X contains infinite values")
    per = X.groupby(level=0).size()
    return {"rows": int(len(X)), "dates": int(per.size), "first": str(dts.min().date()), "last": str(dts.max().date()),
            "tickers_per_date_min": int(per.min()), "tickers_per_date_median": float(per.median()),
            "y_nan_share": float(y.isna().mean()), "x_nan_share": float(num.isna().to_numpy().mean()),
            "features": int(sum(not c.startswith("m_") for c in X.columns))}


def era_of(dates, eras=None):
    """Era label for each date; dates outside every era are 'outside'."""
    eras = eras or ERAS
    d = pd.DatetimeIndex(dates)
    out = np.full(len(d), "outside", dtype=object)
    for name, (a, b) in eras.items():
        out[(d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))] = name
    return pd.Series(out, index=d)


def regime_tags(X, min_obs=26):
    """Per-date regime labels from market columns, using only PAST dates for the thresholds (expanding median of the
    previous dates), so a tag never depends on the future.  Columns absent from X give no tag."""
    first = X.groupby(level=0).head(1)
    first = first.droplevel(1)
    tags = pd.DataFrame(index=first.index)
    def vs_past_median(col, hi, lo):
        v = first[col].astype(float)
        med = v.expanding(min_periods=min_obs).median().shift(1)
        return pd.Series(np.where(med.isna(), "unk", np.where(v > med, hi, lo)), index=first.index)
    if "m_vix" in first:
        tags["vol"] = vs_past_median("m_vix", "high_vol", "low_vol")
    if "m_dispersion" in first:
        tags["dispersion"] = vs_past_median("m_dispersion", "high_disp", "low_disp")
    if "m_spy_ma200" in first:
        v = first["m_spy_ma200"].astype(float)
        tags["trend"] = np.where(v.isna(), "unk", np.where(v > 0, "bull", "bear"))
    return tags


# ------------------------------------------------------------------ statistics
def newey_west_t(x, lag=4):
    """t-stat of the mean with Newey-West (Bartlett) standard error: IC series of overlapping windows are serially
    correlated, and the naive t overstates the evidence."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for l in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(max(s, 1e-18) / n))


def series_stats(x, lag=4):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return {"n": int(len(x)), "mean": np.nan, "sd": np.nan, "t": np.nan, "t_nw": np.nan, "pos_share": np.nan}
    sd = float(x.std(ddof=1))
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": sd,
            "t": float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else (0.0 if x.mean() == 0 else np.nan),
            "t_nw": newey_west_t(x, lag), "pos_share": float((x > 0).mean())}


def decay_fit(age_weeks, ic):
    """Linear decay of the IC with model age: intercept, slope per week, half-life in weeks (nan unless it decays)."""
    a, v = np.asarray(age_weeks, float), np.asarray(ic, float)
    ok = np.isfinite(a) & np.isfinite(v)
    if ok.sum() < 8 or a[ok].std() == 0:
        return {"intercept": np.nan, "slope_per_week": np.nan, "half_life_weeks": np.nan}
    b, c = np.polyfit(a[ok], v[ok], 1)
    hl = (c / 2) / -b if c > 0 and b < -1e-9 else np.nan
    return {"intercept": float(c), "slope_per_week": float(b), "half_life_weeks": float(hl)}


# ------------------------------------------------------------------ walk-forward
def origin_schedule(dates, cfg):
    """Rolling origins: each tests the next `step_dates` dates with a miner fitted on dates <= origin - gap."""
    ud = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    if cfg["gap_days"] < 0:
        raise AssertionError("look-ahead: gap_days must be >= 0")
    gap = pd.Timedelta(days=cfg["gap_days"])
    rows = []
    for i in range(cfg["min_train_dates"], len(ud), cfg["step_dates"]):
        origin = ud[i]
        train = ud[ud <= origin - gap]
        if cfg["train_max_dates"]:
            train = train[-cfg["train_max_dates"]:]
        test = ud[i:i + cfg["step_dates"]]
        if len(train) < cfg["min_train_dates"] // 2 or len(test) == 0:
            continue
        if not train.max() + gap <= test.min():
            raise AssertionError("look-ahead: training window reaches into the test block")   # cannot happen; guards edits
        rows.append({"origin": origin, "train_start": train.min(), "train_end": train.max(), "test_start": test.min(),
                     "test_end": test.max(), "n_train_dates": int(len(train)), "n_test_dates": int(len(test)),
                     "purge_days": int((test.min() - train.max()).days)})
    return pd.DataFrame(rows)


def _oos_pattern_check(bank, Xt, y_ex, dcode_t):
    """For every live pattern: its out-of-sample mean excess return when it fires, week-clustered t, and whether the
    sign of the mined effect held.  A live pattern that does not hold on its own next block is a false discovery."""
    rows = []
    for i, m in pm.fire_masks(bank, Xt):
        per = pd.Series(y_ex[m]).groupby(dcode_t[m]).agg(["mean", "size"])
        per = per[per["size"] >= 3]
        if len(per) < 3:
            rows.append({"name": bank.names[i], "effect": float(bank.effects[i]), "oos_mean": np.nan, "oos_t": np.nan,
                         "checked": False, "held": False, "wrong_sign": False})
            continue
        mu = float(per["mean"].mean()); t = float(mu / (per["mean"].std(ddof=1) / np.sqrt(len(per)) + 1e-18))
        sgn = np.sign(bank.effects[i])
        rows.append({"name": bank.names[i], "effect": float(bank.effects[i]), "oos_mean": mu, "oos_t": t, "checked": True,
                     "held": bool(np.sign(mu) == sgn and t * sgn >= 1.0), "wrong_sign": bool(np.sign(mu) != sgn)})
    return pd.DataFrame(rows, columns=["name", "effect", "oos_mean", "oos_t", "checked", "held", "wrong_sign"])


def _score_block(bank, Xt, y_t, cfg, thr, mov_bank, prev_top, rng):
    """Per-date measurements for one test block."""
    s_dir, _ = pm.score_panel(bank, Xt)
    s_mov = pm.score_panel(mov_bank, Xt)[0] if mov_bank is not None else None
    dcode, ud = pm._date_codes(Xt.index)
    sv = s_dir.to_numpy(); yv = y_t.to_numpy(float)
    yex = pm.demeaned(y_t).to_numpy(float)
    lab = np.abs(yv) >= thr
    order = np.argsort(dcode, kind="stable")
    tick = np.asarray(Xt.index.get_level_values(1))
    rows = []
    hits = 0
    for g in np.split(order, np.flatnonzero(np.diff(dcode[order])) + 1):
        if len(g) < cfg["min_rows_date"]:
            continue
        s, ye, y_ = sv[g], yex[g], yv[g]
        rank_ic = ic = 0.0                                # no view (all scores equal) is an IC of exactly zero
        if s.std() > 0 and ye.std() > 0:
            ic = float(np.corrcoef(s, ye)[0, 1])
            rank_ic = float(np.corrcoef(rankdata(s), rankdata(ye))[0, 1])
        k = min(cfg["topk"], len(g))
        pick = np.lexsort((rng.random(len(g)), -s))[:k]                 # random tie-break: most scores are equal
        names = frozenset(tick[g][pick])
        turn = 1 - len(names & prev_top) / k if prev_top is not None and len(prev_top) == k else np.nan
        prev_top = names
        row = {"date": ud[dcode[g[0]]], "n": int(len(g)), "ic": ic, "rank_ic": rank_ic,
               "top_excess": float(y_[pick].mean() - y_.mean()), "turnover": turn}
        if s_mov is not None:
            mv = s_mov.to_numpy()[g]
            row["mov_auc"] = pm.auc(mv, lab[g]) if mv.std() > 0 else np.nan
        # direction: among the date's realised movers where the pattern has a view
        m = lab[g] & (s != 0) & (ye != 0)
        row["dir_n"] = int(m.sum()); row["dir_hit"] = int((np.sign(s[m]) == np.sign(ye[m])).sum())
        rows.append(row)
    return rows, prev_top, dcode, yex


def walk_forward(X, y, cfg=None, seed=7, null=False, eras=None):
    """One seed's rolling-origin walk-forward -> {frame (one row per test date), origins, patterns}.
    null=True shuffles outcomes within each date first: the whole pipeline then has nothing to find, so whatever
    IC it reports is the harness's own false-positive level."""
    c = {**HEAVY_DEFAULT, **(cfg or {})}
    c["miner"] = {**HEAVY_DEFAULT["miner"], **((cfg or {}).get("miner") or {})}
    rng = np.random.default_rng([seed, 4242])
    dts = X.index.get_level_values(0)
    if null:
        dc, _ = pm._date_codes(X.index)
        y = pd.Series(pm.shuffle_within_dates(y.to_numpy(float), dc, np.random.default_rng([seed, 999])), y.index)
    sched = origin_schedule(dts.unique(), c)
    cc = pm.ctx_cols_of(X)
    frames, org_rows, pat_rows, prev_top, prev_names = [], [], [], None, None
    for _, o in sched.iterrows():
        in_tr = (dts >= o["train_start"]) & (dts <= o["train_end"]) & y.notna().to_numpy()
        in_te = (dts >= o["test_start"]) & (dts <= o["test_end"]) & y.notna().to_numpy()
        Xtr, ytr, Xte, yte = X[in_tr], y[in_tr], X[in_te], y[in_te]
        if len(Xte) < c["min_rows_date"] or len(Xtr) < 1000:
            continue
        p = {**c["miner"], "seed": seed}
        M = PatternMiner(p).fit(Xtr, pm.demeaned(ytr), now=o["train_end"])
        bank = pm.bank_from_miner(M, cc)
        mov_bank = None
        if c["movement"]:
            Mm = PatternMiner(p).fit(Xtr, pm.demeaned_abs(ytr), now=o["train_end"])
            mov_bank = pm.bank_from_miner(Mm, cc)
        thr = float(np.quantile(ytr.abs().to_numpy(), c["mover_q"]))
        rows, prev_top, dcode, yex = _score_block(bank, Xte, yte, c, thr, mov_bank, prev_top, rng)
        for r in rows:
            r.update(origin=o["origin"], seed=seed, age_weeks=(r["date"] - o["train_end"]).days / 7)
        frames.extend(rows)
        chk = _oos_pattern_check(bank, Xte, yex, dcode)
        names = set(bank.names)
        surv = np.nan if prev_names is None or not prev_names else len(prev_names & names) / len(prev_names)
        union = len(prev_names | names) if prev_names is not None else 0
        org_rows.append({**o.to_dict(), "seed": seed, "n_live": len(bank), "n_live_movement": 0 if mov_bank is None else len(mov_bank),
                         "tested": M.report.get("tested", 0), "fdr_pass": M.report.get("fdr_pass", 0),
                         "null_t_95": M.report.get("null_t_95pct", np.nan), "real_t_95": M.report.get("real_t_95pct", np.nan),
                         "gate_corr": M.report.get("gate_corr_confirm", np.nan),
                         "survival_next": np.nan, "jaccard_prev": (len(prev_names & names) / union) if union else np.nan,
                         "kept_from_prev": surv, "oos_checked": int(chk["checked"].sum()), "oos_held": int(chk["held"].sum()),
                         "oos_wrong_sign": int(chk["wrong_sign"].sum())})
        prev_names = names
        chk["origin"], chk["seed"] = o["origin"], seed
        pat_rows.append(chk)
    frame = pd.DataFrame(frames)
    origins = pd.DataFrame(org_rows)
    if len(origins):
        origins["survival_next"] = origins["kept_from_prev"].shift(-1)       # of this origin's live set, share still live next
    if len(frame):
        frame["era"] = era_of(frame["date"], eras).to_numpy()
        frame["cost"] = frame["turnover"] * 2 * c["cost_bps"] / 1e4
        frame["top_net"] = frame["top_excess"] - frame["cost"].fillna(0)
    pats = pd.concat(pat_rows, ignore_index=True) if pat_rows else pd.DataFrame()
    return {"frame": frame, "origins": origins, "patterns": pats, "seed": seed, "null": null, "cfg": c}


# ------------------------------------------------------------------ tables cut from the walk-forward frame
def _agg_row(f, c):
    ic = series_stats(f["rank_ic"], c["nw_lag"])
    dn = f["dir_n"].sum() if "dir_n" in f else 0
    row = {"dates": int(f["rank_ic"].notna().sum()), "rank_ic": ic["mean"], "rank_ic_t": ic["t"], "rank_ic_t_nw": ic["t_nw"],
           "ic_pos_share": ic["pos_share"], "ic": float(f["ic"].mean()),
           "top_excess": float(f["top_excess"].mean()), "top_net": float(f["top_net"].mean()),
           "turnover": float(f["turnover"].mean()), "cost": float(f["cost"].mean()),
           "dir_acc": float(f["dir_hit"].sum() / dn) if dn >= 30 else np.nan, "dir_n": int(dn)}
    if "mov_auc" in f:
        row["mov_auc"] = float(f["mov_auc"].mean())
    return row


def by_era(frame, cfg=None):
    c = {**HEAVY_DEFAULT, **(cfg or {})}
    rows = []
    for era, f in frame.groupby("era", sort=False):
        r = _agg_row(f, c); r["era"] = era
        r["enough"] = r["dates"] >= c["min_era_dates"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("era") if rows else pd.DataFrame()


def by_regime(frame, tags, cfg=None):
    """IC within each regime group (bull/bear, high/low vol, high/low dispersion).  Regime sensitivity per family =
    widest IC gap between its groups that have enough dates."""
    c = {**HEAVY_DEFAULT, **(cfg or {})}
    rows = []
    fr = frame.merge(tags, left_on="date", right_index=True, how="left")
    for fam in tags.columns:
        for grp, f in fr.groupby(fr[fam].fillna("unk")):
            if grp == "unk":
                continue
            r = _agg_row(f, c); r.update(family=fam, group=grp, enough=r["dates"] >= c["min_era_dates"])
            rows.append(r)
    if not rows:
        return pd.DataFrame(), {}
    T = pd.DataFrame(rows).set_index(["family", "group"])
    gap = {}
    for fam, t in T.groupby(level=0):
        t = t[t["enough"]]
        gap[fam] = float(t["rank_ic"].max() - t["rank_ic"].min()) if len(t) >= 2 else np.nan
    return T, gap


def by_age(frame, edges=(0, 4, 13, 26, 52, 1e9)):
    """IC by weeks since the miner was fitted, plus the fitted decay line."""
    lab = [f"{int(a)}-{int(b) if b < 1e8 else 'inf'}w" for a, b in zip(edges[:-1], edges[1:])]
    g = pd.cut(frame["age_weeks"], edges, labels=lab, right=False)
    t = frame.groupby(g, observed=True).agg(dates=("rank_ic", "count"), rank_ic=("rank_ic", "mean"), top_net=("top_net", "mean"))
    return t, decay_fit(frame["age_weeks"], frame["rank_ic"])


def hidden_windows(frame, n, length, seed):
    """n random contiguous windows of `length` OOS dates.  The walk-forward already guarantees the miner never saw
    these dates; the windows only show how wide the spread of outcomes is when one is drawn blind."""
    dates = np.sort(frame["date"].unique())
    if len(dates) < length or n <= 0:
        return pd.DataFrame(columns=["start", "end", "dates", "rank_ic", "t_nw", "top_net"])
    rng = np.random.default_rng([seed, 31337])
    rows = []
    for s in rng.integers(0, len(dates) - length + 1, size=n):
        w = frame[(frame["date"] >= dates[s]) & (frame["date"] <= dates[s + length - 1])]
        st = series_stats(w["rank_ic"])
        rows.append({"start": str(pd.Timestamp(dates[s]).date()), "end": str(pd.Timestamp(dates[s + length - 1]).date()),
                     "dates": st["n"], "rank_ic": st["mean"], "t_nw": st["t_nw"], "top_net": float(w["top_net"].mean())})
    return pd.DataFrame(rows)


def cost_sensitivity(frame, grid_bps):
    """Top-k excess return net of cost at several per-side costs, from the measured turnover."""
    t = frame["turnover"].fillna(0).to_numpy()
    return pd.DataFrame({"cost_bps": list(grid_bps), "net": [float((frame["top_excess"].to_numpy() - t * 2 * b / 1e4).mean()) for b in grid_bps]})


def false_discovery_summary(origins, patterns):
    """What survives: how many patterns were live, how many failed their own next block, how many are still live at
    the next refit, and how strong the search's own noise ceiling was relative to the best real pattern."""
    if origins.empty:
        return {}
    chk = int(origins["oos_checked"].sum())
    held = int(origins["oos_held"].sum())
    return {"live_per_origin_mean": float(origins["n_live"].mean()), "tested_per_origin_mean": float(origins["tested"].mean()),
            "fdr_pass_per_origin_mean": float(origins["fdr_pass"].mean()),
            "oos_checked": chk, "oos_held": held, "oos_false_rate": float(1 - held / chk) if chk else np.nan,
            "oos_wrong_sign_rate": float(origins["oos_wrong_sign"].sum() / chk) if chk else np.nan,
            "survival_next_mean": float(origins["survival_next"].mean()) if origins["survival_next"].notna().any() else np.nan,
            "jaccard_prev_mean": float(origins["jaccard_prev"].mean()) if origins["jaccard_prev"].notna().any() else np.nan,
            "noise_ceiling_ratio": float((origins["null_t_95"] / origins["real_t_95"]).mean())}


# ------------------------------------------------------------------ stability, lifetimes, comparisons
def stability(frame, window=26):
    """How steady the edge is through time, not just how big: information ratio of the daily rank IC, its lag-1
    autocorrelation, the share of rolling windows with a positive mean, the worst rolling window, and the maximum
    drawdown of cumulative top-k net excess return."""
    ic = frame.sort_values("date")["rank_ic"].to_numpy(float)
    ic = ic[np.isfinite(ic)]
    if len(ic) < max(10, window // 2):
        return {"icir": np.nan, "ic_autocorr": np.nan, "rolling_pos_share": np.nan, "worst_window_ic": np.nan, "max_drawdown": np.nan}
    roll = pd.Series(ic).rolling(min(window, len(ic))).mean().dropna()
    net = frame.sort_values("date")["top_net"].fillna(0).to_numpy(float)
    cum = np.cumsum(net)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    ac = float(np.corrcoef(ic[:-1], ic[1:])[0, 1]) if ic.std() > 0 else np.nan
    return {"icir": float(ic.mean() / ic.std(ddof=1)) if ic.std() > 0 else np.nan, "ic_autocorr": ac,
            "rolling_pos_share": float((roll > 0).mean()), "worst_window_ic": float(roll.min()), "max_drawdown": dd}


def pattern_lifetimes(patterns):
    """For each seed, how many consecutive refits each pattern name stayed live (bounded below by 1) -> distribution
    summary.  Patterns that come and go every refit are the signature of overfitting the noise."""
    if patterns is None or len(patterns) == 0:
        return {"patterns": 0, "median_life": np.nan, "share_multi_origin": np.nan, "max_life": 0}
    life = []
    for _, g in patterns.groupby("seed"):
        origins = sorted(g["origin"].unique())
        pos = {o: i for i, o in enumerate(origins)}
        for _, h in g.groupby("name"):
            idx = sorted(pos[o] for o in h["origin"])
            run = best = 1
            for a, b in zip(idx[:-1], idx[1:]):
                run = run + 1 if b == a + 1 else 1
                best = max(best, run)
            life.append(best)
    life = np.array(life)
    return {"patterns": int(len(life)), "median_life": float(np.median(life)), "share_multi_origin": float((life > 1).mean()),
            "max_life": int(life.max())}


def invented_ratio(origins_real, null_live_mean):
    """Live patterns per origin the search reports on shuffled outcomes, relative to the real run.  Near 1 means the
    miner reports as many 'patterns' from noise as from data: its count says nothing about what is real."""
    real = float(origins_real["n_live"].mean()) if len(origins_real) else np.nan
    return float(null_live_mean / real) if real and np.isfinite(real) and real > 0 else np.nan


def era_confidence(frame, cfg=None, n=300, seed=0):
    """Block-bootstrap 90% interval of the rank IC inside each era (era means alone hide how noisy a short era is)."""
    c = {**HEAVY_DEFAULT, **(cfg or {})}
    rng = np.random.default_rng([seed, 77])
    rows = []
    for era, f in frame.groupby("era", sort=False):
        m, lo, hi, p0 = pm.block_bootstrap_mean(f.sort_values("date")["rank_ic"].to_numpy(), c["nw_lag"] + 1, n, rng)
        rows.append({"era": era, "rank_ic": m, "lo5": lo, "hi95": hi, "p_le_0": p0})
    return pd.DataFrame(rows).set_index("era") if rows else pd.DataFrame()


def compare_reports(previous, current, tol=0.5):
    """Regression check between two runs (previous = a report JSON dict, current = a summary): list what got worse.
    A gate that flipped to fail, or an era/IC that fell by more than `tol` of its previous size, is flagged."""
    flags = []
    pv, cv = previous.get("verdict", {}).get("gates", {}), current["verdict"]["gates"]
    for k, g in cv.items():
        if k in pv and pv[k].get("pass") and not g["pass"]:
            flags.append(f"gate {k} passed before and fails now")
    p_ic, c_ic = previous.get("overall", {}).get("mean"), current["overall"]["mean"]
    if p_ic and np.isfinite(c_ic) and p_ic > 0 and c_ic < p_ic * (1 - tol):
        flags.append(f"overall rank IC fell from {p_ic:.4f} to {c_ic:.4f}")
    prev_era = {r["era"]: r.get("rank_ic") for r in previous.get("by_era", [])}
    for era, row in current["by_era"].iterrows():
        po = prev_era.get(era)
        if po is not None and po > 0 and np.isfinite(row["rank_ic"]) and row["rank_ic"] < po * (1 - tol):
            flags.append(f"era {era} rank IC fell from {po:.4f} to {row['rank_ic']:.4f}")
    return flags


# ------------------------------------------------------------------ many seeds, the null, the summary
def _combine_seeds(runs):
    """One row per date: the mean over seeds of each date's measurement (not stacked, so seeds are not treated as
    independent evidence)."""
    f = pd.concat([r["frame"] for r in runs], ignore_index=True)
    num = ["n", "ic", "rank_ic", "top_excess", "turnover", "mov_auc", "dir_n", "dir_hit", "age_weeks", "cost", "top_net"]
    num = [c for c in num if c in f]
    out = f.groupby("date", sort=True).agg({**{c: "mean" for c in num}, "era": "first"}).reset_index()
    out["dir_n"], out["dir_hit"] = out["dir_n"].round(), out["dir_hit"].round()
    return out


def run_heavy(X, y, cfg=None, seeds=(7, 11, 13), null_seeds=(101,), n_hidden=20, hidden_len=26, eras=None,
              gates=None, out_dir=None, log=False):
    """The full Phase 7 run.  Returns a summary dict with every table and the verdict; optionally writes the
    machine- and human-readable report and logs the experiment."""
    c = {**HEAVY_DEFAULT, **(cfg or {})}
    c["miner"] = {**HEAVY_DEFAULT["miner"], **((cfg or {}).get("miner") or {})}
    diag = validate_panel(X, y)
    runs = [walk_forward(X, y, c, s, eras=eras) for s in seeds]
    runs = [r for r in runs if len(r["frame"])]
    if not runs:
        return {"ok": False, "reason": "no origin had enough history", "diagnostics": diag, "verdict": {"pass": False, "gates": {}}}
    frame = _combine_seeds(runs)
    tags = regime_tags(X)
    s_all = series_stats(frame["rank_ic"], c["nw_lag"])
    seed_rows = []
    for r in runs:
        st = series_stats(r["frame"]["rank_ic"], c["nw_lag"])
        seed_rows.append({"seed": r["seed"], "rank_ic": st["mean"], "t_nw": st["t_nw"], "live_mean": float(r["origins"]["n_live"].mean()),
                          "top_net": float(r["frame"]["top_net"].mean())})
    seed_tbl = pd.DataFrame(seed_rows)
    origins = pd.concat([r["origins"] for r in runs], ignore_index=True)
    patterns = pd.concat([r["patterns"] for r in runs], ignore_index=True)
    era_tbl = by_era(frame, c)
    reg_tbl, reg_gap = by_regime(frame, tags, c)
    age_tbl, decay = by_age(frame)
    hidden = hidden_windows(frame, n_hidden, hidden_len, seeds[0])
    nulls = []
    for ns in null_seeds:
        nr = walk_forward(X, y, c, ns, null=True, eras=eras)
        if len(nr["frame"]):
            st = series_stats(nr["frame"]["rank_ic"], c["nw_lag"])
            nulls.append({"seed": ns, "rank_ic": st["mean"], "t_nw": st["t_nw"], "live_mean": float(nr["origins"]["n_live"].mean())})
    null_tbl = pd.DataFrame(nulls)
    fdr = false_discovery_summary(origins, patterns)
    if len(null_tbl):
        fdr["invented_ratio"] = invented_ratio(origins, float(null_tbl["live_mean"].mean()))
    summary = {"ok": True, "diagnostics": diag, "config": c, "seeds": list(seeds), "n_origins": int(origins["origin"].nunique()),
               "overall": {**s_all, "top_excess": float(frame["top_excess"].mean()), "top_net": float(frame["top_net"].mean()),
                           "turnover": float(frame["turnover"].mean()), "cost": float(frame["cost"].mean()),
                           "mov_auc": float(frame["mov_auc"].mean()) if "mov_auc" in frame else np.nan,
                           "dir_acc": float(frame["dir_hit"].sum() / frame["dir_n"].sum()) if frame["dir_n"].sum() >= 30 else np.nan},
               "by_era": era_tbl, "by_regime": reg_tbl, "regime_gap": reg_gap, "by_age": age_tbl, "decay": decay,
               "hidden": hidden, "era_ci": era_confidence(frame, c), "stability": stability(frame),
               "lifetimes": pattern_lifetimes(patterns), "cost_sensitivity": cost_sensitivity(frame, c["cost_grid_bps"]), "seed_table": seed_tbl,
               "null": null_tbl, "false_discovery": fdr, "origins": origins, "frame": frame}
    summary["verdict"] = verdict(summary, gates)
    summary["provenance"] = provenance_block(c, seeds, X, y)
    if out_dir is not None:
        write_report(summary, out_dir)
    if log:
        from .improve import log_experiment
        log_experiment({"event": "heavy_algo_test", "run_id": summary["provenance"]["run_id"],
                        "rank_ic": summary["overall"]["mean"], "t_nw": summary["overall"]["t_nw"],
                        "pass": summary["verdict"]["pass"]}, cfg=c, seed=seeds[0])
    return summary


def verdict(summary, gates=None):
    """Every gate must pass.  A high average IC with one dead era, a regime that flips the sign, a noise control that
    scores as well, or patterns that do not survive a refit fails the run: average return is never enough."""
    g = {**GATES_DEFAULT, **(gates or {})}
    o = summary["overall"]; era = summary["by_era"]; res = {}
    def gate(name, ok, value, need):
        res[name] = {"pass": bool(ok), "value": None if value is None or (isinstance(value, float) and np.isnan(value)) else value, "need": need}
    gate("mean_ic", o["mean"] >= g["min_ic"], o["mean"], f">= {g['min_ic']}")
    gate("t_newey_west", (o["t_nw"] if np.isfinite(o["t_nw"]) else -9) >= g["min_t_nw"], o["t_nw"], f">= {g['min_t_nw']}")
    ok_eras = era[era["enough"]] if len(era) else era
    share = float((ok_eras["rank_ic"] > 0).mean()) if len(ok_eras) else np.nan
    gate("era_positive_share", np.isfinite(share) and share >= g["min_pos_era_share"], share, f">= {g['min_pos_era_share']}")
    worst = float(ok_eras["rank_ic"].min()) if len(ok_eras) else np.nan
    gate("worst_era_ic", np.isfinite(worst) and worst >= g["min_worst_era_ic"], worst, f">= {g['min_worst_era_ic']}")
    fd = summary["false_discovery"]
    gate("oos_false_rate", np.isfinite(fd.get("oos_false_rate", np.nan)) and fd["oos_false_rate"] <= g["max_oos_false_rate"],
         fd.get("oos_false_rate"), f"<= {g['max_oos_false_rate']}")
    gate("net_spread", o["top_net"] > g["min_net_spread"], o["top_net"], f"> {g['min_net_spread']}")
    st = summary["seed_table"]
    agree = float((st["rank_ic"] > 0).mean()) if len(st) else np.nan
    gate("seed_agreement", np.isfinite(agree) and agree >= g["min_seed_agree"], agree, f">= {g['min_seed_agree']}")
    surv = fd.get("survival_next_mean", np.nan)
    gate("pattern_survival", np.isfinite(surv) and surv >= g["min_survival"], surv, f">= {g['min_survival']}")
    gaps = [v for v in summary["regime_gap"].values() if np.isfinite(v)]
    gate("regime_sensitivity", (max(gaps) <= g["max_regime_gap"]) if gaps else False, max(gaps) if gaps else None, f"<= {g['max_regime_gap']}")
    nl = summary["null"]
    if len(nl):
        beats = bool(o["mean"] > nl["rank_ic"].max() and o["t_nw"] > nl["t_nw"].abs().max())
        gate("beats_noise_control", beats, float(nl["t_nw"].abs().max()), "real IC and |t| above the shuffled-outcome run")
    else:
        gate("beats_noise_control", False, None, "a shuffled-outcome control must be run")
    failed = [k for k, v in res.items() if not v["pass"]]
    return {"pass": not failed, "failed": failed, "gates": res}


# ------------------------------------------------------------------ provenance and reports
def data_fingerprint(X, y):
    h = hashlib.sha256()
    h.update(str((X.shape, list(X.columns))).encode())
    h.update(pd.util.hash_pandas_object(y, index=True).to_numpy().tobytes()[:1 << 20])
    return h.hexdigest()[:16]


def provenance_block(cfg, seeds, X, y):
    from .provenance import stamp, config_hash
    st = stamp(cfg, list(seeds))
    st["panel_fingerprint"] = data_fingerprint(X, y)
    st["run_id"] = "H" + config_hash({"cfg": cfg, "seeds": list(seeds), "panel": st["panel_fingerprint"], "code": st["code_hash"]})
    return st


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return json.loads(o.reset_index().to_json(orient="records", date_format="iso"))
    if isinstance(o, pd.Series):
        return json.loads(o.to_json(date_format="iso"))
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.4f}" if isinstance(v, float) else str(v)


def _md_table(df, cols=None):
    if df is None or len(df) == 0:
        return "(none)\n"
    d = df.reset_index()
    cols = cols or list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) if not isinstance(r[c], (str, bool)) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def render_markdown(s):
    if not s.get("ok"):
        return f"# Heavy algorithm test\n\nDid not run: {s.get('reason')}\n"
    p, o, v = s["provenance"], s["overall"], s["verdict"]
    L = [f"# Heavy algorithm test {p['run_id']}", "",
         f"code {p['code_hash']}  git {p['git_commit']}  canon {p['canon_sha']}  bible {p['bible_sha']}  blueprint {p['blueprint_version']}",
         f"config {p['config_hash']}  data {p['data_snapshot']}  panel {p['panel_fingerprint']}  seeds {s['seeds']}", "",
         f"Panel: {s['diagnostics']['rows']:,} rows, {s['diagnostics']['dates']} dates, {s['diagnostics']['first']} to {s['diagnostics']['last']}; "
         f"{s['n_origins']} walk-forward origins.", "",
         f"## Verdict: {'PASS' if v['pass'] else 'FAIL'}", ""]
    L += [f"- {k}: {'pass' if g['pass'] else 'FAIL'} (value {_fmt(g['value'])}, need {g['need']})" for k, g in v["gates"].items()]
    L += ["", "## Overall (out of sample)", "",
          f"rank IC {_fmt(o['mean'])}, t {_fmt(o['t'])}, Newey-West t {_fmt(o['t_nw'])}, positive-date share {_fmt(o['pos_share'])}",
          f"top-{s['config']['topk']} excess {_fmt(o['top_excess'])}/date gross, {_fmt(o['top_net'])} net; turnover {_fmt(o['turnover'])}, "
          f"cost {_fmt(o['cost'])}; movement AUC {_fmt(o['mov_auc'])}; direction accuracy {_fmt(o['dir_acc'])}", "",
          "## By era", "", _md_table(s["by_era"]), "## By regime", "", _md_table(s["by_regime"]),
          f"Regime gaps (widest IC gap within a family): {s['regime_gap']}", "",
          "Era intervals (block bootstrap):", "", _md_table(s["era_ci"]), f"Stability: {_jsonable(s['stability'])}",
          f"Pattern lifetimes: {_jsonable(s['lifetimes'])}", "",
          "## Decay with model age", "", _md_table(s["by_age"]), f"Fit: {s['decay']}", "",
          "## False discoveries and survival", "", json.dumps(_jsonable(s["false_discovery"]), indent=1), "",
          "## Costs", "", _md_table(s["cost_sensitivity"]), "## Seeds", "", _md_table(s["seed_table"]),
          "## Noise control (outcomes shuffled within each date)", "", _md_table(s["null"]),
          "## Random hidden windows", "", _md_table(s["hidden"]),
          f"Positive-IC share of hidden windows: {_fmt(float((s['hidden']['rank_ic'] > 0).mean())) if len(s['hidden']) else 'n/a'}", ""]
    return "\n".join(L)


def write_report(summary, out_dir):
    """Machine-readable JSON + human-readable Markdown (Phase 36).  Returns the two paths."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rid = summary.get("provenance", {}).get("run_id", "H_failed")
    slim = {k: v for k, v in summary.items() if k not in ("frame",)}
    jp, mp = out / f"{rid}.json", out / f"{rid}.md"
    jp.write_text(json.dumps(_jsonable(slim), indent=1), encoding="utf-8")
    mp.write_text(render_markdown(summary), encoding="utf-8")
    return jp, mp


# ------------------------------------------------------------------ ablation across eras (Phase 6 x Phase 7)
def ablation_across_eras(X, y, eras=None, cfg=None, seed=7, min_train_dates=104):
    """Run the six-arm mover ablation with each era as the test window (trained on everything before it).  A pattern
    family that only beats its controls in one era is an era effect, not a pattern."""
    eras = eras or ERAS
    ud = pd.DatetimeIndex(sorted(X.index.get_level_values(0).unique()))
    rows, results = [], {}
    for name, (a, b) in eras.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        first = ud[ud >= a]
        if len(first) == 0 or (ud < a).sum() < min_train_dates:
            rows.append({"era": name, "ran": False, "why": "not enough history before the era"}); continue
        r = pm.run_ablation(X, y, first[0], cfg, seed, test_end=b + pd.Timedelta(days=1))
        results[name] = r
        if not r.get("ok"):
            rows.append({"era": name, "ran": False, "why": r.get("reason")}); continue
        rows.append({"era": name, "ran": True, "why": "", "base": r["arms"]["base"]["auc_daily"],
                     "base+pattern": r["arms"]["base+pattern"]["auc_daily"], "gain": r["gain_vs_base"]["base+pattern"],
                     "best_control": max(v for k, v in r["gain_vs_base"].items() if "#" in k or k.endswith("noise_mined")),
                     "deploy": r["deploy"]})
    return pd.DataFrame(rows).set_index("era"), results
