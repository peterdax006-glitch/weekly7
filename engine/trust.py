"""Per-stock-type trust tables (Bible PHASE 12; canons: never invent reliability from thin data, week-clustered
statistics, shrinkage to parent, recent relevance).

A stock TYPE is a label built from SIC division, size / volatility terciles, trend state, theme momentum and attention
state. Terciles are cut per date across that date's cross-section only (no look-ahead, no global cut-points).
Types nest: ALL > sic > sic|size > sic|size|vol > +trend > +theme|attention. For every (type, indicator) we estimate
the indicator's reliability = mean of z(indicator) * z(forward return) (an IC-like covariance), from WEEK-CLUSTERED cells
(one observation per type per week, so many stocks in one week are not counted as independent), weighted by recency,
then shrunk to the parent type's posterior with a normal-normal empirical-Bayes update. A cell that lacks support
(weeks, observations) or whose posterior is not distinguishable from zero is NEUTRALIZED: weight 0, reason recorded.
Time discipline: rows are used only if date + horizon <= now (their forward return was public by `now`)."""
import json

import numpy as np
import pandas as pd
from scipy import stats

LEVELS = ["ALL", "sic", "sic|size", "sic|size|vol", "sic|size|vol|trend", "sic|size|vol|trend|theme|attn"]
NUMERIC = {"size": "size", "vol": "vol", "trend": "trend", "theme": "theme_mom", "attn": "attention"}
LABELS3 = {"size": ("S", "M", "L"), "vol": ("lowvol", "midvol", "highvol"),
           "trend": ("down", "flat", "up"), "theme": ("cold", "neutral", "hot"), "attn": ("quiet", "normal", "hyped")}

# SEC SIC divisions by 2-digit major group
_DIV = [(1, 9, "A"), (10, 14, "B"), (15, 17, "C"), (20, 39, "D"), (40, 49, "E"),
        (50, 51, "F"), (52, 59, "G"), (60, 67, "H"), (70, 89, "I"), (91, 99, "J")]


def sic_division(sic):
    """SIC code (int, str or NaN) -> division letter A-J, or 'NA' when unknown/invalid. Accepts 2-4 digit codes."""
    try:
        s = int(float(sic))
    except (TypeError, ValueError, OverflowError):
        return "NA"
    if s <= 0:
        return "NA"
    major = s // 100 if s >= 100 else s
    for lo, hi, d in _DIV:
        if lo <= major <= hi:
            return d
    return "NA"


def _tercile_labels(v: pd.Series, labels):
    """Per-date tercile by rank; NaN -> 'NA'. Needs >= 3 valid values that day, else everything is 'NA'."""
    out = pd.Series("NA", index=v.index, dtype=object)
    ok = v.notna()
    if ok.sum() < 3:
        return out
    r = v[ok].rank(method="first")
    q = np.ceil(r / ok.sum() * 3).clip(1, 3).astype(int)
    out[ok] = q.map({1: labels[0], 2: labels[1], 3: labels[2]}).values
    return out


def assign_types(info: pd.DataFrame) -> pd.DataFrame:
    """info: index (date, ticker); columns sic (or sic_div), size, vol, trend, theme_mom, attention (any may be
    missing -> 'NA'). Returns facet labels plus one string key per nesting level (columns named as LEVELS)."""
    if not isinstance(info.index, pd.MultiIndex) or info.index.nlevels != 2:
        raise ValueError("info must be indexed by (date, ticker)")
    L = pd.DataFrame(index=info.index)
    if "sic_div" in info:
        L["sic"] = info["sic_div"].fillna("NA").astype(str)
    elif "sic" in info:
        L["sic"] = info["sic"].map(sic_division)
    else:
        L["sic"] = "NA"
    date = info.index.get_level_values(0)
    for facet, col in NUMERIC.items():
        if col not in info:
            L[facet] = "NA"
            continue
        L[facet] = info[col].groupby(date).transform(lambda s, f=facet: _tercile_labels(s, LABELS3[f]))
    L["ALL"] = "ALL"
    L["sic|size"] = L["sic"] + "|" + L["size"]
    L["sic|size|vol"] = L["sic|size"] + "|" + L["vol"]
    L["sic|size|vol|trend"] = L["sic|size|vol"] + "|" + L["trend"]
    L["sic|size|vol|trend|theme|attn"] = L["sic|size|vol|trend"] + "|" + L["theme"] + "|" + L["attn"]
    return L


def parent_key(key: str):
    """'D|L|highvol' -> 'D|L'; 'D' -> 'ALL'; 'ALL' -> None. The six-part key drops theme and attention together."""
    if key == "ALL":
        return None
    parts = key.split("|")
    if len(parts) == 1:
        return "ALL"
    if len(parts) == 6:
        return "|".join(parts[:4])
    return "|".join(parts[:-1])


def _zscore_by_date(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(level=0)
    sd = g.transform("std").replace(0, np.nan)
    return (df - g.transform("mean")) / sd


def _week_label(dates) -> np.ndarray:
    iso = pd.DatetimeIndex(dates).isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).values


def _weighted_mean_se(vals: np.ndarray, w: np.ndarray):
    """Recency-weighted mean and its standard error using the Kish effective n (reliability weights)."""
    sw = w.sum()
    m = (w * vals).sum() / sw
    n_eff = sw ** 2 / (w ** 2).sum()
    if n_eff <= 1.5:
        return m, np.inf, n_eff
    var = (w * (vals - m) ** 2).sum() / sw * n_eff / (n_eff - 1)
    return m, float(np.sqrt(var / n_eff)), n_eff


class TrustTable:
    def __init__(self, min_weeks=26, min_obs=200, half_life_weeks=104, tau=0.03, z_min=1.645, min_per_week=3,
                 horizon_days=7):
        self.p = dict(min_weeks=min_weeks, min_obs=min_obs, half_life_weeks=half_life_weeks, tau=tau, z_min=z_min,
                      min_per_week=min_per_week, horizon_days=horizon_days)
        self.table = self._empty()
        self.now = None
        self.indicators = []
        self._idx = {}
        self._zbar = z_min

    @staticmethod
    def _empty():
        return pd.DataFrame(columns=["level", "type", "indicator", "raw", "se", "n_weeks", "n_eff", "n_obs", "post",
                                     "post_sd", "z", "confidence", "reliable", "reason"])

    def _reindex(self):
        self._idx = {(r.type, r.indicator): i for i, r in enumerate(self.table.itertuples())}

    # ---- fitting -------------------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, info: pd.DataFrame, now, indicators=None):
        """X: (date,ticker) x indicator columns; y: forward return on the same index; info as in assign_types."""
        now = pd.Timestamp(now)
        self.now = now
        ind = list(indicators) if indicators is not None else [c for c in X.columns if not c.startswith("m_")]
        self.indicators = ind
        self.table, self._idx = self._empty(), {}
        if not ind or len(X) == 0:
            return self
        dates = pd.DatetimeIndex(X.index.get_level_values(0))
        keep = (dates + pd.Timedelta(days=self.p["horizon_days"]) <= now) & (dates <= now)
        X = X.loc[keep, ind]
        y = y.reindex(X.index)
        ok = y.notna().values
        X, y = X[ok], y[ok]
        if len(X) == 0:
            return self
        info = info.reindex(X.index)
        prod = _zscore_by_date(X).mul(_zscore_by_date(y.to_frame("y"))["y"], axis=0)
        labels = assign_types(info)
        weeks = _week_label(X.index.get_level_values(0))
        age = {}
        for k in np.unique(weeks):
            monday = pd.Timestamp.fromisocalendar(int(k) // 100, int(k) % 100, 1)
            age[k] = max(0.0, (now - monday).days / 7.0)
        rows, post, tested = [], {}, 0
        for lvl in LEVELS:
            keys = labels[lvl].values
            g = prod.groupby([keys, weeks])
            cell = g.mean()
            cell["_n"] = g.size()
            cell = cell[cell["_n"] >= self.p["min_per_week"]]
            if cell.empty:
                continue
            obs = prod.notna().groupby(keys).sum()
            stat = []                      # (typ, indicator, raw, se, n_eff, n_weeks, n_obs)
            for typ, sub in cell.groupby(level=0):
                wk = sub.index.get_level_values(1).values
                w = 0.5 ** (np.array([age[k] for k in wk]) / self.p["half_life_weeks"])
                for c in ind:
                    v = sub[c].values
                    m = ~np.isnan(v)
                    if m.any():
                        raw, se, n_eff = _weighted_mean_se(v[m], w[m])
                        stat.append((typ, c, raw, se, n_eff, int(m.sum()), int(obs.loc[typ, c])))
            tau2 = self._level_tau2(stat, post)
            tested += len({r[0] for r in stat})
            # multiplicity: every type at this and coarser levels is a test on the same indicator, so the bar rises
            # with the running count (Bonferroni, 5% family-wise two-sided)
            self._zbar = max(self.p["z_min"], float(stats.norm.isf(0.025 / max(tested, 1))))
            for typ, c, raw, se, n_eff, n_weeks, n_obs in stat:
                rows.append(self._row(lvl, typ, c, raw, se, n_eff, n_weeks, n_obs, post, tau2.get(c)))
        if rows:
            self.table = pd.DataFrame(rows)
            self._reindex()
        return self

    def _level_tau2(self, stat, post) -> dict:
        """Method-of-moments spread of children around their parent, per indicator: var(raw - parent) - mean(se^2).
        With < 4 children or a non-positive estimate we fall back to the configured tau (a floor, never below it)."""
        floor = self.p["tau"] ** 2
        out = {}
        by = {}
        for typ, c, raw, se, *_ in stat:
            par = parent_key(typ)
            if par is not None and np.isfinite(se):
                by.setdefault(c, []).append((raw - post.get((par, c), (0.0, 0))[0], se ** 2))
        for c, lst in by.items():
            if len(lst) >= 4:
                d, v = np.array(lst).T
                out[c] = max(floor, float(np.mean(d ** 2) - np.mean(v)))
        return out

    def _row(self, lvl, typ, ind, raw, se, n_eff, n_weeks, n_obs, post, tau2=None):
        """Empirical-Bayes update toward the parent's posterior; levels are processed coarse -> fine.
        Reliable only if the cell's OWN data clear the z bar too: a parent's evidence never certifies a child."""
        tau2 = self.p["tau"] ** 2 if tau2 is None else tau2
        par = parent_key(typ)
        if par is None:
            pm, prior_var = 0.0, tau2
        else:
            pm, pv = post.get((par, ind), (0.0, tau2))
            prior_var = tau2 + pv          # child deviates ~tau from parent, and the parent itself is uncertain
        if not np.isfinite(se) or se <= 0:
            pmean, pvar = pm, prior_var    # no usable data: inherit, unearned
        else:
            prec = 1 / se ** 2 + 1 / prior_var
            pmean, pvar = (raw / se ** 2 + pm / prior_var) / prec, 1 / prec
        post[(typ, ind)] = (pmean, pvar)
        psd = float(np.sqrt(pvar))
        z = pmean / psd
        reason = ""
        if n_weeks < self.p["min_weeks"]:
            reason = f"support: {n_weeks} weeks < {self.p['min_weeks']}"
        elif n_obs < self.p["min_obs"]:
            reason = f"support: {n_obs} obs < {self.p['min_obs']}"
        elif not np.isfinite(se):
            reason = "no variance estimate"
        elif abs(z) < self._zbar:
            reason = f"indistinguishable from zero (|z|={abs(z):.2f})"
        elif abs(raw / se) < self._zbar:
            reason = f"own data indistinguishable from zero (|z|={abs(raw / se):.2f}); parent evidence does not transfer"
        return dict(level=lvl, type=typ, indicator=ind, raw=raw, se=se, n_weeks=n_weeks, n_eff=n_eff, n_obs=n_obs,
                    post=pmean, post_sd=psd, z=z, confidence=float(2 * stats.norm.cdf(abs(z)) - 1),
                    reliable=reason == "", reason=reason)

    # ---- use -----------------------------------------------------------------------------------------------
    def lookup(self, keys, indicator: str) -> dict:
        """Trust of `indicator` for a type given its keys finest -> coarsest. The finest existing cell decides; if it
        is unreliable the indicator is neutralized (weight 0) rather than borrowing a coarser cell, because the finer
        cell's posterior already contains the coarser evidence."""
        for k in keys:
            i = self._idx.get((k, indicator))
            if i is not None:
                r = self.table.iloc[i]
                return dict(type=k, weight=float(r["post"]) if r["reliable"] else 0.0, reliable=bool(r["reliable"]),
                            confidence=float(r["confidence"]), reason=r["reason"])
        return dict(type=None, weight=0.0, reliable=False, confidence=0.0, reason="type never observed")

    def weights(self, info_day: pd.DataFrame, indicators=None) -> pd.DataFrame:
        """info_day: ticker-indexed features for ONE day. Returns ticker x indicator posterior reliability (0 =
        neutralized). Terciles are cut on this day's cross-section only."""
        ind = list(indicators) if indicators is not None else self.indicators
        info = info_day.copy()
        d = self.now if self.now is not None else pd.Timestamp("1970-01-01")
        info.index = pd.MultiIndex.from_arrays([[d] * len(info), info_day.index])
        L = assign_types(info)
        cols = [L[lv].values for lv in reversed(LEVELS)]
        # weight per (type, indicator): reliable cells carry their posterior, everything else is 0 (neutralized)
        t = self.table
        wm = dict(zip(zip(t["type"], t["indicator"]), np.where(t["reliable"], t["post"], 0.0))) if len(t) else {}
        W = np.zeros((len(info_day), len(ind)))
        for i in range(len(info_day)):
            keys = [c[i] for c in cols]
            for j, c in enumerate(ind):
                for k in keys:
                    w = wm.get((k, c))
                    if w is not None or (k, c) in wm:
                        W[i, j] = w
                        break
        return pd.DataFrame(W, index=info_day.index, columns=ind)

    def neutralize(self, scores: pd.DataFrame, info_day: pd.DataFrame) -> pd.DataFrame:
        """Scale each indicator's score by its type weight / the largest |weight| for that indicator. An unreliable
        type contributes exactly 0; a negative weight means the indicator works inverted there, and the sign is kept."""
        W = self.weights(info_day, list(scores.columns))
        scale = W.abs().max().replace(0, 1.0)
        return scores * (W / scale)

    def reliable_share(self) -> pd.DataFrame:
        if self.table.empty:
            return pd.DataFrame(columns=["level", "cells", "reliable", "share"])
        g = self.table.groupby("level")["reliable"].agg(["size", "sum"]).reindex(LEVELS).dropna()
        g.columns = ["cells", "reliable"]
        g["share"] = g["reliable"] / g["cells"]
        return g.reset_index()

    def report(self, top=10) -> str:
        if self.table.empty:
            return "trust table empty: no indicator has any support"
        t = self.table
        lines = [f"trust table as of {self.now.date()}: {len(t)} cells, {int(t['reliable'].sum())} reliable",
                 self.reliable_share().to_string(index=False)]
        rel = t[t["reliable"]]
        for r in rel.reindex(rel["z"].abs().sort_values(ascending=False).index).head(top).itertuples():
            lines.append(f"  {r.type:<34}{r.indicator:<22}post={r.post:+.4f} z={r.z:+.2f} weeks={r.n_weeks}")
        return "\n".join(lines)

    def to_json(self, path):
        rec = dict(params=self.p, now=str(self.now), indicators=self.indicators,
                   table=json.loads(self.table.to_json(orient="records")))
        with open(path, "w") as f:
            json.dump(rec, f)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            rec = json.load(f)
        tt = cls(**rec["params"])
        tt.now = pd.Timestamp(rec["now"])
        tt.indicators = rec["indicators"]
        if rec["table"]:
            tt.table = pd.DataFrame(rec["table"])
            tt._reindex()
        return tt
