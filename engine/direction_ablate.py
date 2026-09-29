"""Direction-input ablation (Bible PHASE 13: "evidence where justified"; generic ablation pattern: drop one input
family at a time, refit everything, compare out-of-sample Brier with a bootstrap over WEEKS).

Every configuration is fitted by the same DirectionEngine on the same rows with the same time split, so the held-out
block contains identical rows and the Brier difference is paired. Rows are clustered by week (a week's movers share a
market regime), and the bootstrap resamples whole weeks. An input family "earns its place" only if removing it makes
Brier significantly worse (CI of ablated - full entirely above 0); if removal makes it significantly better the input is
harmful; otherwise its contribution is not distinguishable from zero and the report says so rather than guessing."""
import numpy as np
import pandas as pd

from . import direction as D

FAMILIES = {
    "pattern": ["pattern", "trust"],          # pattern and its trust scaling go together
    "trust": ["trust"],                        # keep the pattern, remove per-type neutralization (trust = 1)
    "analog": ["analog"],
    "mw": ["mw"],
    "model": ["model"],
    "evidence": ["evidence"],
}


def drop_family(F: pd.DataFrame, fam: str) -> pd.DataFrame:
    """Remove a family. 'trust' is replaced by 1.0 (no neutralization); others become NaN (missing)."""
    G = F.copy()
    for c in FAMILIES[fam]:
        if c not in G:
            continue
        if fam == "trust":
            G[c] = 1.0
        elif c == "trust":
            G[c] = 0.0                         # pattern gone, so its trust is moot
        else:
            G[c] = np.nan
    return G


def _fit_predict(F, up, now, movers, **kw):
    eng = D.DirectionEngine(**kw).fit(F, up, now, movers)
    if eng.stack is None or "test_start" not in eng.diag:
        return None, eng
    d = F.index.get_level_values(0)
    te = (d >= pd.Timestamp(eng.diag["test_start"])) & (d + pd.Timedelta(days=eng.p["horizon_days"]) <= pd.Timestamp(now))
    Fte = F[te]
    return eng.predict(Fte), eng


def week_bootstrap(diff: pd.Series, n_boot=2000, seed=0, alpha=0.05) -> dict:
    """diff indexed by (date, ticker) -> paired per-row loss difference. Resample weeks with replacement; the statistic
    is the row-weighted mean difference, so weeks with more movers count more, as they do in the pooled Brier."""
    dates = pd.DatetimeIndex(diff.index.get_level_values(0))
    iso = dates.isocalendar()
    wk = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).values
    df = pd.DataFrame({"wk": wk, "d": diff.values})
    g = df.groupby("wk")["d"].agg(["sum", "count"])
    if len(g) < 2:
        return dict(mean=float(diff.mean()) if len(diff) else float("nan"), lo=float("nan"), hi=float("nan"), n_weeks=len(g))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    s, c = g["sum"].values, g["count"].values
    boots = s[idx].sum(1) / c[idx].sum(1)
    return dict(mean=float(diff.mean()), lo=float(np.quantile(boots, alpha / 2)),
                hi=float(np.quantile(boots, 1 - alpha / 2)), n_weeks=len(g))


def ablate(F: pd.DataFrame, up: pd.Series, now, movers=None, families=None, n_boot=2000, seed=0, **engine_kw) -> pd.DataFrame:
    """Returns one row per family: brier_full, brier_drop, delta (drop - full, positive = the family helps), CI over
    weeks, verdict in {helps, harms, no_detectable_effect}. Empty frame when the full engine could not be fitted."""
    cols = ["family", "n", "n_weeks", "brier_full", "brier_drop", "delta", "lo", "hi", "verdict"]
    p_full, eng = _fit_predict(F, up, now, movers, **engine_kw)
    if p_full is None or len(p_full) == 0:
        return pd.DataFrame(columns=cols)
    y = up.reindex(p_full.index).astype(float)
    if movers is not None:
        keep = movers.reindex(p_full.index).fillna(False).astype(bool)
        p_full, y = p_full[keep], y[keep]
    loss_full = (p_full - y) ** 2
    rows = []
    for fam in (families or list(FAMILIES)):
        G = drop_family(F, fam)
        p_d, _ = _fit_predict(G, up, now, movers, **engine_kw)
        if p_d is None:
            continue
        p_d = p_d.reindex(p_full.index)
        diff = ((p_d - y) ** 2 - loss_full).dropna()
        bs = week_bootstrap(diff, n_boot, seed)
        verdict = ("helps" if bs["lo"] > 0 else "harms" if bs["hi"] < 0 else "no_detectable_effect")
        rows.append(dict(family=fam, n=len(diff), n_weeks=bs["n_weeks"], brier_full=float(loss_full.mean()),
                         brier_drop=float(loss_full.mean() + diff.mean()), delta=bs["mean"], lo=bs["lo"], hi=bs["hi"],
                         verdict=verdict))
    return pd.DataFrame(rows, columns=cols)
