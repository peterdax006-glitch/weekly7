"""Bible phases 3, 4 (lifecycle / rescoping), 9, 10 - PATTERN RELIABILITY: learn WHEN a pattern works (canon C60, C61; C56-C59).

C60: patterns die within a year. The system must (1) find WHAT CHANGED when a pattern flipped or stopped, (2) predict when
a pattern will be unreliable, and (3) gate it - use it only where predicted reliable - instead of discarding it. A pattern
whose reliability cannot be predicted and which keeps breaking is disregarded for the period (C59).

  data      `Timelines`: weekly SIGNED pattern returns (positive = the pattern worked) for many patterns, one shared context
            frame, optional per-pattern-week extras (crowding). Pattern keys only; never a stock identity (C57).
  time      Row i is a decision at the close of week i. ctx_i is known then; ret_i is realised over week i -> i+1 and is
            usable only from row i+1 on (`block` weeks for a block label). Every feature uses rets < i (`.shift(1)`).
  breaks    `break_states` / `find_breaks`: a causal working -> broken state machine per pattern, with a CUSUM onset estimate.
  explain   `explain_breaks`: what the market looked like just before/at the onset of breaks versus while the pattern worked,
            and pooled reliability contrasts by context; family-wise error by a circular-shift max-|t| null over EVERY driver
            searched. Output is plain language with evidence counts, never stock identities.
  model     `walk_forward`: one meta-model POOLED across patterns and time predicting P(pattern holds | context now, pattern
            meta); refit on matured evidence only, Platt-calibrated on a later slice; Brier / ECE against base rates.
  gate      `classify` / `policy_weights`: steady -> always on; gateable (OOS-proven skill) -> gated by P(hold); unstable and
            unpredictable -> disregarded for the period. Nothing is deleted. `LiveGate` + `apply_gate_to_view` plug into
            engine.pattern_memory (B26) so the trader-facing weight is multiplied by the gate.
  evaluate  `evaluate_policies`: always-on vs three discard-after-break baselines vs gated, per era, block-bootstrap CIs.
  health    C61 `health_monitor`: every pattern re-checked every period against newly matured evidence (CUSUM with a designed
            false-alarm rate + discounted posterior); healthy / suspect / broken ledger; a broken pattern is never active.
  investigate  C61 `investigate`: every broken verdict must end EXPLAINED_AND_GATED (a driver that beat the family-wise bar
            AND predicted later, independent data) or DISCARDED_UNPREDICTABLE. A driver that fails either bar is never
            reported as the cause: the verdict is 'unknown cause' and its share is tracked (`unknown_cause_share`).
  guards    `audit_context_builder` (truncation test), `scan_context_for_peeking`, `causality_audit` (rerun on truncated
            data must reproduce the prediction): a context that peeks at the next period is caught (C56).

Deterministic: every draw takes a seed. No clock. No network."""
from __future__ import annotations

import dataclasses
import json
import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

PARAMS = {
    "block": 1,                 # weeks of realised return that make the label "hold"
    "hit_windows": (4, 13, 26),
    "min_age": 8,               # matured weeks before a pattern's own history is used at all
    "break_win": 13,            # trailing window (weeks) of the break t-test
    "t_break": 1.5,             # trailing t at or below -t_break after working = broken
    "t_work": 1.0,              # trailing t at or above t_work over work_win = proven working
    "t_recover": 0.5,           # broken -> working again when the trailing t reaches this
    "work_win": 26,
    "unstable_lookback": 52,    # a break detected within this many weeks makes the pattern "unstable"
    "min_train_rows": 400,
    "refit_every": 13,
    "cal_frac": 0.25,
    "model": "lgbm",            # lgbm | logit
    "lgbm": {"n_estimators": 120, "learning_rate": 0.05, "num_leaves": 7, "min_child_samples": 80, "subsample": 0.8,
             "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 5.0},
    "logit_C": 0.3,
    "thr_margin": 0.0,          # hard gate opens at P >= matured base rate + margin
    "gate_mode": "hard",        # hard | soft
    "soft_width": 0.10,
    "min_pred_n": 40,           # matured predictions before OOS skill is judged
    "t_pat_lo": 0.0,            # a pattern inherits the pooled skill unless its own matured skill t is below this
    "t_pat": 1.28,              # one-sided per-pattern skill bar (used to label a pattern's skill as individually proven)
    "t_mem": 3.0,               # MemoryGate bar: date-free rows cannot be clustered by week, so its t is overstated - demand more
    "t_pool": 1.64,             # one-sided pooled skill bar (must pass before ANY pattern is called gateable)
    "ctx_lead": 2,              # rows up to the onset that describe "just before the break"
    "cluster_gap": 4,           # weeks between onsets that still count as one shared event
    "min_clusters": 4,
    "batch": 8,                 # weeks per cluster-robust batch in the contrast statistic
    "n_perm": 300,
    "explain_alpha": 0.10,
    "peek_auc": 0.78,           # a context alone this good at separating this week's winners is not knowledge, it is a leak
    "peek_rho": 0.55,           # ... or this tightly tied to this week's return AND far tighter than to its neighbours
    "peek_ratio": 1.8,
    "peek_hard_rho": 0.9,       # ... or nearly rank-identical to this week's cross-pattern return, whatever the neighbours
    "boot": 1000,
    "boot_block": 6,
    "seed": 7,
    # C61 pattern health monitor and forced investigation
    "explain_meta": ("age", "crowd", "share"),   # pattern-meta drivers that can CAUSE a break (own hit rate is a symptom)
    "est_win": 26,              # matured weeks after first use that set the pattern's expected effect (no earlier)
    "effect_shrink": 0.7,       # expected effect = shrink x burn-in mean (winner's-curse guard) ...
    "effect_lcb_z": 0.5,        # ... capped at the burn-in mean minus this many standard errors (the pattern was picked BECAUSE its
                                #     burn-in looked good, so the raw mean is an upper estimate of its true effect: W-05)
    "oos_lag": 13,              # cross-fit: a healthy week joins the OOS estimate only this many weeks after it happened, so the estimate
                                #     that sets the bar is disjoint from the recent window the monitor is judging (W-05)
    "oos_min": 26,              # healthy post-burn-in weeks needed before the OOS estimate replaces the burn-in one
    "oos_z": 0.5,               # the OOS expected effect = OOS mean minus this many standard errors (no selection bias left to shrink)
    "effect_floor": 0.2,        # ... but never below this share of the burn-in mean (a positive effect must stay positive)
    "arl0": 500,                # target average weeks between false alarms of the sequential test, per pattern
    "k_floor": 0.08,            # CUSUM reference value floor (in sd units)
    "half_life": 13,            # weeks: discounted posterior that the pattern still works
    "suspect_frac": 0.4,        # suspect when the CUSUM exceeds this share of its alarm threshold
    "release_frac": 0.5,        # a broken pattern is released when the CUSUM falls below this share
    "release_hold": 4,          # ... and only after that condition held this many weeks running
    "p_release": 0.9,           # ... or released early when P(still works) is back above this and the recent mean >= expected
    "release_t": 1.5,           # ... AND the recent mean is this many standard errors above ZERO (the posterior alone leans on the prior
                                #     expected effect, so a lucky streak in a dead pattern used to clear it)
    "p_suspect": 0.5,           # ... or when the posterior P(still works) drops under this
    "review_every": 13,         # weeks between investigation reviews
    "oos_horizon": 52,          # weeks of later, unseen data a proposed driver must predict before it is trusted
    "oos_embargo": 4,           # weeks after the last open episode recovers before the confirming window starts
    "max_open": 130,            # an investigation unresolved after this many weeks violates the invariant (then discarded)
}
META = ["age", "hit4", "hit13", "hit26", "mean13", "t13", "exp_hit", "since_break", "hit_trend", "crowd", "share"]
PHRASES = {
    "m_vix": "the level of market fear (VIX)",
    "m_vix_term": "short-term fear relative to long-term fear (VIX term structure)",
    "m_vix_chg5": "the one-week change in market fear",
    "m_spy_ma200": "how far the market sits above its 200-day trend",
    "m_spy_ma50": "how far the market sits above its 50-day trend",
    "m_spy_r5": "the market's last-week return",
    "m_breadth": "market breadth (share of stocks advancing)",
    "m_dispersion": "cross-stock return dispersion",
    "m_liq": "market liquidity (median dollar volume)",
    "m_atr": "typical daily range of stocks",
    "m_vol20": "typical 20-day stock volatility",
    "m_curve": "the yield-curve slope (10y minus 2y)",
    "m_ffr": "the policy interest rate",
    "m_infl": "market-implied inflation (10y breakeven)",
    "age": "how long the pattern has been observed",
    "hit4": "the pattern's own hit rate over the last month",
    "hit13": "the pattern's own hit rate over the last quarter",
    "hit26": "the pattern's own hit rate over the last half-year",
    "mean13": "the pattern's own recent average return",
    "t13": "the pattern's own recent t-statistic",
    "exp_hit": "the pattern's lifetime hit rate",
    "since_break": "time since the pattern last broke",
    "hit_trend": "the pattern's short-term hit rate versus its half-year hit rate",
    "crowd": "how many other patterns pick the same stocks (crowding)",
    "share": "how many stocks currently satisfy the pattern's condition",
}
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


class ReliabilityError(RuntimeError):
    pass


class LeakError(ReliabilityError):
    """A context/feature knows something that was not available at the decision time (C56)."""


def _cfg(cfg):
    return {**PARAMS, **(cfg or {})}


# ---------------------------------------------------------------- small numerics
def _p_two(t):
    return float(2.0 * sps.norm.sf(abs(float(t))))


def _rolling_t(df, win, min_periods=None):
    """Trailing t-statistic of each column over `win` rows ending at the row itself (caller shifts)."""
    mp = min_periods or max(3, win // 2)
    m = df.rolling(win, min_periods=mp).mean()
    s = df.rolling(win, min_periods=mp).std(ddof=1)
    n = df.rolling(win, min_periods=mp).count()
    return m / (s / np.sqrt(n)).replace(0.0, np.nan)


def _expanding_stats(D):
    """Expanding count / mean / sd / AR(1)-inflated t of each column of a DataFrame (NaN = no observation), by cumsum."""
    v = D.values.astype(float)
    ok = np.isfinite(v)
    x = np.where(ok, v, 0.0)
    n = np.cumsum(ok, axis=0).astype(float)
    s1 = np.cumsum(x, axis=0)
    s2 = np.cumsum(x * x, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / n
        var = (s2 - n * mean ** 2) / (n - 1)
        # lag-1 autocorrelation over consecutive valid rows (weekly overlap / persistence) -> effective-n inflation
        prev = np.vstack([np.zeros((1, v.shape[1])), x[:-1]])
        pok = np.vstack([np.zeros((1, v.shape[1]), bool), ok[:-1]])
        pair = ok & pok
        c1 = np.cumsum(np.where(pair, x * prev, 0.0), axis=0)
        npair = np.cumsum(pair, axis=0).astype(float)
        rho = np.clip((c1 / npair - mean ** 2) / var, -0.5, 0.9)
        infl = np.sqrt((1 + rho) / (1 - rho))
        t = mean / (np.sqrt(var / n) * infl)
    t[n < 3] = np.nan
    mk = lambda a: pd.DataFrame(a, index=D.index, columns=D.columns)
    return mk(n), mk(mean), mk(np.sqrt(np.maximum(var, 0))), mk(t)


def nw_t(x, lags=4):
    """Newey-West t of the mean of a 1-D array (Bartlett kernel)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    xc = x - x.mean()
    g0 = float(xc @ xc) / n
    s = g0
    for k in range(1, min(lags, n - 1) + 1):
        s += 2.0 * (1 - k / (lags + 1.0)) * float(xc[k:] @ xc[:-k]) / n
    if s <= 1e-18:
        return float("nan")
    return float(x.mean() / math.sqrt(s / n))


def stationary_bootstrap_ci(x, block=6, n_boot=1000, level=0.95, seed=0):
    """(mean, lo, hi) of a weekly series by the stationary bootstrap (geometric block lengths). NaN-safe, seeded."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        m = float(x.mean()) if n else float("nan")
        return m, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block, 1)
    idx = np.empty((n, n_boot), np.int64)
    idx[0] = rng.integers(0, n, size=n_boot)
    jump = rng.random((n, n_boot)) < p
    new = rng.integers(0, n, size=(n, n_boot))
    for t in range(1, n):                                   # Politis-Romano: continue the block, or restart at random
        idx[t] = np.where(jump[t], new[t], (idx[t - 1] + 1) % n)
    means = x[idx].mean(axis=0)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def holm(p):
    """Holm step-down adjusted p-values (family-wise)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def ece(p, y, bins=10):
    """Expected calibration error with equal-mass bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) < bins * 3:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    out = 0.0
    for chunk in np.array_split(order, bins):
        out += len(chunk) / len(p) * abs(float(p[chunk].mean()) - float(y[chunk].mean()))
    return float(out)


def auc(score, y):
    """Mann-Whitney AUC (ties averaged)."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    ok = np.isfinite(score)
    score, y = score[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = sps.rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- the data container
@dataclass
class Timelines:
    """Weekly signed pattern returns + shared context. See module docstring for the time convention."""
    rets: pd.DataFrame                      # (weeks x patterns); positive = the pattern worked that week
    ctx: pd.DataFrame                       # (weeks x context features); known at the close of the row's week
    extra: dict = field(default_factory=dict)   # name -> DataFrame like rets ("crowd", "share")
    first_pos: pd.Series | None = None      # per pattern: first row it may be used (default: first finite return)
    truth: dict = field(default_factory=dict)   # planted worlds only

    def __post_init__(self):
        self.validate()

    def validate(self):
        r, c = self.rets, self.ctx
        if not r.index.equals(c.index):
            raise ReliabilityError("rets and ctx must share one index")
        if not r.index.is_monotonic_increasing or not r.index.is_unique:
            raise ReliabilityError("index must be strictly increasing")
        if r.shape[1] == 0 or len(r) == 0:
            raise ReliabilityError("empty timelines")
        if not np.isfinite(r.values[np.isfinite(r.values)]).all():
            raise ReliabilityError("non-finite returns")
        if np.isinf(r.values).any() or np.isinf(c.values.astype(float)).any():
            raise ReliabilityError("infinite values")
        for k in r.columns:
            if _ISO.search(str(k)):
                raise ReliabilityError(f"pattern key contains a date: {k!r}")
        for name, df in self.extra.items():
            if not df.index.equals(r.index) or list(df.columns) != list(r.columns):
                raise ReliabilityError(f"extra {name!r} does not align with rets")
        if self.first_pos is None:
            fv = r.notna().values
            self.first_pos = pd.Series(np.where(fv.any(0), fv.argmax(0), len(r)), index=r.columns)

    @property
    def n_weeks(self):
        return len(self.rets)

    @property
    def patterns(self):
        return list(self.rets.columns)

    def alive(self):
        """(weeks x patterns) bool: the pattern exists at that row (its first row onward)."""
        pos = np.arange(self.n_weeks)[:, None]
        return pd.DataFrame(pos >= self.first_pos.values[None, :], index=self.rets.index, columns=self.rets.columns)

    def truncate(self, pos):
        """Rows 0..pos with ctx_pos known but every return at rows >= pos hidden - what was knowable at the close of row pos."""
        r = self.rets.iloc[: pos + 1].copy()
        r.iloc[pos] = np.nan
        ex = {k: v.iloc[: pos + 1].copy() for k, v in self.extra.items()}
        fp = self.first_pos.clip(upper=pos + 1)
        return Timelines(r, self.ctx.iloc[: pos + 1].copy(), ex, fp, dict(self.truth))


# ---------------------------------------------------------------- context construction (point in time)
def _publication_lag(raw, lags):
    """Shift each column by its publication lag in calendar days (value is usable only `lag` days after its stamp)."""
    out = {}
    for c in raw.columns:
        s = raw[c].dropna()
        lag = int((lags or {}).get(c, 1))
        s = s.copy()
        s.index = s.index + pd.Timedelta(days=lag)
        out[c] = s[~s.index.duplicated(keep="last")]
    return out


def expanding_percentile(s, min_hist=52):
    """Rank of each value among values up to and including itself (point in time); NaN until `min_hist` observations."""
    import bisect
    vals, out = [], np.full(len(s), np.nan)
    for i, v in enumerate(s.values):
        if np.isfinite(v):
            bisect.insort(vals, v)
            if len(vals) >= min_hist:
                out[i] = (bisect.bisect_left(vals, v) + bisect.bisect_right(vals, v)) / 2.0 / len(vals)
    return pd.Series(out, index=s.index)


def build_context(raw, dates, lags=None, pct_cols=None, min_hist=52):
    """Weekly context at `dates` from a raw (date-indexed) frame using only values published by each date.
    Per column: `<c>` level (last value published on or before the date, per-column publication lag), `<c>_pct` expanding
    percentile (past only) for `pct_cols`, and `<c>_d4` the four-week change. Nothing looks forward."""
    dates = pd.DatetimeIndex(dates)
    lagged = _publication_lag(raw, lags)
    cols = {}
    for c, s in lagged.items():
        lvl = s.sort_index().reindex(s.index.union(dates)).ffill().reindex(dates)
        cols[c] = lvl
        cols[f"{c}_d4"] = lvl - lvl.shift(4)
        if pct_cols is None or c in pct_cols:
            cols[f"{c}_pct"] = expanding_percentile(lvl, min_hist)
    return pd.DataFrame(cols, index=dates)


def audit_context_builder(builder, raw, dates, n_cuts=8, seed=0, atol=1e-9):
    """Truncation test. For random cut rows c, build the context from raw data that ends at dates[c] and require the
    row at c to equal the row from the full build. A builder that used a later observation (a forward shift, a centred
    window, a backfill) differs. Raises LeakError naming the columns; returns the cut rows checked otherwise."""
    dates = pd.DatetimeIndex(dates)
    full = builder(raw, dates)
    rng = np.random.default_rng(seed)
    lo = min(60, len(dates) // 2)
    cuts = sorted(set(rng.integers(lo, len(dates), size=n_cuts).tolist()))
    bad = {}
    for c in cuts:
        part = builder(raw.loc[raw.index <= dates[c]], dates[: c + 1])
        a, b = full.iloc[c], part.iloc[c]
        for col in full.columns:
            x, y = a[col], b.get(col, np.nan)
            if np.isfinite(x) != np.isfinite(y) or (np.isfinite(x) and abs(x - y) > atol * max(1.0, abs(x))):
                bad.setdefault(col, []).append(int(c))
    if bad:
        raise LeakError("context builder used information after the row date: "
                        + "; ".join(f"{k} (differs at rows {v[:3]})" for k, v in sorted(bad.items())))
    return cuts


def scan_context_for_peeking(tl, cfg=None):
    """Statistical second line of defence for a context that arrives pre-built (so it cannot be re-derived): a feature
    known at the close of week i cannot legitimately (a) separate this week's outcome almost perfectly, or (b) be far
    more tightly tied to THIS week's realised return than to its neighbours' (a slow regime is tied to all of them).
    Returns a per-column report; `flagged` lists columns to reject."""
    P = _cfg(cfg)
    R = tl.rets
    cross = R.mean(axis=1)
    hold = (R > 0).where(R.notna())
    rows = []
    for c in tl.ctx.columns:
        x = tl.ctx[c].astype(float)
        ok = x.notna() & cross.notna()
        if ok.sum() < 40:
            rows.append({"col": c, "auc": np.nan, "rho0": np.nan, "rho_nb": np.nan, "flag": False, "why": "too short"})
            continue
        stacked = pd.DataFrame({"x": np.repeat(x.values[:, None], R.shape[1], 1).ravel(), "h": hold.values.ravel()}).dropna()
        a = auc(stacked["x"].values, stacked["h"].values.astype(bool)) if len(stacked) > 50 else np.nan
        a = max(a, 1 - a) if np.isfinite(a) else np.nan
        rho0 = float(sps.spearmanr(x[ok], cross[ok])[0])
        nb = []
        for k in (-2, -1, 1, 2):
            y = cross.shift(k)
            m = x.notna() & y.notna()
            if m.sum() > 30:
                nb.append(abs(float(sps.spearmanr(x[m], y[m])[0])))
        rho_nb = float(np.nanmax(nb)) if nb else 0.0
        why = []
        if np.isfinite(a) and a >= P["peek_auc"]:
            why.append(f"AUC {a:.2f} on this week's outcome")
        if abs(rho0) >= P["peek_hard_rho"]:
            why.append(f"|rho|={abs(rho0):.2f} with this week's return: nearly the same series")
        elif abs(rho0) >= P["peek_rho"] and abs(rho0) > P["peek_ratio"] * max(rho_nb, 0.05):
            why.append(f"|rho|={abs(rho0):.2f} with this week's return vs {rho_nb:.2f} with neighbours'")
        rows.append({"col": c, "auc": a, "rho0": rho0, "rho_nb": rho_nb, "flag": bool(why), "why": "; ".join(why)})
    rep = pd.DataFrame(rows)
    return {"table": rep, "flagged": rep.loc[rep["flag"], "col"].tolist() if len(rep) else []}


def assert_no_peeking(tl, cfg=None):
    rep = scan_context_for_peeking(tl, cfg)
    if rep["flagged"]:
        why = rep["table"].set_index("col").loc[rep["flagged"], "why"].to_dict()
        raise LeakError(f"context column(s) look like they peek at the outcome: {why}")
    return rep


# ---------------------------------------------------------------- break detection (causal)
def break_states(R, cfg=None):
    """Per-pattern causal state machine. State at row i uses returns of rows < i only:
      0 unproven   never yet shown to work (trailing work_win t < t_work)
      1 working    trailing work_win t >= t_work
      2 broken     was working, then the trailing break_win t fell to <= -t_break (stays broken until t >= t_recover)
    Returns (states DataFrame of ints, events DataFrame). An event has the row it was DETECTED, an estimated ONSET row
    (CUSUM change point inside the two windows before detection) and the recovery row if any."""
    P = _cfg(cfg)
    bw, ww = P["break_win"], P["work_win"]
    Rs = R.shift(1)
    tb = _rolling_t(Rs, bw, max(4, bw // 2)).values
    tw = _rolling_t(Rs, ww, max(6, ww // 2)).values
    T, N = R.shape
    st = np.zeros((T, N), dtype=np.int8)
    ev = []
    for j in range(N):
        s = 0
        cur = None
        col = R.iloc[:, j].values
        for i in range(T):
            b, w = tb[i, j], tw[i, j]
            if s == 0:
                if np.isfinite(w) and w >= P["t_work"]:
                    s = 1
            elif s == 1:
                if np.isfinite(b) and b <= -P["t_break"]:
                    s = 2
                    cur = {"pattern": R.columns[j], "detect": i, "onset": _cusum_onset(col, i, bw), "recover": np.nan}
                    ev.append(cur)
            elif s == 2:
                if np.isfinite(b) and b >= P["t_recover"]:
                    s = 1
                    cur["recover"] = i
            st[i, j] = s
    events = pd.DataFrame(ev, columns=["pattern", "detect", "onset", "recover"])
    return pd.DataFrame(st, index=R.index, columns=R.columns), events


def _cusum_onset(col, detect, bw):
    """Row at which the trailing evidence turned: the split of rows [detect-2*bw, detect-1] (returns realised before the
    detection) maximising (mean_before - mean_after) * sqrt(k (n-k) / n). Uses only rows < detect."""
    lo = max(0, detect - 2 * bw)
    seg = col[lo:detect]
    idx = np.arange(lo, detect)
    ok = np.isfinite(seg)
    seg, idx = seg[ok], idx[ok]
    n = len(seg)
    if n < 6:
        return max(detect - bw, 0)
    cs = np.cumsum(seg)
    k = np.arange(2, n - 1)
    m1 = cs[k - 1] / k
    m2 = (cs[-1] - cs[k - 1]) / (n - k)
    stat = (m1 - m2) * np.sqrt(k * (n - k) / n)
    kbest = int(k[int(np.argmax(stat))])
    return int(idx[kbest])          # first row of the "after" segment


def find_breaks(tl, cfg=None, as_of=None):
    """Events detected at or before row `as_of` (a break detected later is not knowable). Adds 'recovered_by_as_of'."""
    st, ev = break_states(tl.rets, cfg)
    if as_of is not None and len(ev):
        ev = ev[ev["detect"] <= as_of].copy()
        ev["recover"] = ev["recover"].where(ev["recover"] <= as_of)
    return ev.reset_index(drop=True)


# ---------------------------------------------------------------- pattern meta features and examples
def meta_wide(tl, cfg=None, states=None):
    """Wide (weeks x patterns) meta features; every one uses returns of rows < i only. Includes crowding (from `extra`)."""
    P = _cfg(cfg)
    R = tl.rets
    valid = R.notna()
    H = (R > 0).astype(float).where(valid)
    out = {}
    out["age"] = valid.cumsum().shift(1).fillna(0.0)
    for w in P["hit_windows"]:
        out[f"hit{w}"] = H.rolling(w, min_periods=max(2, w // 2)).mean().shift(1)
    out["n26"] = valid.astype(float).rolling(26, min_periods=1).sum().shift(1).fillna(0.0)
    out["mean13"] = R.rolling(13, min_periods=5).mean().shift(1)
    out["t13"] = _rolling_t(R.shift(1), 13, 5)
    out["exp_hit"] = H.expanding(min_periods=4).mean().shift(1)
    if states is None:
        states, _ = break_states(R, P)
    broken = (states.values == 2)
    last = np.where(broken, np.arange(len(R))[:, None], -1)
    last = np.maximum.accumulate(last, axis=0)
    since = np.where(last >= 0, np.arange(len(R))[:, None] - last, 104).clip(max=104)
    out["since_break"] = pd.DataFrame(since.astype(float), index=R.index, columns=R.columns)
    out["hit_trend"] = out["hit4"] - out["hit26"]
    for name in ("crowd", "share"):
        out[name] = tl.extra[name] if name in tl.extra else pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    return out


NONFEATURE = ("pos", "pat", "ret", "hold", "n26", "alive")


def build_examples(tl, cfg=None, states=None):
    """Long (pattern, week) frame with ctx + meta features, the realised return, and the block label `hold`
    (mean signed return over the next `block` rows > 0; NaN until matured). Rows exist from the pattern's first row."""
    P = _cfg(cfg)
    R = tl.rets
    T, N = R.shape
    meta = meta_wide(tl, P, states)
    blk = P["block"]
    fwd = R.rolling(blk, min_periods=blk).mean().shift(-(blk - 1)) if blk > 1 else R
    hold = (fwd > 0).astype(float).where(fwd.notna())
    alive = tl.alive().values
    data = {"pos": np.repeat(np.arange(T), N), "pat": np.tile(np.arange(N), T)}
    for c in tl.ctx.columns:
        data[c] = np.repeat(tl.ctx[c].values.astype(float), N)
    for m in META + ["n26"]:
        data[m] = meta[m].values.ravel()
    data["ret"] = R.values.ravel()
    data["hold"] = hold.values.ravel()
    data["alive"] = alive.ravel()
    ex = pd.DataFrame(data)
    ex = ex[ex["alive"]].reset_index(drop=True)
    return ex


def feature_columns(tl, use_ctx=True, use_meta=True):
    cols = []
    if use_ctx:
        cols += list(tl.ctx.columns)
    if use_meta:
        cols += META
    return cols


# ---------------------------------------------------------------- the reliability meta-model
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class MetaModel:
    """P(hold | features). `fit` trains on the earlier part of matured rows and Platt-calibrates on a later slice (an
    embargo of `block` rows between them); `predict` returns calibrated probabilities. A constant/degenerate training
    label gives the base rate. kind: lgbm | logit."""

    def __init__(self, kind, cols, cfg=None, seed=0):
        self.kind, self.cols, self.P, self.seed = kind, list(cols), _cfg(cfg), int(seed)
        self.base = 0.5
        self.raw = None
        self.med = None
        self.mu = self.sd = None
        self.platt = None
        self.thr = 0.5              # gate threshold on calibrated P(hold): where the expected signed return turns >= 0
        self.n_train = 0
        self.importance = {}

    def _prep(self, X):
        A = X[self.cols].values.astype(float)
        if self.kind == "logit":
            A = np.where(np.isfinite(A), A, self.med)
            A = (A - self.mu) / self.sd
        return A

    def _raw_predict(self, X):
        if self.raw is None:
            return np.full(len(X), self.base)
        A = self._prep(X)
        return self.raw.predict_proba(A)[:, 1]

    def fit(self, X, y, pos, ret=None):
        P = self.P
        y = np.asarray(y, float)
        pos = np.asarray(pos)
        self.n_train = len(y)
        self.base = float(y.mean()) if len(y) else 0.5
        if len(y) < 60 or y.min() == y.max():
            return self
        cut = np.quantile(pos, 1 - P["cal_frac"])
        tr = pos <= cut - P["block"]
        ca = pos > cut
        if tr.sum() < 40 or len(np.unique(y[tr])) < 2:
            tr, ca = np.ones(len(y), bool), np.zeros(len(y), bool)
        Xtr = X.loc[tr] if isinstance(tr, pd.Series) else X[tr]
        if self.kind == "logit":
            from sklearn.linear_model import LogisticRegression
            A = Xtr[self.cols].values.astype(float)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.med = np.nanmedian(np.where(np.isfinite(A), A, np.nan), axis=0)
            self.med = np.where(np.isfinite(self.med), self.med, 0.0)
            A = np.where(np.isfinite(A), A, self.med)
            self.mu, self.sd = A.mean(0), np.where(A.std(0) > 1e-12, A.std(0), 1.0)
            self.raw = LogisticRegression(C=P["logit_C"], max_iter=300)
            self.raw.fit((A - self.mu) / self.sd, y[tr])
        else:
            import lightgbm as lgb
            prm = dict(P["lgbm"])
            prm["min_child_samples"] = int(max(10, min(prm["min_child_samples"], tr.sum() // 12)))
            self.raw = lgb.LGBMClassifier(random_state=self.seed, n_jobs=1, verbose=-1, deterministic=True,
                                          force_row_wise=True, **prm)
            self.raw.fit(Xtr[self.cols].values.astype(float), y[tr])
            self.importance = dict(zip(self.cols, self.raw.booster_.feature_importance("gain")))
        if ca.sum() >= 60 and len(np.unique(y[ca])) == 2:
            from sklearn.linear_model import LogisticRegression
            pr = self._raw_predict(X[ca])
            lr = LogisticRegression(C=1e3, max_iter=200)
            lr.fit(_logit(pr)[:, None], y[ca])
            a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
            if a < 0:           # anti-informative on the calibration slice: fall back to its base rate
                a, b = 0.0, float(_logit(y[ca].mean()))
            self.platt = (a, b)
            if ret is not None:
                self._fit_threshold(X[ca], np.asarray(ret, float)[ca])
        return self

    def _fit_threshold(self, Xc, rc):
        """Break-even P(hold): isotonic regression of the signed return on the calibrated probability, on the calibration
        slice only (later than the training rows, matured). Gate opens where the fitted expected return is >= 0. Using the
        base hit rate instead would switch off weeks whose expected return is still positive."""
        from sklearn.isotonic import IsotonicRegression
        ok = np.isfinite(rc)
        if ok.sum() < 60:
            return
        pc = self.predict(Xc)[ok]
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(pc, rc[ok])
        grid = np.linspace(pc.min(), pc.max(), 200)
        f = iso.predict(grid)
        nz = np.flatnonzero(f >= 0)
        self.thr = 0.0 if len(nz) and nz[0] == 0 else (float(grid[nz[0]]) if len(nz) else 1.0)

    def predict(self, X):
        p = self._raw_predict(X)
        if self.platt is not None:
            a, b = self.platt
            p = 1.0 / (1.0 + np.exp(-(a * _logit(p) + b)))
        return np.clip(p, 0.01, 0.99)


@dataclass
class PredLog:
    """Out-of-sample predictions, each made at a refit row from matured evidence only."""
    df: pd.DataFrame            # long: pos, pat, p, base_pool, base_own, hold, ret, made_at
    T: int
    patterns: list
    block: int
    refits: list
    kind: str
    importance: dict = field(default_factory=dict)

    def wide(self, col):
        M = np.full((self.T, len(self.patterns)), np.nan)
        M[self.df["pos"].values, self.df["pat"].values] = self.df[col].values
        return pd.DataFrame(M, columns=self.patterns)

    def matured(self, as_of):
        """Predictions whose outcome was realised by the close of row `as_of`."""
        d = self.df
        return d[(d["pos"] + self.block <= as_of) & d["hold"].notna()]


def walk_forward(tl, cfg=None, kind="pooled", start=None, ex=None, states=None, seed=None):
    """Walk-forward, past-only predictions of P(hold) for every alive (pattern, week) after the first affordable refit.
    kind: pooled (ctx + meta, one model for all patterns) | pooled_ctx | own_history (meta only) | per_pattern (one small
    model per pattern on ctx only) | base_own (no model: the pattern's own shrunk trailing hit rate)."""
    P = _cfg(cfg)
    seed = P["seed"] if seed is None else seed
    ex = build_examples(tl, P, states) if ex is None else ex
    T, N = tl.n_weeks, len(tl.patterns)
    blk = P["block"]
    cols = {"pooled": feature_columns(tl), "pooled_ctx": feature_columns(tl, True, False),
            "own_history": feature_columns(tl, False, True), "per_pattern": feature_columns(tl, True, False),
            "base_own": []}[kind]
    lab = ex["hold"].notna().values
    pos = ex["pos"].values
    cum = np.cumsum(np.bincount(pos[lab], minlength=T + 1))          # labelled rows with pos <= k
    lo = P["min_age"] + blk + 4 if start is None else start
    first_r = next((r for r in range(lo, T) if cum[max(r - blk, 0)] >= P["min_train_rows"]), None)
    p = np.full(len(ex), np.nan)
    base_pool = np.full(len(ex), np.nan)
    made = np.full(len(ex), np.nan)
    thr = np.full(len(ex), np.nan)
    refits = []
    if first_r is None:
        raise ReliabilityError("not enough matured pattern-weeks for a first refit "
                               f"({int(cum[-1])} labelled rows, need {P['min_train_rows']})")
    points = list(range(first_r, T, P["refit_every"]))
    imp = {}
    for r in points:
        trm = lab & (pos <= r - blk)
        tr = ex[trm]
        end = min(r + P["refit_every"], T)
        prd = (pos >= r) & (pos < end)
        if not prd.any():
            continue
        pool = float(tr["hold"].mean())
        base_pool[prd] = pool
        made[prd] = r
        Xp = ex[prd]
        if kind == "base_own":
            p[prd] = _base_own(Xp, pool)
            thr[prd] = 0.5
        elif kind == "per_pattern":
            out = np.full(len(Xp), np.nan)
            for j in np.unique(Xp["pat"].values):
                sub = tr[tr["pat"] == j]
                sel = (Xp["pat"].values == j)
                m = MetaModel("logit", cols, {**P, "logit_C": 0.1}, seed).fit(sub, sub["hold"].values, sub["pos"].values) \
                    if len(sub) >= 60 else None
                out[sel] = m.predict(Xp[sel]) if m is not None else _base_own(Xp[sel], pool)
            p[prd] = out
        else:
            m = MetaModel(P["model"], cols, P, seed).fit(tr, tr["hold"].values, tr["pos"].values, tr["ret"].values)
            p[prd] = m.predict(Xp)
            thr[prd] = m.thr
            for k, v in m.importance.items():
                imp[k] = imp.get(k, 0.0) + float(v)
        refits.append({"row": int(r), "n_train": int(len(tr)), "pool_base": pool})
    df = ex[["pos", "pat", "hold", "ret"]].copy()
    df["p"], df["base_pool"], df["thr"] = p, base_pool, thr
    df["base_own"] = _base_own(ex, base_pool)
    df["made_at"] = made
    df = df[np.isfinite(df["p"])].reset_index(drop=True)
    tot = sum(imp.values()) or 1.0
    return PredLog(df, T, tl.patterns, blk, refits, kind, {k: v / tot for k, v in sorted(imp.items(), key=lambda kv: -kv[1])})


def _base_own(ex, pool, k=10.0):
    """The pattern's trailing 26-week hit rate shrunk toward the pooled base rate (the honest per-pattern baseline)."""
    h = ex["hit26"].values if isinstance(ex, pd.DataFrame) else ex
    n = ex["n26"].values
    pool = np.asarray(pool, float)
    h = np.where(np.isfinite(h), h, pool)
    return (h * n + k * pool) / (n + k)


def score_predictions(pl, as_of=None, seed=0, n_boot=500):
    """Brier / ECE of the model versus both base rates on matured predictions, with a week-block bootstrap CI on the
    Brier skill (1 - model/base). `as_of` None = everything matured by the end."""
    d = pl.matured(pl.T if as_of is None else as_of)
    if len(d) < 30:
        return {"n": int(len(d)), "note": "too few matured predictions"}
    y = d["hold"].values
    out = {"n": int(len(d)), "hit_rate": float(y.mean()), "brier_model": brier(d["p"], y),
           "brier_pool": brier(d["base_pool"], y), "brier_own": brier(d["base_own"], y),
           "ece_model": ece(d["p"], y), "ece_pool": ece(d["base_pool"], y)}
    out["skill_vs_pool"] = 1 - out["brier_model"] / out["brier_pool"]
    out["skill_vs_own"] = 1 - out["brier_model"] / out["brier_own"]
    diff = (d["base_pool"].values - y) ** 2 - (d["p"].values - y) ** 2
    wk = pd.Series(diff).groupby(d["pos"].values).mean()
    out["skill_t_vs_pool"] = nw_t(wk.values, lags=4)
    mean, lo, hi = stationary_bootstrap_ci(wk.values, 6, n_boot, seed=seed)
    out["gain_vs_pool_mean"], out["gain_vs_pool_lo"], out["gain_vs_pool_hi"] = mean, lo, hi
    out["auc"] = auc(d["p"].values, y.astype(bool))
    return out


def reliability_table(pl, bins=10):
    """Calibration table (equal-mass bins): mean predicted vs realised hold rate per bin, matured predictions only."""
    d = pl.matured(pl.T)
    if len(d) < bins * 5:
        return pd.DataFrame(columns=["bin", "n", "p_mean", "hit_rate"])
    order = np.argsort(d["p"].values, kind="mergesort")
    rows = []
    for b, ix in enumerate(np.array_split(order, bins)):
        rows.append({"bin": b, "n": len(ix), "p_mean": float(d["p"].values[ix].mean()),
                     "hit_rate": float(d["hold"].values[ix].mean())})
    return pd.DataFrame(rows)


def causality_audit(tl, cfg=None, kind="pooled", rows=None, seed=0, atol=1e-9):
    """The strongest guard: recompute the prediction for a row from a Timelines TRUNCATED to what was knowable at that
    close (ctx through the row, returns through the previous row, refit schedule unchanged) and require it to equal the
    full-data prediction. Any code path that read a later return or context changes the number. Raises LeakError."""
    P = _cfg(cfg)
    full = walk_forward(tl, P, kind)
    d = full.df
    if d.empty:
        raise ReliabilityError("no predictions to audit")
    rng = np.random.default_rng(seed)
    poss = np.unique(d["pos"].values)
    pick = rows if rows is not None else sorted(rng.choice(poss, size=min(4, len(poss)), replace=False).tolist())
    bad, skipped = [], []
    first = int(d["made_at"].min())
    for r in pick:
        cut = tl.truncate(int(r))
        # the refit schedule is anchored at the first affordable refit: keep it by anchoring on the same start row
        try:
            part = walk_forward(cut, P, kind, start=first)
        except ReliabilityError:
            skipped.append(int(r))                 # too little matured evidence in the truncated copy: nothing to compare
            continue
        a = d[d["pos"] == r].set_index("pat")["p"]
        b = part.df[part.df["pos"] == r].set_index("pat")["p"]
        common = a.index.intersection(b.index)
        if len(common) == 0 or (a[common] - b[common]).abs().max() > atol:
            bad.append(int(r))
    if bad:
        raise LeakError(f"prediction changed when later data was removed at rows {bad}: something read the future")
    checked = [int(r) for r in pick if int(r) not in skipped]
    if not checked:
        raise ReliabilityError("causality audit could not compare any row")
    return {"rows_checked": checked, "skipped": skipped, "ok": True}


# ---------------------------------------------------------------- break explanation: what changed? (family-wise controlled)
def driver_values(tl, meta=None, cfg=None):
    """Every candidate driver as a (weeks x patterns) matrix: context features (same value across patterns) and pattern
    meta (own hit rate, age, crowding, ...). Drivers with no data are omitted. Returns {name: (kind, matrix)}."""
    N = len(tl.patterns)
    meta = meta if meta is not None else meta_wide(tl, cfg)
    out = {}
    for c in tl.ctx.columns:
        v = tl.ctx[c].values.astype(float)
        if np.isfinite(v).sum() >= 60 and np.nanstd(v) > 1e-12:
            out[c] = ("context", np.repeat(v[:, None], N, 1))
    for m in _cfg(cfg)["explain_meta"]:
        v = meta[m].values.astype(float)
        if np.isfinite(v).sum() >= 200 and np.nanstd(v) > 1e-12:
            out[m] = ("pattern", v)
    return out


def _contrast_parts(Rz, V, Hf, Lf):
    return ((Rz * Hf).sum(1), (V * Hf).sum(1), (Rz * Lf).sum(1), (V * Lf).sum(1))


def _contrast_t(parts, batch_ids, min_side=20):
    """(t, gap, testable) of mean(R | high) - mean(R | low) with batch-clustered (consecutive weeks) influence-function
    variance. testable is False when either side has too few pattern-weeks or sits in fewer than 5 batches."""
    a1, n1, a2, n2 = parts
    N1, N2 = n1.sum(), n2.sum()
    if N1 < min_side or N2 < min_side:
        return 0.0, 0.0, False
    m1, m2 = a1.sum() / N1, a2.sum() / N2
    psi = (a1 - m1 * n1) / N1 - (a2 - m2 * n2) / N2
    nb = int(batch_ids.max()) + 1
    pb = np.bincount(batch_ids, weights=psi, minlength=nb)
    used = np.bincount(batch_ids, weights=(n1 + n2 > 0).astype(float), minlength=nb) > 0
    k = int(used.sum())
    # a group that sits inside one or two batches has no measurable between-cluster variance: untestable, never "certain"
    k1 = int((np.bincount(batch_ids, weights=n1, minlength=nb) > 0).sum())
    k2 = int((np.bincount(batch_ids, weights=n2, minlength=nb) > 0).sum())
    if k < 4 or min(k1, k2) < 5:
        return 0.0, m1 - m2, False
    se = math.sqrt(float((pb ** 2).sum()) * k / (k - 1))
    return ((m1 - m2) / se if se > 1e-15 else 0.0), float(m1 - m2), True


def driver_table(tl, cfg=None, as_of=None, meta=None, seed=None, n_perm=None):
    """Pooled reliability contrast for EVERY candidate driver, on evidence matured by the close of row `as_of`
    (returns of rows <= as_of - block only; tercile cuts from the same rows). Statistic: mean signed return when the
    driver is in its top tercile minus when it is in its bottom tercile, pooled over patterns and weeks, batch-clustered.
    Family-wise error: a circular shift of the returns against ALL drivers at once (autocorrelation and the
    cross-driver dependence are kept) gives the null distribution of the MAXIMUM |t| over every driver searched;
    p_fwer is the share of shifts whose maximum is at least the driver's |t|. No driver escapes the bar by being one
    of many."""
    P = _cfg(cfg)
    n_perm = P["n_perm"] if n_perm is None else n_perm
    seed = P["seed"] if seed is None else seed
    as_of = tl.n_weeks - 1 if as_of is None else int(as_of)
    m = as_of - P["block"]
    cols = ["driver", "kind", "contrast", "t", "p_fwer", "n_high", "n_low", "cut_lo", "cut_hi", "works_when"]
    if m < 60:
        return pd.DataFrame(columns=cols)
    drv = driver_values(tl, meta, P)
    R = tl.rets.values[: m + 1].astype(float)
    alive = tl.alive().values[: m + 1]
    V = (np.isfinite(R) & alive).astype(float)
    Rz = np.where(V > 0, R, 0.0)
    bid = np.arange(m + 1) // P["batch"]
    names, kinds, Hs, Ls, cuts = [], [], [], [], []
    for name, (kind, mat) in drv.items():
        vals = mat[: m + 1]
        ok = (V > 0) & np.isfinite(vals)
        if ok.sum() < 200:
            continue
        lo, hi = np.percentile(vals[ok], [100 / 3, 200 / 3])
        if not hi > lo:
            continue
        H = (ok & (vals >= hi)).astype(float)
        L = (ok & (vals <= lo)).astype(float)
        if H.sum() < 60 or L.sum() < 60:
            continue
        names.append(name); kinds.append(kind); Hs.append(H); Ls.append(L); cuts.append((float(lo), float(hi)))
    if not names:
        return pd.DataFrame(columns=cols)
    obs = [_contrast_t(_contrast_parts(Rz, V, H, L), bid) for H, L in zip(Hs, Ls)]
    rng = np.random.default_rng(seed)
    lo_s = max(26, (m + 1) // 10)
    hi_s = m + 1 - lo_s
    maxes = np.zeros(n_perm)
    if hi_s > lo_s:
        for b in range(n_perm):
            k = int(rng.integers(lo_s, hi_s))
            Rk, Vk = np.roll(Rz, k, axis=0), np.roll(V, k, axis=0)
            maxes[b] = max(abs(_contrast_t(_contrast_parts(Rk, Vk, H, L), bid)[0]) for H, L in zip(Hs, Ls))
    else:
        maxes[:] = np.inf
    rows = []
    for i, name in enumerate(names):
        t, c, _ok = obs[i]
        p = (1 + int((maxes >= abs(t)).sum())) / (1 + n_perm)
        rows.append({"driver": name, "kind": kinds[i], "contrast": c, "t": t, "p_fwer": p, "n_high": int(Hs[i].sum()),
                     "n_low": int(Ls[i].sum()), "cut_lo": cuts[i][0], "cut_hi": cuts[i][1],
                     "works_when": "high" if c > 0 else "low"})
    out = pd.DataFrame(rows, columns=cols)
    out = out.assign(_a=-out["t"].abs()).sort_values(["p_fwer", "_a"]).drop(columns="_a").reset_index(drop=True)
    out.attrs["n_candidates"] = len(names)
    out.attrs["as_of"] = as_of
    return out


def onset_shifts(tl, events, cfg=None, as_of=None):
    """For each break (detected by `as_of`): z-shift of every CONTEXT feature at the onset (mean of the `ctx_lead` rows up
    to and including the onset) relative to the same feature while the pattern was alive earlier, scaled by the feature's
    own history up to the onset. Events whose onsets lie within `cluster_gap` weeks share one cluster (a shared driver
    hits many patterns at once and must not be counted as many independent events). Returns (per-driver table, n_events,
    n_clusters). Holm-adjusted over the context features searched."""
    P = _cfg(cfg)
    ev = events if as_of is None else events[events["detect"] <= as_of]
    cols = ["driver", "mean_z", "t", "p", "p_holm", "n_clusters"]
    if len(ev) == 0:
        return pd.DataFrame(columns=cols), 0, 0
    order = ev.sort_values("onset").reset_index(drop=True)
    cl = (order["onset"].diff().fillna(1e9) > P["cluster_gap"]).cumsum().values
    C = tl.ctx
    zs = {}
    for c in C.columns:
        v = C[c].values.astype(float)
        per_event = []
        for _, e in order.iterrows():
            o = int(e["onset"])
            lead = v[max(0, o - P["ctx_lead"] + 1): o + 1]
            ref = v[max(0, o - 104): max(o - P["ctx_lead"] - 4, 0)]
            hist = v[: o + 1]
            if not (np.isfinite(lead).any() and np.isfinite(ref).sum() >= 10 and np.isfinite(hist).sum() >= 30):
                per_event.append(np.nan)
                continue
            sd = np.nanstd(hist)
            per_event.append((np.nanmean(lead) - np.nanmean(ref)) / sd if sd > 1e-12 else np.nan)
        zs[c] = per_event
    rows = []
    for c, ze in zs.items():
        s = pd.Series(ze).groupby(cl).mean().dropna()
        k = len(s)
        if k < P["min_clusters"]:
            continue
        t = float(s.mean() / (s.std(ddof=1) / math.sqrt(k))) if s.std(ddof=1) > 1e-12 else 0.0
        rows.append({"driver": c, "mean_z": float(s.mean()), "t": t, "p": float(2 * sps.t.sf(abs(t), k - 1)), "n_clusters": k})
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        out["p_holm"] = holm(out["p"].values)
    return out, int(len(order)), int(len(set(cl)))


@dataclass
class Explanation:
    driver: str
    kind: str
    works_when: str                 # 'high' | 'low'
    cut_lo: float
    cut_hi: float
    contrast: float
    t: float
    p_fwer: float
    n_high: int
    n_low: int
    n_candidates: int
    onset_z: float = float("nan")
    onset_p_holm: float = float("nan")
    n_break_clusters: int = 0
    statement: str = ""

    def as_dict(self):
        return dataclasses.asdict(self)


def _statement(name, works_when, lo, hi, contrast, n_high, n_low, n_cand, p, onset):
    ph = PHRASES.get(name)
    if ph is None:
        base = re.sub(r"_(pct|d4)$", "", name)
        suffix = {"pct": " (its rank within its own history)", "d4": " (its four-week change)"}.get(name[len(base) + 1:], "")
        ph = PHRASES.get(base, base.replace("_", " ")) + suffix
    bad = ("below", lo) if works_when == "high" else ("above", hi)
    s = (f"Patterns stop working when {ph} is {bad[0]} {bad[1]:.3g}; they work when it is "
         f"{'above' if works_when == 'high' else 'below'} {(hi if works_when == 'high' else lo):.3g}. "
         f"Evidence: {n_low if works_when == 'high' else n_high} bad-zone vs {n_high if works_when == 'high' else n_low} good-zone "
         f"pattern-weeks, weekly return gap {abs(contrast):.2%}, family-wise p={p:.3f} over {n_cand} candidate drivers")
    if onset is not None and onset[2] > 0:
        s += f"; {onset[2]} independent break clusters began in the bad zone (mean shift {onset[0]:+.2f} sd, Holm p={onset[1]:.3f})"
    return s + "."


def explain_breaks(tl, cfg=None, as_of=None, events=None, meta=None, seed=None):
    """Plain-language reasons patterns stop working, on evidence matured by row `as_of`, with the family-wise bar applied
    over EVERY driver searched. Never names a stock; a driver is reported only if p_fwer <= explain_alpha.
    Returns {'explanations': [Explanation...], 'table': driver_table, 'onset': onset table, 'n_events', 'n_clusters'}."""
    P = _cfg(cfg)
    as_of = tl.n_weeks - 1 if as_of is None else int(as_of)
    if events is None:
        events = find_breaks(tl, P, as_of)
    tab = driver_table(tl, P, as_of, meta, seed)
    on, n_ev, n_cl = onset_shifts(tl, events, P, as_of)
    onset = on.set_index("driver") if len(on) else pd.DataFrame(columns=["mean_z", "p_holm", "n_clusters"]).set_index(pd.Index([], name="driver"))
    ex = []
    ncand = int(tab.attrs.get("n_candidates", len(tab)))
    for _, r in tab.iterrows():
        if r["p_fwer"] > P["explain_alpha"]:
            continue
        o = None
        if r["driver"] in onset.index:
            oz, op, ok_ = float(onset.loc[r["driver"], "mean_z"]), float(onset.loc[r["driver"], "p_holm"]), int(onset.loc[r["driver"], "n_clusters"])
            # onset evidence corroborates only if it points into the bad zone
            bad_high = r["works_when"] == "low"
            if (oz > 0) == bad_high and op <= 0.5:
                o = (oz, op, ok_)
        e = Explanation(r["driver"], r["kind"], r["works_when"], r["cut_lo"], r["cut_hi"], r["contrast"], r["t"], r["p_fwer"],
                        int(r["n_high"]), int(r["n_low"]), ncand, *(o if o else (float("nan"), float("nan"), 0)))
        e.statement = _statement(e.driver, e.works_when, e.cut_lo, e.cut_hi, e.contrast, e.n_high, e.n_low, ncand, e.p_fwer, o)
        ex.append(e)
    for e in ex:
        if _ISO.search(e.statement):
            raise ReliabilityError("explanation contains a date")
    return {"explanations": ex, "table": tab, "onset": on, "n_events": n_ev, "n_clusters": n_cl}


def confirm_driver(tl, driver, cut_lo, cut_hi, works_when, lo_row, hi_row, cfg=None, meta=None, min_side=15):
    """OUT-OF-SAMPLE check of a frozen driver rule: on rows in (lo_row, hi_row] (labels matured by hi_row) is the mean
    signed return in the bad zone lower than in the good zone? Cuts and direction are fixed BEFORE these rows. Returns
    t (positive = the rule predicted correctly), counts and whether it passes the one-sided t_pat bar."""
    P = _cfg(cfg)
    last = min(hi_row - P["block"], tl.n_weeks - 1)
    lo = lo_row + 1
    if last - lo < 12:
        return {"ok": False, "t": float("nan"), "n_good": 0, "n_bad": 0, "testable": False, "why": "window too short"}
    drv = driver_values(tl, meta, P)
    if driver not in drv:
        return {"ok": False, "t": float("nan"), "n_good": 0, "n_bad": 0, "testable": False, "why": "driver unavailable"}
    vals = drv[driver][1][lo: last + 1]
    R = tl.rets.values[lo: last + 1].astype(float)
    V = (np.isfinite(R) & tl.alive().values[lo: last + 1] & np.isfinite(vals)).astype(float)
    Rz = np.where(V > 0, R, 0.0)
    good = V * ((vals >= cut_hi) if works_when == "high" else (vals <= cut_lo))
    bad = V * ((vals <= cut_lo) if works_when == "high" else (vals >= cut_hi))
    bid = np.arange(last - lo + 1) // 4
    t, c, valid = _contrast_t(_contrast_parts(Rz, V, good, bad), bid, min_side=min_side)
    ng, nb = int(good.sum()), int(bad.sum())
    return {"ok": bool(valid and t >= P["t_pat"] and c > 0), "t": float(t), "gap": float(c), "n_good": ng, "n_bad": nb,
            "testable": bool(valid), "why": "" if valid else "a zone has too few pattern-weeks or sits in under 5 batches"}


# ---------------------------------------------------------------- C61: pattern health monitor (no phantom patterns)
def expected_effect(mean, sd, n, P):
    """Winner's-curse-safe expected effect of a pattern that was established on `n` burn-in weeks (W-05). The pattern was
    selected because that stretch looked good, so its mean over-states the effect that will persist. Take the smaller of the
    shrunk mean and a lower confidence bound (mean - z se), and keep it a fixed share of the mean at least so it stays
    positive. An expected effect set too high makes a healthy, stationary pattern look like it is decaying."""
    se = sd / math.sqrt(max(int(n), 1))
    return float(max(P["effect_floor"] * mean, min(P["effect_shrink"] * mean, mean - P["effect_lcb_z"] * se)))


def cusum_threshold(k, arl0):
    """Alarm threshold h (in sd units) of a one-sided CUSUM with reference value k whose in-control average run length is
    `arl0` observations (Siegmund's approximation, solved by bisection)."""
    k = max(float(k), 1e-3)

    def arl(h):
        x = 2 * k * (h + 1.166)
        return (math.exp(min(x, 700)) - x - 1) / (2 * k * k)

    lo, hi = 0.5, 200.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if arl(mid) < arl0 else (lo, mid)
    return 0.5 * (lo + hi)


HEALTHY, SUSPECT, BROKEN = 1, 2, 3
STATUS_NAMES = {0: "unmonitored", HEALTHY: "healthy", SUSPECT: "suspect", BROKEN: "broken"}


@dataclass
class HealthLedger:
    """Result of `health_monitor`: matrices are (weeks x patterns); a row-i entry uses returns of rows < i only."""
    codes: pd.DataFrame          # 0 not yet monitored (burn-in) | 1 healthy | 2 suspect | 3 broken
    cusum: pd.DataFrame          # downward CUSUM in sd units
    pwork: pd.DataFrame          # discounted posterior P(the pattern still works)
    recent: pd.DataFrame         # discounted recent mean signed return
    mu0: pd.Series               # expected effect per pattern (from the burn-in, shrunk)
    sd: pd.DataFrame             # expanding sd used to scale
    h: pd.Series                 # alarm threshold per pattern
    events: pd.DataFrame         # broken episodes: pattern, detect, onset, recover, phantom
    params: dict
    mu_path: pd.DataFrame = None  # expected effect actually used each week (burn-in estimate, then the cross-fitted OOS one)

    def status(self, pos):
        return self.codes.iloc[pos].map(STATUS_NAMES)

    def ledger(self, pos):
        """One period's health ledger: pattern, status, evidence. The record a reviewer reads."""
        rows = []
        for j, p in enumerate(self.codes.columns):
            c = int(self.codes.iat[pos, j])
            if c == 0:
                continue
            ev = (f"cusum {self.cusum.iat[pos, j]:.1f}/{self.h.iloc[j]:.1f}, P(still works) {self.pwork.iat[pos, j]:.2f}, "
                  f"recent mean {self.recent.iat[pos, j]:+.3%} vs expected {self.mu0.iloc[j]:+.3%}")
            rows.append({"pos": pos, "pattern": p, "status": STATUS_NAMES[c], "evidence": ev})
        return pd.DataFrame(rows, columns=["pos", "pattern", "status", "evidence"])

    def active_mask(self):
        """(weeks x patterns) bool: monitored and not broken (a broken pattern can never stay active)."""
        return (self.codes.values != BROKEN)


def health_monitor(tl, cfg=None, expected=None, as_of=None):
    """Re-check every pattern every period against newly matured evidence (C61).
    Expected effect: mu0 = expected_effect(...) - the smaller of effect_shrink x the mean and its lower confidence bound (the
    pattern was picked for looking good, so the raw mean is inflated) - over the matured weeks since the pattern's first row, once
    at least `est_win` of them exist and the effect is established (mean > 0 with t >= t_work); or `expected[pattern]` if the
    discoverer supplies one. A pattern whose effect is not (yet) established is a PHANTOM: BROKEN and switched off until
    the evidence establishes it. Sequential tests against mu0, per pattern:
      * downward CUSUM S_i = max(0, S_{i-1} + (mu0 - r_{i-1}) / sd_i - k), k = max(mu0 / (2 sd), k_floor), alarm threshold h
        set for an average of `arl0` weeks between false alarms; S >= h => BROKEN;
      * a discounted normal posterior that the current effect is positive, prior centred on mu0 (evidence, and SUSPECT).
    Hysteresis: a broken pattern is released when S falls under release_frac x h, or the posterior is back above p_release
    with a recent mean at least mu0; a released pattern restarts its test. Row i uses returns of rows < i only."""
    P = _cfg(cfg)
    R = tl.rets.values.astype(float)
    T, N = R.shape
    codes = np.zeros((T, N), np.int8)
    cus = np.full((T, N), np.nan)
    pw = np.full((T, N), np.nan)
    rec = np.full((T, N), np.nan)
    sdm = np.full((T, N), np.nan)
    mu0s = np.full(N, np.nan)
    mup = np.full((T, N), np.nan)
    hs = np.full(N, np.nan)
    events = []
    lam = 0.5 ** (1.0 / P["half_life"])
    hcache = {}
    last = T if as_of is None else int(as_of) + 1
    for j in range(N):
        f = int(tl.first_pos.iloc[j])
        b0 = f + P["est_win"]
        if b0 >= T:
            continue
        exp_ = None if expected is None else expected.get(tl.patterns[j])
        est, mu0, h, k, mu_rel = False, np.nan, np.nan, np.nan, np.nan
        S, broken, cur, ok_run = 0.0, False, None, 0
        est_i, mu_cap, mu_lo = 0, np.nan, np.nan
        oc, om, om2 = 0, 0.0, 0.0                      # Welford accumulators of the cross-fitted OOS sample
        A = W = Q = 0.0
        cnt, m1, m2 = 0, 0.0, 0.0
        for i in range(f + 1, min(T, last)):
            r = R[i - 1, j]
            if np.isfinite(r):
                cnt += 1
                dlt = r - m1
                m1 += dlt / cnt
                m2 += dlt * (r - m1)
            if i < b0:
                continue
            sd_run = math.sqrt(m2 / (cnt - 1)) if cnt > 2 else 0.0
            if sd_run <= 1e-12 or cnt < max(8, P["est_win"] // 2):
                continue
            if not est:
                tst = m1 / (sd_run / math.sqrt(cnt))
                if exp_ is not None or (m1 > 0.05 * sd_run and tst >= P["t_work"]):
                    est = True
                    mu0 = float(exp_) if exp_ is not None else expected_effect(m1, sd_run, cnt, P)
                    mu_rel = mu0 if exp_ is not None else P["effect_shrink"] * m1     # release bar: the full shrunk effect, not the lower bound
                    sd0 = sd_run
                    est_i, mu_cap, mu_lo = i, max(m1, mu0), P["effect_floor"] * m1
                    oc, om, om2 = 0, 0.0, 0.0
                    k = max(0.5 * mu0 / sd0, P["k_floor"])
                    kk = round(k, 3)
                    if kk not in hcache:
                        hcache[kk] = cusum_threshold(kk, P["arl0"])
                    h = hcache[kk]
                    mu0s[j], hs[j] = mu0, h
                    S, A, W, Q = 0.0, 0.0, 0.0, 0.0
                    if cur is not None and cur.get("phantom"):
                        cur["recover"] = i
                        cur = None
                else:
                    if cur is None:
                        cur = {"pattern": tl.patterns[j], "detect": i, "onset": f, "recover": np.nan, "phantom": True}
                        events.append(cur)
                    codes[i, j], sdm[i, j] = BROKEN, sd_run
                    continue
            kx = i - 1 - P["oos_lag"]
            if exp_ is None and kx >= est_i and codes[kx, j] == HEALTHY and np.isfinite(R[kx, j]):
                oc += 1                                 # week kx was judged healthy BEFORE its return arrived, and is now old enough
                dx = R[kx, j] - om
                om += dx / oc
                om2 += dx * (R[kx, j] - om)
                if oc >= P["oos_min"]:
                    se_o = math.sqrt(om2 / (oc - 1) / oc)
                    mu0 = float(min(mu_cap, max(mu_lo, om - P["oos_z"] * se_o)))
            mup[i, j] = mu0
            sd = max(sd_run, 0.25 * math.sqrt(m2 / (cnt - 1)))
            if np.isfinite(r):
                S = max(0.0, S + (mu0 - r) / sd - k)
                A, W, Q = lam * A + r, lam * W + 1.0, lam * lam * Q + 1.0
            else:
                A, W, Q = lam * A, lam * W, lam * lam * Q
            neff = W * W / Q if Q > 0 else 0.0
            mean_d = A / W if W > 0 else 0.0
            pv = mu0 * mu0
            prec = 1.0 / pv + neff / (sd * sd)
            post = (mu0 / pv + neff * mean_d / (sd * sd)) / prec
            pwork = float(sps.norm.cdf(post * math.sqrt(prec)))
            t_rec = mean_d * math.sqrt(neff) / sd if neff > 0 else 0.0
            early = pwork >= P["p_release"] and mean_d >= mu_rel and t_rec >= P["release_t"]
            ok_run = ok_run + 1 if (broken and early) else 0
            if broken and (S < P["release_frac"] * h or ok_run >= P["release_hold"]):
                broken = False
                S = min(S, P["release_frac"] * h * 0.999)      # a released pattern starts a fresh test
                cur["recover"] = i
            if not broken and S >= h:
                broken = True
                z = i - 1
                while z > b0 and np.isfinite(cus[z, j]) and cus[z, j] > 0:
                    z -= 1
                cur = {"pattern": tl.patterns[j], "detect": i, "onset": int(z), "recover": np.nan, "phantom": False}
                events.append(cur)
            code = BROKEN if broken else (SUSPECT if (S >= P["suspect_frac"] * h or pwork < P["p_suspect"]) else HEALTHY)
            codes[i, j], cus[i, j], pw[i, j], rec[i, j], sdm[i, j] = code, S, pwork, mean_d, sd
    idx, cols = tl.rets.index, tl.rets.columns
    ev = pd.DataFrame(events, columns=["pattern", "detect", "onset", "recover", "phantom"])
    return HealthLedger(pd.DataFrame(codes, index=idx, columns=cols), pd.DataFrame(cus, index=idx, columns=cols),
                        pd.DataFrame(pw, index=idx, columns=cols), pd.DataFrame(rec, index=idx, columns=cols),
                        pd.Series(mu0s, index=cols), pd.DataFrame(sdm, index=idx, columns=cols), pd.Series(hs, index=cols),
                        ev, P, pd.DataFrame(mup, index=idx, columns=cols))


def false_alarm_budget(h, n_pattern_weeks):
    """Expected number of false broken alarms in `n_pattern_weeks` monitored pattern-weeks at the configured ARL0."""
    return n_pattern_weeks / float(h.params["arl0"])


# ---------------------------------------------------------------- C61: forced investigation of every break
VERDICTS = ("EXPLAINED_AND_GATED", "DISCARDED_UNPREDICTABLE", "PHANTOM_DISCARDED", "OPEN")


def _onset_zone_bad(tl, driver, e, works_when, cut_lo, cut_hi, meta, P):
    """Was the driver in its BAD zone while this break developed - a majority of the rows from just before the estimated
    onset up to the row the break was recognised? (The CUSUM onset estimate is early by construction, so the onset row
    alone under-reports; nothing after the detection row is read.)"""
    o, d = int(e["onset"]), int(e["detect"])
    lo = max(0, o - P["ctx_lead"] + 1)
    if driver in tl.ctx.columns:
        v = tl.ctx[driver].values[lo: d].astype(float)
    else:
        j = tl.patterns.index(e["pattern"])
        v = meta[driver].values[lo: d, j].astype(float)
    v = v[np.isfinite(v)]
    if not len(v):
        return False
    bad = (v <= cut_lo) if works_when == "high" else (v >= cut_hi)
    return bool(bad.mean() >= 0.5)


def investigate(tl, cfg=None, as_of=None, health=None, meta=None):
    """Every BROKEN verdict opens an investigation that must end in EXPLAINED_AND_GATED or DISCARDED_UNPREDICTABLE (C60,
    C59, C61). Review rows recur every `review_every` weeks; an event is reviewed at the first review row on or after
    its detection using ONLY evidence matured by that row:
      1. drivers searched: every context feature plus crowding/age/share (family-wise max-|t| circular-shift bar);
      2. a driver counts only if it beats that bar AND was in its bad zone at this break's onset;
      3. it must then PREDICT in later unseen data: cuts/direction frozen at the review; over the next `oos_horizon`
         weeks the bad zone must have a lower signed return than the good zone (one-sided t >= t_pat);
      4. no driver passing 2 + 3 => cause 'unknown': DISCARDED_UNPREDICTABLE, never a spurious driver.
    Until step 3 has data the verdict is OPEN, and it may stay OPEN at most review_every + oos_horizon weeks.
    Returns a DataFrame with the reasoning recorded per break."""
    P = _cfg(cfg)
    as_of = tl.n_weeks - 1 if as_of is None else int(as_of)
    health = health if health is not None else health_monitor(tl, P, as_of=as_of)
    meta = meta if meta is not None else meta_wide(tl, P)
    ev = health.events[health.events["detect"] <= as_of].reset_index(drop=True)
    cols = ["pattern", "detect", "onset", "recover", "review_row", "resolve_row", "verdict", "cause", "driver", "detail",
            "shared_with"]
    if ev.empty:
        return pd.DataFrame(columns=cols)
    step = P["review_every"]
    tables = {}
    rows = []
    for _, e in ev.iterrows():
        det, on = int(e["detect"]), int(e["onset"])
        shared = int(((ev["onset"] - on).abs() <= P["cluster_gap"]).sum() - 1)
        rec = {"pattern": e["pattern"], "detect": det, "onset": on, "recover": e["recover"], "shared_with": shared,
               "driver": "", "review_row": np.nan, "resolve_row": np.nan}
        if e["phantom"]:
            rec.update(review_row=det, resolve_row=det, verdict="PHANTOM_DISCARDED", cause="never_established",
                       detail="the burn-in mean signed return was not positive, so the pattern was never shown to work")
            rows.append(rec)
            continue
        rv = int(math.ceil(det / step) * step)
        rec["review_row"] = rv
        if rv > as_of:
            rec.update(verdict="OPEN", cause="pending", detail="waiting for the next scheduled review")
            rows.append(rec)
            continue
        if rv not in tables:
            tables[rv] = driver_table(tl, P, rv, meta)
        tab = tables[rv]
        ncand = int(tab.attrs.get("n_candidates", len(tab)))
        sig = tab[tab["p_fwer"] <= P["explain_alpha"]] if len(tab) else tab
        if len(sig) == 0:
            rec.update(resolve_row=rv, verdict="DISCARDED_UNPREDICTABLE", cause="unknown",
                       detail=f"none of {ncand} candidate drivers beat the family-wise noise baseline: the cause is "
                              f"not among the indicators the system has (unknown cause - unpredictable)")
            rows.append(rec)
            continue
        cand = [r for _, r in sig.iterrows() if _onset_zone_bad(tl, r["driver"], e, r["works_when"], r["cut_lo"], r["cut_hi"], meta, P)]
        if not cand:
            rec.update(resolve_row=rv, verdict="DISCARDED_UNPREDICTABLE", cause="unknown",
                       detail=f"driver(s) {', '.join(sig['driver'].head(3))} passed the family-wise bar generally but were "
                              f"not in their bad zone at this break's onset: they do not describe it (unknown cause)")
            rows.append(rec)
            continue
        # the confirming window must be INDEPENDENT of the episode(s) that prompted the search: it starts only after most of
        # the episodes of this event (same onset cluster) have recovered, plus an embargo - else a driver that merely
        # coincides with the still-running event would "predict" it. It needs at least oos_horizon // 2 weeks of such data
        # AND enough weeks of the driver's bad zone to be testable; both are awaited (OPEN) up to max_open weeks.
        clus = ev[(ev["detect"] <= rv) & (~ev["phantom"].astype(bool)) & ((ev["onset"] - on).abs() <= 2 * P["oos_horizon"] // 2)]
        found, last_msgs = None, []
        for r_try in range(rv + P["oos_horizon"], min(rv + P["max_open"], as_of) + 1, step):
            rc = np.sort(np.where(clus["recover"].notna() & (clus["recover"] <= r_try), clus["recover"], np.inf))
            q = rc[min(len(rc) - 1, int(math.ceil(0.8 * (len(rc) - 1))))] if len(rc) else rv
            if not np.isfinite(q):
                continue
            st0 = max(rv, int(q) + P["oos_embargo"])
            if r_try - st0 < P["oos_horizon"] // 2:
                continue
            res = [(r, confirm_driver(tl, r["driver"], r["cut_lo"], r["cut_hi"], r["works_when"], st0, r_try, P, meta)) for r in cand[:5]]
            last_msgs = [f"{r['driver']}: out-of-sample t={c['t']:.2f}" if c["testable"] else
                         f"{r['driver']}: bad zone seen in only {c['n_bad']} pattern-weeks" for r, c in res]
            if any(c["testable"] for _, c in res):
                found = (r_try, res)
                break
        if found is None:
            if as_of < rv + P["max_open"]:
                rec.update(verdict="OPEN", cause="pending", driver=cand[0]["driver"],
                           detail=f"candidate {cand[0]['driver']} passed in-sample (p_fwer={cand[0]['p_fwer']:.3f}); waiting for "
                                  f"later data independent of the running episode(s) to test it")
            else:
                rec.update(resolve_row=rv + P["max_open"], verdict="DISCARDED_UNPREDICTABLE", cause="unknown",
                           detail=f"candidate {cand[0]['driver']} passed in-sample but no later independent, testable data "
                                  f"arrived within {P['max_open']} weeks ({'; '.join(last_msgs) or 'none'}): not trusted "
                                  f"(unknown cause - unpredictable)")
            rows.append(rec)
            continue
        rr, res = found
        best = next(((r, c) for r, c in res if c["ok"]), None)
        if best is not None:
            r, c = best
            rec.update(resolve_row=rr, verdict="EXPLAINED_AND_GATED", cause=f"driver:{r['driver']}", driver=r["driver"],
                       detail=f"{r['driver']} (works when {r['works_when']}; cut {r['cut_lo']:.3g}/{r['cut_hi']:.3g}) beat the "
                              f"family-wise bar (p={r['p_fwer']:.3f}, {ncand} searched), was in its bad zone during the break and "
                              f"predicted the later, independent weeks out of sample (t={c['t']:.2f}, {c['n_bad']} bad vs "
                              f"{c['n_good']} good pattern-weeks): gated")
        else:
            rec.update(resolve_row=rr, verdict="DISCARDED_UNPREDICTABLE", cause="unknown",
                       detail="in-sample candidate(s) failed on later unseen data (" + "; ".join(last_msgs) + "): a spurious "
                              "indicator, not the cause (unknown cause - unpredictable)")
        rows.append(rec)
    return pd.DataFrame(rows, columns=cols)


def investigation_invariants(inv, health, as_of, cfg=None):
    """Nothing may fall between the cracks: every broken episode detected by `as_of` has an investigation, none is left
    OPEN beyond review_every + oos_horizon (+ slack) weeks, and every resolved one carries its reasoning. Returns the list
    of violations (empty = fine)."""
    P = _cfg(cfg)
    bad = []
    ev = health.events[health.events["detect"] <= as_of]
    have = set(zip(inv["pattern"], inv["detect"])) if len(inv) else set()
    for _, e in ev.iterrows():
        if (e["pattern"], int(e["detect"])) not in have:
            bad.append(f"uninvestigated break: {e['pattern']} @ {int(e['detect'])}")
    lim = P["max_open"] + P["review_every"]
    for _, r in inv.iterrows():
        if r["verdict"] == "OPEN" and as_of - r["detect"] > lim:
            bad.append(f"investigation open too long: {r['pattern']} @ {int(r['detect'])}")
        if r["verdict"] != "OPEN" and not str(r["detail"]).strip():
            bad.append(f"resolved without reasoning: {r['pattern']} @ {int(r['detect'])}")
    return bad


def unknown_cause_share(inv):
    """Share of RESOLVED, non-phantom breaks whose verdict is 'unknown cause' (nothing predicted them)."""
    res = inv[(inv["verdict"].isin(["EXPLAINED_AND_GATED", "DISCARDED_UNPREDICTABLE"]))] if len(inv) else inv
    if len(res) == 0:
        return {"resolved": 0, "unknown": 0, "share": float("nan")}
    unk = int((res["cause"] == "unknown").sum())
    return {"resolved": int(len(res)), "unknown": unk, "share": unk / len(res)}


def verdict_matrix(inv, health, T, patterns):
    """(weeks x patterns) int codes for policies: 0 no open case | 1 investigation OPEN | 2 EXPLAINED_AND_GATED |
    3 DISCARDED / PHANTOM. A case applies from the row it was detected (open) / resolved (verdict) until the pattern
    recovers. Everything in it was knowable at that row."""
    M = np.zeros((T, len(patterns)), np.int8)
    ix = {p: j for j, p in enumerate(patterns)}
    for _, r in inv.iterrows():
        j = ix[r["pattern"]]
        end = T if not np.isfinite(r["recover"]) else int(r["recover"])
        d = int(r["detect"])
        code = {"OPEN": 1, "EXPLAINED_AND_GATED": 2, "DISCARDED_UNPREDICTABLE": 3, "PHANTOM_DISCARDED": 3}[r["verdict"]]
        if r["verdict"] in ("EXPLAINED_AND_GATED", "DISCARDED_UNPREDICTABLE") and np.isfinite(r["resolve_row"]):
            rs = int(min(r["resolve_row"], end))
            M[d:rs, j] = 1
            M[rs:end, j] = code
        else:
            M[d:end, j] = code
    return M


# ---------------------------------------------------------------- gating: skill test, gate matrices, policies
def skill_tests(pl, cfg=None):
    """Expanding, past-only test of whether the model predicts better than BOTH simple baselines - the pooled base rate and
    the pattern's own shrunk trailing hit rate (which on pure noise is worse than the pooled rate, so beating it alone
    proves nothing). A row-i entry uses only predictions whose outcome had matured by the close of row i. Returns
    DataFrames (weeks x patterns): n, t (the smaller of the two baselines' AR(1)-inflated t), and the pooled
    (cross-pattern weekly mean) t series, again the smaller of the two."""
    Pm, Bo, Bp, H = pl.wide("p"), pl.wide("base_own"), pl.wide("base_pool"), pl.wide("hold")
    Lm = (Pm - H) ** 2
    out = {}
    for tag, B in (("own", Bo), ("pool", Bp)):
        D = ((B - H) ** 2 - Lm).shift(pl.block)
        n, mean, sd, t = _expanding_stats(D)
        nw, _, _, tw = _expanding_stats(D.mean(axis=1).to_frame("pooled"))
        out[tag] = {"n": n, "t": t, "mean": mean, "pooled_t": tw["pooled"], "pooled_n": nw["pooled"], "D": D}
    return {"n": out["own"]["n"], "t": np.minimum(out["own"]["t"], out["pool"]["t"]), "mean": out["own"]["mean"],
            "pooled_t": np.minimum(out["own"]["pooled_t"], out["pool"]["pooled_t"]), "pooled_n": out["own"]["pooled_n"],
            "by_baseline": out}


def gate_matrix(pl, cfg=None, mode=None):
    """Gate weight in [0, 1] per (week, pattern). hard: open iff P(hold) >= the break-even probability learned at the refit + margin.
    soft: linear ramp of width soft_width around that threshold. NaN before the first refit."""
    P = _cfg(cfg)
    mode = mode or P["gate_mode"]
    Pm, thr = pl.wide("p"), pl.wide("thr") + P["thr_margin"]
    if mode == "hard":
        G = (Pm >= thr).astype(float).where(Pm.notna())
    else:
        w = P["soft_width"]
        G = ((Pm - (thr - w)) / (2 * w)).clip(0, 1).where(Pm.notna())
    return G


def gateable_matrix(pl, cfg=None, skill=None):
    """bool (weeks x patterns): reliability of this pattern is predictable so far. Requires the POOLED skill bar first
    (honesty: a model with no proven out-of-sample skill anywhere gates nothing); a pattern then inherits that skill (the
    model is pooled - it sees meta features, never an identity) unless its own matured predictions show it does WORSE than
    the pattern's own trailing hit rate (t < t_pat_lo)."""
    P = _cfg(cfg)
    sk = skill if skill is not None else skill_tests(pl, P)
    pooled_ok = (sk["pooled_t"] >= P["t_pool"]) & (sk["pooled_n"] >= P["min_pred_n"])
    own = (sk["n"] < P["min_pred_n"]) | (sk["t"] >= P["t_pat_lo"])
    return own.mul(pooled_ok, axis=0).fillna(False).astype(bool)


def policy_weights(tl, pl, cfg=None, health=None, inv=None, states=None, skill=None):
    """Weight matrices (weeks x patterns) for every policy compared. Each entry depends only on rows <= its own."""
    P = _cfg(cfg)
    T, N = tl.n_weeks, len(tl.patterns)
    G = gate_matrix(pl, P)
    Gf = G.fillna(1.0)
    gate_ok = gateable_matrix(pl, P, skill)
    health = health if health is not None else health_monitor(tl, P)
    states = states if states is not None else break_states(tl.rets, P)[0]
    inv = inv if inv is not None else investigate(tl, P, health=health)
    codes = health.codes.values
    broken = codes == BROKEN
    ones = np.ones((T, N))
    df = lambda a: pd.DataFrame(a, index=tl.rets.index, columns=tl.rets.columns)
    ever = np.maximum.accumulate(broken, axis=0)
    yrs = (tl.rets.index.year.values if isinstance(tl.rets.index, pd.DatetimeIndex) else np.arange(T) // 52)
    in_year = np.zeros((T, N), bool)
    for y in np.unique(yrs):
        sel = np.where(yrs == y)[0]
        in_year[sel] = np.maximum.accumulate(broken[sel], axis=0)
    unstable = (states.values == 2)
    unstable = pd.DataFrame(unstable).rolling(P["unstable_lookback"], min_periods=1).max().values > 0
    v = verdict_matrix(inv, health, T, tl.patterns)
    Gv, ok = Gf.values, gate_ok.values
    W = {
        "always_on": ones,
        "discard_forever": np.where(ever, 0.0, 1.0),
        "discard_year": np.where(in_year, 0.0, 1.0),
        "discard_revive": np.where(broken, 0.0, 1.0),
        "gated_raw": Gv,
        "gated_c60": np.where(unstable & ~ok, 0.0, np.where(ok, Gv, 1.0)),
        "gated_c61": np.where(broken, np.where((v == 2) | ok, Gv, 0.0), np.where(ok, Gv, 1.0)),
        "oracle": (tl.rets.values > 0).astype(float),
    }
    return {k: df(a) for k, a in W.items()}


def portfolio_series(tl, W, rows):
    """Weekly portfolio return of each policy = sum(weight x signed return) / (patterns alive that week). A switched-off
    pattern is cash (0). Also the mean return per USED pattern-week."""
    R = tl.rets.values
    live = tl.alive().values & np.isfinite(R)
    Rz = np.where(live, R, 0.0)
    nlive = live.sum(1)
    out, used = {}, {}
    for k, w in W.items():
        wv = np.where(live, w.values, 0.0)
        num = (wv * Rz).sum(1)
        out[k] = np.where(nlive > 0, num / np.maximum(nlive, 1), np.nan)
        used[k] = np.where(wv.sum(1) > 0, num / np.maximum(wv.sum(1), 1e-12), np.nan)
    idx = tl.rets.index
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # weeks with no live pattern have no average weight
        avg = pd.DataFrame({k: np.nanmean(np.where(live, w.values, np.nan), axis=1) for k, w in W.items()}, index=idx)
    return pd.DataFrame(out, index=idx).iloc[rows], pd.DataFrame(used, index=idx).iloc[rows], avg.iloc[rows]


def _metrics(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return {"weeks": len(x)}
    sd = x.std(ddof=1)
    return {"weeks": len(x), "mean": float(x.mean()), "sd": float(sd), "sharpe": float(x.mean() / sd * math.sqrt(52)) if sd > 0 else float("nan"),
            "pos_share": float((x > 0).mean()), "worst5": float(np.quantile(x, 0.05)), "cum": float(np.prod(1 + x) - 1)}


def era_edges(index, era_years=3, n_blocks=4):
    """List of (label, first_row, last_row) eras: calendar blocks of `era_years` for a DatetimeIndex, else equal blocks."""
    T = len(index)
    if isinstance(index, pd.DatetimeIndex):
        yrs = index.year.values
        y0 = int(yrs.min())
        lab = (yrs - y0) // era_years
        out = []
        for e in np.unique(lab):
            sel = np.where(lab == e)[0]
            out.append((f"{y0 + e * era_years}-{min(y0 + (e + 1) * era_years - 1, int(yrs.max()))}", int(sel[0]), int(sel[-1])))
        return out
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    return [(f"block{b + 1}", int(edges[b]), int(edges[b + 1] - 1)) for b in range(n_blocks)]


def evaluate_policies(tl, pl, cfg=None, health=None, inv=None, states=None, eras=None, seed=None, baseline="always_on"):
    """Judge gated vs always-on vs discard-after-break out of sample: every weight at row i used only evidence matured by
    row i, every return is a later realised return. Reports mean weekly portfolio return, Sharpe-like ratio, positive-week
    share, worst-5% week, and the GAIN of each policy over `baseline` (and over the best discard baseline) with stationary
    bootstrap 95% intervals; per era; plus whether the switched-off pattern-weeks really were worse than the used ones."""
    P = _cfg(cfg)
    seed = P["seed"] if seed is None else seed
    W = policy_weights(tl, pl, P, health, inv, states)
    first = int(pl.df["pos"].min())
    rows = np.arange(first, tl.n_weeks)
    ret, per_used, avgw = portfolio_series(tl, W, rows)
    ok = ret.notna().all(axis=1)
    ret, per_used, avgw = ret[ok], per_used[ok], avgw[ok]
    summ = []
    discards = [k for k in ("discard_forever", "discard_year", "discard_revive") if k in ret]
    best_disc = max(discards, key=lambda k: ret[k].mean()) if discards else None
    for k in ret.columns:
        m = _metrics(ret[k].values)
        m.update(policy=k, avg_weight=float(avgw[k].mean()), per_used_mean=float(per_used[k].mean()))
        for ref_name, ref in (("always_on", "always_on"), ("best_discard", best_disc)):
            if ref is None or k == ref:
                continue
            d = (ret[k] - ret[ref]).values
            mean, lo, hi = stationary_bootstrap_ci(d, P["boot_block"], P["boot"], seed=seed)
            m[f"gain_vs_{ref_name}"], m[f"gain_vs_{ref_name}_lo"], m[f"gain_vs_{ref_name}_hi"] = mean, lo, hi
        summ.append(m)
    summary = pd.DataFrame(summ).set_index("policy")
    era_rows = []
    pos_arr = rows[ok.values]
    for lab, a, b in (eras if eras is not None else era_edges(tl.rets.index)):
        sub = ret[(pos_arr >= a) & (pos_arr <= b)]
        if len(sub) < 12:
            continue
        for k in ret.columns:
            if k == "always_on":
                continue
            d = (sub[k] - sub["always_on"]).values
            mean, lo, hi = stationary_bootstrap_ci(d, P["boot_block"], min(P["boot"], 400), seed=seed + 1)
            era_rows.append({"era": lab, "policy": k, "weeks": len(sub), "policy_mean": float(sub[k].mean()),
                             "always_on_mean": float(sub["always_on"].mean()), "gain": mean, "gain_lo": lo, "gain_hi": hi})
    era_tab = pd.DataFrame(era_rows)
    # is the gate's "off" set worse than its "on" set? (pattern-week level, matured returns)
    R = tl.rets.values
    live = tl.alive().values & np.isfinite(R)
    sel_rows = np.zeros(tl.n_weeks, bool)
    sel_rows[rows] = True
    diag = {}
    for k in ("gated_raw", "gated_c60", "gated_c61"):
        w = W[k].values
        on = live & (w >= 0.5) & sel_rows[:, None]
        off = live & (w < 0.5) & sel_rows[:, None]
        diag[k] = {"n_on": int(on.sum()), "n_off": int(off.sum()), "mean_on": float(R[on].mean()) if on.any() else float("nan"),
                   "mean_off": float(R[off].mean()) if off.any() else float("nan")}
    return {"weekly": ret, "per_used": per_used, "summary": summary, "eras": era_tab, "on_off": diag, "weights": W,
            "best_discard": best_disc}


# ---------------------------------------------------------------- status at a moment; live gate; adapters
def classify(tl, pl, cfg=None, as_of=None, health=None, inv=None, states=None, skill=None):
    """Where every pattern stands at the close of row `as_of` (default: last). status:
      insufficient  too little matured history for a verdict (weight 1, flagged)
      steady        healthy, never broken lately (weight 1, or gated if gate skill proven)
      gated         reliability predictable: used only where predicted reliable
      broken_gated  broken, cause explained and confirmed out of sample: used only where predicted reliable
      disregarded   broken/unstable and unpredictable (or phantom): weight 0 for the period, kept in memory"""
    P = _cfg(cfg)
    as_of = tl.n_weeks - 1 if as_of is None else int(as_of)
    health = health if health is not None else health_monitor(tl, P)
    inv = inv if inv is not None else investigate(tl, P, as_of, health)
    sk = skill if skill is not None else skill_tests(pl, P)
    gok = gateable_matrix(pl, P, sk)
    G = gate_matrix(pl, P).fillna(1.0)
    v = verdict_matrix(inv, health, tl.n_weeks, tl.patterns)
    rows = []
    for j, p in enumerate(tl.patterns):
        code = int(health.codes.iat[as_of, j])
        age = int(tl.rets[p].iloc[: as_of + 1].notna().sum())
        vc = int(v[as_of, j])
        gateable = bool(gok.iat[as_of, j])
        g = float(G.iat[as_of, j])
        if code == 0 and age < P["est_win"]:
            status, w = "insufficient", 1.0
        elif code == BROKEN:
            status, w = ("broken_gated", g) if vc == 2 else ("disregarded", 0.0)
        elif gateable:
            status, w = "gated", g
        else:
            status, w = "steady", 1.0
        rows.append({"pattern": p, "status": status, "weight_now": w, "health": STATUS_NAMES[code], "age": age,
                     "gateable": gateable, "skill_t": float(sk["t"].iat[as_of, j]) if np.isfinite(sk["t"].iat[as_of, j]) else float("nan"),
                     "n_pred": int(sk["n"].iat[as_of, j]),
                     "case": {0: "", 1: "investigating", 2: "explained", 3: "discarded"}[vc]})
    return pd.DataFrame(rows)


class LiveGate:
    """Frozen reliability gate as of one row of a Timelines: the trained pooled model + each pattern's latest meta and
    status. `weight(pattern, ctx_now)` -> multiplier in [0, 1] for a context dict with the same feature names as tl.ctx.
    Trained ONLY on evidence matured by the as_of row."""

    def __init__(self, tl, cfg=None, as_of=None, kind="pooled"):
        P = self.P = _cfg(cfg)
        as_of = tl.n_weeks - 1 if as_of is None else int(as_of)
        tr = tl.truncate(as_of)             # the returns of row as_of itself are not yet known
        self.cols = feature_columns(tr)
        ex = build_examples(tr, P)
        lab = ex["hold"].notna() & (ex["pos"] <= as_of - P["block"])
        t = ex[lab]
        if len(t) < P["min_train_rows"]:
            raise ReliabilityError("not enough matured evidence to train a gate")
        self.model = MetaModel(P["model"], self.cols, P, P["seed"]).fit(t, t["hold"].values, t["pos"].values)
        self.base = float(t["hold"].mean())
        last = ex[ex["pos"] == as_of].set_index("pat")
        self.meta = {tl.patterns[j]: last.loc[j, META].to_dict() for j in last.index}
        self.pl = walk_forward(tr, P, kind)
        self.status = classify(tr, self.pl, P, as_of).set_index("pattern")
        self.as_of = as_of

    def p_hold(self, pattern, ctx_now):
        if pattern not in self.meta:
            return self.base
        row = {c: ctx_now.get(c, np.nan) for c in self.cols if c not in META}
        row.update(self.meta[pattern])
        return float(self.model.predict(pd.DataFrame([row])[self.cols])[0])

    def weight(self, pattern, ctx_now=None):
        ctx_now = ctx_now or {}
        if pattern not in self.status.index:
            return 1.0
        st = self.status.loc[pattern]
        if st["status"] in ("disregarded",):
            return 0.0
        if st["status"] in ("insufficient", "steady"):
            return 1.0
        p = self.p_hold(pattern, ctx_now)
        thr = self.model.thr + self.P["thr_margin"]
        if self.P["gate_mode"] == "hard":
            return 1.0 if p >= thr else 0.0
        w = self.P["soft_width"]
        return float(np.clip((p - (thr - w)) / (2 * w), 0, 1))


# ---------------------------------------------------------------- B26 pattern-memory adapters
class MemoryGate:
    """`gate` for `PatternMemory.view(real_now, ctx_now, gate=MemoryGate.fit(mem, real_now))`.
    Trains ONE pooled reliability model over the date-free matured `TimelineSummary` of every pattern in the memory:
    each observation k>=2 of a pattern is a row (its context; the pattern's own past hit rate / mean t / count), the label
    is 'did it hold' (signed t > 0). It gates only if its skill was proven OUT OF SAMPLE (each pattern's last 30% of
    observations against the base rate; pooled one-sided t >= t_pool); otherwise every multiplier is 1.0 - a gate without
    proven skill would only add noise. Returns multipliers in [0, 1]; a pattern is never deleted by it."""

    FEATS = ("hit_prev", "mean_t_prev", "n_prev", "last_t")

    def __init__(self, model, base, cols, skill, cfg):
        self.model, self.base, self.cols, self.skill, self.P = model, base, list(cols), skill, cfg

    @classmethod
    def _rows(cls, s, cols):
        t = np.asarray(s.t, float)
        out = []
        for k in range(2, len(t)):
            prev = t[:k]
            row = {"hit_prev": float((prev > 0).mean()), "mean_t_prev": float(prev.mean()), "n_prev": float(k),
                   "last_t": float(prev[-1]), "hold": float(t[k] > 0), "frac": k / len(t)}
            row.update(cls._ctx_of(s, k, cols))
            out.append(row)
        return out

    @staticmethod
    def _ctx_of(s, k, cols):
        d = {c: np.nan for c in cols}
        for i, c in enumerate(s.ctx_cols):
            if c in d and s.ctx.size:
                d[c] = float(s.ctx[k, i]) if k < len(s.ctx) else np.nan
        return d

    @classmethod
    def fit(cls, mem, real_now, cfg=None, ctx_cols=None):
        P = _cfg(cfg)
        sums = _summaries(mem, real_now, min_obs=6)
        cols = sorted(ctx_cols if ctx_cols is not None else {c for s in sums for c in s.ctx_cols})
        rows = [r for s in sums for r in cls._rows(s, cols)]
        base_cols = list(cls.FEATS) + cols
        if len(rows) < 120:
            return cls(None, 0.5, base_cols, {"proven": False, "why": f"only {len(rows)} rows"}, P)
        df = pd.DataFrame(rows)
        train, test = df[df["frac"] <= 0.7], df[df["frac"] > 0.7]
        m = MetaModel("logit", base_cols, P, P["seed"]).fit(train, train["hold"].values, np.arange(len(train)))
        pt = m.predict(test)
        base = float(train["hold"].mean())
        d = (base - test["hold"].values) ** 2 - (pt - test["hold"].values) ** 2
        t = float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))) if len(d) > 5 and d.std(ddof=1) > 0 else float("nan")
        proven = bool(np.isfinite(t) and t >= P["t_mem"])
        final = MetaModel("logit", base_cols, P, P["seed"]).fit(df, df["hold"].values, np.arange(len(df)))
        return cls(final, float(df["hold"].mean()), base_cols,
                   {"proven": proven, "t": t, "n_test": int(len(test)), "brier_model": brier(pt, test["hold"].values),
                    "brier_base": brier(np.full(len(test), base), test["hold"].values)}, P)

    def __call__(self, key, ctx_now, summary):
        if self.model is None or not self.skill.get("proven"):
            return 1.0
        t = np.asarray(summary.t, float)
        if len(t) < 3:
            return 1.0
        row = {"hit_prev": float((t > 0).mean()), "mean_t_prev": float(t.mean()), "n_prev": float(len(t)), "last_t": float(t[-1])}
        for c in self.cols:
            if c not in row:
                v = (ctx_now or {}).get(c, np.nan)
                row[c] = float(v) if v is not None and np.isfinite(v) else np.nan
        p = float(self.model.predict(pd.DataFrame([row])[self.cols])[0])
        thr = self.model.thr + self.P["thr_margin"]
        if self.P["gate_mode"] == "hard":
            return 1.0 if p >= thr else 0.0
        w = self.P["soft_width"]
        return float(np.clip((p - (thr - w)) / (2 * w), 0.0, 1.0))


def _summaries(mem, real_now, min_obs=6):
    """Date-free TimelineSummary of every pattern with >= min_obs matured observations, from ONE pass over the matured
    records (timeline_summary per key would rescan the whole store for each key)."""
    try:
        from .pattern_memory import _summary, check_no_future
        mat = mem._matured(real_now)
        check_no_future(mat, real_now)
        by = {}
        for m in mat:
            by.setdefault(m["key"], []).append(m)
        return [_summary(k, v) for k, v in sorted(by.items()) if len(v) >= min_obs]
    except ImportError:
        out = [mem.timeline_summary(k, real_now) for k in mem.keys()]
        return [s for s in out if s is not None and len(s.t) >= min_obs]


def timelines_from_memory(mem, real_now, cfg=None):
    """Owner-side conversion of a PatternMemory to weekly Timelines (only observations matured strictly before real_now):
    signed return = direction x effect at the observation's week; context = mean of recorded ctx per week. Sparse (most
    weeks empty) - meant for reports, not for the trader."""
    recs = [r for r in mem._matured(real_now)]
    if not recs:
        raise ReliabilityError("no matured observations")
    df = pd.DataFrame(recs)
    sign = df.groupby("key")["t"].sum().apply(lambda x: 1.0 if x >= 0 else -1.0)
    df["sret"] = df["effect"] * df["key"].map(sign)
    df["week"] = pd.to_datetime(df["obs_date"]).dt.to_period("W-FRI").dt.end_time.dt.normalize()
    grid = pd.date_range(df["week"].min(), df["week"].max(), freq="W-FRI")
    rets = df.pivot_table(index="week", columns="key", values="sret", aggfunc="mean").reindex(grid)
    cx = pd.DataFrame([{**r["ctx"], "week": w} for r, w in zip(recs, df["week"])]).groupby("week").mean().reindex(grid).ffill()
    cx = cx.dropna(axis=1, how="all")
    return Timelines(rets, cx if cx.shape[1] else pd.DataFrame({"_none": np.zeros(len(grid))}, index=grid))


# ---------------------------------------------------------------- planted worlds (tests and calibration of the whole stack)
def _ar1(rng, T, phi, sd=1.0):
    x = np.empty(T)
    x[0] = rng.normal(0, sd)
    e = rng.normal(0, sd * math.sqrt(1 - phi * phi), T)
    for i in range(1, T):
        x[i] = phi * x[i - 1] + e[i]
    return x


def planted_world(kind, seed=0, n_weeks=520, n_patterns=8, mu=0.006, sigma=0.010, n_noise=6, life=None, opts=None):
    """Synthetic Timelines with a known truth. kinds:
      healthy        constant edge mu, no breaks (false-alarm budget)
      regime         edge = +mu when a hidden-but-observable regime variable ('m_regime', noisy) is above -0.4, -mu below
      random_breaks  each pattern flips between +mu and -mu at random times unrelated to any context
      family         like regime but every pattern lives only `life` weeks (pooling across patterns must help)
      shock          all patterns take an UNOBSERVED loss in three windows; decoy 'm_spur' coincides with the first two only
      phantom        edge mu for 100 weeks, then zero forever (a pattern that stopped existing)
      leak           regime world plus 'm_peek' = the cross-pattern mean of the COMING week's return
    opts (shock only): windows [(a, b)...] of the unobserved losses, decoy_on (the windows the decoy marks; default the first
    two), bumps (decoy excursions with no shock). Returns Timelines whose .truth holds the generating variables."""
    rng = np.random.default_rng(seed)
    T, N = n_weeks, n_patterns
    idx = pd.date_range("2005-01-07", periods=T, freq="W-FRI")
    ctx = {f"m_n{k}": _ar1(rng, T, 0.8) for k in range(n_noise)}
    common = rng.normal(0, 1, T)
    eps = rng.normal(0, 1, (T, N))
    noise = sigma * (0.5 * common[:, None] + 0.866 * eps)
    truth = {"kind": kind, "mu": mu, "sigma": sigma}
    fp = np.zeros(N, int)
    if kind in ("regime", "leak", "family"):
        z = _ar1(rng, T, 0.95)
        s = np.where(z > -0.4, 1.0, -1.0)
        ctx["m_regime"] = z + 0.3 * rng.normal(0, 1, T)
        S = np.repeat(s[:, None], N, 1)
        truth.update(regime=z, state=s)
    elif kind == "healthy":
        S = np.ones((T, N))
    elif kind == "random_breaks":
        S = np.ones((T, N))
        for j in range(N):
            st = 1.0
            for i in range(T):
                if rng.random() < (0.015 if st > 0 else 0.03):
                    st = -st
                S[i, j] = st
    elif kind == "shock":
        S = np.ones((T, N))
        o = opts or {}
        wins = o.get("windows", [(150, 190), (300, 340), (450, 490)])
        for a, b in wins:
            S[a:b] = -1.2
        spur = _ar1(rng, T, 0.7)
        for a, b in o.get("decoy_on", wins[:2]):
            spur[a:b] += 2.5
        for a, b in o.get("bumps", [(230, 250), (380, 400)]):   # excursions with no shock: the decoy is not the cause
            spur[a:b] += 2.5
        ctx["m_spur"] = spur
        truth.update(shock_windows=wins)
    elif kind == "phantom":
        S = np.ones((T, N))
        S[100:] = 0.0
    else:
        raise ValueError(kind)
    rets = mu * S + noise
    if kind == "leak":
        cross = rets.mean(1)
        ctx["m_peek"] = np.roll(cross, 0) / sigma + 0.05 * rng.normal(0, 1, T)
        truth["leak_col"] = "m_peek"
    R = pd.DataFrame(rets, index=idx, columns=[f"pat_{j:02d}" for j in range(N)])
    if kind == "family":
        life = life or 120
        starts = np.linspace(0, T - life - 1, N).astype(int)
        for j in range(N):
            R.iloc[: starts[j], j] = np.nan
            R.iloc[starts[j] + life:, j] = np.nan
        fp = starts
    elif kind in ("regime", "leak"):
        fp = (rng.integers(0, 60, N) * (np.arange(N) % 2)).astype(int)
        for j in range(N):
            R.iloc[: fp[j], j] = np.nan
    first = pd.Series(np.where(R.notna().values.any(0), R.notna().values.argmax(0), T), index=R.columns)
    C = pd.DataFrame(ctx, index=idx)
    return Timelines(R, C, {}, first, truth)


# ---------------------------------------------------------------- the façade: one call runs the whole C60/C61 pipeline
def run_reliability(tl, cfg=None, kinds=("pooled",), guards=True, audit_rows=None, explain=True, seed=None):
    """Guards -> walk-forward models -> health monitor -> forced investigations -> explanations -> policy evaluation.
    guards: peeking scan (raises LeakError) and, if `audit_rows` is not None, the truncation causality audit."""
    P = _cfg(cfg)
    res = {"params": {k: v for k, v in P.items()}, "n_weeks": tl.n_weeks, "n_patterns": len(tl.patterns)}
    if guards:
        res["peek_scan"] = assert_no_peeking(tl, P)["table"]
        if audit_rows is not None:
            res["causality"] = causality_audit(tl, P, kinds[0], rows=audit_rows)
    states = break_states(tl.rets, P)
    ex = build_examples(tl, P, states[0])
    res["pred"] = {k: walk_forward(tl, P, k, ex=ex, states=states[0], seed=seed) for k in kinds}
    main = res["pred"][kinds[0]]
    res["scores"] = {k: score_predictions(v, seed=P["seed"]) for k, v in res["pred"].items()}
    res["calibration"] = reliability_table(main)
    res["health"] = health_monitor(tl, P)
    meta = meta_wide(tl, P, states[0])
    res["investigations"] = investigate(tl, P, None, res["health"], meta)
    res["invariants"] = investigation_invariants(res["investigations"], res["health"], tl.n_weeks - 1, P)
    res["unknown_cause"] = unknown_cause_share(res["investigations"])
    res["evaluation"] = evaluate_policies(tl, main, P, res["health"], res["investigations"], states[0], seed=seed)
    res["status"] = classify(tl, main, P, None, res["health"], res["investigations"], states[0])
    if explain:
        res["explain"] = explain_breaks(tl, P, None, find_breaks(tl, P), meta, seed)
    return res


def render_report(res, title="Pattern reliability study"):
    """Markdown report of a `run_reliability` result: how well reliability is predicted, what it buys, why patterns broke."""
    L = [f"# {title}", "", f"{res['n_patterns']} patterns, {res['n_weeks']} weeks. Every number below is out of sample: "
         "predictions, gates and verdicts use only evidence matured before the week they act on.", ""]
    L.append("## Can reliability be predicted? (walk-forward, calibrated)")
    L.append("| model | n | hit rate | Brier model | Brier pooled base | Brier own base | skill vs pool | skill t | ECE | AUC |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, s in res["scores"].items():
        if "brier_model" not in s:
            L.append(f"| {k} | {s['n']} | too few | | | | | | | |")
            continue
        L.append(f"| {k} | {s['n']} | {s['hit_rate']:.3f} | {s['brier_model']:.4f} | {s['brier_pool']:.4f} | {s['brier_own']:.4f} | "
                 f"{s['skill_vs_pool']:+.2%} | {s['skill_t_vs_pool']:.2f} | {s['ece_model']:.3f} | {s['auc']:.3f} |")
    ev = res["evaluation"]
    L += ["", "## What gating buys (mean weekly portfolio return of equal-weighted patterns; off = cash)", ""]
    cols = ["mean", "sharpe", "pos_share", "worst5", "avg_weight", "gain_vs_always_on", "gain_vs_always_on_lo",
            "gain_vs_always_on_hi", "gain_vs_best_discard", "gain_vs_best_discard_lo", "gain_vs_best_discard_hi"]
    s = ev["summary"]
    L.append("| policy | weeks | mean | Sharpe | +weeks | worst 5% | avg weight | gain vs always-on [95% CI] | gain vs best discard [95% CI] |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, r in s.iterrows():
        g1 = f"{r['gain_vs_always_on']:+.3%} [{r['gain_vs_always_on_lo']:+.3%}, {r['gain_vs_always_on_hi']:+.3%}]" if "gain_vs_always_on" in r and pd.notna(r.get("gain_vs_always_on")) else ""
        g2 = f"{r['gain_vs_best_discard']:+.3%} [{r['gain_vs_best_discard_lo']:+.3%}, {r['gain_vs_best_discard_hi']:+.3%}]" if "gain_vs_best_discard" in r and pd.notna(r.get("gain_vs_best_discard")) else ""
        L.append(f"| {k} | {int(r['weeks'])} | {r['mean']:+.3%} | {r['sharpe']:.2f} | {r['pos_share']:.1%} | {r['worst5']:+.2%} | "
                 f"{r['avg_weight']:.2f} | {g1} | {g2} |")
    L += ["", f"Best discard baseline: {ev['best_discard']}. Switched-off vs used pattern-weeks (mean signed return):"]
    for k, d in ev["on_off"].items():
        L.append(f"- {k}: used {d['mean_on']:+.3%} (n={d['n_on']}), switched off {d['mean_off']:+.3%} (n={d['n_off']})")
    if len(ev["eras"]):
        L += ["", "## Per era: gain over always-on", "| era | policy | weeks | policy mean | always-on mean | gain [95% CI] |", "|---|---|---|---|---|---|"]
        for _, r in ev["eras"][ev["eras"]["policy"].isin(["gated_c61", "gated_c60", "discard_revive"])].iterrows():
            L.append(f"| {r['era']} | {r['policy']} | {int(r['weeks'])} | {r['policy_mean']:+.3%} | {r['always_on_mean']:+.3%} | "
                     f"{r['gain']:+.3%} [{r['gain_lo']:+.3%}, {r['gain_hi']:+.3%}] |")
    inv = res["investigations"]
    L += ["", "## Pattern health and investigations (C61)", ""]
    h = res["health"]
    broken_now = int((h.codes.iloc[-1] == BROKEN).sum())
    L.append(f"Broken episodes detected: {len(h.events)} (phantom at birth: {int(h.events['phantom'].sum()) if len(h.events) else 0}); "
             f"broken now: {broken_now}; false-alarm design: one per {h.params['arl0']} pattern-weeks.")
    if len(inv):
        L.append("Verdicts: " + ", ".join(f"{k}={v}" for k, v in inv["verdict"].value_counts().items()) + ".")
    u = res["unknown_cause"]
    L.append(f"Unknown-cause share of resolved breaks: {u['unknown']}/{u['resolved']}" + (f" = {u['share']:.0%}." if u["resolved"] else "."))
    L.append(f"Invariant violations: {len(res['invariants'])}" + ("" if not res["invariants"] else " - " + "; ".join(res["invariants"][:5])))
    if "explain" in res:
        ex = res["explain"]
        L += ["", "## Why patterns stop working (family-wise controlled over every candidate driver)", ""]
        if ex["explanations"]:
            for e in ex["explanations"]:
                L.append(f"- {e.statement}")
        else:
            L.append(f"No driver among {int(ex['table'].attrs.get('n_candidates', 0))} searched beat the family-wise noise baseline "
                     f"({ex['n_events']} breaks, {ex['n_clusters']} independent clusters): the honest answer is 'unknown cause'.")
    st = res["status"]
    L += ["", "## Status at the last week", "", "Counts: " + ", ".join(f"{k}={v}" for k, v in st["status"].value_counts().items())]
    return "\n".join(L) + "\n"
