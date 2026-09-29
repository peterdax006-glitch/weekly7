"""Cross-sectional learning (research contract C66 section 24; supports 4, 9, 13, 35; canon C56, C63, C64, C66, C67).
STATUS: IMPLEMENTED - NOT VALIDATED (synthetic-world unit tests only; C63: no real-data run was made).

The contract: do not only ask "what happens next to this stock"; ask "why is this stock behaving differently from the rest of the
market", against its sector, industry, the market, and its volatility / liquidity / momentum / event cohorts, so that a move is
labelled stock-specific, sector-specific, market-wide or regime-wide - and that label feeds BOTH the volatility and the direction model.

What lives here
  * assign_cohorts(): identity-free cohort labels for one day's universe (sector family and nested industry from SIC, within-day
    quantile cohorts for volatility, liquidity and momentum, a live-event cohort). No ticker or date enters a label.
  * decompose_day(): a hierarchical, leave-one-out, shrunk decomposition ret = market + sector + industry + style + idiosyncratic.
    Leave-one-out so a stock never explains itself; shrinkage by cohort size so a three-name industry cannot claim a big effect;
    winsorised inputs so one 80% outlier cannot define its cohort; a ridge fit for the style block. Sequential sums of squares give the
    variance SHARE each level explains, and chance_shares() shuffles the cohort labels to say how much a level would "explain" by luck.
  * classify_scope(): STOCK_SPECIFIC / SECTOR_SPECIFIC / MARKET_WIDE / REGIME_WIDE (plus COHORT_WIDE for style-cohort moves, MIXED,
    NO_MOVE, UNKNOWN). REGIME_WIDE needs the market or style component to persist across days versus its own PAST history; with too
    little history the honest answer is MARKET_WIDE, never a guess.
  * scope_features() / volatility_features() / direction_features(): the columns both models consume - shares, relative strength versus
    each cohort, standardised residual - split into magnitude features (volatility) and signed features (direction).
  * ScopeOutcomes: what each scope did NEXT (continuation for direction, |forward move| for volatility), date-clustered with an
    overlap-corrected error and Benjamini-Hochberg across scope x metric. Nothing is trusted until matured and established.
  * CrossSectionLab + step(): the ONE public entry. Streams day by day (keeps the last window of residuals only, never the
    38M-name-day panel), so it runs year by year on the trusted side (rule 27).

Firewall: scopes and outcomes are MATURED_RESEARCH_STATE. Records leave through to_matured_records() (MaturedRecord.gate) and are dropped
while their evidence year is being replayed in disguise. Trader-side rows are numeric and ticker-free (to_trader_rows +
engine.learning.trader_view.assert_trader_safe).

Built on: engine.learning.situation (sic_family, correlation_structure, clean_number), engine.learning.context.welch_t,
engine.learning.trader_view, engine.research.core, engine.research.multiscale (hac_se, benjamini_hochberg, t_to_p)."""
from __future__ import annotations

import dataclasses
import math
from collections import deque
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning.context import welch_t
from engine.learning.core import (FirewallBreach, Provenance, Unknown, _StrEnum, as_date, current_code_hash, require_past,
                                  stable_hash)
from engine.learning.situation import clean_number, correlation_structure, sic_family
from engine.learning.trader_view import assert_trader_safe
from engine.research.core import MaturedRecord, Namespace
from engine.research.multiscale import benjamini_hochberg, hac_se, t_to_p

SCHEMA_VERSION = "cross_section.v1"
UNK = "unknown"


class CohortKind(_StrEnum):                   # section 24 list: stock vs each of these
    MARKET = "market"
    SECTOR = "sector"
    INDUSTRY = "industry"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    MOMENTUM = "momentum"
    EVENT = "event"


STYLE_KINDS = (CohortKind.VOLATILITY, CohortKind.LIQUIDITY, CohortKind.MOMENTUM, CohortKind.EVENT)


class MoveScope(_StrEnum):
    STOCK_SPECIFIC = "STOCK_SPECIFIC"
    SECTOR_SPECIFIC = "SECTOR_SPECIFIC"       # sector or industry
    MARKET_WIDE = "MARKET_WIDE"
    REGIME_WIDE = "REGIME_WIDE"               # a market or style move that has persisted across days
    COHORT_WIDE = "COHORT_WIDE"               # a volatility / liquidity / momentum / event cohort moved together
    MIXED = "MIXED"                           # no level explains half of the move
    NO_MOVE = "NO_MOVE"                       # too small to attribute
    UNKNOWN = "UNKNOWN"                       # decomposition unavailable (too few names, missing inputs)


SCOPE_CODE = {MoveScope.NO_MOVE: 0, MoveScope.STOCK_SPECIFIC: 1, MoveScope.SECTOR_SPECIFIC: 2, MoveScope.MARKET_WIDE: 3,
              MoveScope.REGIME_WIDE: 4, MoveScope.COHORT_WIDE: 5, MoveScope.MIXED: 6, MoveScope.UNKNOWN: 7}


@dataclasses.dataclass(frozen=True)
class CrossConfig:
    """Column names of one day's universe frame and every threshold, in one validated record."""
    ret: str = "ret"                          # the day's return (point in time: through the decision close)
    sector: str = "sic"
    vol: str = "vol20"
    liq: str = "log_dv"
    mom: str = "mom20"
    events: tuple[str, ...] = ()              # columns > 0 when a filing/event is live for the name
    n_quantiles: int = 5
    min_names: int = 30                       # fewer names than this and the day is UNKNOWN
    min_cohort: int = 8                       # cohorts below this are flagged small (they are still shrunk, never trusted)
    shrink_k: float = 6.0                     # pseudo-observations pulling a cohort effect to zero
    ridge: float = 4.0
    winsor_z: float = 5.0                     # MAD-scaled clip applied to estimation inputs only
    share_bar: float = 0.5                    # a level must explain this share of the move to name it
    quiet_z: float = 0.75
    persist_window: int = 10
    persist_z: float = 1.5
    min_history: int = 40                     # past days before persistence / market z are answered
    history_days: int = 260

    def validate(self) -> list[str]:
        errs = []
        if self.n_quantiles < 2:
            errs.append("n_quantiles < 2")
        if self.min_names < 10 or self.min_cohort < 2:
            errs.append("min_names >= 10 and min_cohort >= 2 required")
        if not 0 < self.share_bar <= 1:
            errs.append("share_bar outside (0, 1]")
        if self.shrink_k < 0 or self.ridge < 0 or self.winsor_z <= 0:
            errs.append("shrink_k, ridge >= 0 and winsor_z > 0 required")
        if self.persist_window < 2 or self.min_history < self.persist_window + 5:
            errs.append("persist_window >= 2 and min_history >= persist_window + 5 required")
        if len({self.ret, self.sector, self.vol, self.liq, self.mom, *self.events}) != 5 + len(self.events):
            errs.append("column names must be distinct")
        return errs


def validate_day_frame(frame: pd.DataFrame, cfg: CrossConfig) -> list[str]:
    """Problems with one day's universe frame (ticker-indexed). Missing cohort columns are reported, not fatal: that cohort
    becomes 'unknown' and the decomposition skips it."""
    errs = list(cfg.validate())
    if frame is None or frame.empty:
        return errs + ["empty day frame"]
    if cfg.ret not in frame.columns:
        errs.append(f"return column {cfg.ret!r} missing")
    if frame.index.has_duplicates:
        errs.append("duplicate tickers in the day frame")
    if isinstance(frame.index, pd.MultiIndex):
        errs.append("day frame must be indexed by ticker only (one date per frame)")
    if cfg.ret in frame.columns:
        r = pd.to_numeric(frame[cfg.ret], errors="coerce")
        if r.notna().sum() == 0:
            errs.append("no finite returns")
        elif (r.abs() > 5).any():
            errs.append(f"{int((r.abs() > 5).sum())} return(s) beyond +-500%: units are not fractions")
    for c in (cfg.sector, cfg.vol, cfg.liq, cfg.mom):
        if c not in frame.columns:
            errs.append(f"cohort column {c!r} missing (cohort will be unknown)")
    return errs


# ------------------------------------------------------------------------------------------------ cohorts

def quantile_labels(x: pd.Series, q: int, prefix: str) -> pd.Series:
    """Within-day quantile cohort '<prefix>_q1'..'<prefix>_q<q>' by mid-rank (ties share a cohort, like situation.CrossSection);
    non-finite values are 'unknown', never a middle bucket."""
    v = pd.to_numeric(x, errors="coerce")
    v = v.where(np.isfinite(v))
    r = v.rank(pct=True, method="average")
    lab = np.ceil(r * q).clip(1, q)
    out = lab.map(lambda k: f"{prefix}_q{int(k)}" if pd.notna(k) else UNK)
    return out.astype(object)


def _sic_text(code: Any) -> str | None:
    try:
        s = str(int(float(code))).zfill(4)
    except (TypeError, ValueError):
        return None
    return s if len(s) >= 3 else None


def assign_cohorts(frame: pd.DataFrame, cfg: CrossConfig) -> pd.DataFrame:
    """Cohort label per kind and ticker: market, sector (SIC family), industry (family + 3-digit SIC, nested in the sector),
    volatility / liquidity / momentum quantiles, and event (live vs none). Labels carry no ticker, date or year."""
    out = pd.DataFrame(index=frame.index)
    out[CohortKind.MARKET.value] = "mkt"
    if cfg.sector in frame.columns:
        codes = frame[cfg.sector]
        fam = codes.map(lambda c: sic_family(c) if _sic_text(c) else UNK)
        three = codes.map(lambda c: (_sic_text(c) or UNK)[:3])
        out[CohortKind.SECTOR.value] = np.where(fam == UNK, UNK, "sec_" + fam.astype(str))
        out[CohortKind.INDUSTRY.value] = np.where((fam == UNK) | (three == UNK), UNK, "ind_" + fam.astype(str) + "_" + three.astype(str))
    else:
        out[CohortKind.SECTOR.value] = UNK
        out[CohortKind.INDUSTRY.value] = UNK
    for kind, col, pre in ((CohortKind.VOLATILITY, cfg.vol, "vol"), (CohortKind.LIQUIDITY, cfg.liq, "liq"),
                           (CohortKind.MOMENTUM, cfg.mom, "mom")):
        out[kind.value] = quantile_labels(frame[col], cfg.n_quantiles, pre) if col in frame.columns else UNK
    ev_cols = [c for c in cfg.events if c in frame.columns]
    if ev_cols:
        ev = frame[ev_cols].apply(pd.to_numeric, errors="coerce")
        live = (ev.fillna(0) > 0).any(axis=1)
        known = ev.notna().any(axis=1)
        out[CohortKind.EVENT.value] = np.where(~known, UNK, np.where(live, "ev_live", "ev_none"))
    else:
        out[CohortKind.EVENT.value] = UNK
    return out


def cohort_sizes(labels: pd.DataFrame) -> dict[str, dict[str, int]]:
    """{kind: {label: members}} excluding 'unknown'."""
    return {k: {lab: int(c) for lab, c in labels[k].value_counts().items() if lab != UNK} for k in labels.columns}


def small_cohorts(labels: pd.DataFrame, min_cohort: int) -> dict[str, list[str]]:
    """Cohorts below min_cohort: they are still shrunk toward zero but reported so nothing leans on them."""
    return {k: sorted(lab for lab, c in sz.items() if c < min_cohort) for k, sz in cohort_sizes(labels).items()}


def cohort_turnover(prev: pd.DataFrame | None, now: pd.DataFrame) -> dict[str, float | None]:
    """Share of names (present on both days) that changed cohort, per kind. Sector/industry are near 0; volatility, liquidity
    and momentum cohorts churn - a cohort that churns fast cannot carry a slow pattern. None when nothing overlaps."""
    if prev is None or prev.empty or now.empty:
        return {k: None for k in now.columns}
    common = prev.index.intersection(now.index)
    if len(common) < 5:
        return {k: None for k in now.columns}
    out = {}
    for k in now.columns:
        if k not in prev.columns:
            out[k] = None
            continue
        a, b = prev.loc[common, k], now.loc[common, k]
        ok = (a != UNK) & (b != UNK)
        out[k] = float((a[ok] != b[ok]).mean()) if ok.sum() >= 5 else None
    return out


# ------------------------------------------------------------------------------------------------ robust building blocks

def robust_scale(x: Sequence[float]) -> float:
    """Median-absolute-deviation scale (x1.4826). 0.0 for fewer than three finite values."""
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return 0.0
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def winsorize(x: pd.Series, z: float) -> pd.Series:
    """Clip to median +- z robust sigmas. A zero scale (a constant day) leaves the values alone."""
    s = robust_scale(x.to_numpy())
    if s <= 0:
        return x
    med = float(np.nanmedian(x.to_numpy(dtype="float64")))
    return x.clip(lower=med - z * s, upper=med + z * s)


def robust_center(x: Sequence[float], trim: float = 0.1) -> float:
    """Trimmed mean (10% each tail) - the market component; NaN for an empty input."""
    from scipy import stats as sps
    a = np.asarray(x, dtype="float64")
    a = a[np.isfinite(a)]
    return float(sps.trim_mean(a, trim)) if len(a) else float("nan")


def loo_group_effect(values: pd.Series, labels: pd.Series, shrink_k: float) -> tuple[pd.Series, pd.Series]:
    """Leave-one-out cohort effect of each element: the mean of the OTHER members, shrunk by (n-1)/(n-1+k) toward zero.
    Returns (effect, others) with others = n-1 (0 for singletons and 'unknown', whose effect is 0.0, not NaN: a name with no
    cohort is simply not adjusted). A stock never contributes to its own benchmark."""
    v = pd.to_numeric(values, errors="coerce")
    ok = v.notna() & (labels != UNK)
    eff = pd.Series(0.0, index=v.index)
    others = pd.Series(0, index=v.index, dtype="int64")
    if ok.sum() == 0:
        return eff, others
    g = v[ok].groupby(labels[ok])
    s, n = g.transform("sum"), g.transform("count")
    denom = (n - 1).where(n > 1)
    loo = (s - v[ok]) / denom
    w = (n - 1) / (n - 1 + shrink_k) if shrink_k > 0 else (n > 1).astype(float)
    eff.loc[ok] = (loo * w).fillna(0.0)
    others.loc[ok] = (n - 1).astype("int64")
    return eff, others


# ------------------------------------------------------------------------------------------------ decomposition

@dataclasses.dataclass(frozen=True)
class Decomposition:
    """One day's decomposition. table columns: ret, market, sector, industry, style, idio and the four style parts
    (style_volatility, style_liquidity, style_momentum, style_event); n_sector / n_industry are the cohort sizes the effects
    were estimated from. stage_ss are the sequential sums of squares (after market, sector, industry, style)."""
    table: pd.DataFrame
    market: float
    dispersion: float
    stage_ss: tuple[float, float, float, float]
    n: int
    status: str = "OK"                        # OK | INSUFFICIENT_DATA | STYLE_SKIPPED
    note: str = ""

    def usable(self) -> bool:
        return self.status in ("OK", "STYLE_SKIPPED") and self.n > 0

    def shares(self) -> dict[str, float | None]:
        """Sequential (Type I) variance shares of the move net of the market: sector, industry, style, idio; they sum to 1.
        A level can be NEGATIVE (a shrunk effect that fits worse than nothing) - kept, because hiding it would flatter the model."""
        ss0, ss1, ss2, ss3 = self.stage_ss
        if not self.usable() or ss0 <= 0:
            return {"sector": None, "industry": None, "style": None, "idio": None, "explained": None}
        return {"sector": (ss0 - ss1) / ss0, "industry": (ss1 - ss2) / ss0, "style": (ss2 - ss3) / ss0,
                "idio": ss3 / ss0, "explained": 1.0 - ss3 / ss0}


def _empty_table(index: pd.Index) -> pd.DataFrame:
    cols = ["ret", "market", "sector", "industry", "style", "idio", "style_volatility", "style_liquidity", "style_momentum",
            "style_event", "n_sector", "n_industry"]
    return pd.DataFrame(np.nan, index=index, columns=cols)


def _dummies(labels: pd.DataFrame, kinds: Sequence[CohortKind]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One-hot design of the style cohorts (unknown labels get an all-zero row for that kind). Returns (X, columns per kind)."""
    parts, cols = [], {}
    for k in kinds:
        d = pd.get_dummies(labels[k.value].where(labels[k.value] != UNK), dtype="float64")
        d.columns = [f"{k.value}:{c}" for c in d.columns]
        parts.append(d)
        cols[k.value] = list(d.columns)
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=labels.index)
    return X, cols


def decompose_day(frame: pd.DataFrame, cfg: CrossConfig, labels: pd.DataFrame | None = None,
                  beta: pd.Series | None = None) -> Decomposition:
    """ret = market + sector + industry + style + idio, estimated cross-sectionally on ONE day, leave-one-out and shrunk.
    market: trimmed mean of the returns. sector / industry: leave-one-out shrunk cohort means of the winsorised remainder.
    style: ridge regression of what is left on the volatility / liquidity / momentum / event cohort dummies (skipped, with
    status STYLE_SKIPPED, when there are fewer than 8 names per dummy). idio: the rest.
    beta (ticker -> market beta from PAST days, see EWBeta): a name's market component is beta x market; None means 1."""
    errs = [e for e in validate_day_frame(frame, cfg) if "missing (cohort" not in e]
    if errs and any("return column" in e or "empty" in e or "no finite" in e or "duplicate" in e for e in errs):
        raise ValueError("bad day frame: " + "; ".join(errs))
    lab = labels if labels is not None else assign_cohorts(frame, cfg)
    r = pd.to_numeric(frame[cfg.ret], errors="coerce")
    r = r.where(np.isfinite(r))
    ok = r.notna()
    tab = _empty_table(frame.index)
    if ok.sum() < cfg.min_names:
        tab["ret"] = r
        return Decomposition(tab, float("nan"), float("nan"), (0.0, 0.0, 0.0, 0.0), int(ok.sum()), "INSUFFICIENT_DATA",
                             f"{int(ok.sum())} finite returns < min_names {cfg.min_names}")
    rr = r[ok]
    lb = lab.loc[rr.index]
    m = robust_center(rr.to_numpy())
    mk = pd.Series(m, index=rr.index) if beta is None else m * beta.reindex(rr.index).fillna(1.0)
    e0 = rr - mk
    disp = robust_scale(e0.to_numpy())
    s_eff, n_sec = loo_group_effect(winsorize(e0, cfg.winsor_z), lb[CohortKind.SECTOR.value], cfg.shrink_k)
    e1 = e0 - s_eff
    i_eff, n_ind = loo_group_effect(winsorize(e1, cfg.winsor_z), lb[CohortKind.INDUSTRY.value], cfg.shrink_k)
    e2 = e1 - i_eff
    X, kind_cols = _dummies(lb, STYLE_KINDS)
    style = pd.Series(0.0, index=rr.index)
    parts = {k.value: pd.Series(0.0, index=rr.index) for k in STYLE_KINDS}
    status, note = "OK", ""
    if X.shape[1] == 0 or len(rr) < 8 * max(X.shape[1], 1):
        status, note = "STYLE_SKIPPED", f"{len(rr)} names for {X.shape[1]} style dummies"
    else:
        y = winsorize(e2, cfg.winsor_z)
        y = y - y.mean()
        Xc = X - X.mean()
        beta = np.linalg.solve(Xc.T.to_numpy() @ Xc.to_numpy() + cfg.ridge * np.eye(Xc.shape[1]), Xc.T.to_numpy() @ y.to_numpy())
        b = pd.Series(beta, index=Xc.columns)
        for k, cols in kind_cols.items():
            if cols:
                parts[k] = Xc[cols] @ b[cols]
        style = sum(parts.values())
    idio = e2 - style
    tab.loc[rr.index, "ret"] = rr
    tab.loc[rr.index, "market"] = mk
    tab.loc[rr.index, "sector"] = s_eff
    tab.loc[rr.index, "industry"] = i_eff
    tab.loc[rr.index, "style"] = style
    tab.loc[rr.index, "idio"] = idio
    for k, v in parts.items():
        tab.loc[rr.index, f"style_{k}"] = v
    tab.loc[rr.index, "n_sector"] = n_sec
    tab.loc[rr.index, "n_industry"] = n_ind
    ss = (float((e0 ** 2).sum()), float((e1 ** 2).sum()), float((e2 ** 2).sum()), float((idio ** 2).sum()))
    return Decomposition(tab, m, disp, ss, int(len(rr)), status, note)


def chance_shares(frame: pd.DataFrame, cfg: CrossConfig, seed: int, n_shuffles: int = 20,
                  beta: pd.Series | None = None) -> dict[str, float | None]:
    """Mean variance shares when cohort labels are randomly reassigned among names (same sizes, no structure). This is what
    each level 'explains' by luck; the excess over it is the honest share. Deterministic in `seed`."""
    lab = assign_cohorts(frame, cfg)
    rng = np.random.default_rng(seed)
    acc: dict[str, list[float]] = {"sector": [], "industry": [], "style": [], "idio": []}
    for _ in range(int(n_shuffles)):
        perm = rng.permutation(len(lab))
        sh = lab.copy()
        for c in lab.columns:
            if c != CohortKind.MARKET.value:
                sh[c] = lab[c].to_numpy()[perm]
        s = decompose_day(frame, cfg, sh, beta).shares()
        for k in acc:
            if s[k] is not None:
                acc[k].append(float(s[k]))
    return {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}


def excess_shares(dec: Decomposition, null: Mapping[str, float | None]) -> dict[str, float | None]:
    """Shares minus the chance shares of the same day. sector + industry + style excess = -(idio excess)."""
    sh = dec.shares()
    return {k: (None if sh[k] is None or null.get(k) is None else sh[k] - null[k]) for k in ("sector", "industry", "style", "idio")}


def market_structure(returns: pd.DataFrame, now, market_col: str | None = None, window: int = 60) -> dict[str, float | None]:
    """How market-wide the universe is behaving as of `now`: average pair correlation and the first principal component's share
    of variance (engine.learning.situation.correlation_structure over the last `window` days of a date x ticker return frame).
    A high first-component share means most moves in this window are market-wide by construction."""
    col = returns.columns[0] if len(returns.columns) else None
    cs = correlation_structure(returns, col, now, market_col, window) if col is not None else {}
    return {"avg_pair_corr": cs.get("avg_pair_corr"), "top_eig_share": cs.get("top_eig_share")}


def sector_dispersion(dec: Decomposition) -> dict[str, Any]:
    """Sector leadership of the day: sectors ranked by their (shrunk) effect, with the spread between best and worst - a large
    spread makes a sector-specific label plausible for many names at once; a tiny one says sector effects are noise today."""
    t = dec.table.dropna(subset=["ret"])
    if t.empty:
        return {"n_sectors": 0, "spread": None, "leaders": [], "laggards": []}
    return {"n_sectors": int(t["sector"].round(12).nunique()), "spread": float(t["sector"].max() - t["sector"].min()),
            "leaders": [round(float(x), 6) for x in sorted(t["sector"].unique())[-3:][::-1]],
            "laggards": [round(float(x), 6) for x in sorted(t["sector"].unique())[:3]]}


# ------------------------------------------------------------------------------------------------ persistence history

@dataclasses.dataclass(frozen=True)
class DayStat:
    """What the lab remembers about one processed day (no tickers): the market and style components and the dispersion."""
    date: str
    market: float
    dispersion: float
    style: Mapping[str, float]                # mean style component of each cohort's top-versus-bottom spread
    shares: Mapping[str, float | None]


class ScopeHistory:
    """Bounded history of past days' market / style components. Persistence and z-scores are answered from days STRICTLY
    BEFORE the day being classified (call classify first, push after), and only once min_history days exist."""

    def __init__(self, cfg: CrossConfig):
        self.cfg = cfg
        self.days: deque[DayStat] = deque(maxlen=cfg.history_days)

    def __len__(self) -> int:
        return len(self.days)

    def last_date(self) -> str | None:
        return self.days[-1].date if self.days else None

    def push(self, stat: DayStat) -> None:
        if self.days and as_date(stat.date) <= as_date(self.days[-1].date):
            raise FirewallBreach(f"history push {stat.date} is not after {self.days[-1].date}")
        self.days.append(stat)

    def _series(self, key: str | None = None) -> np.ndarray:
        if key is None:
            return np.array([d.market for d in self.days], dtype="float64")
        return np.array([d.style.get(key, np.nan) for d in self.days], dtype="float64")

    def ready(self) -> bool:
        return len(self.days) >= self.cfg.min_history

    def z(self, value: float, key: str | None = None) -> float | None:
        """Robust z of today's component against the past distribution of the same component."""
        if not self.ready():
            return None
        a = self._series(key)
        a = a[np.isfinite(a)]
        s = robust_scale(a)
        if len(a) < self.cfg.min_history or s <= 0:
            return None
        return float((value - np.median(a)) / s)

    def persistent(self, value: float, key: str | None = None) -> bool | None:
        """Has this component been pushing the same way? True when today's value and the mean of the last (window-1) past days
        share a sign and the window's total exceeds persist_z robust sigmas of a sum of that many independent days. None if too
        little history (an honest 'do not know', which classify_scope reads as not-regime)."""
        if not self.ready():
            return None
        w = self.cfg.persist_window
        a = self._series(key)
        past = a[-(w - 1):]
        allv = a[np.isfinite(a)]
        if len(past) < w - 1 or not np.isfinite(past).all() or not np.isfinite(value):
            return None
        s = robust_scale(allv)
        if s <= 0:
            return None
        total = float(past.sum() + value)
        return bool(np.sign(value) == np.sign(past.mean()) and abs(total) >= self.cfg.persist_z * math.sqrt(w) * s)

    def dispersion_z(self, value: float) -> float | None:
        if not self.ready():
            return None
        a = np.array([d.dispersion for d in self.days], dtype="float64")
        a = a[np.isfinite(a)]
        s = robust_scale(a)
        return None if s <= 0 or len(a) < self.cfg.min_history else float((value - np.median(a)) / s)


def style_spreads(dec: Decomposition, labels: pd.DataFrame) -> dict[str, float]:
    """Signed cohort-level style move per style kind: the mean style component of the top-quantile cohort minus the bottom-quantile
    cohort (event: live minus none). This is what persistence is measured on - a cohort moving together, with a sign."""
    out: dict[str, float] = {}
    t = dec.table
    for k in STYLE_KINDS:
        col = f"style_{k.value}"
        lab = labels[k.value].reindex(t.index)
        ok = t[col].notna() & (lab != UNK)
        if ok.sum() < 4:
            continue
        g = t.loc[ok, col].groupby(lab[ok]).mean()
        if len(g) >= 2:
            out[k.value] = float(g.iloc[-1] - g.iloc[0]) if k != CohortKind.EVENT else float(g.get("ev_live", g.iloc[-1]) - g.get("ev_none", g.iloc[0]))
    return out


@dataclasses.dataclass(frozen=True)
class ScopeResult:
    """Scope of every name for one day, with the shares behind each call. Identity is the ticker index (trusted side)."""
    scope: pd.Series                          # MoveScope value per ticker
    shares: pd.DataFrame                      # market_share, sector_share, style_share, idio_share
    dominant_style: pd.Series                 # style kind with the largest |component| per ticker ('' if none)
    persistent_market: bool | None
    persistent_style: Mapping[str, bool | None]
    market_z: float | None
    total_scale: float

    def counts(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.scope.value_counts().items()}


def classify_day(dec: Decomposition, labels: pd.DataFrame, history: ScopeHistory, cfg: CrossConfig,
                 spreads: Mapping[str, float] | None = None) -> ScopeResult:
    """Label every name's move for the day. A name is
      NO_MOVE          |ret| under quiet_z x the day's robust total scale (sqrt(market^2 + dispersion^2));
      MARKET_WIDE      the market component is >= share_bar of its attributed move (REGIME_WIDE if it has persisted);
      SECTOR_SPECIFIC  sector + industry components >= share_bar;
      COHORT_WIDE      the style block >= share_bar (REGIME_WIDE if that cohort's spread has persisted);
      STOCK_SPECIFIC   the idiosyncratic residual >= share_bar;
      MIXED            nobody reaches share_bar.
    Persistence is answered from days strictly before this one; without enough history it is None and the label stays
    MARKET_WIDE / COHORT_WIDE - never REGIME_WIDE on a guess."""
    t = dec.table
    idx = t.index
    empty_sh = pd.DataFrame(np.nan, index=idx, columns=["market_share", "sector_share", "style_share", "idio_share"])
    if not dec.usable():
        return ScopeResult(pd.Series(MoveScope.UNKNOWN.value, index=idx), empty_sh, pd.Series("", index=idx), None, {}, None, float("nan"))
    a_m = t["market"].abs()
    a_sec = t["sector"].abs() + t["industry"].abs()
    a_sty = t["style"].abs()
    a_id = t["idio"].abs()
    tot = (a_m + a_sec + a_sty + a_id)
    sh = pd.DataFrame({"market_share": a_m / tot, "sector_share": a_sec / tot, "style_share": a_sty / tot, "idio_share": a_id / tot})
    scale = math.sqrt(dec.market ** 2 + dec.dispersion ** 2)
    pers_m = history.persistent(dec.market)
    pers_s = {k: history.persistent(v, k) for k, v in (spreads or {}).items()}
    style_cols = [f"style_{k.value}" for k in STYLE_KINDS]
    dom = t[style_cols].abs().idxmax(axis=1, skipna=True).astype("string").str.replace("style_", "", regex=False).fillna("")
    winner = sh.to_numpy().argmax(axis=1)
    top = sh.to_numpy().max(axis=1)
    names = np.array(["market", "sector", "style", "idio"])[winner]
    scope = np.full(len(t), MoveScope.MIXED.value, dtype=object)
    bar = cfg.share_bar
    scope[(names == "market") & (top >= bar)] = MoveScope.MARKET_WIDE.value
    scope[(names == "sector") & (top >= bar)] = MoveScope.SECTOR_SPECIFIC.value
    scope[(names == "style") & (top >= bar)] = MoveScope.COHORT_WIDE.value
    scope[(names == "idio") & (top >= bar)] = MoveScope.STOCK_SPECIFIC.value
    scope = pd.Series(scope, index=idx)
    if pers_m:
        scope[scope == MoveScope.MARKET_WIDE.value] = MoveScope.REGIME_WIDE.value
    for kind, flag in pers_s.items():
        if flag:
            scope[(scope == MoveScope.COHORT_WIDE.value) & (dom == kind)] = MoveScope.REGIME_WIDE.value
    scope[t["ret"].abs() < cfg.quiet_z * scale] = MoveScope.NO_MOVE.value
    scope[t["ret"].isna() | tot.isna() | (tot <= 0)] = MoveScope.UNKNOWN.value
    return ScopeResult(scope, sh, dom, pers_m, pers_s, history.z(dec.market), scale)


# ------------------------------------------------------------------------------------------------ features for both models

VOL_COLUMNS = ("xs_abs_resid_z", "xs_idio_share", "xs_sector_share", "xs_market_share", "xs_style_share", "xs_abs_rel_sector",
               "xs_abs_rel_market", "xs_abs_rel_volatility", "xs_day_disp_z", "xs_scope_code")
DIR_COLUMNS = ("xs_resid_z", "xs_rel_market", "xs_rel_sector", "xs_rel_industry", "xs_rel_volatility", "xs_rel_liquidity",
               "xs_rel_momentum", "xs_rel_event", "xs_day_mkt_z", "xs_scope_code")


def loo_deviation(values: pd.Series, labels: pd.Series) -> pd.Series:
    """values - mean of the OTHER members of the same cohort (NaN for singletons, 'unknown' and missing values): how far a
    stock is from its cohort peers, with the stock itself excluded from the peer average."""
    v = pd.to_numeric(values, errors="coerce")
    ok = v.notna() & (labels != UNK)
    out = pd.Series(np.nan, index=v.index)
    if ok.sum() == 0:
        return out
    g = v[ok].groupby(labels[ok])
    s, n = g.transform("sum"), g.transform("count")
    out.loc[ok] = (v[ok] - (s - v[ok]) / (n - 1).where(n > 1))
    return out


def scope_features(dec: Decomposition, scopes: ScopeResult, labels: pd.DataFrame, history: ScopeHistory) -> pd.DataFrame:
    """Per-name features describing WHY the stock moved as it did, from data through today's close only. Ticker-indexed,
    float32, no identity in any value. Missing cohorts give NaN (never 0). Two consumers: VOL_COLUMNS (magnitudes, for the
    volatility model) and DIR_COLUMNS (signed, for the direction model)."""
    t = dec.table
    f = pd.DataFrame(index=t.index)
    idio_s = robust_scale(t["idio"].to_numpy())
    f["xs_resid_z"] = t["idio"] / idio_s if idio_s > 0 else np.nan
    f["xs_abs_resid_z"] = f["xs_resid_z"].abs()
    for c in scopes.shares.columns:
        f["xs_" + c] = scopes.shares[c]
    f["xs_rel_market"] = t["ret"] - dec.market
    for k in (CohortKind.SECTOR, CohortKind.INDUSTRY, *STYLE_KINDS):
        f[f"xs_rel_{k.value}"] = loo_deviation(t["ret"], labels[k.value].reindex(t.index))
    f["xs_abs_rel_sector"] = f["xs_rel_sector"].abs()
    f["xs_abs_rel_market"] = f["xs_rel_market"].abs()
    f["xs_abs_rel_volatility"] = f["xs_rel_volatility"].abs()
    disp_z = history.dispersion_z(dec.dispersion)
    f["xs_day_disp_z"] = np.nan if disp_z is None else disp_z
    f["xs_day_mkt_z"] = np.nan if scopes.market_z is None else scopes.market_z
    f["xs_scope_code"] = scopes.scope.map(lambda s: SCOPE_CODE[MoveScope(s)]).astype("float64")
    return f.astype("float32")


def volatility_features(feats: pd.DataFrame) -> pd.DataFrame:
    """The subset the volatility model reads: how large the stock's private move is, not which way it went."""
    return feats.reindex(columns=list(VOL_COLUMNS))


def direction_features(feats: pd.DataFrame) -> pd.DataFrame:
    """The subset the direction model reads: signed strength versus each cohort, and the standardised residual."""
    return feats.reindex(columns=list(DIR_COLUMNS))


def divergence_table(feats: pd.DataFrame) -> pd.DataFrame:
    """z of the stock's deviation from each cohort's peers, scaled by that day's robust spread of the same deviation: 'how
    unusual is this stock's gap to its sector / industry / volatility cohort ... today'. Columns are cohort kinds."""
    cols = {k: f"xs_rel_{k}" for k in ("market", "sector", "industry", "volatility", "liquidity", "momentum", "event")}
    out = {}
    for k, c in cols.items():
        if c not in feats.columns:
            continue
        s = robust_scale(feats[c].to_numpy(dtype="float64"))
        out[k] = feats[c] / s if s > 0 else pd.Series(np.nan, index=feats.index)
    return pd.DataFrame(out).astype("float32")


@dataclasses.dataclass(frozen=True)
class Divergence:
    cohort: str
    z: float
    direction: str                           # 'above' | 'below' the cohort


def explain_divergence(z_row: Mapping[str, float] | pd.Series, top: int = 3, min_abs_z: float = 1.0) -> list[Divergence]:
    """Answer 'why is this stock behaving differently from the rest of the market': the cohorts it deviates from most, ranked
    by |z|, above the reporting bar. Identity-free (cohort kinds and numbers only)."""
    items = [(k, float(v)) for k, v in dict(z_row).items() if v is not None and math.isfinite(float(v)) and abs(float(v)) >= min_abs_z]
    items.sort(key=lambda kv: (-abs(kv[1]), kv[0]))
    return [Divergence(k, round(v, 4), "above" if v > 0 else "below") for k, v in items[:top]]


# ------------------------------------------------------------------------------------------------ what each scope did next

@dataclasses.dataclass(frozen=True)
class ScopeEffect:
    """Does a scope carry information about what happens next? metric 'continuation' = fwd x sign(move) (direction model),
    'abs_fwd' = |fwd| (volatility model). effect is the scope's per-date mean MINUS the same date's all-scope mean, so a
    market-wide drift on the day does not read as a scope effect. se is HAC-corrected for label overlap; q is BH across all
    scope x metric rows."""
    scope: str
    metric: str
    effect: float | None
    se: float | None
    t: float | None
    q_value: float | None
    n_dates: int
    n_rows: int
    verdict: str                             # ESTABLISHED | NOT_ESTABLISHED | INSUFFICIENT_DATA

    def established(self) -> bool:
        return self.verdict == "ESTABLISHED"


class ScopeOutcomes:
    """Research-world table of matured (scope, next-h outcome) rows. Rows are added only once their outcome is strictly behind
    `now`; reads drop records whose evidence year is being replayed in disguise (rule 27). No ticker is stored."""

    COLS = ("date", "scope", "sign", "fwd", "sessions", "matured_at", "year", "band")
    BANDS = ((0.10, ">10%"), (0.05, "5-10%"), (0.0, "<5%"))      # canon C67 mover bands on |ret|

    def __init__(self, sessions: int = 5):
        self.sessions = int(sessions)
        self._rows: list[tuple] = []

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, date, scope: pd.Series, ret: pd.Series, fwd: pd.Series, matured_at, now) -> int:
        """Settle one decision day: scope/ret are the day's per-ticker scope and return (known at that close), fwd the realised
        forward return from that close (known only at matured_at). Refuses an outcome not strictly before `now`."""
        require_past(matured_at, now, "scope outcome")
        if as_date(matured_at) <= as_date(date):
            raise FirewallBreach(f"outcome matures {matured_at} on/before its decision date {date}")
        d = as_date(date)
        common = scope.index.intersection(ret.index).intersection(fwd.index)
        n = 0
        for tkr in common:
            f, r, s = fwd[tkr], ret[tkr], scope[tkr]
            if s in (MoveScope.UNKNOWN.value, MoveScope.NO_MOVE.value) or not (np.isfinite(f) and np.isfinite(r)) or r == 0:
                continue
            self._rows.append((d.isoformat(), s, 1 if r > 0 else -1, float(f), self.sessions, as_date(matured_at).isoformat(), d.year,
                               next(name for lo, name in self.BANDS if abs(r) >= lo)))
            n += 1
        return n

    def frame(self, now, replay_years: Iterable[int] = ()) -> pd.DataFrame:
        ys = {int(y) for y in replay_years}
        rows = [r for r in self._rows if as_date(r[5]) < as_date(now) and r[6] not in ys]
        return pd.DataFrame(rows, columns=list(self.COLS))

    def effects(self, now, replay_years: Iterable[int] = (), min_dates: int = 30, min_per_date: int = 3,
                t_bar: float = 2.0, q_bar: float = 0.1, by_band: bool = False) -> list[ScopeEffect]:
        """Established-or-not effect per scope (and per mover band when by_band: scope keys read 'SCOPE@band')."""
        df = self.frame(now, replay_years)
        out: list[ScopeEffect] = []
        if df.empty:
            return out
        if by_band:
            df["scope"] = df["scope"] + "@" + df["band"]
        df["cont"] = df["fwd"] * df["sign"]
        df["absf"] = df["fwd"].abs()
        for metric, col in (("continuation", "cont"), ("abs_fwd", "absf")):
            day_mean = df.groupby("date")[col].mean()
            for scope, g in df.groupby("scope"):
                per = g.groupby("date")[col].agg(["mean", "count"])
                per = per[per["count"] >= min_per_date]
                spread = (per["mean"] - day_mean.reindex(per.index)).dropna()
                if len(spread) < min_dates:
                    out.append(ScopeEffect(scope, metric, None, None, None, None, int(len(spread)), int(len(g)), "INSUFFICIENT_DATA"))
                    continue
                full = spread.reindex(pd.Index(sorted(df["date"].unique()))).to_numpy()
                se = hac_se(full, self.sessions - 1)
                eff = float(spread.mean())
                tt = eff / se if se and se > 1e-15 else None
                out.append(ScopeEffect(scope, metric, eff, se, tt, None, int(len(spread)), int(len(g)), "PENDING"))
        q = benjamini_hochberg([t_to_p(e.t) for e in out])
        final = []
        for e, qq in zip(out, q):
            if e.verdict == "INSUFFICIENT_DATA":
                final.append(e)
                continue
            ok = e.t is not None and abs(e.t) >= t_bar and qq is not None and qq <= q_bar
            final.append(dataclasses.replace(e, q_value=qq, verdict="ESTABLISHED" if ok else "NOT_ESTABLISHED"))
        return final

    def prior(self, scope: str, metric: str, now, replay_years: Iterable[int] = ()) -> float | None:
        """The established effect of a scope on a metric, or None (UNKNOWN) - never a default of 0 that a model could mistake
        for 'measured no effect'."""
        for e in self.effects(now, replay_years):
            if e.scope == scope and e.metric == metric and e.established():
                return e.effect
        return None

    def matured_records(self, now, replay_years: Iterable[int] = ()) -> list[MaturedRecord]:
        """One identity-free MaturedRecord per established effect (no ticker, date or year in the payload)."""
        recs = []
        df = self.frame(now, replay_years)
        through = df["matured_at"].max() if len(df) else None
        for e in self.effects(now, replay_years):
            if not e.established() or through is None:
                continue
            prov = Provenance(created_real=str(as_date(now)), learned_at=through, code_hash=current_code_hash(), outcomes_seen_through=through)
            recs.append(MaturedRecord("XS" + stable_hash([e.scope, e.metric, self.sessions], 10), through,
                                      {"scope": e.scope, "metric": e.metric, "effect": clean_number(e.effect), "t": clean_number(e.t),
                                       "sessions": self.sessions, "n_dates": e.n_dates}, prov, Namespace.MATURED_RESEARCH))
        return recs


def apply_scope_prior(feats: pd.DataFrame, scopes: pd.Series, outcomes: ScopeOutcomes, now,
                      replay_years: Iterable[int] = ()) -> pd.DataFrame:
    """Add the learned per-scope priors to a feature frame: xs_prior_cont (direction) and xs_prior_absmove (volatility). NaN
    where the scope's effect is not established. Built only from outcomes matured before `now`."""
    eff = {(e.scope, e.metric): e.effect for e in outcomes.effects(now, replay_years) if e.established()}
    out = feats.copy()
    out["xs_prior_cont"] = scopes.map(lambda s: eff.get((s, "continuation"), np.nan)).astype("float32")
    out["xs_prior_absmove"] = scopes.map(lambda s: eff.get((s, "abs_fwd"), np.nan)).astype("float32")
    return out


# ------------------------------------------------------------------------------------------------ data health of one day

@dataclasses.dataclass(frozen=True)
class DayHealth:
    """A universe frame can be broken in ways that would fabricate a 'market-wide' move: a holiday with stale prices, a feed that
    repeats one return, a unit error. Health is checked BEFORE decomposition; a failed day is DATA_FAILURE, not a scope."""
    n: int
    finite_share: float
    zero_share: float
    modal_share: float                        # share of names sharing the single most common return value
    extreme_share: float                      # |ret| > 50%
    missing_cohorts: tuple[str, ...]
    flags: tuple[str, ...]

    @property
    def failed(self) -> bool:
        return any(f in ("STALE_PRICES", "REPEATED_VALUE", "TOO_FEW_FINITE", "UNIT_ERROR", "EMPTY") for f in self.flags)


def day_health(frame: pd.DataFrame, cfg: CrossConfig) -> DayHealth:
    if frame is None or frame.empty or cfg.ret not in frame.columns:
        return DayHealth(0, 0.0, 0.0, 0.0, 0.0, (), ("EMPTY",))
    r = pd.to_numeric(frame[cfg.ret], errors="coerce")
    n = len(r)
    fin = r[np.isfinite(r)]
    finite_share = len(fin) / n
    zero_share = float((fin == 0).mean()) if len(fin) else 0.0
    modal = float(fin.round(10).value_counts(normalize=True).iloc[0]) if len(fin) else 0.0
    extreme = float((fin.abs() > 0.5).mean()) if len(fin) else 0.0
    missing = tuple(c for c in (cfg.sector, cfg.vol, cfg.liq, cfg.mom) if c not in frame.columns)
    flags = []
    if finite_share < 0.5 or len(fin) < cfg.min_names:
        flags.append("TOO_FEW_FINITE")
    if zero_share > 0.5 and len(fin) >= cfg.min_names:
        flags.append("STALE_PRICES")
    if modal > 0.5 and zero_share <= 0.5 and len(fin) >= cfg.min_names:
        flags.append("REPEATED_VALUE")
    if extreme > 0.05:
        flags.append("UNIT_ERROR")
    if missing:
        flags.append("MISSING_COHORT_COLUMNS")
    return DayHealth(n, finite_share, zero_share, modal, extreme, missing, tuple(flags))


# ------------------------------------------------------------------------------------------------ beta from the past

class EWBeta:
    """Exponentially weighted market beta per ticker, updated from PAST days only and shrunk toward 1 by effective sample size
    (Vasicek-style: weight n/(n+prior)). With it the market component of a stock is beta x market, so a high-beta name in a
    market-wide rally is not mislabelled stock-specific. Memory O(names). Call beta() BEFORE update() for the same day."""

    def __init__(self, halflife: float = 40.0, prior_weight: float = 20.0):
        if halflife <= 0 or prior_weight < 0:
            raise ValueError("halflife > 0 and prior_weight >= 0 required")
        self.decay = 0.5 ** (1.0 / halflife)
        self.prior = float(prior_weight)
        self.sxy = pd.Series(dtype="float64")
        self.n = pd.Series(dtype="float64")
        self.smm = 0.0

    def update(self, ret: pd.Series, market: float) -> None:
        if not np.isfinite(market):
            return
        r = ret[np.isfinite(ret)]
        idx = self.sxy.index.union(r.index)
        sxy, n = self.sxy.reindex(idx).fillna(0.0), self.n.reindex(idx).fillna(0.0)
        seen = idx.isin(r.index)
        sxy[seen] = sxy[seen] * self.decay + r.reindex(idx[seen]).to_numpy() * market
        n[seen] = n[seen] * self.decay + 1.0
        self.sxy, self.n = sxy, n
        self.smm = self.smm * self.decay + market * market

    def beta(self, index: pd.Index) -> pd.Series:
        """Shrunk beta for each ticker in index; 1.0 for a ticker never seen or before any market variance exists."""
        if self.smm <= 1e-14 or self.sxy.empty:
            return pd.Series(1.0, index=index)
        raw = (self.sxy / self.smm).reindex(index)
        n = self.n.reindex(index).fillna(0.0)
        w = n / (n + self.prior) if self.prior > 0 else (n > 0).astype(float)
        return (w * raw.fillna(1.0) + (1.0 - w) * 1.0).clip(-1.0, 4.0).fillna(1.0)


# ------------------------------------------------------------------------------------------------ multi-day relative strength

def relative_strength_features(ret_window: pd.DataFrame, labels: pd.DataFrame, windows: Sequence[int] = (5, 20)) -> pd.DataFrame:
    """Cumulative (log) return over the last w days minus the leave-one-out mean of the stock's cohort peers over the same days,
    for sector, volatility and the whole market. `ret_window` is (date x ticker) of daily returns through today and nothing after;
    a stock without w days of history gets NaN. Columns xs_rs<w>_<cohort>."""
    out = pd.DataFrame(index=labels.index)
    if ret_window.empty:
        return out.astype("float32")
    lr = np.log1p(ret_window.clip(lower=-0.99))
    for w in windows:
        if len(lr) < w:
            for k in ("market", "sector", "volatility"):
                out[f"xs_rs{w}_{k}"] = np.nan
            continue
        tail = lr.iloc[-w:]
        cum = tail.sum(axis=0, min_count=w).reindex(labels.index)
        out[f"xs_rs{w}_market"] = cum - cum.mean()
        for k in (CohortKind.SECTOR, CohortKind.VOLATILITY):
            out[f"xs_rs{w}_{k.value}"] = loo_deviation(cum, labels[k.value])
    return out.astype("float32")


# ------------------------------------------------------------------------------------------------ do the features carry information?

@dataclasses.dataclass(frozen=True)
class FeatureVerdict:
    feature: str
    target: str                               # 'signed' (direction) or 'abs' (volatility)
    mean_ic: float | None
    t: float | None
    q_value: float | None
    n_days: int
    verdict: str                              # ESTABLISHED | NOT_ESTABLISHED | INSUFFICIENT_DATA

    def established(self) -> bool:
        return self.verdict == "ESTABLISHED"


class FeatureIC:
    """Daily cross-sectional rank information coefficient of each cross-sectional feature against the matured forward return
    (signed: does it predict direction) and its absolute value (does it predict movement). Only the per-day IC is stored,
    never a panel. Outcomes must be strictly behind `now`; replayed evidence years are dropped on read. t uses a HAC error
    for the label overlap, and Benjamini-Hochberg runs across every feature x target."""

    def __init__(self, sessions: int = 5, min_names: int = 30):
        self.sessions = int(sessions)
        self.min_names = min_names
        self._rows: list[tuple] = []

    def __len__(self) -> int:
        return len(self._rows)

    @staticmethod
    def _rank_ic(x: np.ndarray, y: np.ndarray) -> float | None:
        """Spearman correlation of two aligned arrays over their jointly finite entries (None if < 3 or a constant side)."""
        from scipy.stats import rankdata
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            return None
        rx, ry = rankdata(x[ok]), rankdata(y[ok])
        sx, sy = rx.std(), ry.std()
        if sx <= 0 or sy <= 0:
            return None
        return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))

    def add(self, day, feats: pd.DataFrame, fwd: pd.Series, matured_at, now) -> int:
        require_past(matured_at, now, "feature outcome")
        if as_date(matured_at) <= as_date(day):
            raise FirewallBreach(f"outcome matures {matured_at} on/before decision day {day}")
        common = feats.index.intersection(fwd.index)
        if len(common) < self.min_names:
            return 0
        f, y = feats.loc[common], fwd.loc[common].astype("float64")
        yv, ya = y.to_numpy(), y.abs().to_numpy()
        d = as_date(day)
        n = 0
        for c in f.columns:
            x = f[c].to_numpy(dtype="float64")
            if np.isfinite(x).sum() < self.min_names:
                continue
            s, a = self._rank_ic(x, yv), self._rank_ic(x, ya)
            if s is None and a is None:
                continue
            self._rows.append((d.isoformat(), c, s, a, int(np.isfinite(x).sum()), as_date(matured_at).isoformat(), d.year))
            n += 1
        return n

    def summary(self, now, replay_years: Iterable[int] = (), min_days: int = 30, t_bar: float = 2.0, q_bar: float = 0.1) -> list[FeatureVerdict]:
        ys = {int(y) for y in replay_years}
        rows = [r for r in self._rows if as_date(r[5]) < as_date(now) and r[6] not in ys]
        if not rows:
            return []
        df = pd.DataFrame(rows, columns=["date", "feature", "signed", "abs", "n", "matured_at", "year"])
        all_days = pd.Index(sorted(df["date"].unique()))
        out: list[FeatureVerdict] = []
        for feat, g in df.groupby("feature"):
            for target in ("signed", "abs"):
                ic = g.set_index("date")[target].dropna()
                if len(ic) < min_days:
                    out.append(FeatureVerdict(feat, target, None, None, None, int(len(ic)), "INSUFFICIENT_DATA"))
                    continue
                se = hac_se(ic.reindex(all_days).to_numpy(), self.sessions - 1)
                m = float(ic.mean())
                out.append(FeatureVerdict(feat, target, m, m / se if se and se > 1e-15 else None, None, int(len(ic)), "PENDING"))
        q = benjamini_hochberg([t_to_p(v.t) for v in out])
        final = []
        for v, qq in zip(out, q):
            if v.verdict == "INSUFFICIENT_DATA":
                final.append(v)
                continue
            ok = v.t is not None and abs(v.t) >= t_bar and qq is not None and qq <= q_bar
            final.append(dataclasses.replace(v, q_value=qq, verdict="ESTABLISHED" if ok else "NOT_ESTABLISHED"))
        return final

    def keep_for(self, model: str, now, replay_years: Iterable[int] = ()) -> tuple[str, ...]:
        """Feature names with an ESTABLISHED information coefficient for the model ('direction' -> signed, 'volatility' -> abs)."""
        target = {"direction": "signed", "volatility": "abs"}[model]
        return tuple(sorted(v.feature for v in self.summary(now, replay_years) if v.target == target and v.established()))


# ------------------------------------------------------------------------------------------------ which levels explain the moves

class ShareLedger:
    """Per-day variance shares of each level (sector, industry, style, idiosyncratic) and, when available, their excess over
    chance. Summaries answer 'which cohort structure really explains moves' and, given a per-day label (a regime from
    regimes.py, duck-typed as a mapping date -> label), 'how does that change by regime'."""

    LEVELS = ("sector", "industry", "style", "idio")

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, day, shares: Mapping[str, float | None], excess: Mapping[str, float | None] | None = None,
            dispersion: float | None = None, market: float | None = None) -> None:
        if any(shares.get(k) is None for k in self.LEVELS):
            return
        row = {"date": as_date(day).isoformat(), "dispersion": dispersion, "market": market}
        for k in self.LEVELS:
            row[k] = float(shares[k])
            row[f"{k}_excess"] = None if not excess or excess.get(k) is None else float(excess[k])
        if self._rows and row["date"] <= self._rows[-1]["date"]:
            raise FirewallBreach(f"share ledger: {row['date']} is not after {self._rows[-1]['date']}")
        self._rows.append(row)

    def frame(self, now=None) -> pd.DataFrame:
        df = pd.DataFrame(self._rows)
        if now is not None and not df.empty:
            df = df[df["date"].map(lambda d: as_date(d) <= as_date(now))]
        return df

    def summary(self, now=None) -> pd.DataFrame:
        """Per level: mean share, spread, mean excess over chance and the share of days the excess is positive."""
        df = self.frame(now)
        rows = []
        for k in self.LEVELS:
            if df.empty:
                rows.append({"level": k, "mean_share": None, "sd_share": None, "mean_excess": None, "days_above_chance": None, "n": 0})
                continue
            ex = df[f"{k}_excess"].dropna()
            rows.append({"level": k, "mean_share": float(df[k].mean()), "sd_share": float(df[k].std(ddof=1)) if len(df) > 1 else None,
                         "mean_excess": float(ex.mean()) if len(ex) else None,
                         "days_above_chance": float((ex > 0).mean()) if len(ex) else None, "n": int(len(df))})
        return pd.DataFrame(rows).set_index("level")

    def by_label(self, labels: Mapping[str, str], now=None) -> pd.DataFrame:
        """Mean share of each level under each externally supplied day label (e.g. a regime). Days without a label are left out."""
        df = self.frame(now)
        if df.empty:
            return pd.DataFrame()
        df = df.assign(label=df["date"].map(lambda d: labels.get(d)))
        df = df.dropna(subset=["label"])
        return df.groupby("label")[list(self.LEVELS)].agg(["mean", "count"]) if not df.empty else pd.DataFrame()


# ------------------------------------------------------------------------------------------------ sector rotation

class SectorRotation:
    """Do sector leaders keep leading? Stores each day's raw sector mean return (net of the market) by sector family, keeps a
    bounded window, and reports the rank persistence of sector effects at lags 1..k (cross-sectional Spearman between today's
    sector ranking and the ranking `lag` days earlier, averaged over the window). Sector families are coarse labels, not names."""

    def __init__(self, window: int = 120, min_sectors: int = 4):
        self.window = window
        self.min_sectors = min_sectors
        self._days: deque[tuple[str, pd.Series]] = deque(maxlen=window)

    def __len__(self) -> int:
        return len(self._days)

    def push(self, day, dec: Decomposition, labels: pd.DataFrame, min_members: int = 5) -> bool:
        if not dec.usable():
            return False
        sec = labels[CohortKind.SECTOR.value].reindex(dec.table.index)
        e0 = dec.table["ret"] - dec.market
        ok = e0.notna() & (sec != UNK)
        g = e0[ok].groupby(sec[ok]).agg(["mean", "count"])
        g = g[g["count"] >= min_members]
        if len(g) < self.min_sectors:
            return False
        d = as_date(day).isoformat()
        if self._days and d <= self._days[-1][0]:
            raise FirewallBreach(f"sector rotation: {d} is not after {self._days[-1][0]}")
        self._days.append((d, g["mean"]))
        return True

    def rank_persistence(self, max_lag: int = 5) -> dict[int, float | None]:
        days = list(self._days)
        out: dict[int, float | None] = {}
        for lag in range(1, max_lag + 1):
            cs = []
            for i in range(lag, len(days)):
                a, b = days[i][1], days[i - lag][1]
                both = a.index.intersection(b.index)
                if len(both) >= self.min_sectors:
                    x, y = a[both].rank(), b[both].rank()
                    if x.std() > 0 and y.std() > 0:
                        cs.append(float(np.corrcoef(x, y)[0, 1]))
            out[lag] = float(np.mean(cs)) if len(cs) >= 5 else None
        return out

    def leaders(self, top: int = 3) -> list[tuple[str, float]]:
        """Sectors with the highest mean effect over the window (family labels and numbers only)."""
        if not self._days:
            return []
        m = pd.concat([s for _, s in self._days], axis=1).mean(axis=1).sort_values(ascending=False)
        return [(k, round(float(v), 6)) for k, v in m.head(top).items()]


# ------------------------------------------------------------------------------------------------ cohort-level evidence

def event_abnormal(frame: pd.DataFrame, cfg: CrossConfig, min_live: int = 5) -> pd.DataFrame:
    """Stock versus EVENT cohort, one row per event column: names with the event live vs not live on the day - counts, mean
    return of each group net of the day's trimmed market, the difference, and its Welch t (engine.learning.context.welch_t).
    A live-event cohort that moves differently from the rest on a day is evidence for an event-specific (not stock-specific)
    scope. Columns with fewer than min_live live names report a NaN t, not zero."""
    cols = ["event", "n_live", "n_none", "mean_live", "mean_none", "diff", "t"]
    if cfg.ret not in frame.columns:
        return pd.DataFrame(columns=cols)
    r = pd.to_numeric(frame[cfg.ret], errors="coerce")
    r = r.where(np.isfinite(r))
    r = r - robust_center(r.dropna().to_numpy()) if r.notna().any() else r
    rows = []
    for c in cfg.events:
        if c not in frame.columns:
            continue
        ev = pd.to_numeric(frame[c], errors="coerce")
        live, none = r[(ev > 0) & r.notna()], r[(ev <= 0) & r.notna()]
        if len(live) < min_live or len(none) < min_live:
            rows.append({"event": c, "n_live": len(live), "n_none": len(none), "mean_live": np.nan, "mean_none": np.nan, "diff": np.nan, "t": np.nan})
            continue
        t = welch_t(float(live.mean()), float(live.var(ddof=1)), len(live), float(none.mean()), float(none.var(ddof=1)), len(none))
        rows.append({"event": c, "n_live": len(live), "n_none": len(none), "mean_live": float(live.mean()), "mean_none": float(none.mean()),
                     "diff": float(live.mean() - none.mean()), "t": t})
    return pd.DataFrame(rows, columns=cols)


@dataclasses.dataclass(frozen=True)
class CohortMove:
    kind: str
    label: str
    n: int
    mean_dev: float                          # mean of (ret - trimmed market) over the cohort
    z: float                                 # mean_dev / (day's robust dispersion / sqrt(n))


def abnormal_cohorts(dec: Decomposition, labels: pd.DataFrame, z_bar: float = 3.0, min_members: int = 5) -> list[CohortMove]:
    """Cohorts (any kind but market) whose members moved together more than dispersion allows: the cohort mean deviation
    divided by the standard error dispersion/sqrt(n). |z| >= z_bar is evidence of a cohort-wide (sector / industry / style)
    scope for the day that no single stock explains. Sorted by |z|."""
    out: list[CohortMove] = []
    if not dec.usable() or not dec.dispersion or dec.dispersion <= 0:
        return out
    t = dec.table
    dev = t["ret"] - dec.market
    for kind in labels.columns:
        if kind == CohortKind.MARKET.value:
            continue
        lab = labels[kind].reindex(t.index)
        ok = dev.notna() & (lab != UNK)
        g = dev[ok].groupby(lab[ok]).agg(["mean", "count"])
        for label, row in g.iterrows():
            if row["count"] < min_members:
                continue
            z = float(row["mean"] / (dec.dispersion / math.sqrt(row["count"])))
            if abs(z) >= z_bar:
                out.append(CohortMove(kind, str(label), int(row["count"]), float(row["mean"]), z))
    return sorted(out, key=lambda c: -abs(c.z))


def dispersion_components(dec: Decomposition) -> dict[str, float | None]:
    """Robust spread of each component across names: where the day's cross-sectional dispersion comes from. A day whose spread
    is mostly 'sector' has many sector-specific moves; mostly 'idio', many stock-specific ones."""
    if not dec.usable():
        return {k: None for k in ("market_beta", "sector", "industry", "style", "idio", "total")}
    t = dec.table
    out = {k: robust_scale(t[k].to_numpy()) for k in ("sector", "industry", "style", "idio")}
    out["market_beta"] = robust_scale((t["market"] - dec.market).to_numpy())
    out["total"] = robust_scale(t["ret"].to_numpy())
    return out


def mover_scope_table(dec: Decomposition, scopes: ScopeResult, bands: Sequence[float] = (0.05, 0.10)) -> pd.DataFrame:
    """Canon C67 link: of the names that moved 5-10% and more than 10% (up and down, close to close), how many were
    stock-specific vs sector vs market-wide? Rows: band x direction; columns: scope counts. A day's 'hundreds of movers' are
    mostly one scope or the other - and the scopes have different futures (see ScopeOutcomes)."""
    lo, hi = sorted(bands)[0], sorted(bands)[-1]
    r = dec.table["ret"]
    edges = {f"{lo:.0%}-{hi:.0%}": (r.abs() >= lo) & (r.abs() < hi), f">{hi:.0%}": r.abs() >= hi}
    rows = []
    for band, mask in edges.items():
        for direction, dmask in (("up", r > 0), ("down", r < 0)):
            sel = scopes.scope[mask & dmask]
            row = {"band": band, "direction": direction, "n": int(len(sel))}
            row.update({s.value: int((sel == s.value).sum()) for s in MoveScope})
            rows.append(row)
    return pd.DataFrame(rows).set_index(["band", "direction"])


# ------------------------------------------------------------------------------------------------ own idiosyncratic history

class IdioVol:
    """Exponentially weighted idiosyncratic variance per ticker from PAST days: what a normal day of stock-specific movement
    looks like for THIS name. surprise = |idio today| / sqrt(own past idio variance): a stock-specific move that is large
    for the name, not just large in the cross-section. rank = the name's typical idio size versus the other names."""

    def __init__(self, halflife: float = 30.0, min_obs: float = 10.0):
        if halflife <= 0:
            raise ValueError("halflife > 0 required")
        self.decay = 0.5 ** (1.0 / halflife)
        self.min_obs = min_obs
        self.var = pd.Series(dtype="float64")
        self.n = pd.Series(dtype="float64")

    def features(self, idio_today: pd.Series) -> pd.DataFrame:
        """(xs_idio_surprise, xs_idio_vol_rank) using variance from days BEFORE today; NaN until min_obs effective days exist."""
        v = self.var.reindex(idio_today.index)
        n = self.n.reindex(idio_today.index).fillna(0.0)
        ok = (n >= self.min_obs) & (v > 0)
        sd = np.sqrt(v.where(ok))
        out = pd.DataFrame(index=idio_today.index)
        out["xs_idio_surprise"] = idio_today.abs() / sd
        out["xs_idio_vol_rank"] = sd.rank(pct=True) - 0.5
        return out.astype("float32")

    def update(self, idio: pd.Series) -> None:
        x = idio[np.isfinite(idio)]
        idx = self.var.index.union(x.index)
        var, n = self.var.reindex(idx), self.n.reindex(idx).fillna(0.0)
        seen = idx.isin(x.index)
        new = x.reindex(idx[seen]).to_numpy() ** 2
        old = var[seen].to_numpy()
        w = self.decay
        var[seen] = np.where(np.isfinite(old), w * old + (1 - w) * new, new)
        n[seen] = n[seen] * w + 1.0
        self.var, self.n = var, n


# ------------------------------------------------------------------------------------------------ discovered cohorts

def kmeans(X: np.ndarray, k: int, seed: int, n_iter: int = 50, n_init: int = 4) -> tuple[np.ndarray, np.ndarray, float]:
    """Seeded k-means with k-means++ starts. Returns (labels, centroids, inertia) of the best of n_init starts. Deterministic
    in `seed`; empty clusters are re-seeded at the farthest point. Shared with engine.research.regimes."""
    X = np.asarray(X, dtype="float64")
    n = len(X)
    if k < 1 or n < k:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    rng = np.random.default_rng(seed)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(n_init):
        cent = np.empty((k, X.shape[1]))
        cent[0] = X[rng.integers(n)]
        d2 = ((X - cent[0]) ** 2).sum(axis=1)
        for j in range(1, k):
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
            cent[j] = X[rng.choice(n, p=p)]
            d2 = np.minimum(d2, ((X - cent[j]) ** 2).sum(axis=1))
        lab = np.zeros(n, dtype=int)
        for _it in range(n_iter):
            dist = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
            new = dist.argmin(axis=1)
            for j in range(k):
                if not (new == j).any():
                    far = dist[np.arange(n), new].argmax()
                    new[far] = j
            if _it > 0 and (new == lab).all():
                break
            lab = new
            cent = np.vstack([X[lab == j].mean(axis=0) for j in range(k)])
        inertia = float(((X - cent[lab]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (lab.copy(), cent.copy(), inertia)
    return best


def silhouette(X: np.ndarray, labels: np.ndarray, max_n: int = 400, seed: int = 0) -> float | None:
    """Mean silhouette on a seeded subsample (O(n^2)); None if fewer than two clusters or too few points."""
    X = np.asarray(X, dtype="float64")
    if len(np.unique(labels)) < 2 or len(X) < 10:
        return None
    if len(X) > max_n:
        keep = np.random.default_rng(seed).choice(len(X), max_n, replace=False)
        X, labels = X[keep], labels[keep]
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    s = []
    for i in range(len(X)):
        same = (labels == labels[i]) & (np.arange(len(X)) != i)
        if not same.any():
            continue
        a = d[i, same].mean()
        b = min(d[i, labels == c].mean() for c in np.unique(labels) if c != labels[i])
        s.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(s)) if s else None


class DiscoveredCohorts:
    """Data-driven cohorts (nothing hard-coded is treated as final): principal components of the standardised return window,
    loadings clustered with seeded k-means. Fit ONLY on a past return window (date x ticker) that excludes the day being labelled.
    Cluster ids are ordered by size (largest first) so they are stable across refits of similar structure; names outside the fit
    window are labelled 'unknown', never assigned to a nearest cluster on no evidence."""

    def __init__(self, k: int = 6, n_components: int = 5, min_days: int = 40, min_names: int = 60, seed: int = 0):
        self.k, self.n_components, self.min_days, self.min_names, self.seed = k, n_components, min_days, min_names, seed
        self.assign: dict[Any, str] = {}
        self.silhouette: float | None = None
        self.explained: float | None = None

    def fit(self, window: pd.DataFrame) -> bool:
        """Fit on a (date x ticker) window. False (and previous fit kept) when the window is too small or degenerate."""
        w = window.loc[:, window.notna().mean() > 0.9].dropna(axis=0, how="any")
        if len(w) < self.min_days or w.shape[1] < max(self.min_names, self.k * 8):
            return False
        Z = (w - w.mean()) / w.std().replace(0, np.nan)
        Z = Z.dropna(axis=1)
        if Z.shape[1] < max(self.min_names, self.k * 8):
            return False
        U, S, Vt = np.linalg.svd(Z.to_numpy(dtype="float64"), full_matrices=False)
        p = min(self.n_components, len(S))
        load = Vt[:p].T * S[:p]                                     # names x p, scaled by component strength
        lab, _cent, _ = kmeans(load, self.k, self.seed)
        order = {c: i for i, (c, _) in enumerate(sorted(zip(*np.unique(lab, return_counts=True)), key=lambda cn: (-cn[1], cn[0])))}
        self.assign = {t: f"disc_{order[c]}" for t, c in zip(Z.columns, lab)}
        self.silhouette = silhouette(load, lab, seed=self.seed)
        self.explained = float((S[:p] ** 2).sum() / (S ** 2).sum())
        return True

    def labels(self, index: pd.Index) -> pd.Series:
        return pd.Series([self.assign.get(t, UNK) for t in index], index=index, dtype=object)

    def fitted(self) -> bool:
        return bool(self.assign)


def compare_cohort_systems(frame: pd.DataFrame, cfg: CrossConfig, discovered: pd.Series) -> dict[str, float | None]:
    """Does the discovered cohort system explain the day's moves better than SIC sectors? Explained share of the move net of
    the market, sector-only (SIC) vs discovered-only, each leave-one-out and shrunk the same way. Positive `gain` means the
    data-driven cohorts carry structure the SIC families miss."""
    r = pd.to_numeric(frame[cfg.ret], errors="coerce")
    r = r.where(np.isfinite(r))
    if r.notna().sum() < cfg.min_names:
        return {"sic": None, "discovered": None, "gain": None}
    e0 = r - robust_center(r.dropna().to_numpy())
    lab = assign_cohorts(frame, cfg)
    res = {}
    for name, labels in (("sic", lab[CohortKind.SECTOR.value]), ("discovered", discovered.reindex(frame.index).fillna(UNK))):
        eff, _ = loo_group_effect(winsorize(e0.dropna(), cfg.winsor_z), labels.reindex(e0.dropna().index), cfg.shrink_k)
        base = float((e0.dropna() ** 2).sum())
        res[name] = None if base <= 0 else 1.0 - float(((e0.dropna() - eff) ** 2).sum()) / base
    res["gain"] = None if res["sic"] is None or res["discovered"] is None else res["discovered"] - res["sic"]
    return res


def day_record(res: DayResult) -> dict[str, Any]:
    """Compact, identity-free summary of one processed day for the daily autopsy: scope counts, shares, persistence flags,
    health flags, the strongest abnormal cohorts (kind and z only) and the dispersion breakdown. No ticker."""
    dec = res.decomposition
    return {"status": dec.status, "n": dec.n, "scope_counts": res.scopes.counts(),
            "shares": {k: (None if v is None else round(v, 6)) for k, v in dec.shares().items()},
            "market_z": None if res.scopes.market_z is None else round(res.scopes.market_z, 4),
            "persistent_market": res.scopes.persistent_market,
            "health": list(res.health.flags), "dispersion": dispersion_components(dec),
            "abnormal": [(c.kind, round(c.z, 2)) for c in abnormal_cohorts(dec, res.labels)[:5]] if dec.usable() else []}


# ------------------------------------------------------------------------------------------------ the lab

@dataclasses.dataclass(frozen=True)
class DayResult:
    """Everything the lab knows about one processed day (trusted side; ticker-indexed frames)."""
    date: str
    decomposition: Decomposition
    scopes: ScopeResult
    features: pd.DataFrame
    labels: pd.DataFrame
    turnover: Mapping[str, float | None]
    small: Mapping[str, list[str]]
    style_spreads: Mapping[str, float]
    health: DayHealth
    excess: Mapping[str, float | None]


class CrossSectionLab:
    """Streams one day at a time and keeps only what a later day needs: a bounded history of market / style components, a short
    window of idiosyncratic residuals, the previous cohort labels, the scopes awaiting their outcome, and the matured outcome
    table. Memory is O(window x names), so a 38M-name-day panel is processed year by year without being held (rule 27)."""

    def __init__(self, cfg: CrossConfig | None = None, sessions: int = 5, idio_window: int = 40, pending_days: int = 20,
                 rs_windows: Sequence[int] = (5, 20), long_window: int = 60, refit_every: int = 20):
        self.cfg = cfg or CrossConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("bad CrossConfig: " + "; ".join(errs))
        self.history = ScopeHistory(self.cfg)
        self.outcomes = ScopeOutcomes(sessions)
        self.sessions = int(sessions)
        self._idio: deque[tuple[str, pd.Series]] = deque(maxlen=idio_window)
        self._retwin: deque[tuple[str, pd.Series]] = deque(maxlen=max(rs_windows) + 1)
        self.rs_windows = tuple(rs_windows)
        self.beta = EWBeta()
        self.feature_ic = FeatureIC(sessions)
        self.share_ledger = ShareLedger()
        self.rotation = SectorRotation()
        self.health_failures: dict[str, int] = {}
        self.idiovol = IdioVol()
        self.discovered = DiscoveredCohorts(seed=0)
        self._longwin: deque[tuple[str, pd.Series]] = deque(maxlen=long_window)
        self.refit_every = refit_every
        self._pending: dict[str, tuple[pd.Series, pd.Series, pd.DataFrame]] = {}
        self._pending_days = pending_days
        self._prev_labels: pd.DataFrame | None = None
        self._prev_scope: pd.Series | None = None
        self._transitions: dict[tuple[str, str], int] = {}
        self.days = 0
        self.last_date: str | None = None

    def process_day(self, day, frame: pd.DataFrame, now=None, null_seed: int | None = None, n_shuffles: int = 10) -> DayResult:
        """Decompose and label one day. `day` is the session whose close the returns run through; it may not be after `now`
        (default: `day` itself) and must be strictly after the previous processed day (a replay or a step backwards raises).
        A day whose data fail day_health() is DATA_FAILURE: no scope, no history update, counted in health_failures.
        null_seed (optional) also measures chance shares so the day's excess is recorded."""
        d = as_date(day)
        if now is not None and d > as_date(now):
            raise FirewallBreach(f"day {d} is after now={as_date(now)}")
        if self.last_date is not None and d <= as_date(self.last_date):
            raise FirewallBreach(f"day {d} is not after the last processed day {self.last_date}")
        health = day_health(frame, self.cfg)
        self.days += 1
        self.last_date = d.isoformat()
        if health.failed:
            for f in health.flags:
                self.health_failures[f] = self.health_failures.get(f, 0) + 1
            idx = frame.index if frame is not None else pd.Index([])
            dec = Decomposition(_empty_table(idx), float("nan"), float("nan"), (0.0, 0.0, 0.0, 0.0), 0, "DATA_FAILURE", ",".join(health.flags))
            scopes = classify_day(dec, pd.DataFrame(index=idx), self.history, self.cfg)
            return DayResult(d.isoformat(), dec, scopes, pd.DataFrame(index=idx), pd.DataFrame(index=idx), {}, {}, {}, health, {})
        labels = assign_cohorts(frame, self.cfg)
        beta = self.beta.beta(frame.index)
        dec = decompose_day(frame, self.cfg, labels, beta)
        spreads = style_spreads(dec, labels) if dec.usable() else {}
        scopes = classify_day(dec, labels, self.history, self.cfg, spreads)
        feats = scope_features(dec, scopes, labels, self.history) if dec.usable() else pd.DataFrame(index=frame.index)
        turnover = cohort_turnover(self._prev_labels, labels)
        excess: dict[str, float | None] = {}
        if dec.usable():
            self._retwin.append((d.isoformat(), dec.table["ret"].astype("float32")))
            win = pd.DataFrame({k: v for k, v in self._retwin}).T
            feats = pd.concat([feats, relative_strength_features(win, labels, self.rs_windows), self.idiovol.features(dec.table["idio"])], axis=1)
            if self.days % self.refit_every == 1 and len(self._longwin) >= self.discovered.min_days:
                self.discovered.fit(pd.DataFrame({k: v for k, v in self._longwin}).T)        # past days only: today is appended below
            if self.discovered.fitted():
                feats["xs_rel_discovered"] = loo_deviation(dec.table["ret"], self.discovered.labels(frame.index)).astype("float32")
            self._longwin.append((d.isoformat(), dec.table["ret"].astype("float32")))
            self.idiovol.update(dec.table["idio"])
            if null_seed is not None:
                excess = excess_shares(dec, chance_shares(frame, self.cfg, null_seed, n_shuffles, beta))
            self.history.push(DayStat(d.isoformat(), dec.market, dec.dispersion, spreads, dec.shares()))
            self.share_ledger.add(d, dec.shares(), excess or None, dec.dispersion, dec.market)
            self.rotation.push(d, dec, labels)
            self.beta.update(dec.table["ret"], dec.market)
            self._idio.append((d.isoformat(), dec.table["idio"].astype("float32")))
            self._pending[d.isoformat()] = (scopes.scope, dec.table["ret"], feats)
            while len(self._pending) > self._pending_days:
                self._pending.pop(next(iter(self._pending)))
            if self._prev_scope is not None:
                both = self._prev_scope.index.intersection(scopes.scope.index)
                pairs = pd.Series(list(zip(self._prev_scope[both], scopes.scope[both]))).value_counts()
                for k, c in pairs.items():
                    self._transitions[k] = self._transitions.get(k, 0) + int(c)
            self._prev_scope = scopes.scope
        self._prev_labels = labels
        return DayResult(d.isoformat(), dec, scopes, feats, labels, turnover, small_cohorts(labels, self.cfg.min_cohort), spreads,
                         health, excess)

    def settle(self, day, fwd: pd.Series, matured_at, now) -> int:
        """Attach a realised forward return to the scopes decided on `day`. Refused unless matured_at is strictly before `now`."""
        key = as_date(day).isoformat()
        if key not in self._pending:
            return 0
        scope, ret, feats = self._pending[key]
        self.feature_ic.add(day, feats, fwd, matured_at, now)
        return self.outcomes.add(day, scope, ret, fwd, matured_at, now)

    def idio_persistence(self, max_lag: int = 5) -> dict[int, float | None]:
        """Mean cross-sectional rank correlation between a day's idiosyncratic residual and the one `lag` days earlier, over the
        stored window. Near 0: stock-specific moves are noise; clearly positive: they persist (or drift on)."""
        days = list(self._idio)
        out: dict[int, float | None] = {}
        for lag in range(1, max_lag + 1):
            cs = []
            for i in range(lag, len(days)):
                a, b = days[i][1], days[i - lag][1]
                both = a.index.intersection(b.index)
                if len(both) >= 30:
                    x, y = a[both].rank(), b[both].rank()
                    if x.std() > 0 and y.std() > 0:
                        cs.append(float(np.corrcoef(x, y)[0, 1]))
            out[lag] = float(np.mean(cs)) if len(cs) >= 3 else None
        return out

    def scope_transitions(self) -> pd.DataFrame:
        """Row-normalised next-day scope given today's scope (how sticky each scope is, per name). Rows sum to 1."""
        if not self._transitions:
            return pd.DataFrame()
        m = pd.Series(self._transitions).unstack(fill_value=0)
        return m.div(m.sum(axis=1), axis=0)

    def state(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, "cfg": dataclasses.asdict(self.cfg), "sessions": self.sessions, "days": self.days,
                "last_date": self.last_date,
                "history": [dataclasses.asdict(s) for s in self.history.days],
                "outcomes": [list(r) for r in self.outcomes._rows], "feature_ic": [list(r) for r in self.feature_ic._rows],
                "shares": list(self.share_ledger._rows), "health_failures": dict(self.health_failures),
                "beta": {"sxy": self.beta.sxy.to_dict(), "n": self.beta.n.to_dict(), "smm": self.beta.smm}}

    @classmethod
    def from_state(cls, st: Mapping[str, Any]) -> "CrossSectionLab":
        if st.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"lab schema {st.get('schema')!r} != {SCHEMA_VERSION}")
        cfg = CrossConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in st["cfg"].items()})
        lab = cls(cfg, st["sessions"])
        for s in st["history"]:
            lab.history.days.append(DayStat(s["date"], s["market"], s["dispersion"], s["style"], s["shares"]))
        lab.outcomes._rows = [tuple(r) for r in st["outcomes"]]
        lab.feature_ic._rows = [tuple(r) for r in st.get("feature_ic", [])]
        lab.share_ledger._rows = list(st.get("shares", []))
        lab.health_failures = dict(st.get("health_failures", {}))
        b = st.get("beta", {})
        lab.beta.sxy = pd.Series(b.get("sxy", {}), dtype="float64")
        lab.beta.n = pd.Series(b.get("n", {}), dtype="float64")
        lab.beta.smm = float(b.get("smm", 0.0))
        lab.days, lab.last_date = st["days"], st["last_date"]
        return lab

    def content_hash(self) -> str:
        return stable_hash(self.state(), 16)


# ------------------------------------------------------------------------------------------------ trader side and streaming

def to_trader_rows(feats: pd.DataFrame, columns: Sequence[str] | None = None) -> list[dict[str, float]]:
    """Ticker-free rows for the blind trader: one dict of finite floats per name, missing values OMITTED (a NaN is not a number
    the trader may see). Verified with engine.learning.trader_view.assert_trader_safe - a year-like value or a forbidden key
    raises FirewallBreach instead of leaking. Row order follows the frame; the caller keeps the ticker mapping on the trusted side."""
    cols = list(columns) if columns is not None else list(feats.columns)
    rows = []
    for rec in feats.reindex(columns=cols).to_dict("records"):
        rows.append({k: round(float(v), 6) for k, v in rec.items() if v is not None and np.isfinite(v)})
    assert_trader_safe(rows, "cross-section trader rows")
    return rows


def stream(lab: CrossSectionLab, days: Iterable[tuple[Any, pd.DataFrame]], on_day=None) -> dict[str, Any]:
    """Process (day, frame) pairs in order, one at a time, never holding more than one frame. Returns per-scope totals. on_day
    (if given) receives each DayResult and may write exception rows to disk; nothing else is retained."""
    totals: dict[str, int] = {}
    n = 0
    for day, frame in days:
        res = lab.process_day(day, frame)
        n += 1
        for k, v in res.scopes.counts().items():
            totals[k] = totals.get(k, 0) + v
        if on_day is not None:
            on_day(res)
    return {"days": n, "scope_totals": totals, "state_hash": lab.content_hash()}


@dataclasses.dataclass(frozen=True)
class CrossStepResult:
    date: str
    status: str
    n_names: int
    scope_counts: Mapping[str, int]
    shares: Mapping[str, float | None]
    excess_shares: Mapping[str, float | None]
    persistent_market: bool | None
    market_z: float | None
    n_settled: int
    small_cohorts: int


def step(lab: CrossSectionLab, now, day_frame: pd.DataFrame,
         settle: Iterable[tuple[Any, pd.Series, Any]] = (), null_seed: int | None = None, n_shuffles: int = 10) -> CrossStepResult:
    """ONE research-loop step at `now`: settle any scopes whose outcome has matured (settle = [(decision_day, fwd, matured_at)],
    each matured strictly before now), decompose and label today's universe (returns through today's close), and, if null_seed
    is given, measure chance shares so the excess is honest. Returns a summary; the lab holds the state."""
    n_set = 0
    for day, fwd, matured_at in settle:
        n_set += lab.settle(day, fwd, matured_at, now)
    res = lab.process_day(now, day_frame, now, null_seed, n_shuffles)
    dec = res.decomposition
    return CrossStepResult(res.date, dec.status, dec.n, res.scopes.counts(), dec.shares(), res.excess, res.scopes.persistent_market,
                           res.scopes.market_z, n_set, sum(len(v) for v in res.small.values()))


def render_report(lab: CrossSectionLab, now, replay_years: Iterable[int] = ()) -> str:
    """Plain-text report: days processed, market/style history depth, idiosyncratic persistence, scope stickiness and the
    established scope effects. Identity-free."""
    lines = [f"CROSS-SECTION LAB ({SCHEMA_VERSION})  days {lab.days}  history {len(lab.history)}  outcomes {len(lab.outcomes)}"]
    ip = lab.idio_persistence()
    lines.append("idio persistence by lag: " + ", ".join(f"{k}:{'n/a' if v is None else round(v, 4)}" for k, v in ip.items()))
    tr = lab.scope_transitions()
    if not tr.empty:
        lines.append("scope stickiness (next-day scope | today):")
        lines.extend("  " + ln for ln in tr.round(3).to_string().splitlines())
    for e in lab.outcomes.effects(now, replay_years):
        lines.append(f"  {e.scope:<16} {e.metric:<12} effect {e.effect if e.effect is None else round(e.effect, 6)}  "
                     f"t {e.t if e.t is None else round(e.t, 2)}  q {e.q_value if e.q_value is None else round(e.q_value, 4)}  {e.verdict}")
    return "\n".join(lines)
