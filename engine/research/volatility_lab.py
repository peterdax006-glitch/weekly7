"""Volatility laboratory: Objective 1 as its own scientific programme (RESEARCH_BRAIN_CONTRACT C66 section 9; canon C66 + C63).
IMPLEMENTED - NOT VALIDATED (code and unit tests only; no real-data run has been made with this module).

Public entry points
    step(state, now, frame=None)            one scheduled unit of research (the wave-2 research loop calls this)
    run_lab(frame, now, cfg, registry)      the whole laboratory on one matured frame -> LabReport
    stream_walk_forward(loader, years, ...) market-wide, year by year, keeping exception rows only (RESEARCH_MAPPING, rule 27)
    VolatilityModel.fit(...).forecast(...)  calibrated P(move), magnitude and timing for a point-in-time frame

What it does
    * competing hypotheses H1-H9 (engine.research.vol_hypotheses) run side by side over the SAME purged walk-forward folds; a
      naive own-volatility baseline B0 is fitted with them, and only INCREMENTAL value over B0 counts (more patterns != better);
    * each contract-section-9 question is a runnable Study (walk-forward, past-only) with a shuffled-input null control that must
      read ~0 - if it does not, the study is INVALID, not positive;
    * transfer across eras / sectors / stocks / regimes / volatility levels goes through engine.learning.transfer's fold harness;
    * competition between hypotheses reuses engine.learning.competition.Arena;
    * every signal is checked for finding volatility WITHOUT direction (a mover with a coin-flip sign is not a trade);
    * discovered regions become H10+ on probation and count only after replicating on later data.
Namespace: everything produced here lives in MATURED_RESEARCH_STATE; it reaches a decision only through MaturedRecord.gate(now)."""
from __future__ import annotations

import dataclasses as dc
import math
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning import transfer as TR
from engine.learning import transfer_score as TS
from engine.learning.core import (FirewallBreach, Provenance, _StrEnum, as_date, current_code_hash, require_past, stable_hash)
from engine.pattern_movers import auc as _auc
from engine.research import vol_hypotheses as VH
from engine.research.core import MaturedRecord, Namespace, Problem
from engine.stops import wilson

HORIZON_BARS = 5


# ---------------------------------------------------------------------------------------------------------------
# configuration and vocabulary
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class LabConfig:
    move: float = 0.10                   # an extreme move touches +-move inside the holding week (matches fv_pipeline.FVConfig.move)
    top_frac: float = 0.10               # the share of names per date a signal 'selects'
    min_train_dates: int = 52
    test_step_dates: int = 13
    n_boot: int = 300
    min_dates_auc: int = 8               # a fold/group with fewer scored dates is reported but not tested
    alpha: float = 0.10                  # BH level for 'supported'
    null_tol: float = 0.01               # a shuffled-input control may move AUC by at most this much (per-date mean) before INVALID
    max_pair_features: int = 6
    max_units: int = 60_000              # cap on transfer units (seeded date-preserving subsample)
    seed: int = 7

    def validate(self) -> list[str]:
        errs = []
        if not 0 < self.move < 1:
            errs.append("move must be in (0,1)")
        if not 0 < self.top_frac < 0.5:
            errs.append("top_frac must be in (0, 0.5)")
        if self.min_train_dates < 8 or self.test_step_dates < 2:
            errs.append("folds need min_train_dates >= 8 and test_step_dates >= 2")
        if not 0 < self.alpha < 1:
            errs.append("alpha must be in (0,1)")
        return errs

    def fingerprint(self) -> str:
        return stable_hash(dc.asdict(self))


class StudyVerdict(_StrEnum):
    SUPPORTED = "SUPPORTED"              # incremental value over the baseline, BH-significant, null control clean
    NOT_SUPPORTED = "NOT_SUPPORTED"      # tested with enough power and found nothing
    INCONCLUSIVE = "INCONCLUSIVE"        # tested but too few dates/events to say
    UNKNOWN = "UNKNOWN"                  # the inputs the question needs are not in the frame (never read as 'no effect')
    INVALID = "INVALID"                  # the null control failed: the study itself cannot be trusted


class DirectionFlag(_StrEnum):
    DIRECTION_BLIND = "DIRECTION_BLIND"      # finds volatility, and the direction of the found moves is a coin flip
    LOSS_SKEWED = "LOSS_SKEWED"              # finds volatility, and the found moves are significantly more often down
    GAIN_SKEWED = "GAIN_SKEWED"
    DIRECTION_INFORMED = "DIRECTION_INFORMED"
    NOT_A_VOL_SIGNAL = "NOT_A_VOL_SIGNAL"    # does not find volatility at all: direction question is moot
    UNTESTED = "UNTESTED"


OUTCOME_COLS = ("touch", "absmove", "tday", "up", "close", "end")


# ---------------------------------------------------------------------------------------------------------------
# frames: labels from the fv_pipeline panel, validation, and the planted worlds used by the tests
# ---------------------------------------------------------------------------------------------------------------
def first_touch_day(entry: np.ndarray, h: np.ndarray, l: np.ndarray, move: float) -> np.ndarray:
    """1..5: the first holding bar whose high/low touched +-move of the entry open; 0 when none did (timing label)."""
    hit = (h / entry[:, None] - 1 >= move) | (l / entry[:, None] - 1 <= -move)
    return np.where(hit.any(1), hit.argmax(1) + 1, 0)


def frame_from_panel(P, cfg: LabConfig = LabConfig(), sector: Mapping[str, str] | None = None, events: pd.DataFrame | None = None,
                     survivor_free: bool = False, chunk: int = 200_000) -> pd.DataFrame:
    """Laboratory frame from an engine.fv_pipeline.Panel: its point-in-time features X plus outcomes of buying the NEXT open and holding
    five sessions (touch, absmove, tday, up, close) and `end`, the date the outcome is known. `sector` maps ticker -> sector label;
    `events` is an optional frame indexed (date, ticker) with any of days_to_event / insider_n30 / filing_n5 / analog_p, which the caller
    must have built point-in-time. Rows whose entry bar is missing get NaN outcomes (never zero)."""
    lab = P.lab
    F = P.X.astype("float32").copy()
    k, j = lab["k"].to_numpy(), lab["j"].to_numpy()
    tday = np.zeros(len(lab), int)
    for s in range(0, len(lab), chunk):
        o, h, l, c, ok, prev = P.gather(k[s:s + chunk], j[s:s + chunk])
        tday[s:s + chunk] = first_touch_day(o[:, 0], h, l, cfg.move)
    fill = lab["fill_ok"].to_numpy(bool)
    hi, lo = lab["hi"].to_numpy(float), lab["lo"].to_numpy(float)
    absmove = np.fmax(hi, -lo)
    F["absmove"] = np.where(fill, absmove, np.nan)
    F["touch"] = np.where(fill, (absmove >= cfg.move).astype(float), np.nan)
    F["tday"] = np.where(fill, tday, np.nan)
    F["close"] = lab["close"].to_numpy(float)
    F["up"] = np.where(fill, (F["close"].to_numpy() > 0).astype(float), np.nan)
    F["end"] = pd.to_datetime(lab["end"].to_numpy())
    if sector is not None:
        F["sector"] = F.index.get_level_values(1).map(lambda t: sector.get(t, "UNKNOWN")).to_numpy()
    if events is not None:
        F = F.join(events, how="left")
    F.attrs["survivor_free"] = bool(survivor_free)
    return F


def validate_frame(F: pd.DataFrame, cfg: LabConfig = LabConfig()) -> list[str]:
    """Every defect that would silently corrupt a study: shape, index, duplicated rows, outcome ranges, an outcome that ends before it
    starts, tday inconsistent with touch, infinite features. The empty frame is valid (every study then reports UNKNOWN)."""
    errs = []
    if not isinstance(F.index, pd.MultiIndex) or F.index.nlevels != 2:
        return ["index must be a (date, ticker) MultiIndex"]
    miss = [c for c in OUTCOME_COLS if c not in F.columns]
    if miss:
        errs.append(f"missing outcome columns {miss}")
    if len(F) == 0 or miss:
        return errs
    if F.index.duplicated().any():
        errs.append("duplicate (date, ticker) rows")
    dates = pd.to_datetime(F.index.get_level_values(0))
    if (pd.to_datetime(F["end"]) <= dates).any():
        errs.append("an outcome ends on or before its decision date")
    t = F["touch"].dropna()
    if not t.isin([0.0, 1.0]).all():
        errs.append("touch must be 0/1")
    a = F["absmove"].dropna()
    if (a < 0).any() or not np.isfinite(a).all():
        errs.append("absmove must be finite and >= 0")
    bad = F[(F["touch"] == 1) & (F["absmove"] < cfg.move - 1e-9)]
    if len(bad):
        errs.append(f"{len(bad)} rows are touch=1 with absmove below the move threshold")
    td = F["tday"].dropna()
    if len(td) and ((td < 0) | (td > HORIZON_BARS)).any():
        errs.append("tday outside 0..5")
    if ((F["tday"] > 0) & (F["touch"] == 0)).any():
        errs.append("tday>0 on a row that never touched")
    num = F.select_dtypes("number")
    if np.isinf(num.to_numpy(float)).any():
        errs.append("infinite values in numeric columns")
    return errs


def mature_only(F: pd.DataFrame, now) -> tuple[pd.DataFrame, int]:
    """The explicit alternative to check_mature's refusal: drop rows whose outcome was not known before `now`, and return how many."""
    m = pd.to_datetime(F["end"]).dt.normalize() < pd.Timestamp(as_date(now))
    return F[m.to_numpy()], int((~m).sum())


TRUTHS = ("null", "H1", "H2", "H3", "H5", "H6", "H7", "hidden")


def planted_frame(truth: str = "null", n_dates: int = 120, n_tickers: int = 80, seed: int = 0, effect: float = 1.1, base_logit: float = -2.1,
                  dir_signal: float = 0.0, first: str = "2012-01-06", n_sectors: int = 5, with_events: bool = True) -> pd.DataFrame:
    """A synthetic market with a KNOWN volatility mechanism. truth: 'null' (movers are unpredictable), H1/H2/H3/H5/H6/H7 (the mechanism
    of that hypothesis drives the mover logit), 'hidden' (a conjunction of near_hi and near_lo that no seeded hypothesis names).
    Direction is independent of everything unless dir_signal > 0. Weekly dates; outcomes mature 8 calendar days later."""
    if truth not in TRUTHS:
        raise ValueError(f"truth must be one of {TRUTHS}")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(first, periods=n_dates, freq="W-FRI")
    tick = [f"P{i:03d}" for i in range(n_tickers)]
    n = n_dates * n_tickers
    tl = np.tile(np.arange(n_tickers), n_dates)
    dl = np.repeat(np.arange(n_dates), n_tickers)
    ar = np.zeros(n_dates)
    for i in range(1, n_dates):
        ar[i] = 0.85 * ar[i - 1] + rng.normal(0, 0.25)
    m_vol_d = 0.012 * np.exp(ar)
    lv_t = np.log(0.02) + rng.normal(0, 0.4, n_tickers)
    vol20 = np.exp(lv_t[tl] + rng.normal(0, 0.25, n) + 0.3 * ar[dl])
    r1 = rng.normal(0, vol20)
    r5 = rng.normal(0, vol20 * np.sqrt(5))
    F = pd.DataFrame({
        "r1": r1, "r5": r5, "r20": rng.normal(0, vol20 * np.sqrt(20)), "r60": rng.normal(0, vol20 * np.sqrt(60)), "vol20": vol20,
        "atr": vol20 * np.exp(rng.normal(0.3, 0.25, n)), "gap": rng.normal(0, vol20 * 0.5), "absr1": np.abs(r1), "absr5": np.abs(r5),
        "range20": vol20 * np.sqrt(20) * np.exp(rng.normal(0, 0.3, n)), "max5": vol20 * np.abs(rng.normal(0, 1, n)) * 1.6,
        "dist_hi": -rng.exponential(0.08, n), "dist_lo": rng.exponential(0.10, n), "logp": rng.normal(3.0, 0.8, n),
        "vol_surge": np.exp(rng.normal(0, 0.5, n)), "log_dv": rng.normal(15.0, 1.2, n),
        "m_r5": np.repeat(rng.normal(0, 0.01, n_dates), n_tickers), "m_vol": np.repeat(m_vol_d, n_tickers),
        "m_r20": np.repeat(rng.normal(0, 0.02, n_dates), n_tickers), "m_breadth": np.repeat(rng.uniform(0.3, 0.7, n_dates), n_tickers),
    }, index=pd.MultiIndex.from_arrays([dates[dl], np.asarray(tick)[tl]], names=["date", "ticker"]))
    F["sector"] = np.array([f"S{i % n_sectors}" for i in range(n_tickers)])[tl]
    if with_events:
        dte = rng.integers(0, 40, n).astype(float)
        dte[rng.random(n) < 0.3] = np.nan
        F["days_to_event"] = dte
        F["insider_n30"] = rng.poisson(0.6, n).astype(float)
        F["filing_n5"] = rng.poisson(0.4, n).astype(float)
        F["analog_p"] = rng.uniform(0.05, 0.25, n)
    D = VH.derive(F, [f for f in VH.DERIVED if not f.startswith(("ix__", "ixnull")) and not VH.missing_columns((f,), F.columns)])
    z = lambda c: ((D[c] - D[c].mean()) / (D[c].std() + 1e-9)).fillna(0.0).to_numpy(float)
    logit = np.full(n, base_logit)
    if truth == "H1":
        logit += effect * (0.7 * z("lv20") + 0.5 * z("vol_ratio_short"))
    elif truth == "H2":
        logit += effect * (1.4 * (D["ev_soon"].fillna(0).to_numpy(float)) + 0.3 * z("insider_recent"))
    elif truth == "H3":
        logit += effect * (-0.8 * z("dv_level") + 0.5 * z("price_low"))
    elif truth == "H5":
        logit += effect * 0.9 * z("lv20_x_stress")
    elif truth == "H6":
        logit += effect * (-0.8 * z("compress") - 0.5 * z("breakout_prox"))
    elif truth == "H7":
        logit += effect * (0.9 * z("surge_x_move") + 0.3 * z("lvol_surge"))
    elif truth == "hidden":
        hid = (D["near_hi"].to_numpy(float) > -0.04) & (D["near_lo"].to_numpy(float) > 0.02) & (D["near_lo"].to_numpy(float) < 0.12)
        logit += effect * 2.2 * hid
    p = 1 / (1 + np.exp(-logit))
    touch = (rng.random(n) < p).astype(float)
    sign = np.where(rng.random(n) < 0.5 + dir_signal * np.tanh(z("rel_r5")) * 0.5, 1.0, -1.0)
    quiet = np.minimum(0.095, vol20 * np.sqrt(5) * np.abs(rng.normal(0, 1, n)) * 0.8)
    absmove = np.where(touch == 1, 0.10 + rng.exponential(0.05, n), quiet)
    tday = np.where(touch == 1, rng.integers(1, HORIZON_BARS + 1, n), 0)
    if truth == "H2":
        ev = np.nan_to_num(F["days_to_event"].to_numpy(float), nan=99.0)
        tday = np.where((touch == 1) & (ev >= 0) & (ev <= 5), np.clip(ev, 1, 5).astype(int), tday)
    F["touch"], F["absmove"], F["tday"] = touch, absmove, tday.astype(float)
    F["close"] = sign * absmove * rng.uniform(0.6, 1.0, n)
    F["up"] = (F["close"] > 0).astype(float)
    F["end"] = pd.to_datetime(F.index.get_level_values(0)) + pd.Timedelta(days=8)
    F.attrs["survivor_free"] = True
    return F


# ---------------------------------------------------------------------------------------------------------------
# walk-forward folds (purged) and the multi-hypothesis walk-forward
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class WFFold:
    idx: int
    test_dates: tuple            # pandas Timestamps
    now: pd.Timestamp            # first test decision date: every training outcome must have ended strictly before it

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_dates[0]


def make_wf_folds(dates: Iterable, cfg: LabConfig = LabConfig()) -> list[WFFold]:
    """Expanding-origin folds over the sorted unique decision dates: the first fold trains on the first min_train_dates and tests on the
    next test_step_dates, and so on. A trailing chunk shorter than half a step is folded into the previous fold's test set."""
    ud = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(dates)).unique()))
    out: list[WFFold] = []
    s = cfg.min_train_dates
    while s < len(ud):
        e = min(s + cfg.test_step_dates, len(ud))
        if len(ud) - e < cfg.test_step_dates // 2:
            e = len(ud)
        out.append(WFFold(len(out), tuple(ud[s:e]), ud[s]))
        s = e
    return out


def baseline_hypothesis() -> VH.Hypothesis:
    """B0: own-volatility persistence. Any hypothesis must beat this before it is said to add anything."""
    return VH.Hypothesis("B0", "own-volatility baseline", "a stock that has been volatile stays volatile; nothing else", ("lv20",),
                         priors=(("lv20", 1),), falsifier="n/a: it is the yardstick")


@dc.dataclass
class WFResult:
    oos: pd.DataFrame                    # out-of-sample rows: outcomes + p_<hid>, m_<hid>, fold
    folds: list[WFFold]
    fits: list[dict]                     # per fold per hypothesis: ok/reason/n_train/coef/signs
    hids: tuple[str, ...]
    unavailable: dict
    survivor_free: bool
    cfg_hash: str

    def fit_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.fits)


def walk_forward(F: pd.DataFrame, hyps: Sequence[VH.Hypothesis], now, cfg: LabConfig = LabConfig(), fit_cfg: VH.FitConfig = VH.FitConfig(),
                 include_baseline: bool = True) -> WFResult:
    """Out-of-sample probabilities for every hypothesis on the same rows. For fold f the training rows are those whose outcome ended
    strictly before the fold's first decision date; test rows are the fold's dates. `now` bounds the whole exercise: no outcome
    ending at/after it is used, as a test row or otherwise. Hypotheses whose inputs are absent from F are listed under `unavailable`."""
    errs = cfg.validate() + validate_frame(F, cfg)
    if errs:
        raise ValueError("; ".join(errs))
    F, _ = mature_only(F, now)
    F = F[F["touch"].notna()].sort_index()
    hyps = [h for h in hyps if h.state != VH.HypState.RETIRED]
    field = ([baseline_hypothesis()] if include_baseline else []) + list(hyps)
    unavailable = {h.hid: h.missing(F.columns) for h in field if h.kind != VH.HypKind.RESIDUAL and h.missing(F.columns)}
    dates = F.index.get_level_values(0)
    folds = make_wf_folds(dates, cfg)
    ends = pd.to_datetime(F["end"]).to_numpy()
    dnp = pd.to_datetime(dates).to_numpy()
    parts, fits = [], []
    for fd in folds:
        tr = F[ends < np.datetime64(fd.now)]
        te = F[np.isin(dnp, np.array(fd.test_dates, dtype="datetime64[ns]"))]
        if len(te) == 0:
            continue
        chunk = te[[c for c in OUTCOME_COLS + ("sector",) if c in te]].copy()
        chunk["fold"] = fd.idx
        for h in field:
            if h.hid in unavailable:
                chunk[f"p_{h.hid}"], chunk[f"m_{h.hid}"] = np.nan, np.nan
                continue
            fh = VH.fit_hypothesis(h, tr, fd.now, fit_cfg, others=[x for x in field if x.hid not in ("B0", h.hid)])
            p, m = fh.predict(te)
            chunk[f"p_{h.hid}"], chunk[f"m_{h.hid}"] = p, m
            fits.append({"fold": fd.idx, "hid": h.hid, "ok": fh.ok, "reason": fh.reason, "n_train": fh.n_train, "n_events": fh.n_events,
                         "trained_through": fh.trained_through, "coef": dict(fh.coef),
                         "sign_agree": (fh.signs.agree_share if fh.signs else float("nan")),
                         "sign_contradicted": (fh.signs.contradicted if fh.signs else ()),
                         "mechanism_consistent": (fh.signs.mechanism_consistent if fh.signs else None)})
        parts.append(chunk)
    oos = pd.concat(parts) if parts else pd.DataFrame(columns=list(OUTCOME_COLS) + ["fold"])
    return WFResult(oos, folds, fits, tuple(h.hid for h in field if h.hid not in unavailable), unavailable,
                    bool(F.attrs.get("survivor_free", False)), cfg.fingerprint())


# ---------------------------------------------------------------------------------------------------------------
# metrics on out-of-sample rows
# ---------------------------------------------------------------------------------------------------------------
def per_date_table(oos: pd.DataFrame, cols: Sequence[str], top_frac: float = 0.10) -> pd.DataFrame:
    """One row per decision date: n, n_touch, and for each score column the cross-sectional AUC of the score against `touch`, the mover
    rate among the date's top `top_frac` names, and the mean predicted probability. This is the point-in-time snapshot the streaming
    path keeps instead of the rows; every summary below reads only this table."""
    rows = []
    d0 = oos.index.get_level_values(0)
    for d, g in oos.groupby(d0, sort=True):
        y = g["touch"].to_numpy(float)
        rec = {"date": d, "n": len(g), "n_touch": int(np.nansum(y)), "base": float(np.nanmean(y)) if len(g) else np.nan}
        k = max(1, int(round(top_frac * len(g))))
        for c in cols:
            s = g[c].to_numpy(float)
            ok = np.isfinite(s) & np.isfinite(y)
            rec[f"auc_{c}"] = _auc(s[ok], y[ok].astype(bool)) if ok.sum() >= 10 else np.nan
            if ok.sum() >= 10:
                top = np.argsort(-s[ok], kind="mergesort")[:k]
                rec[f"prec_{c}"] = float(y[ok][top].mean())
            else:
                rec[f"prec_{c}"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["n", "n_touch", "base"])


def month_cluster(index) -> np.ndarray:
    return pd.to_datetime(pd.Index(index)).strftime("%Y-%m").to_numpy()


@dc.dataclass(frozen=True)
class ScoreCard:
    hid: str
    n_dates: int
    auc: float                # mean per-date cross-sectional AUC
    auc_lo: float
    auc_hi: float
    edge_p: float             # sign-flip p that mean(auc - 0.5) = 0
    lift: float               # mean top-decile mover rate / base rate
    lift_lo: float
    brier: float
    brier_skill: float        # vs a constant base-rate forecast (out-of-sample base rate of the training folds is not available here, so overall rate)
    ece: float
    mag_rho: float            # Spearman(predicted log move, realised log move)

    def as_dict(self) -> dict:
        return dc.asdict(self)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return float("nan")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    ix = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    err = 0.0
    for b in np.unique(ix):
        m = ix == b
        err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def scorecard(oos: pd.DataFrame, hid: str, table: pd.DataFrame | None = None, cfg: LabConfig = LabConfig()) -> ScoreCard:
    """Summarise one hypothesis's out-of-sample rows. AUC is per-date (cross-sectional ranking: 'which names move THIS week'), bootstrapped
    by month so that neighbouring weeks do not pretend to be independent."""
    pc = f"p_{hid}"
    if table is None and oos is None:
        raise ValueError("scorecard needs rows or a per-date table")
    t = table if table is not None else per_date_table(oos, [pc], cfg.top_frac)
    a = t[f"auc_{pc}"].dropna() if f"auc_{pc}" in t else pd.Series(dtype=float)
    nan = float("nan")
    if len(a) < 2:
        return ScoreCard(hid, int(len(a)), nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    bm = TS.cluster_bootstrap_mean(a.to_numpy() - 0.5, month_cluster(a.index), n_boot=cfg.n_boot, seed=cfg.seed)
    p = TS.cluster_signflip_p(a.to_numpy() - 0.5, month_cluster(a.index), n_perm=2000, seed=cfg.seed)
    tb = t.loc[a.index]
    lifts = (tb[f"prec_{pc}"] / tb["base"].replace(0, np.nan)).dropna()
    lb = TS.cluster_bootstrap_mean(lifts.to_numpy(), month_cluster(lifts.index), n_boot=cfg.n_boot, seed=cfg.seed + 1) if len(lifts) >= 2 else None
    brier = ref = ece = rho = nan
    if oos is not None and pc in oos:                 # row-level measures; a stream keeps only the date table and reports NaN here
        s, y = oos[pc].to_numpy(float), oos["touch"].to_numpy(float)
        ok = np.isfinite(s) & np.isfinite(y)
        if ok.any():
            brier = float(np.mean((s[ok] - y[ok]) ** 2))
            ref = float(np.mean((y[ok].mean() - y[ok]) ** 2))
        ece = expected_calibration_error(s, y)
        mc, lm = oos[f"m_{hid}"].to_numpy(float), np.log(oos["absmove"].to_numpy(float) + 1e-4)
        okm = np.isfinite(mc) & np.isfinite(lm)
        rho = float(pd.Series(mc[okm]).corr(pd.Series(lm[okm]), method="spearman")) if okm.sum() > 30 else nan
    return ScoreCard(hid, int(len(a)), bm.mean + 0.5, bm.lo + 0.5, bm.hi + 0.5, p, float(lifts.mean()) if len(lifts) else nan,
                     lb.lo if lb else nan, brier, (1 - brier / ref) if np.isfinite(ref) and ref > 0 else nan, ece, rho)


@dc.dataclass(frozen=True)
class Increment:
    a: str
    b: str
    n_dates: int
    diff: float               # mean per-date AUC(a) - AUC(b)
    lo: float
    hi: float
    p: float
    q: float = float("nan")   # BH-adjusted across the family it was tested in


def auc_increment(t: pd.DataFrame, a: str, b: str, cfg: LabConfig = LabConfig()) -> Increment:
    """Paired per-date AUC difference of two score columns with a month-cluster bootstrap CI and sign-flip p. `t` is per_date_table output."""
    ca, cb = f"auc_p_{a}", f"auc_p_{b}"
    if ca not in t or cb not in t:
        return Increment(a, b, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    d = (t[ca] - t[cb]).dropna()
    if len(d) < 2:
        return Increment(a, b, int(len(d)), float("nan"), float("nan"), float("nan"), float("nan"))
    cl = month_cluster(d.index)
    bm = TS.cluster_bootstrap_mean(d.to_numpy(), cl, n_boot=cfg.n_boot, seed=cfg.seed)
    return Increment(a, b, int(len(d)), bm.mean, bm.lo, bm.hi, TS.cluster_signflip_p(d.to_numpy(), cl, n_perm=2000, seed=cfg.seed))


def increments_vs_baseline(t: pd.DataFrame, hids: Sequence[str], cfg: LabConfig = LabConfig(), base: str = "B0") -> list[Increment]:
    """Each hypothesis against B0 with Benjamini-Hochberg q-values over the family (nine hypotheses tested at once are not nine
    independent chances)."""
    inc = [auc_increment(t, h, base, cfg) for h in hids if h != base]
    ps = np.array([i.p for i in inc], float)
    q = TS.adjust_many(ps, "bh") if len(ps) else np.array([])
    return [dc.replace(i, q=float(qq)) for i, qq in zip(inc, q)]


# ---------------------------------------------------------------------------------------------------------------
# calibration: forward-only isotonic recalibration, reliability with Wilson intervals
# ---------------------------------------------------------------------------------------------------------------
def forward_calibrate(oos: pd.DataFrame, hid: str) -> pd.Series:
    """Calibrated probability for each OOS row using ONLY earlier folds: fold k is recalibrated by an isotonic map fitted on folds < k
    (NaN for fold 0 and for a fold with no usable history). The result is the honest calibrated forecast, not an in-sample fit."""
    from sklearn.isotonic import IsotonicRegression
    out = pd.Series(np.nan, index=oos.index, dtype=float)
    pc = f"p_{hid}"
    for k in sorted(oos["fold"].unique()):
        past = oos[oos["fold"] < k]
        past = past[past[pc].notna() & past["touch"].notna()]
        cur = oos["fold"].to_numpy() == k
        if len(past) < 200 or past["touch"].nunique() < 2:
            continue
        iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(past[pc].to_numpy(float), past["touch"].to_numpy(float))
        out.loc[cur] = iso.predict(oos.loc[cur, pc].to_numpy(float))
    return out


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Equal-count reliability bins: mean forecast, realised rate and its Wilson interval, and whether the forecast sits inside it."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < bins * 5:
        return pd.DataFrame(columns=["bin", "n", "p_mean", "rate", "lo", "hi", "inside"])
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    ix = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = ix == b
        if not m.any():
            continue
        k, n = int(y[m].sum()), int(m.sum())
        lo, hi = wilson(k, n)
        pm = float(p[m].mean())
        rows.append({"bin": b, "n": n, "p_mean": pm, "rate": k / n, "lo": lo, "hi": hi, "inside": bool(lo <= pm <= hi)})
    return pd.DataFrame(rows)


def calibration_verdict(rel: pd.DataFrame, max_bad_share: float = 0.3) -> dict:
    if len(rel) == 0:
        return {"tested": False, "share_outside": float("nan"), "calibrated": None}
    out = float((~rel["inside"]).mean())
    return {"tested": True, "share_outside": out, "calibrated": bool(out <= max_bad_share),
            "worst_gap": float((rel["p_mean"] - rel["rate"]).abs().max())}


# ---------------------------------------------------------------------------------------------------------------
# direction blindness: does the volatility signal say anything about which way?
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class DirectionCheck:
    hid: str
    flag: DirectionFlag
    n_picks: int
    n_moved_picks: int
    up_share: float                 # share of picks that touched the extreme that closed up
    up_lo: float
    up_hi: float
    p_vs_half: float
    dir_auc: float                  # AUC of the volatility score for up-vs-down among actual movers (0.5 = blind)
    dir_auc_lo: float
    dir_auc_hi: float
    mean_signed_close: float        # mean close-to-entry return of the picks that moved
    worst_decile_close: float       # tail of the picks' closes (the loss side)
    reason: str


def direction_check(oos: pd.DataFrame, hid: str, cfg: LabConfig = LabConfig(), vol_edge: float | None = None) -> DirectionCheck:
    """Among each date's top `top_frac` picks by the volatility score, look only at those that DID move and ask whether their direction
    was predictable. A signal that finds movers (per-date AUC above chance) but cannot rank up-movers above down-movers is flagged
    DIRECTION_BLIND: it is dangerous because the extremes it finds are a coin toss. A significantly down-skewed set is LOSS_SKEWED."""
    pc = f"p_{hid}"
    d = oos[oos[pc].notna() & oos["touch"].notna()]
    empty = DirectionCheck(hid, DirectionFlag.UNTESTED, 0, 0, *(float("nan"),) * 9, "no scored rows")
    if len(d) == 0:
        return empty
    rk = d[pc].groupby(level=0).rank(pct=True, method="first")
    picks = d[rk.to_numpy() > 1 - cfg.top_frac]
    mv = picks[(picks["touch"] == 1) & picks["up"].notna()]
    if vol_edge is None:
        t = per_date_table(d, [pc], cfg.top_frac)
        a = t[f"auc_{pc}"].dropna()
        vol_edge = float(a.mean() - 0.5) if len(a) else float("nan")
    nan = float("nan")
    if not np.isfinite(vol_edge) or vol_edge <= 0.01:
        return dc.replace(empty, flag=DirectionFlag.NOT_A_VOL_SIGNAL, n_picks=int(len(picks)), n_moved_picks=int(len(mv)),
                          reason=f"per-date AUC edge {vol_edge:+.3f} is not above chance")
    if len(mv) < 40:
        return dc.replace(empty, n_picks=int(len(picks)), n_moved_picks=int(len(mv)), reason=f"only {len(mv)} moved picks")
    k, n = int(mv["up"].sum()), len(mv)
    lo, hi = wilson(k, n)
    from scipy.stats import binomtest
    pv = float(binomtest(k, n, 0.5).pvalue)
    mm = d[(d["touch"] == 1) & d["up"].notna()]
    dauc, dlo, dhi = nan, nan, nan
    if mm["up"].nunique() == 2 and len(mm) >= 60:
        s, y = mm[pc].to_numpy(float), mm["up"].to_numpy(bool)
        dauc = _auc(s, y)
        cl = month_cluster(mm.index.get_level_values(0))
        boots = _bootstrap_auc(s, y, cl, cfg.n_boot, cfg.seed)
        dlo, dhi = float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))
    blind = np.isfinite(dlo) and dlo <= 0.5 <= dhi
    if pv < 0.01 and lo > 0.5 and not blind:
        flag, why = DirectionFlag.GAIN_SKEWED, "picked movers close up significantly more often"
    elif pv < 0.01 and hi < 0.5:
        flag, why = DirectionFlag.LOSS_SKEWED, "picked movers close down significantly more often: the volatility it finds is mostly loss"
    elif blind:
        flag, why = DirectionFlag.DIRECTION_BLIND, "finds movers, but cannot tell up-movers from down-movers"
    elif np.isfinite(dlo) and dlo > 0.5:
        flag, why = DirectionFlag.DIRECTION_INFORMED, "the volatility score also ranks up-movers above down-movers"
    else:
        flag, why = DirectionFlag.UNTESTED, "direction AUC not estimable"
    cl_ret = mv["close"].to_numpy(float)
    return DirectionCheck(hid, flag, int(len(picks)), n, k / n, lo, hi, pv, dauc, dlo, dhi, float(np.nanmean(cl_ret)),
                          float(np.nanquantile(cl_ret, 0.1)), why)


def _bootstrap_auc(s: np.ndarray, y: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Cluster bootstrap of a pooled AUC: whole clusters (months) are resampled."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(clusters)
    g = len(uniq)
    members = [np.flatnonzero(codes == i) for i in range(g)]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, g, g)
        idx = np.concatenate([members[i] for i in pick])
        a = _auc(s[idx], y[idx])
        if np.isfinite(a):
            out.append(a)
    return np.asarray(out) if out else np.array([0.5])


# ---------------------------------------------------------------------------------------------------------------
# does the signal decay, stop suddenly, or depend on the regime?
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class DecayCheck:
    hid: str
    n_folds: int
    fold_auc: tuple
    slope_per_fold: float
    early: float
    late: float
    late_minus_early: float
    late_lo: float
    late_hi: float
    status: str                    # STABLE | DECAYING | STOPPED | IMPROVING | UNTESTED
    reason: str


def decay_check(t: pd.DataFrame, oos: pd.DataFrame, hid: str, cfg: LabConfig = LabConfig()) -> DecayCheck:
    """Per-fold mean AUC trend and a late-vs-early comparison. STOPPED = the latest quarter of folds is not above chance while the earlier
    ones were clearly above it ('a signal that suddenly stops working'); DECAYING = significantly lower than early and trending down."""
    col = f"auc_p_{hid}"
    fmap = oos.groupby(level=0)["fold"].first()
    a = t[col].dropna() if col in t else pd.Series(dtype=float)
    if len(a) < 4 * cfg.min_dates_auc:
        return DecayCheck(hid, 0, (), float("nan"), *(float("nan"),) * 5, "UNTESTED", f"only {len(a)} scored dates")
    fa = a.groupby(fmap.reindex(a.index)).mean()
    vals = fa.to_numpy()
    if len(vals) < 4:
        return DecayCheck(hid, len(vals), tuple(vals), float("nan"), *(float("nan"),) * 5, "UNTESTED", "fewer than four folds")
    slope = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])
    cut = max(2, int(round(0.75 * len(fa))))
    early_folds, late_folds = set(fa.index[:cut]), set(fa.index[cut:])
    fm = fmap.reindex(a.index)
    ea, la = a[fm.isin(early_folds)], a[fm.isin(late_folds)]
    diff = TS.cluster_bootstrap_diff(la.to_numpy(), ea.to_numpy(), month_cluster(la.index), month_cluster(ea.index), n_boot=cfg.n_boot, level=0.99, seed=cfg.seed)
    lb = TS.cluster_bootstrap_mean(la.to_numpy() - 0.5, month_cluster(la.index), n_boot=cfg.n_boot, seed=cfg.seed)
    early, late = float(ea.mean()), float(la.mean())
    if early > 0.53 and lb.lo <= 0.0 and late < 0.52:
        st, why = "STOPPED", "earlier folds were above chance; the latest folds are not distinguishable from chance"
    elif diff.hi < -0.01 and slope < 0 and len(late_folds) >= 2:
        st, why = "DECAYING", "latest folds significantly weaker than early folds, trend downwards"
    elif diff.lo > 0.01 and slope > 0 and len(late_folds) >= 2:
        st, why = "IMPROVING", "latest folds significantly stronger"
    else:
        st, why = "STABLE", "no significant change between early and late folds"
    return DecayCheck(hid, len(vals), tuple(float(v) for v in vals), slope, early, late, diff.mean, diff.lo, diff.hi, st, why)


@dc.dataclass(frozen=True)
class RegimeCheck:
    hid: str
    by: str
    groups: tuple                  # ((label, n_dates, mean_auc), ...)
    spread: float
    spread_lo: float
    spread_hi: float
    regime_dependent: bool
    reason: str


def regime_labels(oos: pd.DataFrame) -> pd.Series:
    """CALM / NORMAL / STRESS from the market-volatility level's expanding percentile (only the market's own past). Per DATE label."""
    if "m_vol" not in oos:
        return pd.Series(dtype=object)
    d = oos["m_vol"].groupby(level=0).first().sort_index()
    pct = VH.expanding_pct(d, min_periods=8)
    lab = pd.Series(np.where(pct < 1 / 3, "CALM", np.where(pct < 2 / 3, "NORMAL", "STRESS")), index=d.index, dtype=object)
    lab[pct.isna()] = "UNKNOWN"
    return lab


def regime_check(t: pd.DataFrame, oos: pd.DataFrame, hid: str, by: str = "regime", cfg: LabConfig = LabConfig()) -> RegimeCheck:
    """AUC by market regime (by='regime') or by calendar year (by='year'). regime_dependent = the best and worst group differ by at least
    0.03 AUC with a bootstrap interval excluding zero, each group having at least min_dates_auc dates."""
    col = f"auc_p_{hid}"
    a = t[col].dropna() if col in t else pd.Series(dtype=float)
    if by == "regime":
        lab = regime_labels(oos).reindex(a.index)
    elif by == "year":
        lab = pd.Series(pd.to_datetime(a.index).year.astype(str), index=a.index)
    else:
        raise ValueError("by must be 'regime' or 'year'")
    lab = lab.fillna("UNKNOWN")
    groups = []
    for g, sub in a.groupby(lab):
        if g != "UNKNOWN" and len(sub) >= cfg.min_dates_auc:
            groups.append((str(g), int(len(sub)), float(sub.mean())))
    if len(groups) < 2:
        return RegimeCheck(hid, by, tuple(groups), float("nan"), float("nan"), float("nan"), False, "fewer than two groups with enough dates")
    best, worst = max(groups, key=lambda x: x[2]), min(groups, key=lambda x: x[2])
    hi_s, lo_s = a[lab == best[0]], a[lab == worst[0]]
    diff = TS.cluster_bootstrap_diff(hi_s.to_numpy(), lo_s.to_numpy(), month_cluster(hi_s.index), month_cluster(lo_s.index), n_boot=cfg.n_boot, level=0.99, seed=cfg.seed)
    dep = bool(diff.mean >= 0.03 and diff.lo > 0)
    return RegimeCheck(hid, by, tuple(groups), diff.mean, diff.lo, diff.hi, dep,
                       f"{best[0]} {best[2]:.3f} vs {worst[0]} {worst[2]:.3f}" + (": regime-dependent" if dep else ": no reliable difference"))


# ---------------------------------------------------------------------------------------------------------------
# unexplained extremes (H9's honest denominator)
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class UnexplainedShare:
    n_extremes: int
    n_unexplained: int
    share: float
    lo: float
    hi: float
    by_hyp_hit: dict               # hid -> share of extremes that hid ranked in its top decile that day
    reason: str


def unexplained_extremes(oos: pd.DataFrame, hids: Sequence[str], cfg: LabConfig = LabConfig()) -> UnexplainedShare:
    """Share of extreme movers that NONE of the hypotheses ranked in their top `top_frac*3` (a generous net) that day: the size of the
    hole H9 is supposed to describe. Wilson interval; 'unknown' stays a number, not a footnote."""
    hs = [h for h in hids if f"p_{h}" in oos and oos[f"p_{h}"].notna().any()]
    ex = oos[oos["touch"] == 1]
    if not hs or len(ex) == 0:
        return UnexplainedShare(int(len(ex)), 0, float("nan"), float("nan"), float("nan"), {}, "no scored extremes")
    net = min(0.5, cfg.top_frac * 3)
    caught = np.zeros(len(oos), bool)
    per = {}
    for h in hs:
        rk = oos[f"p_{h}"].groupby(level=0).rank(pct=True, method="first").to_numpy()
        top = rk > 1 - net
        caught |= top
        per[h] = float(top[(oos["touch"] == 1).to_numpy()].mean())
    unexpl = int(((oos["touch"] == 1).to_numpy() & ~caught).sum())
    lo, hi = wilson(unexpl, len(ex))
    return UnexplainedShare(int(len(ex)), unexpl, unexpl / len(ex), lo, hi, per, "share of extremes outside every hypothesis's net")


# ---------------------------------------------------------------------------------------------------------------
# competition between hypotheses (engine.learning.competition.Arena)
# ---------------------------------------------------------------------------------------------------------------
def run_competition(oos: pd.DataFrame, hids: Sequence[str], now, *, rows_per_date: int = 30, seed: int = 0, outcome: str = "touch") -> dict:
    """Let the hypotheses compete as explanations of the realised size of moves. Each hypothesis contributes its logit-probability as the
    only feature of a CAUSAL spec in an Arena beside a NULL spec; batches are decision dates in time order, a seeded sample per date.
    The leader, weights, statuses, and the Arena's own account of why it may be undecided are returned; an Arena that cannot separate
    the hypotheses says so - this function never declares a winner the evidence does not support."""
    from engine.learning import competition as CP
    hs = [h for h in hids if f"p_{h}" in oos and oos[f"p_{h}"].notna().all()]
    if len(hs) < 1 or len(oos) == 0:
        return {"tested": False, "reason": "no complete hypothesis columns"}
    d = oos.copy()
    if outcome not in ("touch", "size"):
        raise ValueError("outcome must be 'touch' or 'size'")
    y = d["touch"].to_numpy(float) if outcome == "touch" else np.log(d["absmove"].to_numpy(float) + 1e-3)
    d["y"] = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-9)
    specs = [CP.null_spec()]
    for h in hs:
        zc = f"z_{h}"
        lg = VH._logit(d[f"p_{h}"].to_numpy(float))
        d[zc] = (lg - lg.mean()) / (lg.std() + 1e-9)
        specs.append(CP.causal_spec(zc, hyp_id=h))
    arena = CP.Arena("volatility_size", specs, y_col="y")
    rng = np.random.default_rng(seed)
    dates = d.index.get_level_values(0)
    for dt_, g in d.groupby(dates, sort=True):
        take = g.iloc[np.sort(rng.choice(len(g), min(rows_per_date, len(g)), replace=False))]
        b = take[["y"] + [f"z_{h}" for h in hs]].copy()
        b.index = pd.to_datetime([dt_] * len(b))
        arena.step(b, as_date(now))
    return {"tested": True, "leader": arena.leader(), "weights": arena.weights(), "separated": bool(arena.separated()),
            "status": {k: str(v) for k, v in arena.status().items()}, "winner": arena.winner(), "undecided": arena.undecided_reason(),
            "n_rows": int(arena.n_seen)}


# ---------------------------------------------------------------------------------------------------------------
# the research questions of contract section 9 as runnable studies
# ---------------------------------------------------------------------------------------------------------------
BASE_FEATURES = ("lv20", "shock1")             # what the incremental studies must already have: own volatility and the last shock


@dc.dataclass(frozen=True)
class FeatureGroup:
    features: tuple[str, ...]
    sources: tuple[str, ...]               # base columns shuffled to build the null control
    level: str = "row"                     # 'row' = permute across tickers within a date; 'date' = permute date-level values across dates


FEATURE_GROUPS: dict[str, FeatureGroup] = {
    "volume": FeatureGroup(("lvol_surge", "surge_x_move"), ("vol_surge",)),
    "compression": FeatureGroup(("compress", "squeeze_atr", "breakout_prox"), ("range20", "dist_hi", "dist_lo")),
    "gaps": FeatureGroup(("gap_abs", "gap_ratio"), ("gap",)),
    "intraday_range": FeatureGroup(("latr", "atr_ratio"), ("atr",)),
    "cross_section": FeatureGroup(("xs_vol_rank", "xs_range_rank", "rel_r20", "abs_rel_r20", "rel_r5", "xs_disp"), ("range20", "r20", "r5")),
    "market_vol": FeatureGroup(("mkt_vol", "mkt_stress", "vol_over_mkt", "breadth_dev"), ("m_vol", "m_breadth"), "date"),
    "sector_vol": FeatureGroup(("sector_rel_vol", "sector_vol"), ("sector",)),
    "events": FeatureGroup(("ev_soon", "ev_days"), ("days_to_event",)),
    "insider": FeatureGroup(("insider_recent",), ("insider_n30",)),
    "filings": FeatureGroup(("filing_recent",), ("filing_n5",)),
    "analogs": FeatureGroup(("analog_vol",), ("analog_p",)),
}


@dc.dataclass(frozen=True)
class StudySpec:
    qid: str
    section: str                 # the sentence in contract section 9
    kind: str                    # scan | pairs | group | interaction | transfer
    arg: str = ""                # group name, or transfer axis name
    hypothesis: str = ""         # the hypothesis whose mechanism the question tests
    requires: tuple[str, ...] = ()


STUDIES: tuple[StudySpec, ...] = (
    StudySpec("Q01", "Which features predict extreme movement?", "scan"),
    StudySpec("Q02", "Which combinations predict movement?", "pairs"),
    StudySpec("Q03", "Does volume expansion precede volatility?", "group", "volume", "H7"),
    StudySpec("Q04", "Does unusual return compression precede volatility?", "group", "compression", "H6"),
    StudySpec("Q05", "Do gaps matter?", "group", "gaps", "H3"),
    StudySpec("Q06", "Do intraday ranges matter?", "group", "intraday_range", "H1"),
    StudySpec("Q07", "Do cross-sectional relationships matter?", "group", "cross_section", "H4"),
    StudySpec("Q08", "Does market-wide volatility matter?", "group", "market_vol", "H5"),
    StudySpec("Q09", "Does sector volatility matter?", "group", "sector_vol", "H4"),
    StudySpec("Q10", "Do earnings/event structures matter?", "group", "events", "H2"),
    StudySpec("Q11", "Do insider patterns matter?", "group", "insider", "H2"),
    StudySpec("Q12", "Do filing patterns matter?", "group", "filings", "H2"),
    StudySpec("Q13", "Do previous analogs matter?", "group", "analogs", "H2"),
    StudySpec("Q14", "Do patterns interact?", "interaction", "", "H8"),
    StudySpec("Q15", "Does volatility prediction transfer between eras?", "transfer", "YEAR"),
    StudySpec("Q16", "Does it transfer between sectors?", "transfer", "SECTOR"),
    StudySpec("Q17", "Does it transfer between stocks?", "transfer", "STOCK"),
    StudySpec("Q18", "Does it survive regime changes?", "transfer", "REGIME"),
)


@dc.dataclass(frozen=True)
class StudyResult:
    qid: str
    question: str
    verdict: StudyVerdict
    effect: float                # kind-specific: incremental per-date AUC, or transfer gain in move-size points
    lo: float
    hi: float
    p: float
    q: float = float("nan")      # BH across the family of studies run together
    n_dates: int = 0
    null_effect: float = float("nan")
    detail: Mapping[str, Any] = dc.field(default_factory=dict)
    caveats: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        d = dc.asdict(self)
        d["verdict"] = str(self.verdict)
        return d


def _unknown(spec: StudySpec, why: str, **detail) -> StudyResult:
    nan = float("nan")
    return StudyResult(spec.qid, spec.section, StudyVerdict.UNKNOWN, nan, nan, nan, nan, detail={"why": why, **detail}, caveats=(why,))


def shuffle_sources(F: pd.DataFrame, cols: Sequence[str], level: str, seed: int) -> pd.DataFrame:
    """A copy of F whose source columns are permuted: across tickers within each date (level='row') or, for date-level columns, the whole
    date-level value moved to another date (level='date'). The null control of a group study: same marginals, relation destroyed."""
    out = F.copy()
    rng = np.random.default_rng(seed)
    dates = out.index.get_level_values(0)
    codes, uniq = pd.factorize(dates)
    if level == "date":
        perm = rng.permutation(len(uniq))
        for c in cols:
            first = out[c].groupby(codes).first().to_numpy()
            out[c] = first[perm][codes]
        return out
    for c in cols:
        v = out[c].to_numpy().copy()
        for k in range(len(uniq)):
            ix = np.flatnonzero(codes == k)
            v[ix] = v[ix][rng.permutation(len(ix))]
        out[c] = v
    return out


def _verdict_from_increment(inc: Increment, null: Increment | None, cfg: LabConfig, min_dates: int) -> tuple[StudyVerdict, tuple[str, ...]]:
    cav: list[str] = []
    if inc.n_dates < min_dates or not np.isfinite(inc.diff):
        return StudyVerdict.INCONCLUSIVE, (f"only {inc.n_dates} scored dates",)
    if null is not None and np.isfinite(null.diff) and null.lo > 0 and null.diff > cfg.null_tol:
        return StudyVerdict.INVALID, (f"shuffled-input control gained {null.diff:+.4f} AUC (CI lo {null.lo:+.4f}): the measurement is leaking or mis-specified",)
    if inc.lo > 0 and (not np.isfinite(inc.q) or inc.q <= cfg.alpha):
        return StudyVerdict.SUPPORTED, tuple(cav)
    if inc.hi < 0.01:
        return StudyVerdict.NOT_SUPPORTED, ("interval excludes any gain above 0.01 AUC",)
    return StudyVerdict.INCONCLUSIVE, ("interval includes both no effect and a material effect",)


def feature_increment(F: pd.DataFrame, add: Sequence[str], now, cfg: LabConfig, fit_cfg: VH.FitConfig, base: Sequence[str] = BASE_FEATURES,
                      null_cols: Sequence[str] = (), null_level: str = "row", tag: str = "AUG") -> dict:
    """Walk-forward incremental value of extra derived features over `base` (both logistic). Returns the per-date table, the paired
    increment and, if null_cols is given, the same increment when the source columns are shuffled (must be ~0)."""
    add = tuple(f for f in add if f not in base)
    hb = VH.Hypothesis("BASE", "base", "own volatility and last shock", tuple(base))
    ha = VH.Hypothesis(tag, "augmented", "base plus the group under test", tuple(base) + add)
    wf = walk_forward(F, [hb, ha], now, cfg, fit_cfg, include_baseline=False)
    oos = wf.oos
    cols = ["p_BASE", f"p_{tag}"]
    null_inc = None
    if null_cols and len(oos):
        Fn = shuffle_sources(F.sort_index(), null_cols, null_level, cfg.seed + 101)
        wn = walk_forward(Fn, [dc.replace(ha, hid="NULL")], now, cfg, fit_cfg, include_baseline=False)
        oos = oos.join(wn.oos[["p_NULL"]], how="left")
        cols.append("p_NULL")
    t = per_date_table(oos, cols, cfg.top_frac)
    inc = auc_increment(t, tag, "BASE", cfg)
    if null_cols and "auc_p_NULL" in t:
        null_inc = auc_increment(t, "NULL", "BASE", cfg)
    return {"table": t, "inc": inc, "null": null_inc, "oos": oos, "wf": wf}


def run_group_study(spec: StudySpec, F: pd.DataFrame, now, cfg: LabConfig, fit_cfg: VH.FitConfig) -> StudyResult:
    g = FEATURE_GROUPS[spec.arg]
    miss = VH.missing_columns(g.features, F.columns)
    if miss:
        return _unknown(spec, f"input columns unavailable: {list(miss)}", missing=list(miss))
    r = feature_increment(F, g.features, now, cfg, fit_cfg, null_cols=g.sources, null_level=g.level, tag="AUG")
    inc, null = r["inc"], r["null"]
    v, cav = _verdict_from_increment(inc, null, cfg, cfg.min_dates_auc * 2)
    coefs = [f["coef"] for f in r["wf"].fits if f["hid"] == "AUG" and f["ok"]]
    mean_coef = {k: float(np.mean([c[k] for c in coefs])) for k in g.features if coefs and all(k in c for c in coefs)}
    return StudyResult(spec.qid, spec.section, v, inc.diff, inc.lo, inc.hi, inc.p, float("nan"), inc.n_dates,
                       null.diff if null else float("nan"), {"group": spec.arg, "features": list(g.features), "mean_coef": mean_coef,
                                                              "hypothesis": spec.hypothesis, "null_lo": null.lo if null else float("nan")}, tuple(cav))


def oriented_scan(F: pd.DataFrame, feats: Sequence[str], now, cfg: LabConfig) -> pd.DataFrame:
    """Q01. For every derived feature: its sign is decided on training rows only (pooled AUC on rows ended before each fold), then the
    oriented feature is scored on the fold's test dates. Returns one row per feature with the mean per-date AUC, CI, p and BH q."""
    F, _ = mature_only(F, now)
    F = F[F["touch"].notna()].sort_index()
    D = VH.derive(F, feats)
    folds = make_wf_folds(F.index.get_level_values(0), cfg)
    ends, dn = pd.to_datetime(F["end"]).to_numpy(), pd.to_datetime(F.index.get_level_values(0)).to_numpy()
    y = F["touch"].to_numpy(float)
    signs: dict[str, list[float]] = {f: [] for f in feats}
    cols = {f: np.full(len(F), np.nan) for f in feats}
    for fd in folds:
        tr = np.flatnonzero(ends < np.datetime64(fd.now))
        te = np.flatnonzero(np.isin(dn, np.array(fd.test_dates, dtype="datetime64[ns]")))
        if len(tr) < 200 or len(te) == 0:
            continue
        sub = tr if len(tr) <= 80_000 else np.sort(np.random.default_rng(cfg.seed + fd.idx).choice(tr, 80_000, replace=False))
        for f in feats:
            v = D[f].to_numpy(float)
            ok = np.isfinite(v[sub])
            a = _auc(v[sub][ok], y[sub][ok].astype(bool)) if ok.sum() > 100 else np.nan
            s = 1.0 if not np.isfinite(a) or a >= 0.5 else -1.0
            signs[f].append(s)
            cols[f][te] = s * v[te]
    oos = F[["touch", "absmove", "end"]].copy()
    fold_of = np.full(len(F), -1)
    for fd in folds:
        fold_of[np.isin(dn, np.array(fd.test_dates, dtype="datetime64[ns]"))] = fd.idx
    oos["fold"] = fold_of
    for f in feats:
        oos[f"p_{f}"] = cols[f]
    oos = oos[oos["fold"] >= 0]
    if len(oos) == 0:
        return pd.DataFrame({"feature": list(feats), "n_dates": 0, "auc": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "sign": np.nan, "q": np.nan})
    t = per_date_table(oos, [f"p_{f}" for f in feats], cfg.top_frac)
    rows = []
    for f in feats:
        a = t[f"auc_p_{f}"].dropna()
        if len(a) < 2:
            rows.append({"feature": f, "n_dates": len(a), "auc": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "sign": np.nan})
            continue
        cl = month_cluster(a.index)
        bm = TS.cluster_bootstrap_mean(a.to_numpy(), cl, n_boot=cfg.n_boot, seed=cfg.seed)
        rows.append({"feature": f, "n_dates": len(a), "auc": bm.mean, "lo": bm.lo, "hi": bm.hi,
                     "p": TS.cluster_signflip_p(a.to_numpy() - 0.5, cl, n_perm=1000, seed=cfg.seed),
                     "sign": float(np.mean(signs[f])) if signs[f] else np.nan})
    tab = pd.DataFrame(rows)
    ok = tab["p"].notna()
    tab["q"] = np.nan
    if ok.any():
        tab.loc[ok, "q"] = TS.adjust_many(tab.loc[ok, "p"].to_numpy(), "bh")
    return tab.sort_values("auc", ascending=False, na_position="last").reset_index(drop=True)


def run_scan_study(spec: StudySpec, F: pd.DataFrame, now, cfg: LabConfig) -> StudyResult:
    feats = [f for f in VH.DERIVED if not f.startswith(("ix__", "ixnull")) and not VH.missing_columns((f,), F.columns)]
    absent = [f for f in VH.DERIVED if not f.startswith(("ix__", "ixnull")) and f not in feats]
    if not feats:
        return _unknown(spec, "no derivable features")
    tab = oriented_scan(F, feats, now, cfg)
    ok = tab[tab["auc"].notna()]
    if ok.empty:
        return StudyResult(spec.qid, spec.section, StudyVerdict.INCONCLUSIVE, float("nan"), float("nan"), float("nan"), float("nan"),
                           detail={"table": tab.to_dict("records"), "unavailable": absent}, caveats=("no feature scored on enough dates",))
    top = ok.iloc[0]
    sig = ok[(ok["q"] <= cfg.alpha) & (ok["lo"] > 0.5)]
    v = StudyVerdict.SUPPORTED if len(sig) else (StudyVerdict.NOT_SUPPORTED if top["hi"] < 0.51 else StudyVerdict.INCONCLUSIVE)
    return StudyResult(spec.qid, spec.section, v, float(top["auc"] - 0.5), float(top["lo"] - 0.5), float(top["hi"] - 0.5), float(top["p"]),
                       float(top["q"]), int(top["n_dates"]), detail={"table": tab.to_dict("records"), "significant": sig["feature"].tolist(),
                                                                    "unavailable": absent},
                       caveats=("univariate: a feature can rank movers only because it proxies own volatility (see Q03-Q13 for incremental value)",))


def _feature_edge(v: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(v)
    a = _auc(v[ok], y[ok].astype(bool)) if ok.sum() > 100 else np.nan
    return abs(a - 0.5) if np.isfinite(a) else 0.0


def run_pairs_study(spec: StudySpec, F: pd.DataFrame, now, cfg: LabConfig, fit_cfg: VH.FitConfig) -> StudyResult:
    """Q02. The K strongest features are chosen on rows that ended before the FIRST test fold (so no fold's choice used its own future),
    every pair's product is tested for incremental value over the additive model, and control pairs (second factor permuted within
    date) calibrate what luck looks like."""
    feats = [f for f in VH.DERIVED if not f.startswith(("ix__", "ixnull")) and f not in BASE_FEATURES and not VH.missing_columns((f,), F.columns)]
    Fm, _ = mature_only(F, now)
    Fm = Fm[Fm["touch"].notna()].sort_index()
    folds = make_wf_folds(Fm.index.get_level_values(0), cfg)
    if len(feats) < 3 or not folds:
        return _unknown(spec, "fewer than three derivable features or no folds")
    tr = Fm[pd.to_datetime(Fm["end"]) < folds[0].now]
    if len(tr) < 400:
        return StudyResult(spec.qid, spec.section, StudyVerdict.INCONCLUSIVE, float("nan"), float("nan"), float("nan"), float("nan"),
                           detail={"why": "too little training history for feature selection"}, caveats=("too little history",))
    D = VH.derive(tr, feats)
    y = tr["touch"].to_numpy(float)
    edge = {f: _feature_edge(D[f].to_numpy(float), y) for f in feats}
    top = sorted(feats, key=lambda f: -edge[f])[:cfg.max_pair_features]
    pairs = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]]
    additive = tuple(BASE_FEATURES) + tuple(top)
    rows = []
    for a, b in pairs:
        nm = VH.register_interaction(a, b)
        nn = VH.register_interaction(a, b, null_seed=cfg.seed + 17)
        wf = walk_forward(F, [VH.Hypothesis("ADD", "additive", "top features, additive", additive),
                              VH.Hypothesis("IX", "with pair", "additive plus one product", additive + (nm,)),
                              VH.Hypothesis("NUL", "with control pair", "additive plus a permuted product", additive + (nn,))],
                          now, cfg, fit_cfg, include_baseline=False)
        t = per_date_table(wf.oos, ["p_ADD", "p_IX", "p_NUL"], cfg.top_frac)
        i1, i0 = auc_increment(t, "IX", "ADD", cfg), auc_increment(t, "NUL", "ADD", cfg)
        rows.append({"a": a, "b": b, "diff": i1.diff, "lo": i1.lo, "hi": i1.hi, "p": i1.p, "null_diff": i0.diff, "null_lo": i0.lo, "n_dates": i1.n_dates})
    tab = pd.DataFrame(rows)
    ok = tab["p"].notna()
    tab["q"] = np.nan
    if ok.any():
        tab.loc[ok, "q"] = TS.adjust_many(tab.loc[ok, "p"].to_numpy(), "bh")
    if not ok.any():
        return StudyResult(spec.qid, spec.section, StudyVerdict.INCONCLUSIVE, float("nan"), float("nan"), float("nan"), float("nan"),
                           detail={"pairs": tab.to_dict("records")}, caveats=("no pair scored",))
    best = tab.loc[tab["diff"].idxmax()]
    worst_null = float(tab["null_diff"].max())
    if worst_null > cfg.null_tol and (tab["null_lo"] > 0).any():
        v = StudyVerdict.INVALID
    elif (tab["q"] <= cfg.alpha).any() and best["lo"] > 0:
        v = StudyVerdict.SUPPORTED
    else:
        v = StudyVerdict.NOT_SUPPORTED if best["hi"] < 0.01 else StudyVerdict.INCONCLUSIVE
    return StudyResult(spec.qid, spec.section, v, float(best["diff"]), float(best["lo"]), float(best["hi"]), float(best["p"]), float(best["q"]),
                       int(best["n_dates"]), worst_null, {"pairs": tab.to_dict("records"), "features": top, "best": [best["a"], best["b"]]},
                       ("pairs are searched among the strongest univariate features: a pair of two weak features is never tried",))


def additive_column(oos: pd.DataFrame, hids: Sequence[str]) -> pd.Series:
    """Mean-logit fusion of the additive (non-interacting) hypotheses' OOS probabilities."""
    cols = [f"p_{h}" for h in hids if f"p_{h}" in oos and oos[f"p_{h}"].notna().all()]
    if not cols:
        return pd.Series(np.nan, index=oos.index)
    return pd.Series(VH._expit(np.mean([VH._logit(oos[c].to_numpy(float)) for c in cols], axis=0)), index=oos.index)


def run_interaction_study(spec: StudySpec, wf: WFResult, cfg: LabConfig) -> StudyResult:
    """Q14. Does the interaction-capable model (H8) beat the fused additive models (H1-H7 available)? A gap in H8's favour is evidence
    that combinations carry information the single mechanisms do not; a gap of zero says 'no interactions worth having'."""
    oos = wf.oos
    add_ids = [h for h in wf.hids if h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7")]
    if "H8" not in wf.hids or not add_ids or len(oos) == 0:
        return _unknown(spec, "H8 or the additive hypotheses are not available")
    oos = oos.copy()
    oos["p_ADD"] = additive_column(oos, add_ids)
    if oos["p_ADD"].isna().all():
        return _unknown(spec, "no additive hypothesis has complete out-of-sample probabilities")
    t = per_date_table(oos, ["p_H8", "p_ADD"], cfg.top_frac)
    inc = auc_increment(t, "H8", "ADD", cfg)
    v, cav = _verdict_from_increment(inc, None, cfg, cfg.min_dates_auc * 2)
    return StudyResult(spec.qid, spec.section, v, inc.diff, inc.lo, inc.hi, inc.p, float("nan"), inc.n_dates,
                       detail={"additive": add_ids}, caveats=tuple(cav))


# ---------------------------------------------------------------------------------------------------------------
# transfer across eras / sectors / stocks / regimes / volatility levels (engine.learning.transfer fold harness)
# ---------------------------------------------------------------------------------------------------------------
def transfer_units(F: pd.DataFrame, now, cfg: LabConfig = LabConfig()) -> pd.DataFrame:
    """Decision units for the transfer harness. For every (date, ticker): base = the universe's mean move size that date (what
    'no selection' yields), alt = the name's realised move size; a selector's weight w in [0,1] then gives gain = w*(alt-base), i.e. the
    excess move size captured by the names it selects. Labels: year, regime (market-volatility expanding percentile), sector, and
    vol_bucket (the name's same-date volatility tercile). Rows are a seeded per-date subsample above cfg.max_units."""
    Fm, _ = mature_only(F, now)
    Fm = Fm[Fm["touch"].notna() & Fm["absmove"].notna()].sort_index()
    if len(Fm) == 0:
        return pd.DataFrame(columns=["date", "mature", "ticker", "base", "alt"])
    if len(Fm) > cfg.max_units:
        keep = np.sort(np.random.default_rng(cfg.seed).choice(len(Fm), cfg.max_units, replace=False))
        Fm = Fm.iloc[keep]
    d = Fm.index.get_level_values(0)
    U = Fm.reset_index()
    U = U.rename(columns={"date": "date", "ticker": "ticker"})
    U["mature"] = pd.to_datetime(Fm["end"].to_numpy())
    U["alt"] = Fm["absmove"].to_numpy(float)
    U["base"] = Fm["absmove"].groupby(d).transform("mean").to_numpy(float)
    reg = regime_labels(Fm)
    U["regime"] = reg.reindex(d).to_numpy() if len(reg) else "UNKNOWN"
    if "vol20" in Fm:
        rk = Fm["vol20"].groupby(level=0).rank(pct=True).to_numpy()
        U["vol_bucket"] = np.where(rk < 1 / 3, "V1", np.where(rk < 2 / 3, "V2", "V3"))
    U["year"] = pd.to_datetime(U["date"]).dt.year
    if "sector" in U:
        U["sector"] = U["sector"].fillna("UNKNOWN")
    U["learned"] = U["base"]
    return U


def _to_multi(t: pd.DataFrame) -> pd.DataFrame:
    return t.set_index([pd.to_datetime(t["date"]), t["ticker"].astype(str)]).sort_index()


def selector_fit(h: VH.Hypothesis, fit_cfg: VH.FitConfig, top_frac: float) -> Callable:
    """Adapter from a hypothesis to the harness's Fit protocol: train on the training units, return w = 1 for the names ranked in the
    top `top_frac` of the test rows' own date (else 0). An unfittable hypothesis selects nothing (gain 0), never a random set."""
    def fit(train: pd.DataFrame):
        tm = _to_multi(train)
        fh = VH.fit_hypothesis(h, tm, pd.to_datetime(tm["end"]).max() + pd.Timedelta(days=1), fit_cfg)

        def predict(test: pd.DataFrame) -> np.ndarray:
            if not fh.ok or len(test) == 0:
                return np.zeros(len(test))
            tt = _to_multi(test)
            p, _ = fh.predict(tt)
            rk = pd.Series(p, index=tt.index).groupby(level=0).rank(pct=True, method="first")
            sel = (rk.to_numpy() > 1 - top_frac).astype(float)
            pos = pd.Series(sel, index=pd.MultiIndex.from_arrays([tt.index.get_level_values(0), tt.index.get_level_values(1)]))
            keyed = pd.MultiIndex.from_arrays([pd.to_datetime(test["date"]), test["ticker"].astype(str)])
            return pos.reindex(keyed).fillna(0.0).to_numpy()
        return predict
    return fit


def run_transfer_study(spec: StudySpec, F: pd.DataFrame, h: VH.Hypothesis, now, cfg: LabConfig, fit_cfg: VH.FitConfig) -> StudyResult:
    axis = TR.Axis[spec.arg]
    if h.missing(F.columns):
        return _unknown(spec, f"hypothesis {h.hid} inputs unavailable: {list(h.missing(F.columns))}")
    if axis == TR.Axis.SECTOR and "sector" not in F.columns:
        return _unknown(spec, "no sector column: the sector axis is untestable, not passed")
    U = transfer_units(F, now, cfg)
    if len(U) < 400:
        return StudyResult(spec.qid, spec.section, StudyVerdict.INCONCLUSIVE, float("nan"), float("nan"), float("nan"), float("nan"),
                           detail={"why": f"only {len(U)} units"}, caveats=(f"only {len(U)} units",))
    d = TR.prepare_units(U, now)
    folds = TR.make_folds(d, axis, seed=cfg.seed, n_stock_folds=4, min_train_years=1)
    if not folds:
        return _unknown(spec, f"no folds on the {axis.value} axis (fewer than two labels or no history)")
    res = TR.run_folds(U, selector_fit(h, fit_cfg, cfg.top_frac), folds, now)
    ar = TR.axis_result_from_folds(axis, res, n_boot=cfg.n_boot, seed=cfg.seed)
    lab = ar.verdict.label
    ok_v = (TS.TransferVerdictLabel.GENERALISES, TS.TransferVerdictLabel.TRANSFER_ONLY)
    if lab in ok_v and ar.cross.excludes_zero_above:
        v = StudyVerdict.SUPPORTED
    elif lab == TS.TransferVerdictLabel.INSUFFICIENT_EVIDENCE:
        v = StudyVerdict.INCONCLUSIVE
    elif lab in (TS.TransferVerdictLabel.IDENTITY_DEPENDENT, TS.TransferVerdictLabel.OVER_SPECIALISED, TS.TransferVerdictLabel.HARMFUL,
                 TS.TransferVerdictLabel.NO_LEARNING):
        v = StudyVerdict.NOT_SUPPORTED
    else:
        v = StudyVerdict.INCONCLUSIVE
    cav = [f"folds forward in time: {all(f.forward for f in folds)}", f"selector {h.hid} refitted on each fold's training units only"]
    if not all(f.forward for f in folds):
        cav.append("leave-one-label-out folds train on other labels at all dates (research-side hindsight, not a walk-forward)")
    return StudyResult(spec.qid, spec.section, v, ar.cross.mean, ar.cross.lo, ar.cross.hi,
                       TS.cluster_signflip_p(np.concatenate([r.gain_test for r in res]), np.concatenate([r.cluster_test for r in res]), n_perm=1000, seed=cfg.seed),
                       float("nan"), int(ar.cross.n), detail={"axis": axis.value, "verdict_label": lab.value, "same": ar.same.as_dict(),
                                                              "cross": ar.cross.as_dict(), "ratio": ar.ratio.value, "groups": [list(g) for g in ar.groups],
                                                              "reasons": list(ar.verdict.reasons), "hypothesis": h.hid}, caveats=tuple(cav))


def finalise_family(results: Sequence[StudyResult], cfg: LabConfig = LabConfig()) -> list[StudyResult]:
    """Benjamini-Hochberg over the p-values of every tested study run together; a SUPPORTED whose q exceeds alpha is downgraded to
    INCONCLUSIVE (eighteen questions asked at once are eighteen chances for luck)."""
    ps = np.array([r.p for r in results], float)
    q = np.full(len(results), np.nan)
    ok = np.isfinite(ps)
    if ok.any():
        q[ok] = TS.adjust_many(ps[ok], "bh")
    out = []
    for r, qq in zip(results, q):
        v = r.verdict
        cav = r.caveats
        if v == StudyVerdict.SUPPORTED and np.isfinite(qq) and qq > cfg.alpha:
            v, cav = StudyVerdict.INCONCLUSIVE, cav + (f"BH q={qq:.3f} across {int(ok.sum())} studies exceeds alpha={cfg.alpha}",)
        out.append(dc.replace(r, q=float(qq), verdict=v, caveats=cav))
    return out


# ---------------------------------------------------------------------------------------------------------------
# champion, discovery of H10+, and the full laboratory
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class Champion:
    hid: str
    reason: str
    increment: float
    increment_lo: float
    mechanism_consistent: bool | None


def pick_champion(incs: Sequence[Increment], fits: pd.DataFrame, cfg: LabConfig = LabConfig()) -> Champion:
    """The champion is the hypothesis with the largest LOWER bound of AUC gain over B0 among those BH-significant and (for logistic ones)
    whose fitted signs agreed with their own mechanism in most folds. When none clears the bar the champion is B0 and the reason says so:
    'nothing beats own-volatility persistence' is a finding, not a failure."""
    best = None
    for i in incs:
        if not (np.isfinite(i.lo) and i.lo > 0 and np.isfinite(i.q) and i.q <= cfg.alpha):
            continue
        sub = fits[(fits["hid"] == i.a) & fits["mechanism_consistent"].notna()] if len(fits) else fits
        consistent = None if len(sub) == 0 else bool(sub["mechanism_consistent"].astype(bool).mean() >= 0.5)
        if consistent is False:
            continue
        if best is None or i.lo > best.increment_lo:
            best = Champion(i.a, f"lower bound {i.lo:+.4f} AUC over B0, q={i.q:.3f}", i.diff, i.lo, consistent)
    return best or Champion("B0", "no hypothesis is BH-significantly better than own-volatility persistence with a consistent mechanism", 0.0, 0.0, None)


@dc.dataclass(frozen=True)
class DiscoveryOutcome:
    proposed: int
    registered: tuple[str, ...]
    promoted: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]        # (rule id, reason)
    reason: str


def discover(F: pd.DataFrame, wf: WFResult, registry: VH.HypothesisRegistry, now, cfg: LabConfig = LabConfig(), max_new: int = 2) -> DiscoveryOutcome:
    """Look for H10+. The out-of-sample fused probability of the additive hypotheses says which movers were 'anticipated'; a decision tree
    searches the rest for a region where movers concentrate. Candidates are FOUND on the first two-thirds of the out-of-sample dates and
    must REPLICATE on the last third before promotion. Nothing found -> nothing registered (the honest, common outcome)."""
    oos = wf.oos
    add_ids = [h for h in wf.hids if h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8")]
    if len(oos) == 0 or not add_ids:
        return DiscoveryOutcome(0, (), (), (), "no out-of-sample rows or no known hypotheses to be residual to")
    fused = additive_column(oos, add_ids)
    if fused.isna().all():
        return DiscoveryOutcome(0, (), (), (), "no complete additive probabilities")
    feats = F.loc[F.index.intersection(oos.index)]
    J = feats.join(oos[["fold"]], how="inner")
    J["p_known"] = fused.reindex(J.index).to_numpy()
    J = J.sort_index()
    ud = np.array(sorted(J.index.get_level_values(0).unique()))
    if len(ud) < 30:
        return DiscoveryOutcome(0, (), (), (), "fewer than 30 out-of-sample dates")
    cut = ud[int(len(ud) * 2 / 3)]
    find, rep = J[J.index.get_level_values(0) < cut], J[J.index.get_level_values(0) >= cut]
    rep = rep[pd.to_datetime(rep["end"]) < pd.Timestamp(as_date(now))]
    known_sel = {}
    for h in add_ids:
        c = f"p_{h}"
        if c in oos:
            r = oos[c].groupby(level=0).rank(pct=True, method="first").reindex(find.index)
            known_sel[h] = (r.to_numpy() > 1 - cfg.top_frac * 2)
    cands = VH.propose_rules(find, find["p_known"].to_numpy(), now, seed=cfg.seed, known_selections=known_sel)
    registered: list[str] = []
    promoted: list[str] = []
    rejected: list[tuple[str, str]] = []
    for cand, ev in cands[:max_new + 3]:
        if len(registered) >= max_new:
            break
        rr = VH.replicate_rule(cand, rep, now, seed=cfg.seed)
        if not rr.replicated:
            rejected.append((ev.rule_id, rr.reason))
            continue
        try:
            found = registry.register_discovered(cand, ev.rule_id)
        except ValueError as e:
            rejected.append((ev.rule_id, str(e)))
            continue
        registered.append(found.hid)
        registry.promote(found.hid, f"replicated: lift {rr.lift:.2f} (lo {rr.lift_lo:.2f}) on {rr.n_in} later rows")
        promoted.append(found.hid)
    return DiscoveryOutcome(len(cands), tuple(registered), tuple(promoted), tuple(rejected),
                            "found and replicated" if promoted else ("candidates did not replicate" if cands else "no region beat the search's lift/overlap/size bars"))


@dc.dataclass
class LabReport:
    now: str
    cfg_hash: str
    code_hash: str
    survivor_free: bool
    n_rows: int
    n_dates: int
    hids: tuple
    unavailable: dict
    scorecards: dict
    increments: list
    champion: Champion
    directions: dict
    decay: dict
    regimes: dict
    years: dict
    unexplained: UnexplainedShare
    competition: dict
    calibration: dict
    discovery: DiscoveryOutcome
    studies: list
    registry_fingerprint: str
    caveats: tuple
    data_through: str = ""
    fits: pd.DataFrame = dc.field(default_factory=pd.DataFrame)
    assessments: dict = dc.field(default_factory=dict)
    extremes: dict = dc.field(default_factory=dict)
    health: pd.DataFrame = dc.field(default_factory=pd.DataFrame)
    persistence: pd.DataFrame = dc.field(default_factory=pd.DataFrame)
    thresholds: pd.DataFrame = dc.field(default_factory=pd.DataFrame)
    cells: pd.DataFrame = dc.field(default_factory=pd.DataFrame)
    fused: dict = dc.field(default_factory=dict)

    def study(self, qid: str) -> StudyResult:
        return next(s for s in self.studies if s.qid == qid)

    def verdicts(self) -> dict:
        return {s.qid: str(s.verdict) for s in self.studies}


def survivor_caveats(F: pd.DataFrame) -> tuple[str, ...]:
    if F.attrs.get("survivor_free", False):
        return ()
    return ("price panel is survivor-only: delisted names are absent, so the mover base rate and every AUC here are biased (RG04); "
            "re-measure on a survivor-free panel before trusting any number",)


def run_lab(F: pd.DataFrame, now, cfg: LabConfig = LabConfig(), registry: VH.HypothesisRegistry | None = None,
            fit_cfg: VH.FitConfig = VH.FitConfig(), qids: Sequence[str] | None = None, discover_new: bool = True) -> LabReport:
    """The whole laboratory on one matured frame: walk-forward of B0 and H1-H9(+), scorecards, BH-corrected increments, champion,
    direction-blindness and decay/regime checks for each hypothesis, unexplained share, Arena competition, forward calibration of the
    champion, discovery of H10+ and the eighteen section-9 studies. Deterministic given (frame, now, cfg, seed)."""
    registry = registry or VH.HypothesisRegistry()
    wf = walk_forward(F, registry.all(), now, cfg, fit_cfg)
    oos = wf.oos
    hids = list(wf.hids)
    cols = [f"p_{h}" for h in hids]
    t = per_date_table(oos, cols, cfg.top_frac) if len(oos) else pd.DataFrame()
    cards = {h: scorecard(oos, h, t, cfg) for h in hids} if len(oos) else {}
    incs = increments_vs_baseline(t, hids, cfg) if len(oos) else []
    fits = wf.fit_table()
    champ = pick_champion(incs, fits, cfg)
    directions = {h: direction_check(oos, h, cfg) for h in hids} if len(oos) else {}
    decay = {h: decay_check(t, oos, h, cfg) for h in hids} if len(oos) else {}
    regimes = {h: regime_check(t, oos, h, "regime", cfg) for h in hids} if len(oos) else {}
    years = {h: regime_check(t, oos, h, "year", cfg) for h in hids} if len(oos) else {}
    unexpl = unexplained_extremes(oos, [h for h in hids if h != "B0"], cfg)
    comp = run_competition(oos, [h for h in hids if h != "B0"], now, seed=cfg.seed) if len(oos) else {"tested": False, "reason": "no out-of-sample rows"}
    calib = {}
    if len(oos):
        cal = forward_calibrate(oos, champ.hid)
        m = cal.notna()
        rel = reliability_table(cal[m].to_numpy(), oos.loc[m, "touch"].to_numpy()) if m.any() else pd.DataFrame()
        calib = {"hid": champ.hid, "verdict": calibration_verdict(rel), "table": rel.to_dict("records")}
    disc = discover(F, wf, registry, now, cfg) if (discover_new and len(oos)) else DiscoveryOutcome(0, (), (), (), "discovery not requested or no rows")
    studies = run_studies(F, now, cfg, fit_cfg, wf=wf, champion=champ.hid if champ.hid != "B0" else "H1", registry=registry, qids=qids)
    caveats = survivor_caveats(F)
    if len(oos) == 0:
        caveats += ("no out-of-sample rows: frame too short for the configured folds",)
    health = feature_health(F) if len(F) else pd.DataFrame()
    if len(health) and health["data_failure"].any():
        caveats += (f"input columns failing health checks (DATA_FAILURE, not evidence): {health.loc[health['data_failure'], 'column'].tolist()}",)
    fused = {}
    if len(oos):
        fz, used = fuse_forward(oos, [h for h in hids if h != "B0"], cfg)
        tt = per_date_table(oos.assign(p_FUSED=fz), ["p_FUSED", f"p_{champ.hid}"], cfg.top_frac)
        fi = auc_increment(tt, "FUSED", champ.hid, cfg)
        fused = {"increment_over_champion": dc.asdict(fi), "weights": used, "beats_champion": bool(np.isfinite(fi.lo) and fi.lo > 0)}
    ch = champ.hid if champ.hid != "B0" else (hids[1] if len(hids) > 1 else "B0")
    rep_ = LabReport(str(as_date(now)), cfg.fingerprint(), current_code_hash(), bool(F.attrs.get("survivor_free", False)), int(len(F)),
                     int(F.index.get_level_values(0).nunique()) if len(F) else 0, tuple(hids), wf.unavailable, cards, incs, champ, directions, decay,
                     regimes, years, unexpl, comp, calib, disc, studies, registry.fingerprint(), caveats,
                     str(pd.to_datetime(mature_only(F, now)[0]["end"]).max().date()) if len(F) and len(mature_only(F, now)[0]) else "",
                     fits, assess_all(hids, incs, fits, decay, regimes, directions, studies, cfg), classify_extremes(oos, [h for h in hids if h != "B0"], cfg) if len(oos) else {},
                     health, mover_persistence(F, cfg=cfg) if len(F) else pd.DataFrame(),
                     threshold_curve(oos, ch, cfg=cfg) if len(oos) else pd.DataFrame(), cell_breakdown(oos, F, ch, cfg) if len(oos) else pd.DataFrame(), fused)
    return rep_


def run_studies(F: pd.DataFrame, now, cfg: LabConfig = LabConfig(), fit_cfg: VH.FitConfig = VH.FitConfig(), *, wf: WFResult | None = None,
                champion: str = "H1", registry: VH.HypothesisRegistry | None = None, qids: Sequence[str] | None = None) -> list[StudyResult]:
    """Run the selected section-9 studies (default all eighteen) and BH-correct them as one family."""
    registry = registry or VH.HypothesisRegistry()
    todo = [s for s in STUDIES if qids is None or s.qid in qids]
    out: list[StudyResult] = []
    for s in todo:
        if s.kind == "scan":
            out.append(run_scan_study(s, F, now, cfg))
        elif s.kind == "pairs":
            out.append(run_pairs_study(s, F, now, cfg, fit_cfg))
        elif s.kind == "group":
            out.append(run_group_study(s, F, now, cfg, fit_cfg))
        elif s.kind == "interaction":
            wf = wf or walk_forward(F, registry.all(), now, cfg, fit_cfg)
            out.append(run_interaction_study(s, wf, cfg))
        elif s.kind == "transfer":
            h = registry.get(champion) if champion in registry else registry.get("H1")
            out.append(run_transfer_study(s, F, h, now, cfg, fit_cfg))
    return finalise_family(out, cfg)


# ---------------------------------------------------------------------------------------------------------------
# calibrated forecast: P(move), magnitude quantiles, timing - fitted on matured rows only, applied to a point-in-time frame
# ---------------------------------------------------------------------------------------------------------------
TIMING_DAYS = tuple(range(1, HORIZON_BARS + 1))


@dc.dataclass
class VolatilityModel:
    hyp: VH.Hypothesis
    fitted: VH.FittedHypothesis
    trained_through: str
    calibrator: Any = None
    reliability: pd.DataFrame = dc.field(default_factory=pd.DataFrame)
    mag_models: dict = dc.field(default_factory=dict)
    mag_std: VH.Standardiser | None = None
    mag_features: tuple = ()
    timing: Any = None
    timing_prior: np.ndarray = dc.field(default_factory=lambda: np.full(HORIZON_BARS, 1.0 / HORIZON_BARS))
    direction: DirectionCheck | None = None
    survivor_free: bool = False
    code_hash: str = ""
    cfg_hash: str = ""

    @classmethod
    def fit(cls, F: pd.DataFrame, now, hyp: VH.Hypothesis, cfg: LabConfig = LabConfig(), fit_cfg: VH.FitConfig = VH.FitConfig(),
            calib_oos: pd.DataFrame | None = None, direction: DirectionCheck | None = None) -> "VolatilityModel":
        """Fit on rows whose outcome ended before `now`. `calib_oos` (a walk-forward OOS frame with a p_<hid> column, all earlier than
        `now`) supplies the isotonic calibration and the reliability table; without it the forecast is raw and says so."""
        Fm, _ = mature_only(F, now)
        Fm = Fm[Fm["touch"].notna()].sort_index()
        fh = VH.fit_hypothesis(hyp, Fm, now, fit_cfg)
        m = cls(hyp, fh, fh.trained_through, direction=direction, survivor_free=bool(F.attrs.get("survivor_free", False)),
                code_hash=current_code_hash(), cfg_hash=cfg.fingerprint())
        if not fh.ok:
            return m
        pc = f"p_{hyp.hid}"
        if calib_oos is not None and pc in calib_oos and len(calib_oos):
            if (pd.to_datetime(calib_oos["end"]) >= pd.Timestamp(as_date(now))).any():
                raise FirewallBreach("calibration rows include outcomes that had not ended before now")
            from sklearn.isotonic import IsotonicRegression
            c = calib_oos[calib_oos[pc].notna() & calib_oos["touch"].notna()]
            if len(c) >= 200 and c["touch"].nunique() == 2:
                m.calibrator = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(c[pc].to_numpy(float), c["touch"].to_numpy(float))
                m.reliability = reliability_table(m.calibrator.predict(c[pc].to_numpy(float)), c["touch"].to_numpy(float))
        feats = fh.resid_features if hyp.kind == VH.HypKind.RESIDUAL else hyp.features
        feats = tuple(f for f in feats if not VH.missing_columns((f,), Fm.columns))
        if feats and len(Fm) >= fit_cfg.min_rows:
            import lightgbm as lgb
            sub = VH._subsample(Fm, fit_cfg.max_train_rows, fit_cfg.seed)
            X = VH.derive(sub, feats).to_numpy(float)
            std = VH.Standardiser.fit(X, fit_cfg.winsor)
            Z = std.apply(X)
            lm = np.log(sub["absmove"].to_numpy(float) + 1e-4)
            m.mag_std, m.mag_features = std, feats
            for a in (0.5, 0.9):
                m.mag_models[a] = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=fit_cfg.gbm_trees, num_leaves=fit_cfg.gbm_leaves,
                                                    min_child_samples=fit_cfg.gbm_min_child, learning_rate=0.06, random_state=fit_cfg.seed,
                                                    verbose=-1, n_jobs=1).fit(Z, lm)
            ev = sub[(sub["touch"] == 1) & sub["tday"].between(1, HORIZON_BARS)]
            if len(ev):
                cnt = np.array([(ev["tday"] == d).sum() for d in TIMING_DAYS], float)
                m.timing_prior = (cnt + 1.0) / (cnt.sum() + HORIZON_BARS)
            if len(ev) >= 150 and ev["tday"].nunique() >= 3:
                from sklearn.linear_model import LogisticRegression
                Ze = std.apply(VH.derive(ev, feats).to_numpy(float))
                m.timing = LogisticRegression(C=0.3, max_iter=300).fit(Ze, ev["tday"].astype(int).to_numpy())
        return m

    def forecast(self, Fnow: pd.DataFrame, now) -> pd.DataFrame:
        """One row per (date, ticker) of Fnow. Fails closed if any row is dated after `now` or if the model was trained on outcomes that
        ended at/after the earliest row it is asked to forecast. An unfitted model abstains (NaN), it never guesses."""
        if len(Fnow) == 0:
            return pd.DataFrame(columns=["p_raw", "p_move", "p_lo", "p_hi", "calibrated", "mag_med", "mag_q90", "exp_day", "dir_flag", "abstain"])
        dts = pd.to_datetime(Fnow.index.get_level_values(0))
        if dts.max() > pd.Timestamp(as_date(now)):
            raise FirewallBreach(f"forecast asked for rows dated {dts.max().date()} after now={as_date(now)}")
        if self.trained_through:
            require_past(self.trained_through, dts.min(), "model training data")
        n = len(Fnow)
        out = pd.DataFrame(index=Fnow.index)
        if not self.fitted.ok:
            for c in ("p_raw", "p_move", "p_lo", "p_hi", "mag_med", "mag_q90", "exp_day"):
                out[c] = np.nan
            out["calibrated"], out["dir_flag"], out["abstain"] = False, str(DirectionFlag.UNTESTED), True
            return out
        p, _ = self.fitted.predict(Fnow)
        out["p_raw"] = p
        if self.calibrator is not None:
            pc = self.calibrator.predict(p)
            out["p_move"], out["calibrated"] = pc, True
            if len(self.reliability):
                edges = self.reliability["p_mean"].to_numpy()
                nb = np.abs(pc[:, None] - edges[None, :]).argmin(1)
                out["p_lo"], out["p_hi"] = self.reliability["lo"].to_numpy()[nb], self.reliability["hi"].to_numpy()[nb]
            else:
                out["p_lo"], out["p_hi"] = np.nan, np.nan
        else:
            out["p_move"], out["calibrated"], out["p_lo"], out["p_hi"] = p, False, np.nan, np.nan
        if self.mag_models:
            assert self.mag_std is not None               # fitted together with the magnitude models
            Z = self.mag_std.apply(VH.derive(Fnow, self.mag_features).to_numpy(float))
            out["mag_med"] = np.exp(self.mag_models[0.5].predict(Z))
            out["mag_q90"] = np.maximum(np.exp(self.mag_models[0.9].predict(Z)), out["mag_med"])
            tp = np.tile(self.timing_prior, (n, 1))
            if self.timing is not None:
                pr = self.timing.predict_proba(Z)
                tp = np.zeros((n, HORIZON_BARS))
                for j, cls_ in enumerate(self.timing.classes_):
                    tp[:, int(cls_) - 1] = pr[:, j]
                tp = tp / tp.sum(1, keepdims=True)
        else:
            out["mag_med"] = out["mag_q90"] = np.nan
            tp = np.tile(self.timing_prior, (n, 1))
        for d in TIMING_DAYS:
            out[f"p_day{d}"] = out["p_move"].to_numpy() * tp[:, d - 1]
        out["exp_day"] = tp @ np.array(TIMING_DAYS, float)
        flag = self.direction.flag if self.direction else DirectionFlag.UNTESTED
        out["dir_flag"] = str(flag)
        out["abstain"] = False
        out["dir_blind"] = flag in (DirectionFlag.DIRECTION_BLIND, DirectionFlag.LOSS_SKEWED)
        return out


# ---------------------------------------------------------------------------------------------------------------
# streaming, market-wide: keep per-date snapshots and exception rows, never the whole universe (RESEARCH_MAPPING rule 27)
# ---------------------------------------------------------------------------------------------------------------
class TrainReservoir:
    """A bounded, seeded, per-date reservoir of past rows for refitting during a stream. Holds at most `per_date` rows of every date, so
    memory grows with the number of dates, not with the size of the universe."""

    def __init__(self, per_date: int = 150, seed: int = 0):
        self.per_date, self.seed = per_date, seed
        self._parts: list[pd.DataFrame] = []

    def add(self, F: pd.DataFrame) -> None:
        if len(F) == 0:
            return
        codes = pd.factorize(F.index.get_level_values(0))[0]
        rng = np.random.default_rng(self.seed + len(self._parts))
        keep = []
        for c in np.unique(codes):
            ix = np.flatnonzero(codes == c)
            keep.append(ix if len(ix) <= self.per_date else ix[np.sort(rng.choice(len(ix), self.per_date, replace=False))])
        self._parts.append(F.iloc[np.concatenate(keep)].astype({c: "float32" for c in F.select_dtypes("float64").columns}))

    def rows_ended_before(self, cutoff) -> pd.DataFrame:
        if not self._parts:
            return pd.DataFrame()
        R = pd.concat(self._parts)
        return R[pd.to_datetime(R["end"]) < pd.Timestamp(as_date(cutoff))].sort_index()

    def __len__(self) -> int:
        return int(sum(len(p) for p in self._parts))


@dc.dataclass
class StreamResult:
    date_table: pd.DataFrame
    exceptions: pd.DataFrame
    fits: list
    years_done: list
    rows_seen: int
    peak_rows_in_memory: int
    hids: tuple


def stream_walk_forward(loader: Callable[[int], pd.DataFrame], years: Sequence[int], hyps: Sequence[VH.Hypothesis], now,
                        cfg: LabConfig = LabConfig(), fit_cfg: VH.FitConfig = VH.FitConfig(), *, per_date_reservoir: int = 150,
                        exceptions_per_date: int = 4, min_train_dates: int | None = None) -> StreamResult:
    """Year-by-year walk-forward that never holds more than one year plus a reservoir. For each year: fit every hypothesis on the reservoir
    rows that ENDED before the year's first decision date, score the year, keep (a) the per-date snapshot table (AUC, top-decile
    precision per hypothesis) and (b) at most `exceptions_per_date` exception rows per date - missed extremes (touches that every
    hypothesis ranked in its lower half) and confident false alarms - then add a bounded sample of the year to the reservoir and drop
    the year. `loader(year)` must return a lab frame for that calendar year only."""
    min_dates = min_train_dates or cfg.min_train_dates
    field = [baseline_hypothesis()] + [h for h in hyps if h.state != VH.HypState.RETIRED]
    res = TrainReservoir(per_date_reservoir, cfg.seed)
    tables, exc, fits = [], [], []
    seen, peak, done = 0, 0, []
    for y in years:
        Fy = loader(y)
        errs = validate_frame(Fy, cfg)
        if errs:
            raise ValueError(f"year {y}: " + "; ".join(errs))
        Fy = Fy.sort_index()
        seen += len(Fy)
        peak = max(peak, len(Fy) + len(res))
        Fm, _ = mature_only(Fy, now)
        Fm = Fm[Fm["touch"].notna()]
        if len(Fm):
            first = Fm.index.get_level_values(0).min()
            tr = res.rows_ended_before(first)
            if tr.index.get_level_values(0).nunique() >= min_dates:
                chunk = Fm[[c for c in OUTCOME_COLS + ("sector",) if c in Fm]].copy()
                for h in field:
                    if h.kind != VH.HypKind.RESIDUAL and (h.missing(Fm.columns) or h.missing(tr.columns)):
                        chunk[f"p_{h.hid}"] = np.nan
                        continue
                    fh = VH.fit_hypothesis(h, tr, first, fit_cfg, others=[x for x in field if x.hid not in ("B0", h.hid)])
                    p, _m = fh.predict(Fm)
                    chunk[f"p_{h.hid}"] = p
                    fits.append({"year": y, "hid": h.hid, "ok": fh.ok, "reason": fh.reason, "n_train": fh.n_train})
                cols = [c for c in chunk.columns if c.startswith("p_") and chunk[c].notna().any()]
                tables.append(per_date_table(chunk, cols, cfg.top_frac))
                exc.append(_exception_rows(chunk, cols, exceptions_per_date, cfg))
        res.add(Fy)
        done.append(y)
        del Fy, Fm
    dt_ = pd.concat(tables) if tables else pd.DataFrame()
    ex = pd.concat(exc) if exc else pd.DataFrame()
    return StreamResult(dt_, ex, fits, done, seen, peak, tuple(h.hid for h in field))


def _exception_rows(chunk: pd.DataFrame, cols: Sequence[str], k: int, cfg: LabConfig) -> pd.DataFrame:
    """Missed extremes (touches ranked below the median by every hypothesis) and the most confident false alarms, at most k per date."""
    ranks = pd.concat([chunk[c].groupby(level=0).rank(pct=True, method="first") for c in cols], axis=1)
    best_rank = ranks.max(axis=1)
    strongest = chunk[list(cols)].max(axis=1)
    missed = chunk[(chunk["touch"] == 1) & (best_rank < 0.5)].assign(kind="MISSED_EXTREME")
    false_alarm = chunk[(chunk["touch"] == 0) & (best_rank > 1 - cfg.top_frac / 2)].assign(kind="FALSE_ALARM")
    pick = []
    for part, key in ((missed, missed["absmove"]), (false_alarm, strongest.reindex(false_alarm.index))):
        if len(part):
            g = part.assign(_k=key.to_numpy()).groupby(level=0, group_keys=False).apply(lambda x: x.nlargest(k, "_k")).drop(columns="_k")
            pick.append(g)
    return pd.concat(pick) if pick else chunk.iloc[:0].assign(kind="")


def summarise_stream(sr: StreamResult, cfg: LabConfig = LabConfig()) -> dict:
    """Scorecards and BH increments from a stream's date table alone (no rows needed)."""
    if len(sr.date_table) == 0:
        return {"cards": {}, "increments": []}
    hids = [h for h in sr.hids if f"auc_p_{h}" in sr.date_table]
    return {"cards": {h: scorecard(None, h, sr.date_table, cfg) for h in hids}, "increments": increments_vs_baseline(sr.date_table, hids, cfg)}


def remeasure(old: Mapping[str, Any], new: ScoreCard, tol: float = 0.01) -> dict:
    """RS re-measurement (contract WP3): compare a figure measured on the survivor-only panel with the same hypothesis measured on a
    survivor-free one. Returns the AUC change, its direction, and whether the old headline survives."""
    d = new.auc - float(old["auc"])
    return {"hid": new.hid, "old_auc": float(old["auc"]), "new_auc": new.auc, "delta": d,
            "direction": "SURVIVOR_BIAS_INFLATED" if d < -tol else "SURVIVOR_BIAS_DEFLATED" if d > tol else "UNCHANGED_WITHIN_TOL",
            "old_headline_survives": bool(np.isfinite(new.auc_lo) and new.auc_lo > 0.5 and new.auc >= float(old["auc"]) - tol)}


# ---------------------------------------------------------------------------------------------------------------
# report, matured record, and the research-loop entry point
# ---------------------------------------------------------------------------------------------------------------
def render_report(rep: LabReport) -> str:
    L = [f"VOLATILITY LABORATORY  now={rep.now}  rows={rep.n_rows}  dates={rep.n_dates}  survivor_free={rep.survivor_free}",
         "status: IMPLEMENTED - NOT VALIDATED", f"champion: {rep.champion.hid} ({rep.champion.reason})", ""]
    L.append("hypothesis  auc   [lo,hi]        lift  brier_skill  ece    direction / decay")
    for h in rep.hids:
        c = rep.scorecards.get(h)
        if c is None:
            continue
        d = rep.directions.get(h)
        dk = rep.decay.get(h)
        L.append(f"{h:<10} {c.auc:.3f} [{c.auc_lo:.3f},{c.auc_hi:.3f}] {c.lift:5.2f} {c.brier_skill:+.3f}      {c.ece:.3f}  "
                 f"{d.flag if d else '-'} / {dk.status if dk else '-'}")
    if rep.unavailable:
        L.append("unavailable (UNKNOWN, not failed): " + "; ".join(f"{k} needs {list(v)}" for k, v in rep.unavailable.items()))
    L.append(f"unexplained extremes: {rep.unexplained.share:.1%} [{rep.unexplained.lo:.1%},{rep.unexplained.hi:.1%}] of {rep.unexplained.n_extremes}")
    L.append(f"discovery: {rep.discovery.reason}; registered {list(rep.discovery.registered)}; rejected {len(rep.discovery.rejected)}")
    L.append("")
    for s in rep.studies:
        L.append(f"{s.qid} {str(s.verdict):<13} effect={s.effect:+.4f} [{s.lo:+.4f},{s.hi:+.4f}] q={s.q:.3f}  {s.question}")
        for c in s.caveats[:1]:
            L.append(f"      note: {c}")
    for c in rep.caveats:
        L.append(f"CAVEAT: {c}")
    return "\n".join(L)


def to_matured_record(rep: LabReport, now, data_through, seed: int = 0) -> MaturedRecord:
    """Wrap a report as a MATURED_RESEARCH_STATE record. It can reach the live side only through record.gate(now), which fails closed unless
    `data_through` (the newest outcome the report used) is strictly before that now. Payload is identity-free: no tickers, no dates
    other than the maturity stamp on the record itself."""
    payload = {"kind": "volatility_lab", "problem": str(Problem.VOLATILITY), "champion": rep.champion.hid, "champion_reason": rep.champion.reason,
               "cards": {h: c.as_dict() for h, c in rep.scorecards.items()}, "verdicts": rep.verdicts(),
               "directions": {h: str(d.flag) for h, d in rep.directions.items()}, "decay": {h: d.status for h, d in rep.decay.items()},
               "unexplained_share": rep.unexplained.share, "survivor_free": rep.survivor_free, "caveats": list(rep.caveats),
               "status": "IMPLEMENTED - NOT VALIDATED"}
    rid = "VL" + stable_hash([rep.cfg_hash, rep.registry_fingerprint, str(data_through), payload["verdicts"]], 12)
    from engine.learning.knowledge import wall_stamp            # F31: the caller's injected clock (real time when none is set)
    prov = Provenance(created_real=wall_stamp(), learned_at=str(as_date(data_through)),
                      code_hash=rep.code_hash or current_code_hash(), data_hash=stable_hash([rep.n_rows, rep.n_dates, str(data_through)], 12),
                      config_hash=rep.cfg_hash, experiment_id="volatility_lab", seed=seed, outcomes_seen_through=str(as_date(data_through)))
    return MaturedRecord(rid, str(as_date(data_through)), payload, prov, Namespace.MATURED_RESEARCH)


@dc.dataclass
class LabState:
    """What the loop carries between steps. Holds research-side results only."""
    registry: VH.HypothesisRegistry = dc.field(default_factory=VH.HypothesisRegistry)
    cfg: LabConfig = LabConfig()
    fit_cfg: VH.FitConfig = VH.FitConfig()
    results: dict = dc.field(default_factory=dict)          # qid -> StudyResult
    ran_on: dict = dc.field(default_factory=dict)           # task -> frame key it was last run on
    score: dict = dc.field(default_factory=dict)
    discovery: DiscoveryOutcome | None = None
    history: list = dc.field(default_factory=list)
    _wf: tuple | None = None

    def tasks(self) -> list[str]:
        return ["SCORE"] + [s.qid for s in STUDIES] + ["DISCOVER"]


@dc.dataclass(frozen=True)
class StepResult:
    ran: tuple
    remaining: tuple
    record: MaturedRecord | None
    note: str


def frame_key(F: pd.DataFrame, now) -> str:
    if len(F) == 0:
        return stable_hash(["empty", str(as_date(now))])
    d = F.index.get_level_values(0)
    return stable_hash([len(F), str(d.min()), str(d.max()), float(np.nansum(F["touch"].to_numpy(float))), sorted(F.columns), str(as_date(now))], 16)


def step(state: LabState, now, frame: pd.DataFrame | None = None, max_tasks: int = 2) -> StepResult:
    """One scheduled unit of volatility research for the wave-2 loop. Runs up to `max_tasks` not-yet-done tasks on this frame (SCORE =
    the multi-hypothesis walk-forward with scorecards/direction/decay; Q01-Q18; DISCOVER = H10+ search), folds the results into
    `state`, and returns a MaturedRecord (research namespace) when something new was learned. With no frame or an empty one it does
    nothing and says so. A frame with outcomes that end at/after `now` is refused, not trimmed silently."""
    if frame is None or len(frame) == 0:
        return StepResult((), tuple(state.tasks()), None, "no matured frame supplied")
    errs = validate_frame(frame, state.cfg)
    if errs:
        raise ValueError("; ".join(errs))
    check = pd.to_datetime(frame["end"]).max()
    if check >= pd.Timestamp(as_date(now)):
        raise FirewallBreach(f"frame contains outcomes ending {check.date()} at/after now={as_date(now)}; pass mature_only(frame, now) explicitly")
    key = frame_key(frame, now)
    if state._wf is None or state._wf[0] != key:
        state._wf = (key, walk_forward(frame, state.registry.all(), now, state.cfg, state.fit_cfg))
    wf = state._wf[1]
    due = [t for t in state.tasks() if state.ran_on.get(t) != key][:max(1, max_tasks)]
    for t in due:
        if t == "SCORE":
            cols = [f"p_{h}" for h in wf.hids]
            tb = per_date_table(wf.oos, cols, state.cfg.top_frac) if len(wf.oos) else pd.DataFrame()
            cards = {h: scorecard(wf.oos, h, tb, state.cfg) for h in wf.hids} if len(wf.oos) else {}
            incs = increments_vs_baseline(tb, wf.hids, state.cfg) if len(wf.oos) else []
            state.score = {"cards": cards, "increments": incs, "champion": pick_champion(incs, wf.fit_table(), state.cfg),
                           "directions": {h: direction_check(wf.oos, h, state.cfg) for h in wf.hids} if len(wf.oos) else {}}
        elif t == "DISCOVER":
            state.discovery = discover(frame, wf, state.registry, now, state.cfg)
        else:
            champ = state.score["champion"].hid if state.score else "H1"
            got = run_studies(frame, now, state.cfg, state.fit_cfg, wf=wf, champion=champ if champ != "B0" else "H1", registry=state.registry, qids=[t])
            state.results[t] = got[0]
        state.ran_on[t] = key
    if state.results:
        fam = finalise_family(list(state.results.values()), state.cfg)
        state.results = {r.qid: r for r in fam}
    remaining = tuple(t for t in state.tasks() if state.ran_on.get(t) != key)
    through = pd.to_datetime(frame["end"]).max()
    rep_like = {"n_done": len(due), "through": str(through.date())}
    state.history.append({"now": str(as_date(now)), "ran": due, "key": key, **rep_like})
    rec = None
    if due:
        payload = {"kind": "volatility_lab_step", "tasks": due, "verdicts": {q: str(r.verdict) for q, r in state.results.items()},
                   "champion": state.score["champion"].hid if state.score else None, "survivor_free": bool(frame.attrs.get("survivor_free", False)),
                   "caveats": list(survivor_caveats(frame)), "status": "IMPLEMENTED - NOT VALIDATED"}
        from engine.learning.knowledge import wall_stamp as _wall_stamp      # F31: the caller's injected clock
        prov = Provenance(_wall_stamp(), str(through.date()), current_code_hash(), key,
                          state.cfg.fingerprint(), "volatility_lab.step", "", state.cfg.seed, str(through.date()))
        rec = MaturedRecord("VS" + stable_hash([key, due], 12), str(through.date()), payload, prov, Namespace.MATURED_RESEARCH)
    return StepResult(tuple(due), remaining, rec, "ok" if due else "all tasks already done on this frame")


# ---------------------------------------------------------------------------------------------------------------
# input health: a data failure must not be mistaken for a market finding
# ---------------------------------------------------------------------------------------------------------------
def population_stability_index(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """PSI of b against a's decile bins (a = reference). > 0.25 is a material shift. NaN if either side has too few finite values."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50:
        return float("nan")
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def feature_health(F: pd.DataFrame, cols: Sequence[str] | None = None, psi_limit: float = 0.25, max_missing: float = 0.5) -> pd.DataFrame:
    """One row per input column: missing share, share of exact zeros, infinities, constancy, and the PSI between the first and second half
    of the dates. `data_failure` is True for a column that is mostly missing, constant, infinite, or that shifted so much between halves
    that a study spanning both is comparing two different measurements. A failing column is a DATA_FAILURE (Knowability), not evidence."""
    cols = list(cols) if cols is not None else [c for c in VH.BASE_COLUMNS + VH.EVENT_COLUMNS if c in F.columns and c != "sector"]
    dates = pd.to_datetime(F.index.get_level_values(0))
    half = dates.min() + (dates.max() - dates.min()) / 2 if len(F) else None
    rows = []
    for c in cols:
        v = F[c].to_numpy(float)
        miss = float(np.mean(~np.isfinite(v))) if len(v) else float("nan")
        fin = v[np.isfinite(v)]
        const = bool(len(fin) > 1 and np.nanstd(fin) < 1e-12)
        psi = population_stability_index(v[(dates <= half)], v[(dates > half)]) if len(F) else float("nan")
        rows.append({"column": c, "missing": miss, "zeros": float(np.mean(fin == 0)) if len(fin) else float("nan"), "n_inf": int(np.isinf(v).sum()),
                     "constant": const, "psi_halves": psi,
                     "data_failure": bool(miss > max_missing or const or np.isinf(v).any() or (np.isfinite(psi) and psi > psi_limit))})
    return pd.DataFrame(rows, columns=["column", "missing", "zeros", "n_inf", "constant", "psi_halves", "data_failure"])


# ---------------------------------------------------------------------------------------------------------------
# model-free evidence for H1 and friends: do extreme movers repeat?
# ---------------------------------------------------------------------------------------------------------------
def mover_persistence(F: pd.DataFrame, lags: Sequence[int] = (2, 4, 8), cfg: LabConfig = LabConfig(), max_gap_days: int = 10) -> pd.DataFrame:
    """P(mover now | mover `lag` decisions ago) against P(mover now | not), per lag, with the same contrast inside each own-volatility
    tercile (so 'volatile names move again' is separated from 'names that just moved move again'). A lagged outcome is used only if it had
    ENDED before the current decision date (otherwise it is NaN, never peeked). Weekly frames only: rows whose lag is more than
    max_gap_days*lag calendar days back are skipped."""
    F = F[F["touch"].notna()].sort_index()
    if len(F) == 0:
        return pd.DataFrame(columns=["lag", "n", "p1", "p0", "diff", "lo", "hi", "diff_controlled"])
    G = F.reset_index().sort_values(["ticker", "date"])
    G["date"] = pd.to_datetime(G["date"])
    out = []
    tercile = G.groupby("date")["vol20"].rank(pct=True) if "vol20" in G else pd.Series(0.5, index=G.index)
    G["terc"] = np.minimum((tercile * 3).astype(int), 2) if "vol20" in G else 0
    for lag in lags:
        g = G.groupby("ticker")
        prev_touch, prev_end, prev_date = g["touch"].shift(lag), g["end"].shift(lag), g["date"].shift(lag)
        usable = prev_touch.notna() & (prev_end < G["date"]) & ((G["date"] - prev_date).dt.days <= max_gap_days * lag)
        d = G[usable].assign(prev=prev_touch[usable].to_numpy())
        if len(d) < 200 or d["prev"].nunique() < 2:
            out.append({"lag": lag, "n": int(len(d)), "p1": np.nan, "p0": np.nan, "diff": np.nan, "lo": np.nan, "hi": np.nan, "diff_controlled": np.nan})
            continue
        per_date = d.groupby("date").apply(lambda x: (x.loc[x["prev"] == 1, "touch"].mean() - x.loc[x["prev"] == 0, "touch"].mean())
                                           if (x["prev"] == 1).sum() >= 2 and (x["prev"] == 0).sum() >= 2 else np.nan).dropna()
        ctrl = []
        for tq, x in d.groupby("terc"):
            if (x["prev"] == 1).sum() >= 20 and (x["prev"] == 0).sum() >= 20:
                ctrl.append(x.loc[x["prev"] == 1, "touch"].mean() - x.loc[x["prev"] == 0, "touch"].mean())
        bm = TS.cluster_bootstrap_mean(per_date.to_numpy(), month_cluster(per_date.index), n_boot=cfg.n_boot, seed=cfg.seed) if len(per_date) >= 2 else None
        out.append({"lag": lag, "n": int(len(d)), "p1": float(d.loc[d["prev"] == 1, "touch"].mean()), "p0": float(d.loc[d["prev"] == 0, "touch"].mean()),
                    "diff": bm.mean if bm else np.nan, "lo": bm.lo if bm else np.nan, "hi": bm.hi if bm else np.nan,
                    "diff_controlled": float(np.mean(ctrl)) if ctrl else np.nan})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------------------
# does the signal find only the huge movers, or the 7% ones too? where is it weak?
# ---------------------------------------------------------------------------------------------------------------
def threshold_curve(oos: pd.DataFrame, hid: str, thresholds: Sequence[float] = (0.05, 0.07, 0.10, 0.15, 0.20), cfg: LabConfig = LabConfig()) -> pd.DataFrame:
    """Per-date AUC of the (10%-mover) probability against 'moved at least thr' for several thr. A signal trained on +-10% touches may or
    may not rank 5-7% movers; the portfolio target lives around 7%, so this is asked explicitly."""
    pc = f"p_{hid}"
    rows = []
    d = oos[oos[pc].notna() & oos["absmove"].notna()]
    for thr in thresholds:
        t = d.assign(touch=(d["absmove"] >= thr).astype(float))
        tb = per_date_table(t, [pc], cfg.top_frac)
        a = tb[f"auc_{pc}"].dropna() if f"auc_{pc}" in tb else pd.Series(dtype=float)
        bm = TS.cluster_bootstrap_mean(a.to_numpy(), month_cluster(a.index), n_boot=cfg.n_boot, seed=cfg.seed) if len(a) >= 2 else None
        rows.append({"threshold": thr, "base_rate": float(t["touch"].mean()) if len(t) else np.nan, "n_dates": int(len(a)),
                     "auc": bm.mean if bm else np.nan, "lo": bm.lo if bm else np.nan, "hi": bm.hi if bm else np.nan})
    return pd.DataFrame(rows)


def cell_labels(F: pd.DataFrame) -> dict[str, pd.Series]:
    """Universe cells for the breakdown: price band, liquidity tercile (same-date), own-volatility tercile (same-date), sector."""
    out: dict[str, pd.Series] = {}
    if "logp" in F:
        p = np.exp(F["logp"].to_numpy(float))
        out["price"] = pd.Series(np.where(p < 10, "P<10", np.where(p < 30, "P10-30", "P30+")), index=F.index)
    for name, col in (("liquidity", "log_dv"), ("volatility", "vol20")):
        if col in F:
            r = F[col].groupby(level=0).rank(pct=True)
            out[name] = pd.Series(np.where(r < 1 / 3, f"{name[:3]}_lo", np.where(r < 2 / 3, f"{name[:3]}_mid", f"{name[:3]}_hi")), index=F.index)
    if "sector" in F:
        out["sector"] = F["sector"].astype(str)
    return out


def cell_breakdown(oos: pd.DataFrame, F: pd.DataFrame, hid: str, cfg: LabConfig = LabConfig(), min_names: int = 15) -> pd.DataFrame:
    """Per-cell mean per-date AUC (dates with fewer than min_names names in the cell are skipped) for each universe cell. weak=True when the
    interval's upper end is below 0.52: the signal does not work there and a portfolio should not lean on it there."""
    pc = f"p_{hid}"
    labs = cell_labels(F.loc[F.index.intersection(oos.index)])
    rows = []
    d = oos[oos[pc].notna()]
    for kind, lab in labs.items():
        L = lab.reindex(d.index)
        for name in sorted(L.dropna().unique()):
            sub = d[(L == name).to_numpy()]
            vals = {}
            for dt_, g in sub.groupby(level=0):
                y = g["touch"].to_numpy(float)
                if len(g) >= min_names and 0 < y.sum() < len(y):
                    vals[dt_] = _auc(g[pc].to_numpy(float), y.astype(bool))
            a = pd.Series(vals, dtype=float)
            if len(a) < 2:
                rows.append({"kind": kind, "cell": name, "n_dates": len(a), "auc": np.nan, "lo": np.nan, "hi": np.nan, "weak": None})
                continue
            bm = TS.cluster_bootstrap_mean(a.to_numpy(), month_cluster(a.index), n_boot=cfg.n_boot, seed=cfg.seed)
            rows.append({"kind": kind, "cell": name, "n_dates": len(a), "auc": bm.mean, "lo": bm.lo, "hi": bm.hi, "weak": bool(bm.hi < 0.52)})
    return pd.DataFrame(rows, columns=["kind", "cell", "n_dates", "auc", "lo", "hi", "weak"])


def classify_extremes(oos: pd.DataFrame, hids: Sequence[str], cfg: LabConfig = LabConfig()) -> dict:
    """Volatility-side view of 'could the features have told us': every extreme mover is PREDICTABLE if some hypothesis ranked it in the
    top `top_frac` that day, WEAKLY_PREDICTABLE if in the top 3x that, else UNKNOWN. (The full could-I-have-known test with external
    information is the knowability module's job; this only reports what the price/volume features could see.) Also, per hypothesis, how
    many extremes it alone caught."""
    from engine.research.core import Knowability
    hs = [h for h in hids if f"p_{h}" in oos and oos[f"p_{h}"].notna().any()]
    ex_mask = (oos["touch"] == 1).to_numpy()
    if not hs or not ex_mask.any():
        return {"n": int(ex_mask.sum()), "counts": {}, "shares": {}, "unique_catch": {}}
    rk = {h: oos[f"p_{h}"].groupby(level=0).rank(pct=True, method="first").to_numpy() for h in hs}
    top = {h: rk[h] > 1 - cfg.top_frac for h in hs}
    wide = {h: rk[h] > 1 - min(0.5, 3 * cfg.top_frac) for h in hs}
    any_top = np.any([top[h] for h in hs], axis=0)
    any_wide = np.any([wide[h] for h in hs], axis=0)
    cls = np.where(any_top, str(Knowability.PREDICTABLE), np.where(any_wide, str(Knowability.WEAKLY_PREDICTABLE), str(Knowability.UNKNOWN)))
    labs, cnt = np.unique(cls[ex_mask], return_counts=True)
    n = int(ex_mask.sum())
    uniq = {}
    for h in hs:
        others = np.any([top[o] for o in hs if o != h], axis=0) if len(hs) > 1 else np.zeros(len(oos), bool)
        uniq[h] = int((top[h] & ~others & ex_mask).sum())
    return {"n": n, "counts": dict(zip(labs.tolist(), cnt.tolist())), "shares": {k: v / n for k, v in zip(labs.tolist(), cnt.tolist())}, "unique_catch": uniq}


# ---------------------------------------------------------------------------------------------------------------
# magnitude and timing measured walk-forward (how large, and when)
# ---------------------------------------------------------------------------------------------------------------
def walk_forward_forecast(F: pd.DataFrame, hyp: VH.Hypothesis, now, cfg: LabConfig = LabConfig(), fit_cfg: VH.FitConfig = VH.FitConfig(),
                          wf: WFResult | None = None) -> pd.DataFrame:
    """Out-of-sample VolatilityModel forecasts (calibrated probability, magnitude quantiles, timing) for every test fold. Fold k's model is
    fitted on rows that ended before the fold and calibrated on earlier folds' out-of-sample probabilities that had also ended by then."""
    Fm, _ = mature_only(F, now)
    Fm = Fm[Fm["touch"].notna()].sort_index()
    wf = wf or walk_forward(Fm, [hyp], now, cfg, fit_cfg, include_baseline=False)
    pc = f"p_{hyp.hid}"
    parts = []
    for fd in wf.folds:
        te = Fm[np.isin(pd.to_datetime(Fm.index.get_level_values(0)).to_numpy(), np.array(fd.test_dates, dtype="datetime64[ns]"))]
        if len(te) == 0:
            continue
        cal = wf.oos[(wf.oos["fold"] < fd.idx) & (pd.to_datetime(wf.oos["end"]) < fd.now)] if pc in wf.oos else None
        model = VolatilityModel.fit(Fm, fd.now, hyp, cfg, fit_cfg, calib_oos=cal)
        fc = model.forecast(te, fd.test_dates[-1])
        keep = te[[c for c in OUTCOME_COLS if c in te]]
        parts.append(keep.join(fc).assign(fold=fd.idx))
    return pd.concat(parts) if parts else pd.DataFrame()


def forecast_metrics(fc: pd.DataFrame) -> dict:
    """Score forecasts: probability calibration (calibrated rows only), coverage of the magnitude quantiles (median should cover ~50%,
    q90 ~90%, judged against the realised absolute move), and timing (mean log-loss of the day distribution on movers against a uniform
    guess, argmax hit rate, MAE of the expected day against always guessing the middle)."""
    out: dict[str, Any] = {"n": int(len(fc))}
    if len(fc) == 0 or fc.get("abstain", pd.Series(dtype=bool)).all():
        return {**out, "note": "no forecasts or the model abstained"}
    ok = fc[~fc["abstain"].astype(bool)]
    cal = ok[ok["calibrated"].astype(bool) & ok["touch"].notna()]
    out["calibration"] = calibration_verdict(reliability_table(cal["p_move"].to_numpy(), cal["touch"].to_numpy())) if len(cal) >= 100 else {"tested": False}
    m = ok[ok["mag_med"].notna() & ok["absmove"].notna()]
    if len(m):
        out["cover_median"] = float((m["absmove"] <= m["mag_med"]).mean())
        out["cover_q90"] = float((m["absmove"] <= m["mag_q90"]).mean())
        out["q90_ok"] = bool(abs(out["cover_q90"] - 0.9) <= 0.05)
    mv = ok[(ok["touch"] == 1) & ok["tday"].between(1, HORIZON_BARS)]
    if len(mv) >= 30 and all(f"p_day{d}" in mv for d in TIMING_DAYS):
        P = mv[[f"p_day{d}" for d in TIMING_DAYS]].to_numpy(float)
        P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
        day = mv["tday"].astype(int).to_numpy()
        ll = -np.log(np.clip(P[np.arange(len(mv)), day - 1], 1e-6, 1))
        out["timing_logloss"] = float(ll.mean())
        out["timing_logloss_uniform"] = float(np.log(HORIZON_BARS))
        out["timing_beats_uniform"] = bool(ll.mean() < np.log(HORIZON_BARS))
        out["timing_argmax_hit"] = float((P.argmax(1) + 1 == day).mean())
        out["timing_mae"] = float(np.abs(mv["exp_day"].to_numpy(float) - day).mean())
        out["timing_mae_constant"] = float(np.abs(3.0 - day).mean())
    return out


# ---------------------------------------------------------------------------------------------------------------
# forward-only fusion of hypotheses
# ---------------------------------------------------------------------------------------------------------------
def fuse_forward(oos: pd.DataFrame, hids: Sequence[str], cfg: LabConfig = LabConfig()) -> tuple[pd.Series, dict]:
    """Stack the hypotheses' probabilities. Weights for fold k come from folds < k only (proportional to the positive Brier improvement
    over the constant forecast, so a hypothesis that lost to the base rate gets none); fold 0 is NaN. Returns the fused probability and
    the weights used in each fold. A fused forecast must beat its best member before it is worth having - the caller tests that."""
    hs = [h for h in hids if f"p_{h}" in oos and oos[f"p_{h}"].notna().all()]
    fused = pd.Series(np.nan, index=oos.index, dtype=float)
    used = {}
    for k in sorted(oos["fold"].unique()):
        past = oos[oos["fold"] < k]
        if len(past) < 200 or not hs:
            continue
        y = past["touch"].to_numpy(float)
        ref = float(np.mean((y.mean() - y) ** 2))
        gain = np.array([max(0.0, ref - float(np.mean((past[f"p_{h}"].to_numpy(float) - y) ** 2))) for h in hs])
        if gain.sum() <= 0:
            continue
        w = gain / gain.sum()
        cur = (oos["fold"] == k).to_numpy()
        lg = np.sum([wi * VH._logit(oos.loc[cur, f"p_{h}"].to_numpy(float)) for wi, h in zip(w, hs)], axis=0)
        fused.loc[cur] = VH._expit(lg)
        used[int(k)] = dict(zip(hs, w.tolist()))
    return fused, used


# ---------------------------------------------------------------------------------------------------------------
# evidence for each hypothesis, and the questions the results raise
# ---------------------------------------------------------------------------------------------------------------
def assess_all(hids: Sequence[str], incs: Sequence[Increment], fits: pd.DataFrame, decay: Mapping[str, DecayCheck], regimes: Mapping[str, RegimeCheck],
               directions: Mapping[str, DirectionCheck], studies: Sequence[StudyResult], cfg: LabConfig = LabConfig()) -> dict:
    """Collect each hypothesis's measurements into a HypothesisEvidence and assess it. Transfer evidence exists only for the hypothesis the
    transfer studies used (the champion); every other hypothesis is therefore capped at CONDITIONAL with 'transfer not measured'."""
    by_a = {i.a: i for i in incs}
    out = {}
    for h in hids:
        if h == "B0":
            continue
        inc = by_a.get(h)
        sub = fits[(fits["hid"] == h) & fits["mechanism_consistent"].notna()] if len(fits) else fits
        share = float(sub["mechanism_consistent"].astype(bool).mean()) if len(sub) else None
        transfer = tuple((s.detail.get("axis", s.qid), str(s.verdict)) for s in studies if s.detail.get("hypothesis") == h and s.qid in ("Q15", "Q16", "Q17", "Q18"))
        grp = [s for s in studies if s.detail.get("hypothesis") == h and s.qid not in ("Q15", "Q16", "Q17", "Q18")]
        null_ok = None if not grp else all(s.verdict != StudyVerdict.INVALID for s in grp)
        ev = VH.HypothesisEvidence(h, inc.diff if inc else None, inc.lo if inc else None, inc.hi if inc else None, inc.q if inc else None, share,
                                   decay[h].status if h in decay else None, regimes[h].regime_dependent if h in regimes else None, transfer,
                                   str(directions[h].flag) if h in directions else None, null_ok)
        out[h] = VH.assess_hypothesis(ev, cfg.alpha)
    return out


def follow_up_questions(rep: "LabReport", now) -> list:
    """Research questions raised by this report (contract section 40): inconclusive studies, decaying signals, direction-blind champions,
    a large unexplained share, failed null controls, promoted discoveries, survivor-only data. Identity-free text; each carries a success
    and a failure criterion so the agenda can decide whether to spend compute on it."""
    from engine.research.core import ExperimentValue, ResearchQuestion
    made = str(as_date(now))
    through = rep.data_through or made
    qs = []

    def mk(text, source, problem, ok, bad, ev):
        qs.append(ResearchQuestion.make(text, source, problem, made, through, ok, bad, expected=ev))

    for s in rep.studies:
        if s.verdict == StudyVerdict.INCONCLUSIVE:
            mk(f"Does {s.question.rstrip('?').lower()} once more matured weeks are available (interval currently spans no effect and a material effect)?",
               "inconclusive", Problem.VOLATILITY, "the interval excludes zero or excludes a gain above 0.01 AUC", "still inconclusive after doubling the dates",
               ExperimentValue(information_gain=0.6, transfer_potential=0.4, compute_cost=8.0))
        if s.verdict == StudyVerdict.INVALID:
            mk(f"Why did the shuffled-input control of '{s.question.rstrip('?').lower()}' gain accuracy (measurement leak or mis-specification)?",
               "contradiction", Problem.DATA_QUALITY, "the control returns to zero after the defect is found", "the gain persists with all known leaks removed",
               ExperimentValue(information_gain=0.9, failure_reduction_value=0.8, compute_cost=5.0))
    for h, d in rep.decay.items():
        if d.status in ("STOPPED", "DECAYING") and h != "B0":
            mk(f"What changed when the {h} volatility signal went from AUC {d.early:.2f} to {d.late:.2f} in the latest folds (regime, crowding, data)?",
               "break", Problem.VOLATILITY, "a cause explains at least half of the drop", "the cause stays unknown and is logged as such",
               ExperimentValue(information_gain=0.7, failure_reduction_value=0.6, compute_cost=6.0))
    ch = rep.directions.get(rep.champion.hid)
    if ch is not None and ch.flag in (DirectionFlag.DIRECTION_BLIND, DirectionFlag.LOSS_SKEWED):
        mk(f"Can anything rank the up-movers above the down-movers among the names {rep.champion.hid} selects (it finds movers with no direction)?",
           "discovery", Problem.DIRECTION, "a direction AUC interval above 0.5 out of sample", "direction stays a coin flip",
           ExperimentValue(direction_value=0.8, loss_reduction_value=0.5, compute_cost=12.0))
    if np.isfinite(rep.unexplained.hi) and rep.unexplained.hi > 0.5:
        mk("Do the extreme movers that no hypothesis ranked share any structure (the unexplained share is above half)?", "surprise", Problem.VOLATILITY,
           "a replicated rule covering a material share of them", "no structure beyond chance", ExperimentValue(information_gain=0.8, volatility_value=0.7, compute_cost=10.0))
    if rep.discovery.promoted:
        mk(f"Do the newly promoted rules {list(rep.discovery.promoted)} transfer to other sectors and stocks?", "discovery", Problem.VOLATILITY,
           "lift above one in at least 75% of sectors", "lift only in the discovery sector mix", ExperimentValue(transfer_potential=0.8, information_gain=0.5, compute_cost=4.0))
    if not rep.survivor_free:
        mk("Do the volatility results change on a survivor-free universe including delisted names?", "data", Problem.DATA_QUALITY,
           "AUC and mover base rate unchanged within 0.01", "material change (results were survivor-biased)",
           ExperimentValue(information_gain=0.9, failure_reduction_value=0.9, compute_cost=30.0))
    seen, uniq = set(), []
    for q in qs:
        if q.question_id not in seen:
            seen.add(q.question_id)
            uniq.append(q)
    return uniq


# ---------------------------------------------------------------------------------------------------------------
# persistence and the harness's own self-check
# ---------------------------------------------------------------------------------------------------------------
def to_jsonable(x: Any) -> Any:
    if dc.is_dataclass(x) and not isinstance(x, type):
        return {f.name: to_jsonable(getattr(x, f.name)) for f in dc.fields(x)}
    if isinstance(x, pd.DataFrame):
        return x.reset_index().to_dict("records") if x.index.name or isinstance(x.index, pd.MultiIndex) else x.to_dict("records")
    if isinstance(x, Mapping):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, _StrEnum):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(x))
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    return x


def save_report(rep: LabReport, out_dir, *, cfg: LabConfig | None = None, seed: int | None = None, name: str = "volatility_lab") -> dict:
    """Write <name>.json and <name>.txt into out_dir with an engine.provenance stamp (code hash, config hash, seed). Returns the paths."""
    import json
    from pathlib import Path
    from engine import provenance
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = to_jsonable(rep)
    body["provenance"] = provenance.stamp(dc.asdict(cfg) if cfg else None, seed)
    (out / f"{name}.json").write_text(json.dumps(body, indent=1, sort_keys=True, default=str), encoding="utf-8")
    (out / f"{name}.txt").write_text(render_report(rep) + "\n", encoding="utf-8")
    return {"json": str(out / f"{name}.json"), "txt": str(out / f"{name}.txt")}


def selfcheck(seed: int = 0, n_dates: int = 90, n_tickers: int = 50) -> dict:
    """The laboratory's own calibration (a check that can fail): on a planted H1 world H1 must beat the baseline out of sample and its
    fitted signs must agree with the mechanism; on a null world no hypothesis may be BH-significantly better than the baseline and the
    signal check must not call a random score a volatility signal."""
    cfg = LabConfig(min_train_dates=40, test_step_dates=12, n_boot=150, seed=seed)
    hyps = [h for h in VH.seeded_hypotheses() if h.hid in ("H1", "H6")]
    res: dict[str, Any] = {}
    for truth in ("H1", "null"):
        F = planted_frame(truth, n_dates=n_dates, n_tickers=n_tickers, seed=seed + 5, effect=1.3)
        wf = walk_forward(F, hyps, "2035-01-01", cfg)
        tb = per_date_table(wf.oos, [f"p_{h}" for h in wf.hids], cfg.top_frac)
        incs = increments_vs_baseline(tb, wf.hids, cfg)
        res[truth] = {"increments": {i.a: (i.diff, i.lo, i.q) for i in incs}}
    h1 = res["H1"]["increments"]["H1"]
    res["planted_recovered"] = bool(h1[1] > 0 and h1[2] <= cfg.alpha)
    res["null_clean"] = all(not (v[1] > 0 and v[2] <= cfg.alpha) for v in res["null"]["increments"].values())
    res["passed"] = bool(res["planted_recovered"] and res["null_clean"])
    return res


# ---------------------------------------------------------------------------------------------------------------
# point-in-time event / insider / filing / analog inputs for H2 and Q10-Q13 (built on the engine.edgar tables, never re-fetched)
# ---------------------------------------------------------------------------------------------------------------
def _public_day(accepted: pd.Series) -> np.ndarray:
    """First calendar day a filing was usable: same day if accepted before the close (16:00 ET), else the next day. datetime64[D]."""
    a = pd.to_datetime(accepted, utc=True).dt.tz_convert("America/New_York")
    day = a.dt.tz_localize(None).dt.normalize()
    late = (a.dt.hour * 60 + a.dt.minute) >= 16 * 60
    return (day + pd.to_timedelta(late.astype(int), unit="D")).to_numpy().astype("datetime64[D]")


def event_inputs(index: pd.MultiIndex, events: pd.DataFrame | None, insider: pd.DataFrame | None, *, earn_kind: str = "EARN", min_events: int = 3,
                 max_days: int = 60) -> pd.DataFrame:
    """days_to_event / filing_n5 / insider_n30 for each (decision date, ticker), using only records public strictly before the decision date.
    days_to_event = the ticker's last public earnings-release date + the median of its last three intervals, minus the decision date
    (calendar days; negative = overdue, NaN with fewer than min_events past releases or more than max_days away). insider_n30 counts
    Form-4 purchases filed in the 30 days before (a filing is public the day after it is filed). filing_n5 counts any classified filing
    public in the previous 5 days (0 for every name while the table is present, whether or not the name ever files - which names file
    LATER is future information). A missing table gives NaN columns, never zeros: absence of the table is not absence of events."""
    d = pd.to_datetime(index.get_level_values(0)).to_numpy().astype("datetime64[D]")
    t = np.asarray(index.get_level_values(1))
    out = pd.DataFrame(np.nan, index=index, columns=["days_to_event", "filing_n5", "insider_n30"])
    if events is not None and len(events):
        # F09: coverage is the TABLE (as for insider_n30), never 'this ticker has a row somewhere in it': that test read the future - a
        # name whose first filing came later read 0 on every earlier date and NaN when the data ended first (scripts/feature_leak_audit.py)
        out["filing_n5"] = 0.0
        ev = events.assign(pub=_public_day(events["accepted"]))
        for tk, g in ev.groupby("ticker"):
            sel = np.flatnonzero(t == tk)
            if len(sel) == 0:
                continue
            pubs = np.sort(g["pub"].to_numpy().astype("datetime64[D]"))
            hi = np.searchsorted(pubs, d[sel], side="left")            # public strictly before the decision date
            lo = np.searchsorted(pubs, d[sel] - np.timedelta64(5, "D"), side="left")
            out.iloc[sel, out.columns.get_loc("filing_n5")] = (hi - lo).astype(float)
            er = np.sort(g.loc[g["kind"] == earn_kind, "pub"].to_numpy().astype("datetime64[D]"))
            if len(er) >= min_events:
                k = np.searchsorted(er, d[sel], side="left")
                dte = np.full(len(sel), np.nan)
                for j in np.flatnonzero(k >= min_events):
                    recent = er[max(0, k[j] - 4):k[j]]
                    gap = np.median(np.diff(recent).astype("timedelta64[D]").astype(float))
                    dte[j] = float((recent[-1] + np.timedelta64(int(round(gap)), "D") - d[sel][j]) / np.timedelta64(1, "D"))
                dte[np.abs(dte) > max_days] = np.nan
                out.iloc[sel, out.columns.get_loc("days_to_event")] = dte
    if insider is not None and len(insider):
        ins = insider.assign(pub=(pd.to_datetime(insider["filed"]).dt.normalize() + pd.Timedelta(days=1)).to_numpy().astype("datetime64[D]"))
        col = out.columns.get_loc("insider_n30")
        out.iloc[:, col] = 0.0
        for tk, g in ins.groupby("symbol"):
            sel = np.flatnonzero(t == tk)
            if len(sel) == 0:
                continue
            pubs = np.sort(g["pub"].to_numpy().astype("datetime64[D]"))
            hi = np.searchsorted(pubs, d[sel], side="left")
            lo = np.searchsorted(pubs, d[sel] - np.timedelta64(30, "D"), side="left")
            out.iloc[sel, col] = (hi - lo).astype(float)
    return out


def analog_inputs(index: pd.MultiIndex, analog_prob: Mapping[Any, float] | pd.Series | None) -> pd.Series:
    """analog_p from a caller-supplied per-date probability (e.g. engine.analogs.Analogs.find(...)['prediction'] of a 'big move next
    week' outcome, computed by the caller with its own point-in-time discipline). Dates the analog engine declined to answer are NaN."""
    dates = pd.to_datetime(index.get_level_values(0))
    if analog_prob is None:
        return pd.Series(np.nan, index=index, name="analog_p")
    s = pd.Series(analog_prob)
    s.index = pd.to_datetime(s.index)
    return pd.Series(s.reindex(dates).to_numpy(float), index=index, name="analog_p")


def with_event_inputs(F: pd.DataFrame, events: pd.DataFrame | None = None, insider: pd.DataFrame | None = None,
                      analog_prob: Mapping[Any, float] | pd.Series | None = None) -> pd.DataFrame:
    """Return F with the event columns attached (existing ones replaced). attrs (survivor_free) are preserved."""
    E = event_inputs(F.index, events, insider)
    G = F.drop(columns=[c for c in ("days_to_event", "filing_n5", "insider_n30", "analog_p") if c in F.columns]).join(E)
    if analog_prob is not None:
        G["analog_p"] = analog_inputs(F.index, analog_prob).to_numpy()
    G.attrs.update(F.attrs)
    return G


# ---------------------------------------------------------------------------------------------------------------
# hypothesis ledger across reports: which mechanisms keep winning, which are dying (accumulates over the loop's steps)
# ---------------------------------------------------------------------------------------------------------------
class HypothesisLedger:
    """Append-only record of each hypothesis's increment over B0 at each run of the lab. Reads are point-in-time: trajectory(hid, as_of)
    returns only entries whose data_through precedes as_of. Never deleted; a retired hypothesis keeps its history."""

    def __init__(self):
        self._rows: list[dict] = []

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, rep: LabReport) -> int:
        n = 0
        by = {i.a: i for i in rep.increments}
        for h in rep.hids:
            if h == "B0":
                continue
            c, i, a = rep.scorecards.get(h), by.get(h), rep.assessments.get(h)
            self._rows.append({"now": rep.now, "data_through": rep.data_through, "hid": h, "auc": c.auc if c else np.nan,
                               "inc": i.diff if i else np.nan, "inc_lo": i.lo if i else np.nan, "q": i.q if i else np.nan,
                               "status": str(a.status) if a else "UNKNOWN", "cfg": rep.cfg_hash, "survivor_free": rep.survivor_free})
            n += 1
        return n

    def trajectory(self, hid: str, as_of) -> pd.DataFrame:
        r = [x for x in self._rows if x["hid"] == hid and x["data_through"] and as_date(x["data_through"]) < as_date(as_of)]
        return pd.DataFrame(r, columns=["now", "data_through", "hid", "auc", "inc", "inc_lo", "q", "status", "cfg", "survivor_free"])

    def consistent(self, hid: str, as_of, min_reports: int = 3) -> dict:
        """'Consistent' in the timeline sense: significant (lower bound above 0) in every report so far, with how many there were."""
        t = self.trajectory(hid, as_of)
        if len(t) < min_reports:
            return {"hid": hid, "n": int(len(t)), "consistent": None, "reason": f"only {len(t)} reports"}
        ok = (t["inc_lo"] > 0).to_numpy()
        return {"hid": hid, "n": int(len(t)), "consistent": bool(ok.all()), "share_significant": float(ok.mean()),
                "reason": "significant in every report" if ok.all() else f"significant in {int(ok.sum())}/{len(t)} reports"}

    def league(self, as_of) -> pd.DataFrame:
        rows = []
        for h in sorted({x["hid"] for x in self._rows}):
            t = self.trajectory(h, as_of)
            if len(t):
                rows.append({"hid": h, "reports": len(t), "mean_inc": float(t["inc"].mean()), "worst_lo": float(t["inc_lo"].min()),
                             "last_status": t["status"].iloc[-1]})
        return pd.DataFrame(rows, columns=["hid", "reports", "mean_inc", "worst_lo", "last_status"]).sort_values("mean_inc", ascending=False).reset_index(drop=True)


def retire_candidates(ledger: HypothesisLedger, registry: VH.HypothesisRegistry, as_of, min_reports: int = 4) -> list[str]:
    """Hypotheses to retire: at least min_reports reports, none significant, mean increment not positive. Retirement is a registry state
    change (never deletion) that the caller applies."""
    out = []
    for h in registry.all():
        if h.kind == VH.HypKind.RESIDUAL:
            continue
        t = ledger.trajectory(h.hid, as_of)
        if len(t) >= min_reports and not (t["inc_lo"] > 0).any() and t["inc"].mean() <= 0:
            out.append(h.hid)
    return out


# ---------------------------------------------------------------------------------------------------------------
# C67: the NEXT-DAY PATH of a 5-10% (or >10%) mover as a first-class prediction target, beside the move itself.
# Episode detection and the path taxonomy are owned by engine.research.episodes / episode_paths (R21); this consumes their output
# by duck type: a frame indexed (date, ticker) with an `end` column, a `path` label and point-in-time feature columns.
# ---------------------------------------------------------------------------------------------------------------
PATH_CLASSES = ("CONTINUATION", "REVERSAL", "CONSOLIDATION", "EXPANSION")
_PATH_ALIASES = {"SPIKED_NEXT_DAY": "CONTINUATION", "REVERSED_NEXT_DAY": "REVERSAL", "CONSOLIDATED": "CONSOLIDATION", "EXPANDED": "EXPANSION",
                 "STOPPED": "CONSOLIDATION"}
PATH_FEATURES = ("lv20", "shock1", "vol_ratio_short", "compress", "lvol_surge", "surge_x_move", "gap_ratio", "dv_rel", "mkt_vol", "mkt_stress",
                 "xs_vol_rank", "breakout_prox", "near_hi", "near_lo", "abs_rel_r20")


def normalise_paths(ep: pd.DataFrame, col: str = "path") -> pd.DataFrame:
    """Map another module's path labels onto the four C67 classes. Labels it cannot map (UNCLASSIFIED, unknown) are dropped and
    counted in attrs['unmapped']: an unlabelled path is not a fifth class and is never guessed."""
    lab = ep[col].astype(str).str.upper().map(lambda s: _PATH_ALIASES.get(s, s))
    keep = lab.isin(PATH_CLASSES)
    out = ep[keep.to_numpy()].copy()
    out["path"] = lab[keep].to_numpy()
    out.attrs.update(ep.attrs)
    out.attrs["unmapped"] = int((~keep).sum())
    return out


def planted_path_frame(kind: str = "signal", n_dates: int = 80, per_date: int = 40, seed: int = 0, strength: float = 1.6) -> pd.DataFrame:
    """Episodes with a KNOWN next-day rule. kind='signal': REVERSAL is likelier when the episode day's volume surge and shock are large and
    the name sits far from its high; EXPANSION when own volatility is high; else CONTINUATION/CONSOLIDATION by coin. kind='null': labels
    are drawn from the fixed class prior, independent of every feature."""
    if kind not in ("signal", "null"):
        raise ValueError("kind must be 'signal' or 'null'")
    rng = np.random.default_rng(seed)
    n = n_dates * per_date
    dates = pd.date_range("2013-01-04", periods=n_dates, freq="B")
    idx = pd.MultiIndex.from_arrays([np.repeat(dates, per_date), [f"E{i:03d}" for i in np.tile(np.arange(per_date), n_dates)]], names=["date", "ticker"])
    F = pd.DataFrame({"vol20": np.exp(rng.normal(np.log(0.03), 0.4, n)), "vol_surge": np.exp(rng.normal(0.3, 0.6, n)), "absr1": rng.uniform(0.05, 0.14, n),
                      "atr": np.exp(rng.normal(np.log(0.04), 0.3, n)), "range20": np.exp(rng.normal(np.log(0.2), 0.3, n)), "gap": rng.normal(0, 0.03, n),
                      "dist_hi": -rng.exponential(0.06, n), "dist_lo": rng.exponential(0.1, n), "log_dv": rng.normal(15, 1, n),
                      "r5": rng.normal(0, 0.06, n), "r20": rng.normal(0, 0.1, n), "m_vol": np.repeat(0.01 * np.exp(rng.normal(0, 0.2, n_dates)), per_date),
                      "m_r5": np.repeat(rng.normal(0, 0.01, n_dates), per_date), "m_r20": np.repeat(rng.normal(0, 0.02, n_dates), per_date),
                      "m_breadth": np.repeat(rng.uniform(0.3, 0.7, n_dates), per_date), "logp": rng.normal(2.8, 0.7, n)}, index=idx)
    prior = np.array([0.30, 0.25, 0.30, 0.15])
    logits = np.tile(np.log(prior), (n, 1))
    if kind == "signal":
        z = lambda v: (v - v.mean()) / v.std()
        logits[:, 1] += strength * (0.9 * z(np.log(F["vol_surge"].to_numpy())) + 0.6 * z(F["absr1"].to_numpy()) - 0.5 * z(F["dist_hi"].to_numpy()))
        logits[:, 3] += strength * 1.1 * z(np.log(F["vol20"].to_numpy()))
    pr = np.exp(logits - logits.max(1, keepdims=True))
    pr /= pr.sum(1, keepdims=True)
    cum = pr.cumsum(1)
    F["path"] = np.array(PATH_CLASSES)[(rng.random(n)[:, None] > cum).sum(1).clip(0, 3)]
    F["end"] = pd.to_datetime(F.index.get_level_values(0)) + pd.Timedelta(days=4)
    F.attrs["survivor_free"] = True
    return F


@dc.dataclass(frozen=True)
class PathScore:
    n: int
    logloss: float
    logloss_prior: float
    skill: float                   # 1 - logloss/logloss_prior (0 = no better than the class prior)
    skill_lo: float
    skill_hi: float
    acc: float
    acc_prior: float
    per_class_auc: Mapping[str, float]
    prior: Mapping[str, float]
    verdict: StudyVerdict
    reason: str


def _softmax_fit(Z: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=400).fit(Z, y)


def path_walk_forward(ep: pd.DataFrame, now, cfg: LabConfig = LabConfig(), fit_cfg: VH.FitConfig = VH.FitConfig(), features: Sequence[str] = PATH_FEATURES,
                      min_train: int = 300) -> tuple[pd.DataFrame, PathScore]:
    """Walk-forward multiclass prediction of the next-day path of an episode. Fold k trains on episodes whose path had ended before fold
    k's first date, and is scored against the CLASS PRIOR of the same training rows, so the headline is skill over 'always say the
    base rates'. Returns per-episode probabilities (p_<CLASS>) and the score. Bootstrap by month over per-date log-loss differences."""
    ep = normalise_paths(ep)
    check_end = pd.to_datetime(ep["end"]).dt.normalize() >= pd.Timestamp(as_date(now)) if len(ep) else pd.Series(dtype=bool)
    if len(ep) and check_end.any():
        ep = ep[~check_end.to_numpy()]
    ep = ep.sort_index()
    feats = tuple(f for f in features if not VH.missing_columns((f,), ep.columns))
    nan = float("nan")
    empty = PathScore(0, nan, nan, nan, nan, nan, nan, nan, {}, {}, StudyVerdict.UNKNOWN, "no episodes or no derivable features")
    if len(ep) == 0 or not feats:
        return pd.DataFrame(), empty
    D = VH.derive(ep, feats).to_numpy(float)
    y = ep["path"].map({c: i for i, c in enumerate(PATH_CLASSES)}).to_numpy()
    dn = pd.to_datetime(ep.index.get_level_values(0)).to_numpy()
    ends = pd.to_datetime(ep["end"]).to_numpy()
    folds = make_wf_folds(ep.index.get_level_values(0), cfg)
    P = np.full((len(ep), 4), np.nan)
    Q = np.full((len(ep), 4), np.nan)
    for fd in folds:
        tr = np.flatnonzero(ends < np.datetime64(fd.now))
        te = np.flatnonzero(np.isin(dn, np.array(fd.test_dates, dtype="datetime64[ns]")))
        if len(tr) < min_train or len(te) == 0 or len(np.unique(y[tr])) < 4:
            continue
        std = VH.Standardiser.fit(D[tr], fit_cfg.winsor)
        clf = _softmax_fit(std.apply(D[tr]), y[tr], fit_cfg.C)
        P[te] = clf.predict_proba(std.apply(D[te]))
        Q[te] = np.bincount(y[tr], minlength=4) / len(tr)
    ok = np.isfinite(P).all(1)
    if ok.sum() < 100:
        return pd.DataFrame(P, index=ep.index, columns=[f"p_{c}" for c in PATH_CLASSES]), dc.replace(empty, n=int(ok.sum()), reason="too few scored episodes")
    yo = y[ok]
    ll = -np.log(np.clip(P[ok][np.arange(ok.sum()), yo], 1e-6, 1))
    llq = -np.log(np.clip(Q[ok][np.arange(ok.sum()), yo], 1e-6, 1))
    dts = pd.DatetimeIndex(dn[ok])
    per_date = pd.DataFrame({"a": ll, "b": llq}).groupby(dts.normalize()).mean()
    cl = month_cluster(per_date.index)
    diff = TS.cluster_bootstrap_mean((per_date["b"] - per_date["a"]).to_numpy(), cl, n_boot=cfg.n_boot, seed=cfg.seed)
    skill_scale = float(llq.mean())
    aucs = {c: (_auc(P[ok][:, i], yo == i) if 0 < (yo == i).sum() < len(yo) else nan) for i, c in enumerate(PATH_CLASSES)}
    score = PathScore(int(ok.sum()), float(ll.mean()), float(llq.mean()), float(1 - ll.mean() / llq.mean()), diff.lo / skill_scale, diff.hi / skill_scale,
                      float((P[ok].argmax(1) == yo).mean()), float((Q[ok].argmax(1) == yo).mean()), aucs,
                      dict(zip(PATH_CLASSES, (np.bincount(yo, minlength=4) / len(yo)).tolist())),
                      StudyVerdict.SUPPORTED if diff.lo > 0 else (StudyVerdict.NOT_SUPPORTED if diff.hi < 0.002 else StudyVerdict.INCONCLUSIVE),
                      "log-loss gain over the class prior, month-clustered bootstrap")
    out = pd.DataFrame(P, index=ep.index, columns=[f"p_{c}" for c in PATH_CLASSES])
    out["path"] = ep["path"].to_numpy()
    return out, score


def path_direction_split(ep: pd.DataFrame, probs: pd.DataFrame, up_col: str = "ep_up") -> dict:
    """For episodes with a known side (`ep_up` = the episode day closed up), does the REVERSAL probability separate reversals from
    continuations equally for up and down episodes? A reversal signal that only works on one side is a direction claim in disguise."""
    if up_col not in ep or len(probs) == 0:
        return {"tested": False, "reason": f"no {up_col} column or no probabilities"}
    j = probs.join(ep[[up_col]], how="inner").dropna(subset=["p_REVERSAL"])
    j = j[j["path"].isin(["REVERSAL", "CONTINUATION"])]
    res: dict[str, Any] = {"tested": True}
    for side, name in ((1.0, "up_episodes"), (0.0, "down_episodes")):
        s = j[j[up_col] == side]
        res[name] = {"n": int(len(s)), "auc": _auc(s["p_REVERSAL"].to_numpy(), (s["path"] == "REVERSAL").to_numpy()) if 30 <= len(s) and s["path"].nunique() == 2 else float("nan")}
    a, b = res["up_episodes"]["auc"], res["down_episodes"]["auc"]
    res["asymmetric"] = bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) > 0.08)
    return res


def run_path_study(ep: pd.DataFrame, now, cfg: LabConfig = LabConfig(), fit_cfg: VH.FitConfig = VH.FitConfig()) -> StudyResult:
    """Study P01: can the next-day path class of a mover be predicted better than its base rates? Null-controlled by refitting with the
    features permuted across episodes within each date; a control that gains skill invalidates the study."""
    spec = StudySpec("P01", "Can the next-day path of a 5-10% mover (continuation, reversal, consolidation, expansion) be predicted?", "path")
    probs, sc = path_walk_forward(ep, now, cfg, fit_cfg)
    if sc.verdict == StudyVerdict.UNKNOWN or not np.isfinite(sc.skill):
        return _unknown(spec, sc.reason, n=sc.n)
    perm = shuffle_sources(normalise_paths(ep), [c for c in VH.required_columns([f for f in PATH_FEATURES if not VH.missing_columns((f,), ep.columns)]) if not c.startswith("m_")], "row", cfg.seed + 5)
    _, ctl = path_walk_forward(perm, now, cfg, fit_cfg)
    verdict = sc.verdict
    cav = [f"per-class AUC {({k: round(v, 3) for k, v in sc.per_class_auc.items()})}"]
    if np.isfinite(ctl.skill) and ctl.skill_lo > 0:
        verdict, cav = StudyVerdict.INVALID, cav + [f"shuffled-feature control gained skill {ctl.skill:+.4f}"]
    return StudyResult(spec.qid, spec.section, verdict, sc.skill, sc.skill_lo, sc.skill_hi, float("nan"), float("nan"), int(sc.n),
                       ctl.skill if np.isfinite(ctl.skill) else float("nan"), {"prior": dict(sc.prior), "acc": sc.acc, "acc_prior": sc.acc_prior,
                                                                          "per_class_auc": dict(sc.per_class_auc)}, tuple(cav))


# ---------------------------------------------------------------------------------------------------------------
# C67: long, resumable sweeps over many years and names (the iterator the wave-2 loop schedules forever)
# ---------------------------------------------------------------------------------------------------------------
@dc.dataclass
class SweepCheckpoint:
    """What a sweep needs to resume: the passes finished, the unit (year, universe seed) it is inside, and the verdicts so far. JSON only."""
    passes: int = 0
    done: list = dc.field(default_factory=list)          # [[pass, year, seed], ...]
    verdicts: dict = dc.field(default_factory=dict)      # "pass:year:seed" -> {qid: verdict}
    cfg_hash: str = ""

    def key(self, p: int, year: int, seed: int) -> str:
        return f"{p}:{year}:{seed}"

    def is_done(self, p: int, year: int, seed: int) -> bool:
        return [p, year, seed] in self.done

    def save(self, path) -> None:
        import json
        import os
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dc.asdict(self), fh, sort_keys=True)
        os.replace(tmp, path)                                # atomic: a kill mid-write cannot corrupt the checkpoint

    @classmethod
    def load(cls, path, cfg_hash: str = "") -> "SweepCheckpoint":
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return cls(cfg_hash=cfg_hash)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return cls(cfg_hash=cfg_hash)                    # an unreadable checkpoint restarts the sweep; it never crashes it
        ck = cls(int(d.get("passes", 0)), [list(x) for x in d.get("done", [])], dict(d.get("verdicts", {})), d.get("cfg_hash", ""))
        if cfg_hash and ck.cfg_hash and ck.cfg_hash != cfg_hash:
            return cls(cfg_hash=cfg_hash)                    # different configuration: the old verdicts do not describe this run
        ck.cfg_hash = cfg_hash or ck.cfg_hash
        return ck


@dc.dataclass(frozen=True)
class SweepUnit:
    pass_no: int
    year: int
    seed: int
    ran: tuple
    rows: int
    skipped: str
    checkpoint_path: str


def sweep(loader: Callable[[int, int], pd.DataFrame], years: Sequence[int], state: LabState, now_for: Callable[[int], Any], *,
          checkpoint_path=None, max_passes: int | None = 1, name_seed: int = 0, tasks_per_unit: int = 3) -> Iterator[SweepUnit]:
    """Generator over (pass, year) units; pass p uses universe seed name_seed+p, so successive passes look at different draws of names.
    For each unit it loads `loader(year, seed)` (a lab frame for that calendar year), runs up to tasks_per_unit lab tasks via step(), and
    saves the checkpoint BEFORE yielding, so a kill between units costs nothing. Units already in the checkpoint are skipped, which makes
    resumption exact. max_passes=None never ends (the wave-2 loop pulls from it as compute allows). A frame that fails validation is
    skipped with the reason, not fatal: one bad year must not stop a sweep that is meant to run for weeks."""
    ck = SweepCheckpoint.load(checkpoint_path, state.cfg.fingerprint()) if checkpoint_path else SweepCheckpoint(cfg_hash=state.cfg.fingerprint())
    p = 0
    while max_passes is None or p < max_passes:
        for y in years:
            seed = name_seed + p
            if ck.is_done(p, y, seed):
                continue
            F = loader(y, seed)
            reason = "empty frame" if F is None or len(F) == 0 else ""
            ran: tuple = ()
            if not reason:
                errs = validate_frame(F, state.cfg)
                reason = "; ".join(errs)
            if not reason:
                now = now_for(y)
                Fm, _ = mature_only(F, now)
                if len(Fm) == 0:
                    reason = "no matured rows before now"
                else:
                    state.ran_on.clear()
                    state._wf = None
                    r = step(state, now, Fm, max_tasks=tasks_per_unit)
                    ran = r.ran
                    ck.verdicts[ck.key(p, y, seed)] = {q: str(res.verdict) for q, res in state.results.items()}
            ck.done.append([p, y, seed])
            ck.passes = p
            if checkpoint_path:
                ck.save(checkpoint_path)
            yield SweepUnit(p, y, seed, ran, 0 if F is None else int(len(F)), reason, str(checkpoint_path or ""))
        if all(ck.is_done(p, y, name_seed + p) for y in years):
            ck.passes = max(ck.passes, p + 1)
        p += 1
        if checkpoint_path:
            ck.save(checkpoint_path)


def sweep_consistency(ck: SweepCheckpoint, qid: str) -> dict:
    """Across every (pass, year, seed) unit a study ran in: how often it was SUPPORTED. 'Consistent' means supported in every unit it
    was run in (at least three); one supported year among many is a hypothesis, not a finding."""
    v = [d[qid] for d in ck.verdicts.values() if qid in d]
    if len(v) < 3:
        return {"qid": qid, "units": len(v), "consistent": None, "supported_share": float("nan")}
    share = sum(x == "SUPPORTED" for x in v) / len(v)
    return {"qid": qid, "units": len(v), "consistent": share == 1.0, "supported_share": share}


# ---------------------------------------------------------------------------------------------------------------
# timing, disagreement, direction-safety table, and the hypothesis x transfer-axis matrix
# ---------------------------------------------------------------------------------------------------------------
def run_timing_study(F: pd.DataFrame, hyp: VH.Hypothesis, now, cfg: LabConfig = LabConfig(), fit_cfg: VH.FitConfig = VH.FitConfig()) -> StudyResult:
    """Q19: given that a name will move, can WHICH DAY be predicted better than uniform / always-the-middle? Walk-forward forecasts; the
    verdict rests on the timing log-loss gain over uniform with a month-clustered bootstrap over per-date means."""
    spec = StudySpec("Q19", "When in the holding week does the move come?", "timing")
    fc = walk_forward_forecast(F, hyp, now, cfg, fit_cfg)
    if len(fc) == 0 or "p_day1" not in fc:
        return _unknown(spec, "no out-of-sample forecasts (frame too short or model unfittable)")
    mv = fc[(fc["touch"] == 1) & fc["tday"].between(1, HORIZON_BARS) & fc["p_day1"].notna()]
    if len(mv) < 60:
        return StudyResult(spec.qid, spec.section, StudyVerdict.INCONCLUSIVE, float("nan"), float("nan"), float("nan"), float("nan"),
                           detail={"n_movers": int(len(mv))}, caveats=(f"only {len(mv)} scored movers",))
    P = mv[[f"p_day{d}" for d in TIMING_DAYS]].to_numpy(float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    day = mv["tday"].astype(int).to_numpy()
    gain = np.log(np.clip(P[np.arange(len(mv)), day - 1], 1e-6, 1)) - np.log(1.0 / HORIZON_BARS)
    per = pd.Series(gain).groupby(pd.to_datetime(mv.index.get_level_values(0)).to_numpy()).mean()
    bm = TS.cluster_bootstrap_mean(per.to_numpy(), month_cluster(per.index), n_boot=cfg.n_boot, seed=cfg.seed)
    p = TS.cluster_signflip_p(per.to_numpy(), month_cluster(per.index), n_perm=1000, seed=cfg.seed)
    v = StudyVerdict.SUPPORTED if bm.lo > 0 else (StudyVerdict.NOT_SUPPORTED if bm.hi < 0.02 else StudyVerdict.INCONCLUSIVE)
    return StudyResult(spec.qid, spec.section, v, bm.mean, bm.lo, bm.hi, p, float("nan"), int(len(per)),
                       detail={"n_movers": int(len(mv)), "mae_expected_day": float(np.abs(mv["exp_day"] - day).mean()), "mae_constant_3": float(np.abs(3 - day).mean()),
                               "hypothesis": hyp.hid}, caveats=("gain is nats of log-likelihood per mover over a uniform day distribution",))


def disagreement_slices(oos: pd.DataFrame, a: str, b: str, F: pd.DataFrame | None = None, top_frac: float = 0.15, cfg: LabConfig = LabConfig()) -> dict:
    """Where do two hypotheses disagree most (per-date rank difference) and who is right there? The slice with the largest disagreement is
    the cheapest place to separate them: the realised mover rate among names a ranks high and b low, against the reverse. This is
    the volatility-side counterpart of Arena.next_experiment, in the ranks the lab actually uses."""
    ra = oos[f"p_{a}"].groupby(level=0).rank(pct=True, method="first")
    rb = oos[f"p_{b}"].groupby(level=0).rank(pct=True, method="first")
    hi_a, hi_b = (ra > 1 - top_frac) & (rb < 0.5), (rb > 1 - top_frac) & (ra < 0.5)
    y = oos["touch"]
    out: dict[str, Any] = {"a": a, "b": b, "n_a_only": int(hi_a.sum()), "n_b_only": int(hi_b.sum()), "rate_a_only": float(y[hi_a].mean()) if hi_a.any() else float("nan"),
           "rate_b_only": float(y[hi_b].mean()) if hi_b.any() else float("nan"), "base_rate": float(y.mean())}
    if hi_a.sum() >= 30 and hi_b.sum() >= 30:
        from scipy.stats import fisher_exact
        tab = [[int(y[hi_a].sum()), int((1 - y[hi_a]).sum())], [int(y[hi_b].sum()), int((1 - y[hi_b]).sum())]]
        out["p_diff"] = float(fisher_exact(tab)[1])
        out["winner"] = a if out["rate_a_only"] > out["rate_b_only"] and out["p_diff"] < 0.05 else b if out["rate_b_only"] > out["rate_a_only"] and out["p_diff"] < 0.05 else "UNDECIDED"
    else:
        out["p_diff"], out["winner"] = float("nan"), "UNDECIDED"
    if F is not None and hi_a.sum() >= 30:
        cols = [c for c in ("vol20", "log_dv", "logp", "vol_surge") if c in F.columns]
        sub = F.reindex(oos.index[hi_a.to_numpy()])[cols]
        out["a_only_profile"] = {c: float(sub[c].median()) for c in cols}
    return out


def direction_safety_table(rep: LabReport) -> pd.DataFrame:
    """One row per hypothesis: how well it finds volatility and whether it does so WITHOUT direction (the contract's 'dangerous' signals).
    dangerous = finds volatility (AUC lower bound above 0.5) and is DIRECTION_BLIND or LOSS_SKEWED; that is exactly the signal a long-only
    portfolio cannot use without a separate, proven direction source."""
    rows = []
    for h in rep.hids:
        c, d = rep.scorecards.get(h), rep.directions.get(h)
        if c is None or d is None:
            continue
        finds = bool(np.isfinite(c.auc_lo) and c.auc_lo > 0.5)
        rows.append({"hid": h, "auc": c.auc, "auc_lo": c.auc_lo, "finds_volatility": finds, "direction": str(d.flag), "up_share": d.up_share,
                     "dir_auc": d.dir_auc, "worst_decile_close": d.worst_decile_close,
                     "dangerous": bool(finds and d.flag in (DirectionFlag.DIRECTION_BLIND, DirectionFlag.LOSS_SKEWED))})
    return pd.DataFrame(rows, columns=["hid", "auc", "auc_lo", "finds_volatility", "direction", "up_share", "dir_auc", "worst_decile_close", "dangerous"])


def transfer_matrix(F: pd.DataFrame, hyps: Sequence[VH.Hypothesis], now, cfg: LabConfig = LabConfig(), fit_cfg: VH.FitConfig = VH.FitConfig(),
                    axes: Sequence[str] = ("YEAR", "SECTOR", "STOCK", "REGIME")) -> pd.DataFrame:
    """Hypothesis x axis grid of transfer verdicts: the champion-only transfer studies leave every other hypothesis with 'transfer not
    measured'; this measures each one. A hypothesis that transfers on one axis and fails another is reported as such, not averaged."""
    rows = []
    qmap = {"YEAR": "Q15", "SECTOR": "Q16", "STOCK": "Q17", "REGIME": "Q18"}
    for h in hyps:
        for ax in axes:
            spec = next(s for s in STUDIES if s.qid == qmap[ax])
            r = run_transfer_study(spec, F, h, now, cfg, fit_cfg)
            rows.append({"hid": h.hid, "axis": ax, "verdict": str(r.verdict), "cross_gain": r.effect, "lo": r.lo, "hi": r.hi, "n_units": r.n_dates,
                         "why": "; ".join(r.caveats[:1])})
    return pd.DataFrame(rows, columns=["hid", "axis", "verdict", "cross_gain", "lo", "hi", "n_units", "why"])


def explain_extremes(oos: pd.DataFrame, hids: Sequence[str], top_frac: float = 0.10, limit: int = 50) -> pd.DataFrame:
    """For the most extreme realised movers: which hypotheses had them in their top `top_frac` that day (the 'could the features have known'
    record for the volatility side). An extreme none of them ranked is listed with an empty `anticipated_by`: that is the UNKNOWN bucket."""
    hs = [h for h in hids if f"p_{h}" in oos and oos[f"p_{h}"].notna().any()]
    ex = oos[oos["touch"] == 1].nlargest(limit, "absmove")
    rows = []
    for idx, r in ex.iterrows():
        who = []
        for h in hs:
            day = oos.loc[oos.index.get_level_values(0) == idx[0], f"p_{h}"]
            if r[f"p_{h}"] >= day.quantile(1 - top_frac):
                who.append(h)
        rows.append({"absmove": float(r["absmove"]), "tday": r.get("tday"), "anticipated_by": tuple(who), "n_anticipating": len(who),
                     "best_rank": float(max((oos.loc[oos.index.get_level_values(0) == idx[0], f"p_{h}"].rank(pct=True).loc[idx] for h in hs), default=np.nan))})
    return pd.DataFrame(rows, columns=["absmove", "tday", "anticipated_by", "n_anticipating", "best_rank"])


# ---------------------------------------------------------------------------------------------------------------
# daily use, magnitude honesty, and power planning
# ---------------------------------------------------------------------------------------------------------------
def run_day(model: VolatilityModel, Fnow: pd.DataFrame, now, *, top_n: int = 20, min_p: float = 0.0) -> pd.DataFrame:
    """The day's volatility shortlist from a fitted model and a point-in-time frame (rows dated `now` only). Returns the top_n names by
    calibrated P(move) with magnitude, expected day and the direction-safety flag. Names are ranked WITHIN the day; a model that abstains
    returns an empty frame. `dir_blind` rows are kept and marked, never dropped: whether to trade them is the decision layer's call, and
    it must be told they carry no direction."""
    if len(Fnow) == 0:
        return pd.DataFrame()
    dts = pd.to_datetime(Fnow.index.get_level_values(0))
    if (dts != pd.Timestamp(as_date(now))).any():
        raise FirewallBreach(f"run_day expects rows dated exactly {as_date(now)}")
    fc = model.forecast(Fnow, now)
    if fc["abstain"].all():
        return pd.DataFrame()
    fc = fc[fc["p_move"] >= min_p].sort_values("p_move", ascending=False, kind="mergesort").head(top_n)
    fc["rank"] = np.arange(1, len(fc) + 1)
    fc["expected_abs_move"] = fc["p_move"] * fc["mag_med"]
    return fc


def magnitude_coverage(fc: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    """Coverage of the q90 magnitude forecast inside forecast-size buckets. Overall q90 coverage can look right while being wrong in the tails
    that matter (the biggest forecasts), so it is reported per bucket with a Wilson interval."""
    m = fc[fc["mag_q90"].notna() & fc["absmove"].notna()]
    if len(m) < buckets * 30:
        return pd.DataFrame(columns=["bucket", "n", "coverage", "lo", "hi", "ok"])
    b = pd.qcut(m["mag_q90"].rank(method="first"), buckets, labels=False)
    rows = []
    for k in range(buckets):
        s = m[b == k]
        hit = int((s["absmove"] <= s["mag_q90"]).sum())
        lo, hi = wilson(hit, len(s))
        rows.append({"bucket": k, "n": len(s), "coverage": hit / len(s), "lo": lo, "hi": hi, "ok": bool(lo <= 0.9 <= hi)})
    return pd.DataFrame(rows)


def dates_needed(inc: Increment, target: float = 0.01, z: float = 2.487) -> dict:
    """How many more decision dates before an increment of `target` AUC could be told from zero (planning, not a claim). Uses the observed
    per-date standard error implied by the interval; NaN when the interval is not usable."""
    if not (np.isfinite(inc.lo) and np.isfinite(inc.hi) and inc.n_dates >= 4):
        return {"n_now": int(inc.n_dates), "n_needed": None, "note": "no usable interval"}
    se_now = (inc.hi - inc.lo) / (2 * 1.645)
    sd_per_date = se_now * math.sqrt(inc.n_dates)
    need = int(math.ceil((z * sd_per_date / target) ** 2))
    return {"n_now": int(inc.n_dates), "n_needed": need, "more": max(0, need - int(inc.n_dates)),
            "note": "assumes independent dates; monthly clustering makes the true figure larger"}


def power_table(rep: LabReport, target: float = 0.01) -> pd.DataFrame:
    """dates_needed for every hypothesis increment in a report: which 'not significant yet' results are just under-powered."""
    rows = []
    for i in rep.increments:
        d = dates_needed(i, target)
        rows.append({"hid": i.a, "diff": i.diff, "n_dates": i.n_dates, "n_needed": d["n_needed"], "more": d.get("more"), "significant": bool(np.isfinite(i.lo) and i.lo > 0)})
    return pd.DataFrame(rows, columns=["hid", "diff", "n_dates", "n_needed", "more", "significant"])


# ---------------------------------------------------------------------------------------------------------------
# path study through the loop, profile of the unexplained, and seed stability of the verdict
# ---------------------------------------------------------------------------------------------------------------
def step_paths(state: LabState, now, episodes: pd.DataFrame | None) -> StepResult:
    """Loop entry for the C67 next-day-path question. `episodes` is R21's episode frame (duck-typed: date/ticker index, `end`, `path`, feature
    columns). Runs P01 once per distinct frame, stores it in state.results, and returns a research-namespace record."""
    if episodes is None or len(episodes) == 0:
        return StepResult((), ("P01",), None, "no episode frame supplied")
    if pd.to_datetime(episodes["end"]).max() >= pd.Timestamp(as_date(now)):
        ep, dropped = mature_only(episodes, now)
    else:
        ep, dropped = episodes, 0
    key = frame_key(ep.assign(touch=1.0) if "touch" not in ep else ep, now)
    if state.ran_on.get("P01") == key:
        return StepResult((), (), None, "P01 already run on this frame")
    res = run_path_study(ep, now, state.cfg, state.fit_cfg)
    state.results["P01"] = res
    state.ran_on["P01"] = key
    through = pd.to_datetime(ep["end"]).max() if len(ep) else pd.Timestamp(as_date(now)) - pd.Timedelta(days=1)
    from engine.learning.knowledge import wall_stamp as _wall_stamp      # F31: the caller's injected clock
    prov = Provenance(_wall_stamp(), str(through.date()), current_code_hash(), key, state.cfg.fingerprint(),
                      "volatility_lab.step_paths", "", state.cfg.seed, str(through.date()))
    payload = {"kind": "volatility_lab_paths", "verdict": str(res.verdict), "skill": res.effect, "skill_lo": res.lo, "dropped_immature": dropped,
               "per_class_auc": dict(res.detail.get("per_class_auc", {})), "status": "IMPLEMENTED - NOT VALIDATED"}
    return StepResult(("P01",), (), MaturedRecord("VP" + stable_hash([key], 12), str(through.date()), payload, prov, Namespace.MATURED_RESEARCH), "ok")


def unexplained_profile(oos: pd.DataFrame, F: pd.DataFrame, hids: Sequence[str], cfg: LabConfig = LabConfig()) -> pd.DataFrame:
    """What do the extremes that NO hypothesis ranked look like, next to the extremes that were anticipated and next to non-movers? Median of
    each base column per group, plus the standardised difference (unexplained minus anticipated). Structure here is where a new hypothesis
    (H10+) should be looked for; no structure means the unexplained really is unknown."""
    hs = [h for h in hids if f"p_{h}" in oos and oos[f"p_{h}"].notna().any()]
    if not hs:
        return pd.DataFrame(columns=["column", "unexplained", "anticipated", "quiet", "std_diff"])
    net = min(0.5, cfg.top_frac * 3)
    caught = np.any([oos[f"p_{h}"].groupby(level=0).rank(pct=True, method="first").to_numpy() > 1 - net for h in hs], axis=0)
    touch = (oos["touch"] == 1).to_numpy()
    groups = {"unexplained": touch & ~caught, "anticipated": touch & caught, "quiet": ~touch}
    cols = [c for c in ("vol20", "atr", "range20", "gap", "vol_surge", "log_dv", "logp", "r5", "r20", "dist_hi", "dist_lo") if c in F.columns]
    G = F.reindex(oos.index)[cols]
    rows = []
    for c in cols:
        med = {g: float(G.loc[m, c].median()) if m.any() else np.nan for g, m in groups.items()}
        sd = float(G[c].std())
        rows.append({"column": c, **med, "std_diff": (med["unexplained"] - med["anticipated"]) / sd if sd > 0 else np.nan,
                     "n_unexplained": int(groups["unexplained"].sum())})
    return pd.DataFrame(rows).sort_values("std_diff", key=lambda s: -s.abs()).reset_index(drop=True)


def seed_stability(F: pd.DataFrame, hyps: Sequence[VH.Hypothesis], now, cfg: LabConfig = LabConfig(), fit_cfg: VH.FitConfig = VH.FitConfig(),
                   seeds: Sequence[int] = (0, 1, 2)) -> dict:
    """Is the champion an accident of the fitting seed (row subsampling, boosting)? Re-run the walk-forward under several fit seeds and report
    the champion in each and the spread of each hypothesis's increment. A champion that changes with the seed is not a champion."""
    champs: list[str] = []
    incs: dict[str, list[float]] = {}
    for s in seeds:
        fc = dc.replace(fit_cfg, seed=s)
        wf = walk_forward(F, hyps, now, dc.replace(cfg, seed=cfg.seed), fc)
        if len(wf.oos) == 0:
            return {"tested": False, "reason": "no out-of-sample rows"}
        t = per_date_table(wf.oos, [f"p_{h}" for h in wf.hids], cfg.top_frac)
        ii = increments_vs_baseline(t, wf.hids, cfg)
        champs.append(pick_champion(ii, wf.fit_table(), cfg).hid)
        for i in ii:
            incs.setdefault(i.a, []).append(i.diff)
    spread = {h: float(np.ptp(v)) for h, v in incs.items()}
    return {"tested": True, "champions": champs, "stable": len(set(champs)) == 1, "increment_spread": spread,
            "max_spread": float(max(spread.values())) if spread else float("nan")}


def candle_inputs(bars: dict, index: pd.MultiIndex) -> pd.DataFrame:
    """engine.candles.build features for each (decision date, ticker) row. Candle features at row t use bars up to and including t's close
    (that module's own contract), so they are legal at a close-of-day decision. Tickers or dates absent from the bars give NaN."""
    from engine import candles
    feats = candles.build(bars)
    keep = [c for c in VH.CANDLE_COLUMNS if c in feats]
    d = pd.to_datetime(index.get_level_values(0))
    t = np.asarray(index.get_level_values(1))
    out = {}
    for c in keep:
        W = feats[c]
        ri = W.index.get_indexer(d)
        cj = W.columns.get_indexer(t)
        v = np.full(len(index), np.nan)
        ok = (ri >= 0) & (cj >= 0)
        v[ok] = W.to_numpy()[ri[ok], cj[ok]]
        out[c] = v
    return pd.DataFrame(out, index=index).astype("float32")


def with_candle_inputs(F: pd.DataFrame, bars: dict) -> pd.DataFrame:
    """F plus candle columns (replacing any of the same name). attrs are preserved."""
    C = candle_inputs(bars, F.index)
    G = F.drop(columns=[c for c in C.columns if c in F.columns]).join(C)
    G.attrs.update(F.attrs)
    return G


def candle_study(F: pd.DataFrame, now, cfg: LabConfig = LabConfig(), fit_cfg: VH.FitConfig = VH.FitConfig()) -> StudyResult:
    """Q20: does candle structure (wicks, close location, inside/outside days, streaks, gap fills) add to own volatility? Same incremental
    design and shuffled-input null control as the other group studies; UNKNOWN if the frame carries no candle columns."""
    spec = StudySpec("Q20", "Does candle structure predict extreme movement beyond own volatility?", "group", "candles", "H6")
    feats = tuple(f for f in VH.EXTRA_DERIVED if not VH.missing_columns((f,), F.columns))
    if not feats:
        return _unknown(spec, "no candle columns in the frame (attach them with with_candle_inputs)")
    src = tuple(dict.fromkeys(c for f in feats for c in VH.DERIVED[f][0]))
    r = feature_increment(F, feats, now, cfg, fit_cfg, null_cols=src, tag="CAN")
    v, cav = _verdict_from_increment(r["inc"], r["null"], cfg, cfg.min_dates_auc * 2)
    return StudyResult(spec.qid, spec.section, v, r["inc"].diff, r["inc"].lo, r["inc"].hi, r["inc"].p, float("nan"), r["inc"].n_dates,
                       r["null"].diff if r["null"] else float("nan"), {"features": list(feats), "hypothesis": "H6"}, tuple(cav))


def hypothesis_table(rep: LabReport) -> pd.DataFrame:
    """One row per hypothesis joining score, increment over B0, decay, regime dependence, direction flag and epistemic status."""
    by = {i.a: i for i in rep.increments}
    rows = []
    for h in rep.hids:
        c, i, a = rep.scorecards.get(h), by.get(h), rep.assessments.get(h)
        rows.append({"hid": h, "auc": c.auc if c else np.nan, "inc": i.diff if i else np.nan, "inc_lo": i.lo if i else np.nan, "q": i.q if i else np.nan,
                     "decay": rep.decay[h].status if h in rep.decay else None, "regime_dependent": rep.regimes[h].regime_dependent if h in rep.regimes else None,
                     "direction": str(rep.directions[h].flag) if h in rep.directions else None, "status": str(a.status) if a else None,
                     "untested": ",".join(a.untested) if a else ""})
    return pd.DataFrame(rows)
