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
from typing import Any, Callable, Iterable, Mapping, Sequence

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
    of that hypothesis drives the mover logit), 'hidden' (a conjunction of xs_atr_rank and near_lo that no seeded hypothesis names).
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
    D = VH.derive(F, list(VH.DERIVED) if with_events else [f for f in VH.DERIVED if not VH.missing_columns((f,), F.columns)])
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
        hid = (D["xs_atr_rank"].to_numpy(float) > 0.75) & (D["near_lo"].to_numpy(float) < -0.02) & (D["near_lo"].to_numpy(float) > -0.12)
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
    s, y = oos[pc].to_numpy(float), oos["touch"].to_numpy(float)
    ok = np.isfinite(s) & np.isfinite(y)
    brier = float(np.mean((s[ok] - y[ok]) ** 2)) if ok.any() else nan
    ref = float(np.mean((y[ok].mean() - y[ok]) ** 2)) if ok.any() else nan
    mc, lm = oos[f"m_{hid}"].to_numpy(float), np.log(oos["absmove"].to_numpy(float) + 1e-4)
    okm = np.isfinite(mc) & np.isfinite(lm)
    rho = float(pd.Series(mc[okm]).corr(pd.Series(lm[okm]), method="spearman")) if okm.sum() > 30 else nan
    return ScoreCard(hid, int(len(a)), bm.mean + 0.5, bm.lo + 0.5, bm.hi + 0.5, p, float(lifts.mean()) if len(lifts) else nan,
                     lb.lo if lb else nan, brier, (1 - brier / ref) if ref and ref > 0 else nan, expected_calibration_error(s, y), rho)


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
        per = pd.Series(np.where(y, 1.0, 0.0)).groupby(cl)
        boots = _bootstrap_auc(s, y, cl, cfg.n_boot, cfg.seed)
        dlo, dhi = float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))
        del per
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
    diff = TS.cluster_bootstrap_diff(la.to_numpy(), ea.to_numpy(), month_cluster(la.index), month_cluster(ea.index), n_boot=cfg.n_boot, seed=cfg.seed)
    lb = TS.cluster_bootstrap_mean(la.to_numpy() - 0.5, month_cluster(la.index), n_boot=cfg.n_boot, seed=cfg.seed)
    early, late = float(ea.mean()), float(la.mean())
    if early > 0.53 and lb.lo <= 0.0 and late < 0.52:
        st, why = "STOPPED", "earlier folds were above chance; the latest folds are not distinguishable from chance"
    elif diff.hi < 0 and slope < 0:
        st, why = "DECAYING", "latest folds significantly weaker than early folds, trend downwards"
    elif diff.lo > 0 and slope > 0:
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
    diff = TS.cluster_bootstrap_diff(hi_s.to_numpy(), lo_s.to_numpy(), month_cluster(hi_s.index), month_cluster(lo_s.index), n_boot=cfg.n_boot, seed=cfg.seed)
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
def run_competition(oos: pd.DataFrame, hids: Sequence[str], now, *, rows_per_date: int = 30, seed: int = 0) -> dict:
    """Let the hypotheses compete as explanations of the realised size of moves. Each hypothesis contributes its logit-probability as the
    only feature of a CAUSAL spec in an Arena beside a NULL spec; batches are decision dates in time order, a seeded sample per date.
    The leader, weights, statuses, and the Arena's own account of why it may be undecided are returned; an Arena that cannot separate
    the hypotheses says so - this function never declares a winner the evidence does not support."""
    from engine.learning import competition as CP
    hs = [h for h in hids if f"p_{h}" in oos and oos[f"p_{h}"].notna().all()]
    if len(hs) < 1 or len(oos) == 0:
        return {"tested": False, "reason": "no complete hypothesis columns"}
    d = oos.copy()
    y = np.log(d["absmove"].to_numpy(float) + 1e-3)
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
    cav = []
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
    signs = {f: [] for f in feats}
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
    edge = {f: abs((_auc(D[f].to_numpy(float)[np.isfinite(D[f].to_numpy(float))], y[np.isfinite(D[f].to_numpy(float))].astype(bool)) or 0.5) - 0.5) for f in feats}
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
    registered, promoted, rejected = [], [], []
    for cand, ev in cands[:max_new + 3]:
        if len(registered) >= max_new:
            break
        rr = VH.replicate_rule(cand, rep, now, seed=cfg.seed)
        if not rr.replicated:
            rejected.append((ev.rule_id, rr.reason))
            continue
        try:
            h = registry.register_discovered(cand, ev.rule_id)
        except ValueError as e:
            rejected.append((ev.rule_id, str(e)))
            continue
        registered.append(h.hid)
        registry.promote(h.hid, f"replicated: lift {rr.lift:.2f} (lo {rr.lift_lo:.2f}) on {rr.n_in} later rows")
        promoted.append(h.hid)
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
    return LabReport(str(as_date(now)), cfg.fingerprint(), current_code_hash(), bool(F.attrs.get("survivor_free", False)), int(len(F)),
                     int(F.index.get_level_values(0).nunique()) if len(F) else 0, tuple(hids), wf.unavailable, cards, incs, champ, directions, decay,
                     regimes, years, unexpl, comp, calib, disc, studies, registry.fingerprint(), caveats)


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
