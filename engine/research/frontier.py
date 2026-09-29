"""Conditional accuracy frontier and the 80% question (RESEARCH_BRAIN_CONTRACT C66 section 11; also serves sections 43, 44, 48).

STATUS: IMPLEMENTED - NOT VALIDATED (C63: code and unit tests only; no real-data run has been made).

The system must never report "direction accuracy = 53%". It reports coverage -> accuracy: how accurate the direction call is on the
100 / 50 / 25 / 10 / 5 / 1% (and finer) most confident calls, and for every coverage cell it carries the sample size, an honest
interval, calibration, the base rate, weeks, stocks, independent observations, and era / sector / regime stability. It then asks
whether a high accuracy at small coverage is real or only the winner's curse of picking a tiny sample, and answers the 80% question
as   accuracy + coverage + calibration + out-of-sample validity + transfer + risk   - never accuracy alone (section 11, 43, 48).
When the evidence does not clear every one of those, the answer is exactly:  No reliable 80% directional region has been found.

What it builds on (imported, not copied):
  engine.learning.calibration   wilson, reliability_diagram, ece, brier, platt_slope, ece_null_pvalue, IsotonicCalibrator, sharpness
  engine.direction_calib        venn_abers (a second, distribution-free calibrated lower bound on P(correct))
  engine.direction_features     COVERAGES, CONTROLS naming convention, and pred-frame layout (adapter `from_direction_features`)
  engine.learning.scorecard     Measured (every reported number that has an interval is a Measured, UNTESTED is not zero)
  engine.research.core          MaturedRecord / Namespace / Knowability vocabulary: a report reaches a decision only through gate(now)

Time discipline (C56, rule 3): every function that reads a set of predictions takes `now` (or is handed a set already filtered by
`Predictions.matured_before(now)`); a row whose horizon has not ended strictly before `now` is not visible. Thresholds that define
a coverage region are fitted on EARLIER blocks only and applied to LATER ones, with a purge of rows whose outcome would have been
unknown at the block boundary. Nothing here reads a ticker or a real date into anything handed to the trader: the public summary
holds numbers only (engine.learning.trader_view.assert_trader_safe is applied to it)."""
from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine import direction_calib, pattern_stats
from engine.learning import calibration as CAL
from engine.learning.core import FirewallBreach, Provenance, as_date, canonical_json, current_code_hash, require_past, stable_hash
from engine.learning.scorecard import Measured, MStatus
from engine.learning.trader_view import assert_trader_safe
from engine.research.core import MaturedRecord, Namespace

NOT_FOUND_SENTENCE = "No reliable 80% directional region has been found."
DEFAULT_COVERAGES = (1.0, 0.75, 0.50, 0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001)
SEGMENT_DIMENSIONS = ("era", "sector", "regime")
REQUIRED_COLUMNS = ("date", "ticker", "p", "up", "matured_at")
NL = chr(10)


# ==================================================================================================================
# configuration
# ==================================================================================================================
@dataclass(frozen=True)
class FrontierConfig:
    """Every threshold the frontier uses, in one validated place (so a run can be hashed and compared)."""
    coverages: tuple[float, ...] = DEFAULT_COVERAGES
    gate: float = 0.80                       # the accuracy target of section 11
    level: float = 0.95                      # interval level
    n_boot: int = 400                        # week-cluster bootstrap resamples
    n_sim: int = 300                         # selection-null permutations
    min_n: int = 30                          # below this a cell carries no verdict
    min_weeks: int = 8                       # distinct decision weeks needed in a cell
    min_stocks: int = 10
    min_indep: float = 30.0                  # independent observations after the design effect
    min_coverage: float = 0.02               # an "80% region" narrower than this is a curiosity, not a system
    ece_tol: float = 0.06                    # tolerated calibration error inside a cell
    gap_tol: float = 0.05                    # tolerated |mean stated confidence - accuracy|
    horizon_days: float = 7.0                # outcome horizon; overlapping same-ticker rows are not independent
    ece_bins: int = 5
    concentration_max: float = 0.35          # one week may not supply more than this share of a cell's hits
    stock_concentration_max: float = 0.25
    shrink_strength: float = 20.0            # pseudo-observations of the full-coverage accuracy in the shrunk estimate
    seg_min_n: int = 20                      # a segment smaller than this is reported but not judged
    seg_tol: float = 0.05                    # a segment "holds" the gate if its accuracy >= gate - seg_tol
    transfer_min_share: float = 0.75         # share of held-out segments that must hold the gate
    transfer_min_segments: int = 3
    n_blocks: int = 4                        # walk-forward blocks
    big_loss: float = 0.20                   # directional return below -big_loss is a large loss
    fwer: float = 0.05
    seed: int = 0

    def validate(self) -> list[str]:
        errs = []
        if not self.coverages or any(not (0 < c <= 1) for c in self.coverages):
            errs.append("coverages must be non-empty and inside (0, 1]")
        if len(set(self.coverages)) != len(self.coverages):
            errs.append("duplicate coverage")
        if not (0.5 < self.gate < 1):
            errs.append("gate must lie in (0.5, 1)")
        if not (0.5 < self.level < 1):
            errs.append("level must lie in (0.5, 1)")
        if self.n_boot < 50 or self.n_sim < 50:
            errs.append("n_boot and n_sim must be >= 50")
        if self.min_n < 5 or self.min_weeks < 2 or self.min_stocks < 1:
            errs.append("min_n >= 5, min_weeks >= 2, min_stocks >= 1")
        if not (0 < self.min_coverage <= 1):
            errs.append("min_coverage must lie in (0, 1]")
        if self.n_blocks < 2:
            errs.append("n_blocks must be >= 2")
        if self.horizon_days <= 0:
            errs.append("horizon_days must be positive")
        if not (0 < self.concentration_max <= 1 and 0 < self.stock_concentration_max <= 1):
            errs.append("concentration limits must lie in (0, 1]")
        if not (0 < self.fwer < 1):
            errs.append("fwer must lie in (0, 1)")
        return errs

    def check(self) -> "FrontierConfig":
        errs = self.validate()
        if errs:
            raise ValueError("invalid FrontierConfig: " + "; ".join(errs))
        return self

    @property
    def z(self) -> float:
        return float(sps.norm.ppf(1 - (1 - self.level) / 2))

    def digest(self) -> str:
        return stable_hash(dataclasses.asdict(self), 12)


# ==================================================================================================================
# the prediction set
# ==================================================================================================================
class Predictions:
    """A validated table of direction calls and their resolved outcomes.

    Columns: date (decision day), ticker, p (P(up) in [0, 1]), up (1 = the stock finished up, 0 = down), matured_at (the day the
    outcome became known, strictly after `date`). Optional: sector, regime, era (else the calendar year), fwd (signed forward return).
    Derived: conf = max(p, 1 - p), side = +1 / -1, correct."""

    def __init__(self, frame: pd.DataFrame, *, validate: bool = True):
        f = frame.copy()
        for c in ("date", "matured_at"):
            if c in f:
                f[c] = pd.to_datetime(f[c])
        if validate:
            errs = self._problems(f)
            if errs:
                raise ValueError("invalid Predictions: " + "; ".join(errs))
        f = f.reset_index(drop=True)
        if len(f):
            f["conf"] = np.maximum(f["p"], 1.0 - f["p"])
            f["side"] = np.where(f["p"] >= 0.5, 1, -1)
            f["correct"] = ((f["p"] >= 0.5) == (f["up"] > 0.5)).astype(float)
            if "era" not in f:
                f["era"] = f["date"].dt.year.astype(str)
            f["wk"] = week_codes(f["date"])
        else:
            for c in ("conf", "side", "correct", "era", "wk"):
                f[c] = pd.Series(dtype=float)
        self.frame = f

    @staticmethod
    def _problems(f: pd.DataFrame) -> list[str]:
        errs = [f"missing column {c}" for c in REQUIRED_COLUMNS if c not in f]
        if errs:
            return errs
        if len(f) == 0:
            return []
        if f[["date", "ticker", "p", "up", "matured_at"]].isna().any().any():
            errs.append("NaN in a required column")
            return errs
        if (f["p"] < 0).any() or (f["p"] > 1).any():
            errs.append("p outside [0, 1]")
        if not f["up"].isin((0, 1, 0.0, 1.0)).all():
            errs.append("up must be 0 or 1")
        if (f["matured_at"] <= f["date"]).any():
            errs.append("matured_at must be strictly after date (an outcome cannot mature on the day it is decided)")
        if f.duplicated(["date", "ticker"]).any():
            errs.append("duplicate (date, ticker) rows")
        if "fwd" in f and not np.isfinite(f["fwd"].to_numpy(float)).all():
            errs.append("fwd holds a non-finite value")
        return errs

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def empty(self) -> bool:
        return len(self.frame) == 0

    def has(self, col: str) -> bool:
        return col in self.frame and self.frame[col].notna().any()

    def matured_before(self, now) -> "Predictions":
        """The rows whose outcome was known strictly before `now`; nothing else is visible."""
        cut = pd.Timestamp(as_date(now))
        return Predictions(self.frame[self.frame["matured_at"] < cut].drop(columns=["conf", "side", "correct", "wk"], errors="ignore"),
                           validate=False)

    def subset(self, mask) -> "Predictions":
        keep = self.frame[np.asarray(mask, bool)]
        return Predictions(keep.drop(columns=["conf", "side", "correct", "wk"], errors="ignore"), validate=False)

    def week_codes(self) -> np.ndarray:
        return self.frame["wk"].to_numpy()

    def ticker_codes(self) -> np.ndarray:
        return pd.factorize(self.frame["ticker"])[0]

    def digest(self) -> str:
        f = self.frame
        if f.empty:
            return "empty"
        h = pd.util.hash_pandas_object(f[["date", "ticker", "p", "up"]], index=False).to_numpy()
        return hashlib.sha256(h.tobytes()).hexdigest()[:16]

    @classmethod
    def concat(cls, parts: Sequence["Predictions"]) -> "Predictions":
        frames = [p.frame.drop(columns=["conf", "side", "correct", "wk"], errors="ignore") for p in parts if not p.empty]
        if not frames:
            return cls(pd.DataFrame({c: pd.Series(dtype=float) for c in REQUIRED_COLUMNS}))
        return cls(pd.concat(frames, ignore_index=True))


def from_direction_features(pred: pd.DataFrame, model: str, horizon_days: int = 7) -> Predictions:
    """Adapter for engine.direction_features.walk_forward output (columns row, date, year, model, p, up, q, ..., seg, reg).
    That frame carries no ticker and no maturity date, so a synthetic row-keyed ticker and date + horizon are supplied; segment
    columns 'seg' -> sector and 'reg' -> regime are mapped when present."""
    d = pred[pred["model"] == model]
    out = pd.DataFrame({"date": pd.to_datetime(d["date"]), "ticker": ["r" + str(r) for r in d["row"]], "p": d["p"].to_numpy(float),
                        "up": (d["up"].to_numpy(float) > 0.5).astype(float)})
    out["matured_at"] = out["date"] + pd.Timedelta(days=horizon_days)
    if "seg" in d:
        out["sector"] = d["seg"].astype(str).to_numpy()
    if "reg" in d:
        out["regime"] = d["reg"].astype(str).to_numpy()
    return Predictions(out)


def week_codes(dates) -> np.ndarray:
    """Integer id of the calendar week of each date (the cluster unit: all calls of one week share one market)."""
    d = pd.to_datetime(pd.Series(dates))
    return pd.factorize(d.dt.to_period("W").astype(str))[0]


# ==================================================================================================================
# confidence ranking and coverage thresholds
# ==================================================================================================================
def survival(conf: np.ndarray) -> np.ndarray:
    """q_i = share of rows at least as confident as row i (ties count for the row, so coverage is never understated)."""
    conf = np.asarray(conf, float)
    n = len(conf)
    if n == 0:
        return conf
    s = np.sort(conf)
    return 1.0 - np.searchsorted(s, conf, side="left") / n


def fit_thresholds(conf_cal: np.ndarray, coverages: Sequence[float]) -> dict[float, float]:
    """Confidence cut-off for each coverage, fitted on CALIBRATION rows only: the k-th most confident value, k = ceil(c * n).
    Coverage 1.0 is -inf (take everything). An empty calibration set yields +inf (take nothing): fail closed."""
    conf_cal = np.sort(np.asarray(conf_cal, float))[::-1]
    n = len(conf_cal)
    out = {}
    for c in coverages:
        if n == 0:
            out[float(c)] = math.inf
        elif c >= 1.0:
            out[float(c)] = -math.inf
        else:
            out[float(c)] = float(conf_cal[max(int(math.ceil(c * n)), 1) - 1])
    return out


def select_mask(conf: np.ndarray, coverage: float, thresholds: Mapping[float, float] | None = None) -> np.ndarray:
    """Rows inside the coverage region: by rank in the set itself, or by a threshold fitted elsewhere (out-of-sample)."""
    conf = np.asarray(conf, float)
    if thresholds is not None:
        return conf >= thresholds[float(coverage)]
    return survival(conf) <= coverage + 1e-12


# ==================================================================================================================
# uncertainty: clustering, design effect, bootstrap, jackknife, concentration
# ==================================================================================================================
def group_sums(codes: np.ndarray, values: np.ndarray, mask: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """(count, sum of values) per cluster over the masked rows."""
    cnt = np.bincount(codes[mask], minlength=n_groups).astype(float)
    tot = np.bincount(codes[mask], weights=values[mask], minlength=n_groups)
    return cnt, tot


def intraclass(values: np.ndarray, codes: np.ndarray) -> float:
    """One-way ANOVA estimate of the intraclass correlation of a 0/1 outcome inside clusters, clipped to [0, 1]. High values mean
    the rows of a cluster (a week, a stock) move together, so n rows carry far fewer than n independent observations."""
    values = np.asarray(values, float)
    if len(values) < 4:
        return 0.0
    uniq, inv = np.unique(codes, return_inverse=True)
    k, n = len(uniq), len(values)
    if k < 2 or k >= n:
        return 0.0
    sizes = np.bincount(inv).astype(float)
    means = np.bincount(inv, weights=values) / sizes
    grand = values.mean()
    msb = float((sizes * (means - grand) ** 2).sum() / (k - 1))
    msw = float(((values - means[inv]) ** 2).sum() / (n - k))
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    if denom <= 1e-15 or n0 <= 1:
        return 0.0
    return float(min(max((msb - msw) / denom, 0.0), 1.0))


def design_effect(values: np.ndarray, codes: np.ndarray) -> float:
    """1 + (mean cluster size - 1) * ICC, never below 1."""
    if len(values) == 0:
        return 1.0
    sizes = np.bincount(pd.factorize(codes)[0])
    return float(max(1.0, 1.0 + (sizes.mean() - 1.0) * intraclass(values, codes)))


def overlap_factor(dates: pd.Series, tickers: pd.Series, horizon_days: float) -> float:
    """Same-ticker rows closer together than the outcome horizon share return path. Factor = horizon / median spacing, >= 1."""
    if len(dates) < 3:
        return 1.0
    df = pd.DataFrame({"d": pd.to_datetime(dates).to_numpy(), "t": tickers.to_numpy()}).sort_values(["t", "d"])
    gaps = df.groupby("t")["d"].diff().dt.days.dropna()
    gaps = gaps[gaps > 0]
    if gaps.empty:
        return 1.0
    return float(max(1.0, horizon_days / float(gaps.median())))


def independent_observations(P: Predictions, mask: np.ndarray, cfg: FrontierConfig) -> tuple[float, float]:
    """(n_eff, design effect) of a cell: the larger of the week-cluster and the stock-cluster design effects, times the
    horizon-overlap factor. This is the sample size the interval must be judged on, not the row count."""
    n = int(mask.sum())
    if n == 0:
        return 0.0, 1.0
    f = P.frame[mask]
    corr = f["correct"].to_numpy()
    deff = max(design_effect(corr, f["wk"].to_numpy()), design_effect(corr, pd.factorize(f["ticker"])[0]))
    deff *= overlap_factor(f["date"], f["ticker"], cfg.horizon_days)
    return float(n / deff), float(deff)


def cluster_bootstrap(correct: np.ndarray, codes: np.ndarray, mask: np.ndarray, rng: np.random.Generator, n_boot: int,
                      level: float) -> tuple[float, float, float]:
    """(lo, hi, se) of accuracy by resampling whole clusters (weeks) with replacement. NaN when fewer than 5 clusters or 20 draws."""
    n_groups = int(codes.max()) + 1 if len(codes) else 0
    cnt, hit = group_sums(codes, correct, mask, n_groups)
    live = np.flatnonzero(cnt > 0)
    if len(live) < 5:
        return math.nan, math.nan, math.nan
    cnt, hit = cnt[live], hit[live]
    idx = rng.integers(0, len(live), size=(n_boot, len(live)))
    accs = hit[idx].sum(1) / cnt[idx].sum(1)
    a = (1 - level) / 2
    return float(np.quantile(accs, a)), float(np.quantile(accs, 1 - a)), float(accs.std(ddof=1))


def jackknife_weeks(correct: np.ndarray, codes: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Leave-one-week-out accuracy: how far the headline moves if the single luckiest week is removed."""
    n_groups = int(codes.max()) + 1 if len(codes) else 0
    cnt, hit = group_sums(codes, correct, mask, n_groups)
    live = np.flatnonzero(cnt > 0)
    if len(live) < 3:
        return {"min": math.nan, "max": math.nan, "worst_drop": math.nan, "weeks": float(len(live))}
    tot_n, tot_k = cnt[live].sum(), hit[live].sum()
    loo = (tot_k - hit[live]) / np.maximum(tot_n - cnt[live], 1)
    return {"min": float(loo.min()), "max": float(loo.max()), "worst_drop": float(tot_k / tot_n - loo.min()), "weeks": float(len(live))}


def concentration(correct: np.ndarray, codes: np.ndarray, mask: np.ndarray) -> float:
    """Largest single cluster's share of the cell's correct calls (1.0 = all hits come from one place)."""
    if not mask.any():
        return math.nan
    n_groups = int(codes.max()) + 1
    _, hit = group_sums(codes, correct, mask, n_groups)
    tot = hit.sum()
    return float(hit.max() / tot) if tot > 0 else 0.0


def mean_ci(wilson_k: float, wilson_n: float, z: float) -> tuple[float, float]:
    """Wilson interval on a (possibly fractional) count, used with the effective sample size."""
    if wilson_n <= 0:
        return 0.0, 1.0
    return CAL.wilson(wilson_k, wilson_n, z)


def shrunk_accuracy(k: float, n: float, prior_acc: float, strength: float) -> float:
    """Empirical-Bayes shrinkage toward the full-coverage accuracy: a tiny cell borrows strength, a large one keeps its own."""
    return float((k + strength * prior_acc) / (n + strength)) if (n + strength) > 0 else math.nan


# ==================================================================================================================
# calibration inside a cell
# ==================================================================================================================
@dataclass(frozen=True)
class CellCalibration:
    """How honest the stated confidence is inside one coverage cell. `gap` > 0 means the model claimed more than it delivered."""
    n: int
    mean_conf: float
    accuracy: float
    gap: float
    ece: float
    ece_p: float                 # P(ECE >= observed | perfectly calibrated), by seeded simulation; NaN = untestable
    slope: float                 # logistic recalibration slope of P(correct) on logit(conf); < 1 = overconfident
    slope_lo: float
    slope_hi: float
    brier: float
    brier_skill: float           # 1 - brier / brier_of_constant_accuracy; > 0 means confidence carries information
    auc: float                   # can conf rank correct above incorrect calls at all

    @property
    def measured(self) -> bool:
        return self.n > 0 and math.isfinite(self.gap)

    def acceptable(self, cfg: FrontierConfig) -> bool | None:
        """True / False, or None when the cell is too small to say (never read None as True)."""
        if not self.measured or self.n < cfg.min_n:
            return None
        significant = math.isfinite(self.ece_p) and self.ece_p < 0.05
        return bool(abs(self.gap) <= cfg.gap_tol and (self.ece <= cfg.ece_tol or not significant))


def cell_calibration(conf: np.ndarray, correct: np.ndarray, cfg: FrontierConfig, rng: np.random.Generator) -> CellCalibration:
    """Calibration of `conf` as a forecast of `correct`. conf lives in [0.5, 1]; it is a claim about P(direction right)."""
    n = len(conf)
    nan = math.nan
    if n == 0:
        return CellCalibration(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    conf = np.clip(np.asarray(conf, float), 0.0, 1.0)
    correct = np.asarray(correct, float)
    acc = float(correct.mean())
    gap = float(conf.mean() - acc)
    if n < 10 or correct.min() == correct.max():
        b = CAL.brier(conf, correct)
        return CellCalibration(n, float(conf.mean()), acc, gap, nan, nan, nan, nan, nan, b, nan, nan)
    bins = CAL.reliability_diagram(conf, correct, n_bins=cfg.ece_bins, strategy="quantile")
    e = CAL.ece(bins)
    e_p = CAL.ece_null_pvalue(conf, cfg.ece_bins, CAL.ece_equal_mass(conf, correct, cfg.ece_bins), rng, n_sim=120) if n >= 30 else nan
    pl = CAL.platt_slope(conf, correct)
    b = CAL.brier(conf, correct)
    b0 = CAL.brier(np.full(n, acc), correct)
    skill = float(1.0 - b / b0) if b0 > 1e-12 else nan
    return CellCalibration(n, float(conf.mean()), acc, gap, float(e), float(e_p), float(pl["b"]), float(pl["b_lo"]), float(pl["b_hi"]),
                           float(b), skill, float(CAL.discrimination(conf, correct)))


# ==================================================================================================================
# risk inside a cell
# ==================================================================================================================
@dataclass(frozen=True)
class CellRisk:
    """What being wrong costs. Directional return = side * forward return: positive when the call earned money."""
    measured: bool
    n: int = 0
    mean_return: float = math.nan
    avg_win: float = math.nan
    avg_loss: float = math.nan            # positive number: mean size of a losing call
    payoff_ratio: float = math.nan
    expectancy: float = math.nan          # accuracy * avg_win - (1 - accuracy) * avg_loss, per call
    cvar5: float = math.nan               # mean of the worst 5% of calls
    worst: float = math.nan
    big_loss_rate: float = math.nan       # share of calls losing more than cfg.big_loss
    negative_week_share: float = math.nan  # share of weeks whose equal-weight return is negative
    max_drawdown: float = math.nan        # of the cumulative weekly equal-weight series

    def acceptable(self, cfg: FrontierConfig) -> bool | None:
        if not self.measured or self.n < cfg.min_n:
            return None
        return bool(self.expectancy > 0 and self.big_loss_rate <= 0.02 and self.cvar5 > -2 * cfg.big_loss)


def cell_risk(f: pd.DataFrame, cfg: FrontierConfig) -> CellRisk:
    """Risk of the calls in `f` (already restricted to the cell). Unmeasured when no forward return was supplied."""
    if "fwd" not in f or f["fwd"].isna().all() or len(f) == 0:
        return CellRisk(False)
    r = (f["side"] * f["fwd"]).to_numpy(float)
    win, lose = r[r > 0], -r[r < 0]
    acc = float((r > 0).mean())
    avg_win = float(win.mean()) if len(win) else 0.0
    avg_loss = float(lose.mean()) if len(lose) else 0.0
    k = max(int(math.ceil(0.05 * len(r))), 1)
    wk = pd.Series(r).groupby(f["wk"].to_numpy()).mean()
    cum = (1.0 + wk).cumprod()
    dd = float((cum / cum.cummax() - 1.0).min()) if len(cum) else math.nan
    return CellRisk(True, len(r), float(r.mean()), avg_win, avg_loss, avg_win / avg_loss if avg_loss > 0 else math.inf,
                    acc * avg_win - (1 - acc) * avg_loss, float(np.sort(r)[:k].mean()), float(r.min()),
                    float((r < -cfg.big_loss).mean()), float((wk < 0).mean()), dd)


# ==================================================================================================================
# the selection null: how good would the best small region look if confidence carried no information?
# ==================================================================================================================
@dataclass(frozen=True)
class SelectionNull:
    """Permutation null for coverage cells. Within each week the correct/incorrect labels are shuffled across the calls, which keeps
    week structure and base accuracy but destroys any link between confidence and being right. A cell's `p_cell` is how often a
    same-sized region looks at least this good; `p_family` corrects for having looked at every coverage (max-statistic)."""
    coverages: tuple[float, ...]
    p_cell: dict[float, float]
    p_family: dict[float, float]
    null_mean: dict[float, float]
    null_q95: dict[float, float]
    n_sim: int

    def as_rows(self) -> list[dict[str, float]]:
        return [dict(coverage=c, p_cell=self.p_cell[c], p_family=self.p_family[c], null_mean=self.null_mean[c],
                     null_q95=self.null_q95[c]) for c in self.coverages]


def selection_null(P: Predictions, cfg: FrontierConfig, thresholds: Mapping[float, float] | None = None,
                   rng: np.random.Generator | None = None) -> SelectionNull:
    """Run the within-week permutation null for every coverage of the config. Deterministic given `rng` (default: cfg.seed)."""
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    covs = tuple(cfg.coverages)
    if P.empty:
        nan = {c: math.nan for c in covs}
        return SelectionNull(covs, nan, dict(nan), dict(nan), dict(nan), 0)
    order = np.argsort(week_codes(P.frame["date"]), kind="stable")
    f = P.frame.iloc[order]
    codes = f["wk"].to_numpy()
    conf, corr = f["conf"].to_numpy(), f["correct"].to_numpy()
    masks = {c: select_mask(conf, c, thresholds) for c in covs}
    ns = {c: int(m.sum()) for c, m in masks.items()}
    a0 = float(corr.mean())
    sd0 = math.sqrt(max(a0 * (1 - a0), 1e-9))

    def stat(y: np.ndarray) -> dict[float, float]:
        return {c: (float(y[m].mean()) - a0) / (sd0 / math.sqrt(ns[c])) if ns[c] > 0 else -math.inf for c, m in masks.items()}
    obs = stat(corr)
    obs_acc = {c: float(corr[m].mean()) if ns[c] else math.nan for c, m in masks.items()}
    sims_acc = {c: np.empty(cfg.n_sim) for c in covs}
    max_stat = np.empty(cfg.n_sim)
    for i in range(cfg.n_sim):
        perm = np.lexsort((rng.random(len(corr)), codes))
        y = corr[perm]
        s = stat(y)
        max_stat[i] = max(s.values())
        for c, m in masks.items():
            sims_acc[c][i] = y[m].mean() if ns[c] else math.nan
    p_cell, p_family, mean, q95 = {}, {}, {}, {}
    for c in covs:
        if ns[c] == 0:
            p_cell[c] = p_family[c] = mean[c] = q95[c] = math.nan
            continue
        p_cell[c] = float((1 + np.sum(sims_acc[c] >= obs_acc[c] - 1e-12)) / (cfg.n_sim + 1))
        p_family[c] = float((1 + np.sum(max_stat >= obs[c] - 1e-12)) / (cfg.n_sim + 1))
        mean[c], q95[c] = float(np.nanmean(sims_acc[c])), float(np.nanquantile(sims_acc[c], 0.95))
    return SelectionNull(covs, p_cell, p_family, mean, q95, cfg.n_sim)


def null_max_accuracy(n: int, base: float, rng: np.random.Generator, n_sim: int = 2000, quantile: float = 0.95) -> float:
    """The accuracy a purely random region of n calls exceeds only (1 - quantile) of the time. Small n gives a large value:
    that is the winner's curse the frontier is built to expose (n = 10 needs ~0.9 to be even suggestive)."""
    if n <= 0:
        return math.nan
    return float(np.quantile(rng.binomial(n, base, size=n_sim) / n, quantile))


# ==================================================================================================================
# one coverage cell
# ==================================================================================================================
class Flag(str, enum.Enum):
    TINY_N = "TINY_N"
    FEW_WEEKS = "FEW_WEEKS"
    FEW_STOCKS = "FEW_STOCKS"
    LOW_INDEPENDENT_OBS = "LOW_INDEPENDENT_OBS"
    WEEK_CONCENTRATED = "WEEK_CONCENTRATED"
    STOCK_CONCENTRATED = "STOCK_CONCENTRATED"
    JACKKNIFE_FRAGILE = "JACKKNIFE_FRAGILE"
    SELECTION_ARTIFACT = "SELECTION_ARTIFACT"
    SHRINKAGE_COLLAPSE = "SHRINKAGE_COLLAPSE"
    MISCALIBRATED = "MISCALIBRATED"
    OVERCONFIDENT = "OVERCONFIDENT"
    BASE_RATE_ONLY = "BASE_RATE_ONLY"
    ONE_SIDED = "ONE_SIDED"
    NO_INTERVAL = "NO_INTERVAL"


SEVERE_FLAGS = frozenset({Flag.TINY_N, Flag.FEW_WEEKS, Flag.LOW_INDEPENDENT_OBS, Flag.WEEK_CONCENTRATED, Flag.STOCK_CONCENTRATED,
                          Flag.JACKKNIFE_FRAGILE, Flag.SELECTION_ARTIFACT, Flag.SHRINKAGE_COLLAPSE, Flag.NO_INTERVAL})


@dataclass(frozen=True)
class CoverageCell:
    coverage_target: float
    coverage_real: float
    n_universe: int
    n: int
    weeks: int
    stocks: int
    n_eff: float
    deff: float
    accuracy: float
    ci_lo: float                    # week-cluster bootstrap
    ci_hi: float
    wilson_lo: float                # naive: treats every row as independent (optimistic)
    eff_lo: float                   # Wilson on the effective sample size (the number to trust)
    adj_lo: float                   # eff_lo further reduced for the number of cells examined
    base_rate: float                # share of the selected rows that finished UP
    majority_accuracy: float        # accuracy of always guessing the more common direction inside the cell
    lift: float                     # accuracy - majority_accuracy
    full_accuracy: float            # accuracy at coverage 1.0 of the same set
    shrunk: float
    up_call_accuracy: float
    up_calls: int
    down_call_accuracy: float
    down_calls: int
    calibration: CellCalibration
    risk: CellRisk
    p_cell: float
    p_family: float
    top_week_share: float
    top_stock_share: float
    jackknife_min: float
    flags: tuple[Flag, ...] = ()

    @property
    def severe(self) -> tuple[Flag, ...]:
        return tuple(f for f in self.flags if f in SEVERE_FLAGS)

    def measured(self, which: str = "accuracy", lo: str = "eff_lo") -> Measured:
        """The cell's accuracy as a scorecard Measured (interval = the effective-sample Wilson bound)."""
        if self.n == 0:
            return Measured.untested("empty cell")
        return Measured(self.accuracy, getattr(self, lo), self.ci_hi, self.n, MStatus.MEASURED, f"coverage {self.coverage_target}")

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in dataclasses.asdict(self).items() if k not in ("calibration", "risk", "flags")}
        d["flags"] = [f.value for f in self.flags]
        d.update({f"cal_{k}": v for k, v in dataclasses.asdict(self.calibration).items()})
        d.update({f"risk_{k}": v for k, v in dataclasses.asdict(self.risk).items()})
        return d


def _empty_cell(c: float, n_universe: int, full_acc: float) -> CoverageCell:
    nan = math.nan
    return CoverageCell(c, 0.0, n_universe, 0, 0, 0, 0.0, 1.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, full_acc, nan, nan, 0, nan, 0,
                        cell_calibration(np.array([]), np.array([]), FrontierConfig(), np.random.default_rng(0)), CellRisk(False),
                        nan, nan, nan, nan, nan, (Flag.TINY_N,))


def compute_cell(P: Predictions, coverage: float, cfg: FrontierConfig, rng: np.random.Generator, *,
                 thresholds: Mapping[float, float] | None = None, n_cells: int = 1, null: SelectionNull | None = None,
                 full_accuracy: float | None = None) -> CoverageCell:
    """Everything section 11 asks of one coverage cell. `thresholds` (fitted on earlier data) make it an out-of-sample cell; the
    realised coverage then differs from the target and is reported as such."""
    f = P.frame
    N = len(f)
    full = float(f["correct"].mean()) if N else math.nan
    full = full if full_accuracy is None else full_accuracy
    if N == 0:
        return _empty_cell(coverage, 0, math.nan)
    mask = select_mask(f["conf"].to_numpy(), coverage, thresholds)
    n = int(mask.sum())
    if n == 0:
        return _empty_cell(coverage, N, full)
    sel = f[mask]
    corr = sel["correct"].to_numpy()
    k = float(corr.sum())
    acc = k / n
    wcodes = f["wk"].to_numpy()
    lo, hi, se = cluster_bootstrap(f["correct"].to_numpy(), wcodes, mask, rng, cfg.n_boot, cfg.level)
    n_eff, deff = independent_observations(P, mask, cfg)
    z = cfg.z
    wl = CAL.wilson(k, n, z)[0]
    el = mean_ci(acc * n_eff, n_eff, z)[0]
    za = float(sps.norm.ppf(1 - 0.05 / max(n_cells, 1)))
    adj = float(acc - za * math.sqrt(max(acc * (1 - acc), 1e-9) / max(n_eff, 1e-9)))
    base = float(sel["up"].mean())
    maj = max(base, 1 - base)
    up_m, dn_m = (sel["side"] > 0).to_numpy(), (sel["side"] < 0).to_numpy()
    cal = cell_calibration(sel["conf"].to_numpy(), corr, cfg, rng)
    jk = jackknife_weeks(f["correct"].to_numpy(), wcodes, mask)
    cell_kw = dict(top_week=concentration(f["correct"].to_numpy(), wcodes, mask),
                   top_stock=concentration(f["correct"].to_numpy(), P.ticker_codes(), mask))
    cell = CoverageCell(
        coverage_target=float(coverage), coverage_real=n / N, n_universe=N, n=n, weeks=int(sel["date"].dt.to_period("W").nunique()),
        stocks=int(sel["ticker"].nunique()), n_eff=n_eff, deff=deff, accuracy=acc, ci_lo=lo, ci_hi=hi, wilson_lo=wl, eff_lo=el, adj_lo=adj,
        base_rate=base, majority_accuracy=maj, lift=acc - maj, full_accuracy=full,
        shrunk=shrunk_accuracy(k, n, full, cfg.shrink_strength), up_call_accuracy=float(corr[up_m].mean()) if up_m.any() else math.nan,
        up_calls=int(up_m.sum()), down_call_accuracy=float(corr[dn_m].mean()) if dn_m.any() else math.nan, down_calls=int(dn_m.sum()),
        calibration=cal, risk=cell_risk(sel, cfg), p_cell=null.p_cell.get(float(coverage), math.nan) if null else math.nan,
        p_family=null.p_family.get(float(coverage), math.nan) if null else math.nan, top_week_share=cell_kw["top_week"],
        top_stock_share=cell_kw["top_stock"], jackknife_min=jk["min"])
    return dataclasses.replace(cell, flags=tuple(cell_flags(cell, cfg)))


def cell_flags(c: CoverageCell, cfg: FrontierConfig) -> list[Flag]:
    """Why a cell's headline accuracy must not be believed at face value. Every flag is a computed fact about the cell."""
    out: list[Flag] = []
    if c.n < cfg.min_n:
        out.append(Flag.TINY_N)
    if c.weeks < cfg.min_weeks:
        out.append(Flag.FEW_WEEKS)
    if c.stocks < cfg.min_stocks:
        out.append(Flag.FEW_STOCKS)
    if c.n_eff < cfg.min_indep:
        out.append(Flag.LOW_INDEPENDENT_OBS)
    if not (math.isfinite(c.ci_lo) and math.isfinite(c.ci_hi)):
        out.append(Flag.NO_INTERVAL)
    if math.isfinite(c.top_week_share) and c.weeks >= 3 and c.top_week_share > cfg.concentration_max:
        out.append(Flag.WEEK_CONCENTRATED)
    if math.isfinite(c.top_stock_share) and c.stocks >= 3 and c.top_stock_share > cfg.stock_concentration_max:
        out.append(Flag.STOCK_CONCENTRATED)
    if math.isfinite(c.jackknife_min) and c.accuracy >= cfg.gate > c.jackknife_min:
        out.append(Flag.JACKKNIFE_FRAGILE)
    if math.isfinite(c.p_family) and c.p_family > cfg.fwer and c.accuracy > c.full_accuracy + 0.02:
        out.append(Flag.SELECTION_ARTIFACT)
    if math.isfinite(c.shrunk) and c.accuracy >= cfg.gate > c.shrunk and c.n < 4 * cfg.shrink_strength:
        out.append(Flag.SHRINKAGE_COLLAPSE)
    ok = c.calibration.acceptable(cfg)
    if ok is False:
        out.append(Flag.MISCALIBRATED)
    if math.isfinite(c.calibration.slope_hi) and c.calibration.slope_hi < 0.9 and c.calibration.gap > 0:
        out.append(Flag.OVERCONFIDENT)
    if c.n >= cfg.min_n and c.lift <= 0.0:
        out.append(Flag.BASE_RATE_ONLY)
    small = min(c.up_calls, c.down_calls)
    if c.n >= cfg.min_n and small < max(3, 0.05 * c.n):
        out.append(Flag.ONE_SIDED)
    return out


# ==================================================================================================================
# the frontier
# ==================================================================================================================
@dataclass(frozen=True)
class Frontier:
    """Coverage -> accuracy for one prediction set, plus how the curve behaves."""
    cells: tuple[CoverageCell, ...]
    config_digest: str
    n_rows: int
    n_weeks: int
    n_stocks: int
    full_accuracy: float
    n_cells_examined: int
    null: SelectionNull | None = None

    def cell(self, coverage: float) -> CoverageCell:
        for c in self.cells:
            if abs(c.coverage_target - coverage) < 1e-12:
                return c
        raise KeyError(f"no cell at coverage {coverage}")

    def table(self) -> pd.DataFrame:
        if not self.cells:
            return pd.DataFrame()
        return pd.DataFrame([c.as_dict() for c in self.cells])

    def rising_toward_small_coverage(self) -> float:
        """Spearman correlation between log coverage and accuracy over cells with data. Strongly negative = accuracy climbs as the
        sample shrinks, the signature of either real confidence ordering or of a winner's curse; the selection null decides which."""
        pts = [(math.log(c.coverage_real), c.accuracy) for c in self.cells if c.n >= 5 and c.coverage_real > 0]
        if len(pts) < 4:
            return math.nan
        r = sps.spearmanr([p[0] for p in pts], [p[1] for p in pts]).statistic
        return float(r) if math.isfinite(r) else math.nan

    def monotone_violations(self, tol: float = 0.02) -> int:
        """Times accuracy falls by more than `tol` while moving to a MORE confident (smaller) cell: a non-monotone frontier means the
        confidence ordering is not the ordering of correctness."""
        seq = sorted((c for c in self.cells if c.n >= 10), key=lambda c: -c.coverage_target)
        return sum(1 for a, b in zip(seq[:-1], seq[1:]) if b.accuracy < a.accuracy - tol)

    def best_supported(self, cfg: FrontierConfig) -> CoverageCell | None:
        """Highest lower-bound cell that is not disqualified as tiny: the most accuracy the data can honestly support."""
        ok = [c for c in self.cells if c.n >= cfg.min_n and c.coverage_real >= cfg.min_coverage and math.isfinite(c.eff_lo)]
        return max(ok, key=lambda c: (c.eff_lo, c.coverage_real)) if ok else None

    def highest_accuracy(self) -> CoverageCell | None:
        live = [c for c in self.cells if c.n > 0]
        return max(live, key=lambda c: (c.accuracy, c.n)) if live else None


def compute_frontier(P: Predictions, cfg: FrontierConfig | None = None, *, thresholds: Mapping[float, float] | None = None,
                     extra_cells: int = 0, rng: np.random.Generator | None = None, with_null: bool = True) -> Frontier:
    """Build the whole frontier. `extra_cells` is the number of cells already examined elsewhere (other models, segments, earlier
    sweeps): it is added to this frontier's cell count so the multiple-comparison price follows the true search size."""
    cfg = (cfg or FrontierConfig()).check()
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    if P.empty:
        return Frontier(tuple(_empty_cell(c, 0, math.nan) for c in cfg.coverages), cfg.digest(), 0, 0, 0, math.nan, len(cfg.coverages) + extra_cells)
    null = selection_null(P, cfg, thresholds, rng) if with_null and len(P) >= cfg.min_n else None
    full = float(P.frame["correct"].mean())
    n_cells = len(cfg.coverages) + extra_cells
    cells = tuple(compute_cell(P, c, cfg, rng, thresholds=thresholds, n_cells=n_cells, null=null, full_accuracy=full) for c in cfg.coverages)
    return Frontier(cells, cfg.digest(), len(P), int(P.frame["date"].dt.to_period("W").nunique()), int(P.frame["ticker"].nunique()), full,
                    n_cells, null)


def finer_coverages(P: Predictions, start: float = 0.10, floor_n: int = 20, steps: int = 6) -> tuple[float, ...]:
    """Coverage grid that keeps splitting below `start` but stops where a cell would hold fewer than `floor_n` calls: 'and finer' of
    section 11 without inventing cells no sample can support."""
    if P.empty:
        return ()
    out, c = [], start
    for _ in range(steps):
        if c * len(P) < floor_n:
            break
        out.append(round(c, 6))
        c /= 2.0
    return tuple(out)


def frontier_from_scores(dates, tickers, p, up, matured_at, *, cfg: FrontierConfig | None = None, **extra) -> Frontier:
    """Convenience: raw arrays -> Predictions -> Frontier."""
    df = pd.DataFrame({"date": dates, "ticker": tickers, "p": p, "up": up, "matured_at": matured_at})
    for k, v in extra.items():
        df[k] = v
    return compute_frontier(Predictions(df), cfg)


# ==================================================================================================================
# stability across eras, sectors and regimes
# ==================================================================================================================
class Stability(str, enum.Enum):
    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"
    HETEROGENEOUS = "HETEROGENEOUS"
    UNTESTED = "UNTESTED"


@dataclass(frozen=True)
class SegmentStat:
    segment: str
    n: int
    accuracy: float
    lo: float
    hi: float
    base_rate: float
    weeks: int


@dataclass(frozen=True)
class SegmentStability:
    """Does a coverage cell's accuracy hold in every era / sector / regime, or is it one segment's story?"""
    dimension: str
    coverage: float
    segments: tuple[SegmentStat, ...]
    judged: int
    holding: int
    worst_accuracy: float
    best_accuracy: float
    q_stat: float
    q_p: float
    verdict: Stability

    @property
    def hold_share(self) -> float:
        return self.holding / self.judged if self.judged else math.nan


def segment_stability(P: Predictions, coverage: float, dimension: str, cfg: FrontierConfig,
                      thresholds: Mapping[float, float] | None = None) -> SegmentStability:
    """Per-segment accuracy of the calls inside one coverage region (region defined on the WHOLE set, so segments cannot
    each pick their own favourable slice). Heterogeneity is Cochran's Q on the segment accuracies."""
    if P.empty or dimension not in P.frame or not P.has(dimension):
        return SegmentStability(dimension, coverage, (), 0, 0, math.nan, math.nan, math.nan, math.nan, Stability.UNTESTED)
    f = P.frame
    mask = select_mask(f["conf"].to_numpy(), coverage, thresholds)
    stats: list[SegmentStat] = []
    for seg, g in f[mask].groupby(f.loc[mask, dimension].astype(str)):
        n = len(g)
        k = float(g["correct"].sum())
        lo, hi = CAL.wilson(k, n, cfg.z)
        stats.append(SegmentStat(str(seg), n, k / n, lo, hi, float(g["up"].mean()), int(g["date"].dt.to_period("W").nunique())))
    judged = [s for s in stats if s.n >= cfg.seg_min_n]
    if len(judged) < 2:
        return SegmentStability(dimension, coverage, tuple(stats), len(judged), 0, math.nan, math.nan, math.nan, math.nan, Stability.UNTESTED)
    holding = sum(1 for s in judged if s.accuracy >= cfg.gate - cfg.seg_tol)
    accs = np.array([s.accuracy for s in judged])
    ns = np.array([s.n for s in judged], float)
    pbar = float((accs * ns).sum() / ns.sum())
    if 0 < pbar < 1:
        q = float((ns * (accs - pbar) ** 2).sum() / (pbar * (1 - pbar)))
        qp = float(sps.chi2.sf(q, len(judged) - 1))
    else:
        q, qp = 0.0, 1.0
    if holding == len(judged):
        v = Stability.STABLE if qp >= 0.05 else Stability.HETEROGENEOUS
    else:
        v = Stability.UNSTABLE
    return SegmentStability(dimension, coverage, tuple(stats), len(judged), holding, float(accs.min()), float(accs.max()), q, qp, v)


def stability_all(P: Predictions, coverage: float, cfg: FrontierConfig, dims: Sequence[str] = SEGMENT_DIMENSIONS,
                  thresholds: Mapping[float, float] | None = None) -> dict[str, SegmentStability]:
    return {d: segment_stability(P, coverage, d, cfg, thresholds) for d in dims}


def stability_table(stab: Mapping[str, SegmentStability]) -> pd.DataFrame:
    rows = []
    for d, s in stab.items():
        for seg in s.segments:
            rows.append(dict(dimension=d, coverage=s.coverage, **dataclasses.asdict(seg)))
    return pd.DataFrame(rows)


# ==================================================================================================================
# out-of-sample validity: thresholds fitted on the past, applied to the future
# ==================================================================================================================
def time_blocks(dates: pd.Series, n_blocks: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Equal-week-count blocks as half-open [start, end) date ranges covering the whole span."""
    d = pd.to_datetime(dates)
    weeks = np.sort(d.dt.to_period("W").dt.start_time.unique())
    if len(weeks) < n_blocks:
        return []
    cuts = [weeks[int(round(i * len(weeks) / n_blocks))] for i in range(n_blocks)]
    ends = cuts[1:] + [d.max() + pd.Timedelta(days=1)]
    return [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in zip(cuts, ends)]


@dataclass(frozen=True)
class OOSResult:
    blocks: int
    train_rows: int
    test_rows: int
    purged_rows: int
    cells: tuple[CoverageCell, ...]              # pooled out-of-sample cells with realised coverage
    by_block: tuple[tuple[int, CoverageCell], ...]  # (block index, cell at each block) for the coverage of interest
    in_sample: tuple[CoverageCell, ...]
    decay: dict[float, float]                    # in-sample accuracy - out-of-sample accuracy, per coverage

    def cell(self, coverage: float) -> CoverageCell | None:
        for c in self.cells:
            if abs(c.coverage_target - coverage) < 1e-12:
                return c
        return None


def walk_forward_frontier(P: Predictions, cfg: FrontierConfig, focus: float | None = None, rng: np.random.Generator | None = None) -> OOSResult:
    """Expanding-window walk-forward. For test block b the confidence thresholds come from blocks < b, using ONLY rows whose
    outcome had matured before block b began (the purge); the block's own rows never influence its cut-offs. The pooled
    test rows are then scored with those fixed cut-offs, so coverage is realised, not targeted."""
    rng = rng if rng is not None else np.random.default_rng(cfg.seed + 7)
    blocks = time_blocks(P.frame["date"], cfg.n_blocks) if not P.empty else []
    if len(blocks) < 2:
        return OOSResult(len(blocks), 0, 0, 0, tuple(_empty_cell(c, 0, math.nan) for c in cfg.coverages), (), (), {})
    f = P.frame
    parts, purged, train_total = [], 0, 0
    thr_by_row = np.full((len(f), len(cfg.coverages)), np.nan)
    tested = np.zeros(len(f), bool)
    by_block: list[tuple[int, CoverageCell]] = []
    focus = focus if focus is not None else min(cfg.coverages, key=lambda c: abs(c - 0.10))
    for b in range(1, len(blocks)):
        start, end = blocks[b]
        past = (f["date"] < start)
        usable = past & (f["matured_at"] < start)
        purged += int((past & ~usable).sum())
        te = (f["date"] >= start) & (f["date"] < end)
        if usable.sum() < cfg.min_n or not te.any():
            continue
        thr = fit_thresholds(f.loc[usable, "conf"].to_numpy(), cfg.coverages)
        train_total += int(usable.sum())
        tested |= te.to_numpy()
        for j, c in enumerate(cfg.coverages):
            thr_by_row[te.to_numpy(), j] = thr[float(c)]
        Pt = P.subset(te.to_numpy())
        by_block.append((b, compute_cell(Pt, focus, cfg, rng, thresholds=thr, full_accuracy=float(Pt.frame["correct"].mean()))))
    if not tested.any():
        return OOSResult(len(blocks), 0, 0, purged, tuple(_empty_cell(c, 0, math.nan) for c in cfg.coverages), tuple(by_block), (), {})
    Pt = P.subset(tested)
    conf = Pt.frame["conf"].to_numpy()
    tmat = thr_by_row[tested]
    full = float(Pt.frame["correct"].mean())
    cells = []
    n_cells = len(cfg.coverages)
    for j, c in enumerate(cfg.coverages):
        thr_row = tmat[:, j]
        sub = Pt.subset(conf >= thr_row)
        if sub.empty:
            cells.append(_empty_cell(c, len(Pt), full))
            continue
        cell = compute_cell(sub, 1.0, cfg, rng, n_cells=n_cells, full_accuracy=full)
        cells.append(dataclasses.replace(cell, coverage_target=float(c), coverage_real=len(sub) / len(Pt), n_universe=len(Pt)))
    insample = compute_frontier(Pt, cfg, with_null=False, rng=rng)
    decay = {c.coverage_target: (i.accuracy - c.accuracy) if (c.n and i.n) else math.nan for c, i in zip(cells, insample.cells)}
    return OOSResult(len(blocks), train_total, len(Pt), purged, tuple(cells), tuple(by_block), insample.cells, decay)


def oos_table(o: OOSResult) -> pd.DataFrame:
    rows = []
    for c in o.cells:
        rows.append(dict(coverage=c.coverage_target, coverage_real=c.coverage_real, n=c.n, n_eff=c.n_eff, accuracy=c.accuracy, eff_lo=c.eff_lo,
                         in_sample_decay=o.decay.get(c.coverage_target, math.nan)))
    return pd.DataFrame(rows)


# ==================================================================================================================
# calibrated-probability regions: a second route to "80%" that tests the model's OWN claim
# ==================================================================================================================
@dataclass(frozen=True)
class CalibratedRegion:
    """Rows whose calibrated P(correct) is >= gate, where the calibration map is fitted on earlier rows. If the calibrated claim is
    honest, realised accuracy in the region is >= gate; the shortfall (`claim_gap`) is the cost of trusting the model's own words."""
    method: str
    fit_rows: int
    test_rows: int
    region_rows: int
    coverage: float
    claimed: float
    accuracy: float
    lo: float
    claim_gap: float
    holds: bool | None


def calibrated_region(cal: Predictions, test: Predictions, cfg: FrontierConfig, method: str = "isotonic") -> CalibratedRegion:
    """Fit conf -> P(correct) on `cal` (isotonic / platt / venn_abers lower bound), select test rows whose calibrated value is >= gate."""
    n_test = len(test)
    if cal.empty or test.empty or len(cal) < 2 * cfg.min_n or cal.frame["correct"].nunique() < 2:
        return CalibratedRegion(method, len(cal), n_test, 0, 0.0, math.nan, math.nan, math.nan, math.nan, None)
    cc, ck = cal.frame["conf"].to_numpy(), cal.frame["correct"].to_numpy()
    tc = test.frame["conf"].to_numpy()
    if method == "isotonic":
        pred = CAL.IsotonicCalibrator().fit(cc, ck).predict(tc)
    elif method == "platt":
        pred = CAL.PlattCalibrator().fit(cc, ck).predict(tc)
    elif method == "venn_abers":
        p0, p1, _ = direction_calib.venn_abers(cal.frame["p"].to_numpy(), cal.frame["up"].to_numpy(), test.frame["p"].to_numpy())
        up = test.frame["p"].to_numpy() >= 0.5
        pred = np.where(up, p0, 1.0 - p1)                  # pessimistic P(direction right)
    else:
        raise ValueError(f"unknown calibration method {method!r}")
    m = pred >= cfg.gate
    k = int(m.sum())
    if k == 0:
        return CalibratedRegion(method, len(cal), n_test, 0, 0.0, cfg.gate, math.nan, math.nan, math.nan, None)
    corr = test.frame["correct"].to_numpy()[m]
    acc = float(corr.mean())
    lo = CAL.wilson(float(corr.sum()), k, cfg.z)[0]
    gap = float(pred[m].mean() - acc)
    return CalibratedRegion(method, len(cal), n_test, k, k / n_test, float(pred[m].mean()), acc, lo, gap,
                            bool(lo >= cfg.gate) if k >= cfg.min_n else None)


def calibrated_regions_walk_forward(P: Predictions, cfg: FrontierConfig) -> list[CalibratedRegion]:
    """All three calibration routes, fitted on the first half of the weeks (purged by maturity) and scored on the second half."""
    if P.empty:
        return []
    blocks = time_blocks(P.frame["date"], 2)
    if len(blocks) < 2:
        return []
    start = blocks[1][0]
    f = P.frame
    cal = P.subset(((f["date"] < start) & (f["matured_at"] < start)).to_numpy())
    test = P.subset((f["date"] >= start).to_numpy())
    return [calibrated_region(cal, test, cfg, m) for m in ("isotonic", "platt", "venn_abers")]


# ==================================================================================================================
# transfer: does a region defined here hold on unseen stocks, sectors, regimes and eras?
# ==================================================================================================================
@dataclass(frozen=True)
class HeldOut:
    segment: str
    n: int
    accuracy: float
    lo: float
    threshold: float
    holds: bool | None


@dataclass(frozen=True)
class TransferResult:
    dimension: str
    coverage: float
    held: tuple[HeldOut, ...]
    judged: int
    holding: int
    pooled_accuracy: float
    pooled_lo: float
    worst: float
    verdict: Stability

    @property
    def share(self) -> float:
        return self.holding / self.judged if self.judged else math.nan


def stock_folds(tickers: pd.Series, k: int, salt: str = "fold") -> np.ndarray:
    """Deterministic ticker -> fold id in [0, k) (hash based, so the same stock is always in the same fold)."""
    out = {t: int(stable_hash({"s": salt, "t": str(t)}, 8), 16) % k for t in tickers.unique()}
    return tickers.map(out).to_numpy()


def leave_one_out_transfer(P: Predictions, coverage: float, dimension: str, cfg: FrontierConfig) -> TransferResult:
    """Leave one segment out: the confidence cut-off is fitted on all OTHER segments, then applied to the held-out one.
    `dimension` is a column ('era', 'sector', 'regime') or 'stock' (hash folds of tickers)."""
    if P.empty:
        return TransferResult(dimension, coverage, (), 0, 0, math.nan, math.nan, math.nan, Stability.UNTESTED)
    f = P.frame
    if dimension == "stock":
        seg = pd.Series(stock_folds(f["ticker"], max(cfg.transfer_min_segments, 5)), index=f.index).astype(str)
    elif dimension in f and P.has(dimension):
        seg = f[dimension].astype(str)
    else:
        return TransferResult(dimension, coverage, (), 0, 0, math.nan, math.nan, math.nan, Stability.UNTESTED)
    held: list[HeldOut] = []
    hits = trials = 0
    for s in sorted(seg.unique()):
        te = (seg == s).to_numpy()
        rest = ~te
        if rest.sum() < cfg.min_n:
            continue
        thr = fit_thresholds(f.loc[rest, "conf"].to_numpy(), [coverage])[float(coverage)]
        m = te & (f["conf"].to_numpy() >= thr)
        n = int(m.sum())
        if n == 0:
            held.append(HeldOut(s, 0, math.nan, math.nan, thr, None))
            continue
        k = float(f["correct"].to_numpy()[m].sum())
        lo = CAL.wilson(k, n, cfg.z)[0]
        holds = bool(k / n >= cfg.gate - cfg.seg_tol) if n >= cfg.seg_min_n else None
        held.append(HeldOut(s, n, k / n, lo, thr, holds))
        hits += k
        trials += n
    judged = [h for h in held if h.holds is not None]
    holding = sum(1 for h in judged if h.holds)
    pooled = hits / trials if trials else math.nan
    plo = CAL.wilson(hits, trials, cfg.z)[0] if trials else math.nan
    worst = min((h.accuracy for h in judged), default=math.nan)
    if len(judged) < cfg.transfer_min_segments:
        v = Stability.UNTESTED
    elif holding / len(judged) >= cfg.transfer_min_share:
        v = Stability.STABLE
    else:
        v = Stability.UNSTABLE
    return TransferResult(dimension, coverage, tuple(held), len(judged), holding, pooled, plo, worst, v)


def transfer_all(P: Predictions, coverage: float, cfg: FrontierConfig, dims: Sequence[str] = ("stock",) + SEGMENT_DIMENSIONS
                 ) -> dict[str, TransferResult]:
    return {d: leave_one_out_transfer(P, coverage, d, cfg) for d in dims}


def transfer_ratio(oos_acc: float, in_acc: float, base: float) -> float:
    """Share of the in-sample excess over the base rate that survives out of sample (1 = all of it, 0 or below = none)."""
    excess = in_acc - base
    if not (math.isfinite(oos_acc) and math.isfinite(in_acc)) or excess <= 1e-9:
        return math.nan
    return float((oos_acc - base) / excess)


# ==================================================================================================================
# the cumulative multiple-testing ledger (C67: with searches this large, small real effects must be separated from chance)
# ==================================================================================================================
@dataclass(frozen=True)
class TestEntry:
    __test__ = False                      # not a pytest class
    test_id: str
    family: str
    p: float
    logged_real: str
    matured_through: str
    count: int = 1                        # number of tests this entry stands for (a block of screened-out tests has p = 1)


class TestLedger:
    """Every hypothesis test ever run in a sweep, kept for good. Adjusted significance always uses the CUMULATIVE count, so the
    bar rises as the search grows and a p = 0.01 found on the 5,000th look is not celebrated as if it were the first.
    Entries are append-only; re-logging the same test_id keeps the first p (a test cannot be re-rolled until it passes)."""
    __test__ = False                      # not a pytest class

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._entries: dict[str, TestEntry] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = TestEntry(**json.loads(line))
                    self._entries.setdefault(e.test_id, e)

    def __len__(self) -> int:
        return sum(e.count for e in self._entries.values())

    def log_null(self, test_id: str, family: str, count: int, matured_through, now) -> None:
        """Register `count` tests that were examined but screened out (each counts as p = 1): the search size is part of the price."""
        require_past(matured_through, now, f"test block {test_id}")
        if count <= 0 or test_id in self._entries:
            return
        e = TestEntry(test_id, family, 1.0, str(as_date(now)), str(as_date(matured_through)), int(count))
        self._entries[test_id] = e
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dataclasses.asdict(e), sort_keys=True) + NL)

    def log(self, test_id: str, family: str, p: float, matured_through, now) -> TestEntry:
        """Record one test. `matured_through` (newest outcome date used) must be strictly before `now`, else FirewallBreach."""
        require_past(matured_through, now, f"test {test_id}")
        if not (isinstance(p, (int, float)) and 0.0 <= float(p) <= 1.0):
            raise ValueError(f"p-value {p!r} outside [0, 1]")
        if test_id in self._entries:
            return self._entries[test_id]
        e = TestEntry(test_id, family, float(p), str(as_date(now)), str(as_date(matured_through)))
        self._entries[test_id] = e
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dataclasses.asdict(e), sort_keys=True) + NL)
        return e

    def entries(self, family: str | None = None) -> list[TestEntry]:
        return [e for e in self._entries.values() if family is None or e.family == family]

    def families(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self._entries.values():
            out[e.family] = out.get(e.family, 0) + 1
        return out

    def bonferroni(self, test_id: str) -> float:
        e = self._entries[test_id]
        return float(min(1.0, e.p * len(self)))

    def bh_adjusted(self) -> dict[str, float]:
        """Benjamini-Hochberg q-values over EVERY test in the ledger, counting screened-out blocks as p = 1 tests
        (engine.pattern_stats.bh_qvalues on the real p-values padded with ones up to the cumulative count)."""
        real = [e for e in self._entries.values() if e.count == 1]
        if not real:
            return {}
        p = np.array([e.p for e in real] + [1.0] * (len(self) - len(real)))
        q = pattern_stats.bh_qvalues(p)[:len(real)]
        return {e.test_id: float(x) for e, x in zip(real, q)}

    def survivors(self, q: float = 0.05, strict: bool = True) -> list[str]:
        """Tests that survive. strict=True demands BOTH BH q-value <= q and Bonferroni <= 0.5 (a big-search guard); False is BH only."""
        adj = self.bh_adjusted()
        out = [i for i, a in adj.items() if a <= q and (not strict or self.bonferroni(i) <= 0.5)]
        return sorted(out, key=lambda i: adj[i])

    def expected_false_positives(self, alpha: float = 0.05) -> float:
        """How many entries would clear p <= alpha by luck alone, given the ledger size: the noise floor of the search."""
        return alpha * len(self)

    def excess_small_p(self, alpha: float = 0.05) -> dict[str, float]:
        """Observed count of p <= alpha versus the count luck predicts, with a one-sided binomial test: is the sweep finding MORE small p
        than chance, i.e. is any real signal present at all, even when no single test survives correction?"""
        n = len(self)
        if n == 0:
            return {"n": 0, "observed": 0, "expected": 0.0, "p": math.nan}
        obs = sum(e.count for e in self._entries.values() if e.p <= alpha and e.count == 1)
        return {"n": n, "observed": obs, "expected": alpha * n, "p": float(sps.binom.sf(obs - 1, n, alpha))}

    def digest(self) -> str:
        return stable_hash(sorted((e.test_id, round(e.p, 12)) for e in self._entries.values()), 16)


# ==================================================================================================================
# tiny-sample detection
# ==================================================================================================================
class TinyVerdict(str, enum.Enum):
    ROBUST = "ROBUST"                      # survives every check
    TINY_SAMPLE_ARTIFACT = "TINY_SAMPLE_ARTIFACT"
    INSUFFICIENT = "INSUFFICIENT"          # cannot be judged, not the same as fine
    NO_EFFECT = "NO_EFFECT"


@dataclass(frozen=True)
class TinyAssessment:
    coverage: float
    verdict: TinyVerdict
    reasons: tuple[str, ...]
    accuracy: float
    null_q95: float                        # accuracy a random region of this size beats only 5% of the time
    exceeds_null: bool | None
    shrunk: float
    n: int
    n_eff: float


def assess_tiny_sample(F: Frontier, cfg: FrontierConfig, rng: np.random.Generator | None = None) -> list[TinyAssessment]:
    """For every cell decide whether its accuracy is real or a consequence of selecting few rows. A cell is ROBUST only if it has
    no severe flag, beats the size-matched random region, keeps its accuracy after shrinkage, and its interval clears the base."""
    rng = rng if rng is not None else np.random.default_rng(cfg.seed + 11)
    out = []
    for c in F.cells:
        if c.n == 0:
            out.append(TinyAssessment(c.coverage_target, TinyVerdict.INSUFFICIENT, ("no calls in the cell",), math.nan, math.nan, None, math.nan, 0, 0.0))
            continue
        q95 = null_max_accuracy(int(max(round(c.n_eff), 1)), max(c.majority_accuracy, 0.5), rng)
        exceeds = bool(c.accuracy > q95) if math.isfinite(q95) else None
        reasons = [f.value for f in c.severe]
        if exceeds is False:
            reasons.append("accuracy within what a random region of this many independent calls reaches by luck")
        if math.isfinite(c.shrunk) and c.shrunk < c.accuracy - 0.05:
            reasons.append("shrinkage toward full-coverage accuracy removes more than 5 points")
        if c.n < cfg.min_n or c.n_eff < cfg.min_indep:
            v = TinyVerdict.INSUFFICIENT if c.accuracy < cfg.gate else TinyVerdict.TINY_SAMPLE_ARTIFACT
        elif reasons:
            v = TinyVerdict.TINY_SAMPLE_ARTIFACT
        elif not (c.eff_lo > c.majority_accuracy):
            v = TinyVerdict.NO_EFFECT
            reasons.append("interval does not clear the majority-direction base rate")
        else:
            v = TinyVerdict.ROBUST
        out.append(TinyAssessment(c.coverage_target, v, tuple(reasons), c.accuracy, q95, exceeds, c.shrunk, c.n, c.n_eff))
    return out


def high_accuracy_from_tiny_sample(F: Frontier, cfg: FrontierConfig) -> list[float]:
    """Coverages where the gate is reached on the point estimate but the cell is judged a tiny-sample artifact."""
    tiny = {t.coverage: t for t in assess_tiny_sample(F, cfg)}
    return [c.coverage_target for c in F.cells if c.n > 0 and c.accuracy >= cfg.gate and tiny[c.coverage_target].verdict == TinyVerdict.TINY_SAMPLE_ARTIFACT]


# ==================================================================================================================
# the 80% question
# ==================================================================================================================
class Eighty(str, enum.Enum):
    NOT_FOUND = "NOT_FOUND"
    CANDIDATE = "CANDIDATE"                # every criterion passed on this data; it still needs a fresh holdout (section 48)
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class Criterion:
    name: str
    passed: bool | None                   # None = untested, which blocks the answer exactly like a failure
    value: float
    threshold: float
    note: str = ""


CRITERIA = ("accuracy", "coverage", "independent_observations", "selection", "tiny_sample", "calibration", "out_of_sample", "transfer",
            "stability", "risk")


@dataclass(frozen=True)
class RegionVerdict:
    coverage: float
    criteria: tuple[Criterion, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed is True for c in self.criteria)

    @property
    def failing(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.criteria if c.passed is False)

    @property
    def untested(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.criteria if c.passed is None)


@dataclass(frozen=True)
class EightyAnswer:
    status: Eighty
    headline: str
    regions: tuple[RegionVerdict, ...]
    best_supported_coverage: float
    best_supported_accuracy: float
    best_supported_lo: float
    best_supported_failing: tuple[str, ...]
    highest_point_accuracy: float
    highest_point_coverage: float
    tiny_sample_coverages: tuple[float, ...]
    cells_examined: int
    n_rows: int

    def lines(self) -> list[str]:
        out = [self.headline]
        if self.status == Eighty.NOT_FOUND and math.isfinite(self.best_supported_accuracy):
            out.append(f"Best supported region: coverage {self.best_supported_coverage:.3f}, accuracy {self.best_supported_accuracy:.3f} "
                       f"(lower bound {self.best_supported_lo:.3f}); it fails: {', '.join(self.best_supported_failing) or 'none recorded'}.")
        if self.tiny_sample_coverages:
            out.append("High point accuracy at coverage " + ", ".join(f"{c:.3f}" for c in self.tiny_sample_coverages) +
                       " comes from a tiny sample and is not evidence.")
        return out


def _criterion(name: str, passed: bool | None, value: float, threshold: float, note: str = "") -> Criterion:
    return Criterion(name, passed, float(value) if value is not None else math.nan, float(threshold), note)


def judge_region(cell: CoverageCell, tiny: TinyAssessment, oos: OOSResult, transfers: Mapping[str, TransferResult],
                 stab: Mapping[str, SegmentStability], cfg: FrontierConfig) -> RegionVerdict:
    """All ten section-11 / 43 / 48 requirements for ONE coverage region. Anything not measured is None and blocks the answer."""
    oc = oos.cell(cell.coverage_target)
    cal_ok = cell.calibration.acceptable(cfg)
    risk_ok = cell.risk.acceptable(cfg)
    tr = [t for t in transfers.values() if t.verdict != Stability.UNTESTED]
    st = [s for s in stab.values() if s.verdict != Stability.UNTESTED]
    crit = [
        _criterion("accuracy", bool(cell.eff_lo >= cfg.gate and cell.adj_lo >= cfg.gate - 0.05) if cell.n and math.isfinite(cell.eff_lo) else None,
                   cell.eff_lo, cfg.gate, "effective-sample lower bound; multiple-comparison adjusted bound within 5 points"),
        _criterion("coverage", cell.coverage_real >= cfg.min_coverage, cell.coverage_real, cfg.min_coverage),
        _criterion("independent_observations", cell.n_eff >= cfg.min_indep, cell.n_eff, cfg.min_indep),
        _criterion("selection", ((cell.p_family <= cfg.fwer) or (cell.full_accuracy >= cfg.gate and abs(cell.accuracy - cell.full_accuracy) <= 0.02))
                   if math.isfinite(cell.p_family) else None, cell.p_family, cfg.fwer,
                   "family-wise permutation p over all coverages (moot when the un-selected set already reaches the gate)"),
        _criterion("tiny_sample", tiny.verdict == TinyVerdict.ROBUST if tiny.verdict != TinyVerdict.INSUFFICIENT else None,
                   tiny.n_eff, cfg.min_indep, "; ".join(tiny.reasons)),
        _criterion("calibration", cal_ok, cell.calibration.gap, cfg.gap_tol, "stated confidence vs realised accuracy"),
        _criterion("out_of_sample", (oc.eff_lo >= cfg.gate - cfg.seg_tol) if oc is not None and oc.n >= cfg.min_n and math.isfinite(oc.eff_lo) else None,
                   oc.eff_lo if oc is not None else math.nan, cfg.gate - cfg.seg_tol, "walk-forward with purged, earlier-only thresholds"),
        _criterion("transfer", (all(t.verdict == Stability.STABLE for t in tr)) if len(tr) >= 2 else None,
                   min((t.share for t in tr), default=math.nan), cfg.transfer_min_share, "held-out stocks / sectors / regimes / eras"),
        _criterion("stability", (all(s.verdict in (Stability.STABLE, Stability.HETEROGENEOUS) and s.hold_share >= cfg.transfer_min_share for s in st))
                   if len(st) >= 1 else None, min((s.hold_share for s in st), default=math.nan), cfg.transfer_min_share, "era / sector / regime"),
        _criterion("risk", risk_ok, cell.risk.expectancy, 0.0, "expectancy, large-loss rate and tail of the calls' directional return"),
    ]
    return RegionVerdict(cell.coverage_target, tuple(crit))


def answer_80_question(F: Frontier, tiny: Sequence[TinyAssessment], oos: OOSResult, transfers: Mapping[float, Mapping[str, TransferResult]],
                       stab: Mapping[float, Mapping[str, SegmentStability]], cfg: FrontierConfig) -> EightyAnswer:
    """Assemble the answer. CANDIDATE only if some region with coverage >= min_coverage passes every criterion; the wording never
    says validated, because a fresh holdout is still required. Otherwise the exact not-found sentence (section 11, 53)."""
    tmap = {t.coverage: t for t in tiny}
    verdicts = []
    for c in F.cells:
        if c.n == 0 or c.coverage_target not in transfers:
            continue
        verdicts.append(judge_region(c, tmap[c.coverage_target], oos, transfers[c.coverage_target], stab[c.coverage_target], cfg))
    best = F.best_supported(cfg)
    high = F.highest_accuracy()
    tiny_cov = tuple(high_accuracy_from_tiny_sample(F, cfg))
    passing = [v for v in verdicts if v.passed]
    enough = F.n_rows >= cfg.min_n and F.n_weeks >= cfg.min_weeks and any(c.n >= cfg.min_n for c in F.cells)
    if not enough:
        return EightyAnswer(Eighty.INSUFFICIENT_DATA, NOT_FOUND_SENTENCE + " The data are too small to say more than that.", tuple(verdicts),
                            math.nan, math.nan, math.nan, (), high.accuracy if high else math.nan, high.coverage_real if high else math.nan,
                            tiny_cov, F.n_cells_examined, F.n_rows)
    if passing:
        top = max(passing, key=lambda v: v.coverage)
        cell = F.cell(top.coverage)
        head = (f"A candidate 80% directional region exists at coverage {cell.coverage_real:.3f}: accuracy {cell.accuracy:.3f}, lower bound "
                f"{cell.eff_lo:.3f}, {cell.n} calls ({cell.n_eff:.0f} independent). It passed accuracy, coverage, calibration, out-of-sample, "
                f"transfer, stability and risk on THIS data only; it is not validated and needs a fresh holdout before any use.")
        return EightyAnswer(Eighty.CANDIDATE, head, tuple(verdicts), cell.coverage_real, cell.accuracy, cell.eff_lo, (),
                            high.accuracy if high else math.nan, high.coverage_real if high else math.nan, tiny_cov, F.n_cells_examined, F.n_rows)
    fail = ()
    if best is not None:
        bv = next((v for v in verdicts if abs(v.coverage - best.coverage_target) < 1e-12), None)
        fail = (bv.failing + bv.untested) if bv else ()
    return EightyAnswer(Eighty.NOT_FOUND, NOT_FOUND_SENTENCE, tuple(verdicts), best.coverage_real if best else math.nan,
                        best.accuracy if best else math.nan, best.eff_lo if best else math.nan, tuple(fail),
                        high.accuracy if high else math.nan, high.coverage_real if high else math.nan, tiny_cov, F.n_cells_examined, F.n_rows)


# ==================================================================================================================
# full assessment: the one object a caller wants
# ==================================================================================================================
@dataclass(frozen=True)
class FrontierReport:
    now: str
    config_digest: str
    data_digest: str
    frontier: Frontier
    tiny: tuple[TinyAssessment, ...]
    oos: OOSResult
    transfers: dict[float, dict[str, TransferResult]]
    stability: dict[float, dict[str, SegmentStability]]
    calibrated: tuple[CalibratedRegion, ...]
    answer: EightyAnswer
    excluded_unmatured: int
    provenance: Provenance

    def record_id(self) -> str:
        return "FR" + stable_hash({"n": self.now, "c": self.config_digest, "d": self.data_digest}, 12)

    def to_matured_record(self) -> MaturedRecord:
        """Wrap for the trusted side: it reaches a decision only through record.gate(now) once `now` is past the newest outcome."""
        return MaturedRecord(self.record_id(), self.provenance.learned_at, public_summary(self), self.provenance, Namespace.MATURED_RESEARCH)


def assess(P: Predictions, now, cfg: FrontierConfig | None = None, *, code_hash: str | None = None, extra_cells: int = 0,
           focus_coverages: Sequence[float] | None = None) -> FrontierReport:
    """Coverage -> accuracy frontier and the 80% answer for everything that had matured strictly before `now`. Rows dated at or
    after `now` are dropped and counted (`excluded_unmatured`), never used."""
    cfg = (cfg or FrontierConfig()).check()
    visible = P.matured_before(now)
    excluded = len(P) - len(visible)
    rng = np.random.default_rng(cfg.seed)
    F = compute_frontier(visible, cfg, extra_cells=extra_cells, rng=rng)
    tiny = assess_tiny_sample(F, cfg, rng)
    oos = walk_forward_frontier(visible, cfg, rng=rng)
    focus = tuple(focus_coverages) if focus_coverages is not None else tuple(c.coverage_target for c in F.cells if c.n >= cfg.min_n)
    transfers = {c: transfer_all(visible, c, cfg) for c in focus}
    stab = {c: stability_all(visible, c, cfg) for c in focus}
    ans = answer_80_question(F, tiny, oos, transfers, stab, cfg)
    newest = visible.frame["matured_at"].max() if not visible.empty else pd.Timestamp(as_date(now)) - pd.Timedelta(days=1)
    prov = Provenance(created_real=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), learned_at=str(newest.date()),
                      code_hash=code_hash or current_code_hash(), data_hash=visible.digest(), config_hash=cfg.digest(),
                      seed=cfg.seed, outcomes_seen_through=str(newest.date()))
    return FrontierReport(str(as_date(now)), cfg.digest(), visible.digest(), F, tuple(tiny), oos, transfers, stab,
                          tuple(calibrated_regions_walk_forward(visible, cfg)), ans, excluded, prov)


# ==================================================================================================================
# identity-free output and rendering
# ==================================================================================================================
def _num(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 6)


def public_summary(r: FrontierReport) -> dict[str, Any]:
    """Numbers and verdict words only: no ticker, no date, no year. Checked by trader_view.assert_trader_safe before it is returned."""
    a = r.answer
    lg = lambda x: _num(math.log10(x)) if x and x > 0 else None      # counts are sent as log10: a raw 1,990 would read as a year to the blind-view scan
    out = {
        "status": a.status.value, "headline": a.headline, "rows_log10": lg(r.frontier.n_rows), "weeks_log10": lg(r.frontier.n_weeks),
        "stocks_log10": lg(r.frontier.n_stocks),
        "full_accuracy": _num(r.frontier.full_accuracy), "cells_examined_log10": lg(a.cells_examined),
        "best_supported": {"coverage": _num(a.best_supported_coverage), "accuracy": _num(a.best_supported_accuracy),
                           "lower_bound": _num(a.best_supported_lo), "failing": list(a.best_supported_failing)},
        "tiny_sample_coverages": [_num(c) for c in a.tiny_sample_coverages],
        "cells": [{"coverage": _num(c.coverage_target), "n_log10": lg(c.n), "n_eff_log10": lg(c.n_eff), "accuracy": _num(c.accuracy), "lower_bound": _num(c.eff_lo),
                   "base_rate": _num(c.base_rate), "calibration_gap": _num(c.calibration.gap), "flags": [f.value for f in c.flags]}
                  for c in r.frontier.cells],
    }
    assert_trader_safe(out, "frontier public summary")
    return out


def render_report(r: FrontierReport) -> str:
    """Plain-text report: the frontier table, the tiny-sample verdicts, out-of-sample decay, and the 80% answer."""
    L = ["COVERAGE -> ACCURACY FRONTIER  (IMPLEMENTED - NOT VALIDATED)", f"rows {r.frontier.n_rows}, weeks {r.frontier.n_weeks}, stocks {r.frontier.n_stocks}, "
         f"full-coverage accuracy {r.frontier.full_accuracy:.3f}, cells examined {r.answer.cells_examined}, unmatured rows dropped {r.excluded_unmatured}", ""]
    L.append(f"{'cov':>7} {'n':>7} {'n_eff':>7} {'acc':>6} {'eff_lo':>7} {'base':>6} {'gap':>7} {'p_fam':>6}  verdict / flags")
    tmap = {t.coverage: t for t in r.tiny}
    for c in r.frontier.cells:
        if c.n == 0:
            L.append(f"{c.coverage_target:7.3f} {'0':>7}  (empty)")
            continue
        L.append(f"{c.coverage_target:7.3f} {c.n:7d} {c.n_eff:7.0f} {c.accuracy:6.3f} {c.eff_lo:7.3f} {c.base_rate:6.3f} {c.calibration.gap:7.3f} "
                 f"{c.p_family:6.3f}  {tmap[c.coverage_target].verdict.value} {' '.join(f.value for f in c.flags)}")
    L += ["", "OUT OF SAMPLE (thresholds fitted on earlier, matured rows only)"]
    for c in r.oos.cells:
        if c.n:
            L.append(f"  target {c.coverage_target:6.3f}  realised {c.coverage_real:6.3f}  n {c.n:6d}  acc {c.accuracy:.3f}  eff_lo {c.eff_lo:.3f}  "
                     f"decay {r.oos.decay.get(c.coverage_target, math.nan):+.3f}")
    L += [""] + r.answer.lines()
    return NL.join(L)


# ==================================================================================================================
# planted worlds: instruments must be shown to see a real region, and not to see a fake one
# ==================================================================================================================
def synthetic_predictions(seed: int = 0, n_weeks: int = 80, per_week: int = 60, kind: str = "planted", region_share: float = 0.10,
                          region_acc: float = 0.90, base_acc: float = 0.52, with_returns: bool = True, start: str = "2015-01-05") -> Predictions:
    """Deterministic labelled worlds.
    planted   a `region_share` slice of calls has high confidence and is right with probability `region_acc`, spread over every week,
              stock, sector and regime; the rest sit near 0.5 and are right with probability `base_acc`.
    null      confidence carries no information: correct with probability `base_acc` whatever conf says.
    tiny      the tempting fake: the ~12 most confident calls, all in the SAME week, are all right; nothing else is informative.
    lopsided  the planted region exists in only one sector (so stability and transfer must refuse it).
    overconf  confidence is very high but accuracy is only `base_acc`+0.06: right direction of ordering, wildly overconfident."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start, periods=n_weeks, freq="7D")
    n = n_weeks * per_week
    date = np.repeat(weeks.to_numpy(), per_week)
    tick = np.tile(np.array([f"S{i:03d}" for i in range(per_week * 3)]), n_weeks)[: n] if False else \
        np.array([f"S{int(i):03d}" for i in rng.integers(0, per_week * 3, n)])
    df = pd.DataFrame({"date": date, "ticker": tick})
    df = df.drop_duplicates(["date", "ticker"]).reset_index(drop=True)
    n = len(df)
    sector = np.array(["tech", "health", "energy", "fin"])[rng.integers(0, 4, n)]
    regime = np.where(pd.to_datetime(df["date"]).dt.year.to_numpy() % 2 == 0, "calm", "stress")
    truth_up = rng.random(n) < 0.5
    in_region = rng.random(n) < region_share
    if kind == "lopsided":
        in_region &= sector == "tech"
    if kind == "tiny":
        p_correct = np.full(n, 0.5)
        in_region = np.zeros(n, bool)
    elif kind == "null":
        p_correct = np.full(n, base_acc)
        in_region = np.zeros(n, bool)
    elif kind == "overconf":
        p_correct = np.where(in_region, base_acc + 0.06, base_acc)
    else:
        p_correct = np.where(in_region, region_acc, base_acc)
    correct = rng.random(n) < p_correct
    if kind == "tiny":
        wk0 = np.sort(pd.to_datetime(df["date"]).unique())[n_weeks // 2]
        top = np.flatnonzero(pd.to_datetime(df["date"]).to_numpy() == wk0)[:12]
        in_region[top] = True
        correct[top] = True
    predicted_up = np.where(correct, truth_up, ~truth_up)
    conf = np.where(in_region, rng.uniform(0.86, 0.94, n), rng.uniform(0.50, 0.70, n))
    if kind == "null":
        conf = rng.uniform(0.5, 0.99, n)
    p = np.where(predicted_up, conf, 1.0 - conf)
    df["p"] = p
    df["up"] = truth_up.astype(float)
    df["matured_at"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=7)
    df["sector"], df["regime"] = sector, regime
    if with_returns:
        mag = np.abs(rng.normal(0.06, 0.03, n)) + 0.01
        df["fwd"] = np.where(truth_up, mag, -mag)
    return Predictions(df)


def frontier_selfcheck(seed: int = 0, cfg: FrontierConfig | None = None, n_weeks: int = 80) -> dict[str, Any]:
    """Run the instrument on worlds whose truth is known. `ok` is True only if the planted region is found (as a CANDIDATE), and none of
    null / tiny / lopsided / overconfident is. A frontier that cannot fail this is worthless (CONTEXT rule 5)."""
    cfg = cfg or FrontierConfig(n_boot=200, n_sim=150, seed=seed)
    out: dict[str, Any] = {}
    for kind in ("planted", "null", "tiny", "lopsided", "overconf"):
        P = synthetic_predictions(seed, n_weeks=n_weeks, kind=kind)
        r = assess(P, "2030-01-01", cfg, code_hash="selfcheck")
        out[kind] = r.answer.status.value
    out["ok"] = out["planted"] == Eighty.CANDIDATE.value and all(out[k] == Eighty.NOT_FOUND.value for k in ("null", "tiny", "lopsided", "overconf"))
    return out


# ==================================================================================================================
# coverage tracking and the always-on sweep (C67)
# ==================================================================================================================
class CoverageTracker:
    """Which (source family, era, cohort) units the sweep has examined, and how much. `next_unit` always returns the LEAST covered
    candidate, so a 24/7 sweep spreads over the whole space instead of re-mining the easy corner. State is one JSON file, written
    atomically, so a killed process resumes exactly where it stopped (CONTEXT rule 11)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.counts: dict[str, int] = {}
        self.last: dict[str, str] = {}
        if self.path and self.path.exists():
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.counts, self.last = {k: int(v) for k, v in d.get("counts", {}).items()}, dict(d.get("last", {}))

    @staticmethod
    def key(family: str, era: str = "*", cohort: str = "*") -> str:
        return f"{family}|{era}|{cohort}"

    def mark(self, family: str, era: str, cohort: str, now, weight: int = 1) -> None:
        k = self.key(family, era, cohort)
        self.counts[k] = self.counts.get(k, 0) + int(weight)
        self.last[k] = str(as_date(now))

    def count(self, family: str, era: str = "*", cohort: str = "*") -> int:
        return self.counts.get(self.key(family, era, cohort), 0)

    def next_unit(self, candidates: Iterable[tuple[str, str, str]]) -> tuple[str, str, str] | None:
        """Least-covered candidate; ties broken by hash so the choice is deterministic but not alphabetical."""
        cands = list(candidates)
        if not cands:
            return None
        return min(cands, key=lambda c: (self.count(*c), stable_hash({"u": c}, 8)))

    def gaps(self, candidates: Iterable[tuple[str, str, str]]) -> list[tuple[tuple[str, str, str], int]]:
        return sorted(((c, self.count(*c)) for c in candidates), key=lambda t: (t[1], stable_hash({"u": t[0]}, 8)))

    def evenness(self, candidates: Iterable[tuple[str, str, str]]) -> float:
        """1 = perfectly even coverage of the candidate space, 0 = all effort in one unit (normalised entropy)."""
        v = np.array([self.count(*c) for c in candidates], float)
        if len(v) < 2 or v.sum() == 0:
            return 0.0
        p = v / v.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum() / math.log(len(v)))

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"counts": self.counts, "last": self.last}, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


@dataclass(frozen=True)
class PrecursorSource:
    """A candidate precursor handed in as a discovery input (e.g. from engine/research/precursors.py). Duck-typed on purpose: this module
    must not import a sibling being built in parallel. `frame` holds the columns Predictions needs (p = the precursor's P(up) call)."""
    source_id: str
    family: str
    frame: pd.DataFrame

    @classmethod
    def coerce(cls, obj: Any) -> "PrecursorSource":
        missing = [a for a in ("precursor_id", "source_family", "frame") if not hasattr(obj, a)]
        if missing:
            raise ValueError(f"precursor candidate lacks {missing}")
        return cls(str(obj.precursor_id), str(obj.source_family), obj.frame)


@dataclass(frozen=True)
class SweepStep:
    unit: tuple[str, str, str] | None
    rows: int
    tests_logged: int
    status: str
    best_accuracy: float
    best_family_p: float
    survivors_total: int


class FrontierSweep:
    """Always-on sweep over many prediction sources. Each `run_once` picks the least-covered (family, era) unit, assesses it, logs one
    test per coverage cell into the cumulative TestLedger, and saves. Because every cell of every unit ever swept sits in one ledger,
    the multiple-comparison price of any new 'finding' grows with the whole search, exactly as C67 requires."""

    def __init__(self, tracker: CoverageTracker, ledger: TestLedger, cfg: FrontierConfig | None = None):
        self.tracker, self.ledger, self.cfg = tracker, ledger, (cfg or FrontierConfig()).check()
        self.sources: dict[str, Predictions] = {}

    def add_source(self, family: str, P: Predictions) -> None:
        self.sources[family] = P

    def add_precursors(self, candidates: Iterable[Any]) -> int:
        """Accept precursor candidates as sources. Malformed candidates are refused (fail closed), valid ones become a family each."""
        n = 0
        for obj in candidates:
            s = PrecursorSource.coerce(obj)
            self.sources[f"precursor:{s.family}:{s.source_id}"] = Predictions(s.frame)
            n += 1
        return n

    def units(self) -> list[tuple[str, str, str]]:
        out = []
        for fam, P in self.sources.items():
            eras = sorted(P.frame["era"].astype(str).unique()) if not P.empty else []
            out += [(fam, e, "all") for e in eras]
        return out

    def run_once(self, now) -> SweepStep:
        unit = self.tracker.next_unit(self.units())
        if unit is None:
            return SweepStep(None, 0, 0, "no sources", math.nan, math.nan, len(self.ledger.survivors()))
        fam, era, cohort = unit
        P = self.sources[fam]
        vis = P.matured_before(now)
        sub = vis.subset((vis.frame["era"].astype(str) == era).to_numpy()) if not vis.empty else vis
        self.tracker.mark(fam, era, cohort, now, weight=max(len(sub), 1))
        if len(sub) < self.cfg.min_n:
            self.tracker.save()
            return SweepStep(unit, len(sub), 0, "too few matured rows", math.nan, math.nan, len(self.ledger.survivors()))
        F = compute_frontier(sub, self.cfg, extra_cells=len(self.ledger))
        newest = sub.frame["matured_at"].max()
        logged, best_acc, best_p = 0, math.nan, 1.0
        for c in F.cells:
            if c.n < self.cfg.min_n or not math.isfinite(c.p_family):
                continue
            self.ledger.log(f"{fam}|{era}|{c.coverage_target}|{sub.digest()}", fam, c.p_family, newest, now)
            logged += 1
            if c.p_family < best_p:
                best_p, best_acc = c.p_family, c.accuracy
        self.tracker.save()
        return SweepStep(unit, len(sub), logged, "assessed", best_acc, best_p if logged else math.nan, len(self.ledger.survivors()))

    def run(self, now, steps: int) -> list[SweepStep]:
        return [self.run_once(now) for _ in range(steps)]


# ==================================================================================================================
# state and the single public entry the research loop calls
# ==================================================================================================================
class FrontierState:
    """Accumulates resolved direction calls across the run (research side, MATURED_RESEARCH_STATE)."""

    def __init__(self, cfg: FrontierConfig | None = None, ledger_path: str | Path | None = None):
        self.cfg = (cfg or FrontierConfig()).check()
        self.batches: list[Predictions] = []
        self.history: list[dict[str, Any]] = []
        self.ledger_path = Path(ledger_path) if ledger_path else None

    def add(self, frame: pd.DataFrame | Predictions) -> int:
        P = frame if isinstance(frame, Predictions) else Predictions(frame)
        self.batches.append(P)
        return len(P)

    def all(self) -> Predictions:
        return Predictions.concat(self.batches)

    def append_history(self, r: FrontierReport) -> None:
        a = r.answer
        row = {"now": r.now, "status": a.status.value, "best_acc": _num(a.best_supported_accuracy), "best_lo": _num(a.best_supported_lo),
               "best_cov": _num(a.best_supported_coverage), "rows": r.frontier.n_rows, "data": r.data_digest, "config": r.config_digest}
        self.history.append(row)
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + NL)

    def trend(self) -> dict[str, Any]:
        """Has the honestly supported accuracy moved between snapshots? Slope of best lower bound over snapshots (NaN < 3 points)."""
        pts = [(i, h["best_lo"]) for i, h in enumerate(self.history) if h["best_lo"] is not None]
        if len(pts) < 3:
            return {"snapshots": len(self.history), "slope": math.nan, "ever_found": any(h["status"] == "CANDIDATE" for h in self.history)}
        s = sps.linregress([p[0] for p in pts], [p[1] for p in pts])
        return {"snapshots": len(self.history), "slope": float(s.slope), "p": float(s.pvalue), "ever_found": any(h["status"] == "CANDIDATE" for h in self.history)}


def step(state: FrontierState, now, seed: int | None = None, *, code_hash: str | None = None) -> FrontierReport:
    """ONE public entry (CONTEXT rule 25): assess everything in `state` that matured strictly before `now`, record the snapshot, return
    the report. The report reaches the trader only through report.to_matured_record().gate(now)."""
    cfg = state.cfg if seed is None else dataclasses.replace(state.cfg, seed=int(seed))
    r = assess(state.all(), now, cfg, code_hash=code_hash)
    state.append_history(r)
    return r


# ==================================================================================================================
# planning: how much independent evidence would certification take? (separates NOT FOUND from NOT YET MEASURABLE)
# ==================================================================================================================
def required_independent_obs(true_acc: float, gate: float = 0.80, level: float = 0.95, power: float = 0.80) -> float:
    """Independent calls needed for a region whose true accuracy is `true_acc` to show a `level` Wilson lower bound >= gate with the given
    power. Infinite when true_acc <= gate: no amount of data certifies a region that is not really that good."""
    if true_acc <= gate:
        return math.inf
    z_a, z_b = float(sps.norm.ppf(level)), float(sps.norm.ppf(power))
    sd_gate, sd_true = math.sqrt(gate * (1 - gate)), math.sqrt(true_acc * (1 - true_acc))
    return float(((z_a * sd_gate + z_b * sd_true) / (true_acc - gate)) ** 2)


def detectable_accuracy(n_eff: float, gate: float = 0.80, level: float = 0.95, power: float = 0.80) -> float:
    """Smallest true accuracy a region with n_eff independent calls could show as >= gate with the given power (the reverse question).
    Infinity means even a perfect region could not be certified with this much evidence."""
    if n_eff < 2:
        return math.inf
    if required_independent_obs(1.0 - 1e-9, gate, level, power) > n_eff:
        return math.inf                                   # even a perfect region could not be certified
    lo, hi = gate, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if required_independent_obs(mid, gate, level, power) > n_eff:
            lo = mid
        else:
            hi = mid
    return float(hi)


@dataclass(frozen=True)
class EvidenceGap:
    coverage: float
    n_eff: float
    observed_accuracy: float
    needed_if_true: float                 # independent calls needed if the observed accuracy were the truth
    shortfall: float                      # needed - have (<= 0: enough evidence exists)
    detectable: float                     # best accuracy this cell's evidence could certify

    @property
    def measurable(self) -> bool:
        return self.detectable <= 1.0


def evidence_gaps(F: Frontier, cfg: FrontierConfig) -> list[EvidenceGap]:
    """Per cell: is 'not found' a finding, or only a shortage of data? A cell that could not have certified even a perfect region
    says nothing about 80%; the answer for that coverage is UNMEASURED, not NO."""
    out = []
    for c in F.cells:
        if c.n == 0:
            out.append(EvidenceGap(c.coverage_target, 0.0, math.nan, math.inf, math.inf, math.inf))
            continue
        need = required_independent_obs(c.accuracy, cfg.gate, cfg.level)
        out.append(EvidenceGap(c.coverage_target, c.n_eff, c.accuracy, need, need - c.n_eff, detectable_accuracy(c.n_eff, cfg.gate, cfg.level)))
    return out


# ==================================================================================================================
# comparing runs and models
# ==================================================================================================================
@dataclass(frozen=True)
class CellChange:
    coverage: float
    acc_before: float
    acc_after: float
    delta: float
    z: float
    significant: bool


def compare_frontiers(before: Frontier, after: Frontier, cfg: FrontierConfig) -> list[CellChange]:
    """Did the frontier really move between two snapshots? Two-proportion z on effective counts; a change inside noise is reported as
    not significant (section 43: a higher number is not automatically better)."""
    out = []
    for a in before.cells:
        try:
            b = after.cell(a.coverage_target)
        except KeyError:
            continue
        if a.n == 0 or b.n == 0:
            continue
        na, nb = max(a.n_eff, 1.0), max(b.n_eff, 1.0)
        pool = (a.accuracy * na + b.accuracy * nb) / (na + nb)
        se = math.sqrt(max(pool * (1 - pool) * (1 / na + 1 / nb), 1e-12))
        z = (b.accuracy - a.accuracy) / se
        out.append(CellChange(a.coverage_target, a.accuracy, b.accuracy, b.accuracy - a.accuracy, float(z), bool(abs(z) > sps.norm.ppf(1 - 0.025))))
    return out


@dataclass(frozen=True)
class ModelComparison:
    model_a: str
    model_b: str
    coverage: float
    n_shared: int
    diff: float
    lo: float
    hi: float
    favours: str


def paired_model_comparison(Pa: Predictions, Pb: Predictions, names: tuple[str, str], coverage: float, cfg: FrontierConfig,
                            rng: np.random.Generator | None = None) -> ModelComparison:
    """Accuracy of model A minus model B on the SAME (date, ticker) rows, each choosing its own coverage-c region, with a paired
    week-cluster bootstrap. Comparing on shared rows removes the row-difficulty variance that fools unpaired comparisons."""
    rng = rng if rng is not None else np.random.default_rng(cfg.seed + 3)
    a = Pa.frame[["date", "ticker", "conf", "correct"]].rename(columns={"conf": "ca", "correct": "ka"})
    b = Pb.frame[["date", "ticker", "conf", "correct"]].rename(columns={"conf": "cb", "correct": "kb"})
    m = a.merge(b, on=["date", "ticker"])
    if len(m) < cfg.min_n:
        return ModelComparison(names[0], names[1], coverage, len(m), math.nan, math.nan, math.nan, "insufficient")
    ma, mb = select_mask(m["ca"].to_numpy(), coverage), select_mask(m["cb"].to_numpy(), coverage)
    codes = week_codes(m["date"])
    G = int(codes.max()) + 1
    na, ka = group_sums(codes, m["ka"].to_numpy(), ma, G)
    nb, kb = group_sums(codes, m["kb"].to_numpy(), mb, G)
    live = np.flatnonzero((na > 0) & (nb > 0))
    if len(live) < 5:
        return ModelComparison(names[0], names[1], coverage, len(m), math.nan, math.nan, math.nan, "insufficient")
    na, ka, nb, kb = na[live], ka[live], nb[live], kb[live]
    idx = rng.integers(0, len(live), size=(cfg.n_boot, len(live)))
    d = ka[idx].sum(1) / na[idx].sum(1) - kb[idx].sum(1) / nb[idx].sum(1)
    lo, hi = np.quantile(d, [(1 - cfg.level) / 2, 1 - (1 - cfg.level) / 2])
    point = float(ka.sum() / na.sum() - kb.sum() / nb.sum())
    fav = names[0] if lo > 0 else names[1] if hi < 0 else "neither"
    return ModelComparison(names[0], names[1], coverage, len(m), point, float(lo), float(hi), fav)


def random_control(P: Predictions, seed: int = 0) -> Predictions:
    """Same rows, confidence and side carry no information: outcomes are kept, p is redrawn. A frontier run on this must find nothing."""
    rng = np.random.default_rng(seed)
    f = P.frame.drop(columns=["conf", "side", "correct", "wk"], errors="ignore").copy()
    conf = rng.uniform(0.5, 0.99, len(f))
    f["p"] = np.where(rng.random(len(f)) < 0.5, conf, 1.0 - conf)
    return Predictions(f)


def shuffled_outcome_control(P: Predictions, seed: int = 0) -> Predictions:
    """Same calls, outcomes permuted within each week: real confidence structure, no link to what happened."""
    rng = np.random.default_rng(seed)
    f = P.frame.drop(columns=["conf", "side", "correct", "wk"], errors="ignore").copy()
    codes = week_codes(f["date"])
    perm = np.arange(len(f))
    for g in np.unique(codes):
        ix = np.flatnonzero(codes == g)
        perm[ix] = rng.permutation(ix)
    f["up"] = f["up"].to_numpy()[perm]
    if "fwd" in f:
        f["fwd"] = f["fwd"].to_numpy()[perm]
    return Predictions(f)


def leak_canary(P: Predictions, seed: int = 0, strength: float = 0.9) -> Predictions:
    """A deliberately leaking model: p is a noisy copy of the OUTCOME. The frontier must find an 80% region on it (it does not prove
    the region is tradable, only that the instrument is not blind); reachable through the same `assess` path as a real model."""
    rng = np.random.default_rng(seed)
    f = P.frame.drop(columns=["conf", "side", "correct", "wk"], errors="ignore").copy()
    conf = rng.uniform(0.75, 0.99, len(f))
    right = rng.random(len(f)) < strength
    says_up = np.where(right, f["up"].to_numpy() > 0.5, f["up"].to_numpy() <= 0.5)
    f["p"] = np.where(says_up, conf, 1.0 - conf)
    return Predictions(f)


def instrument_controls(P: Predictions, now, cfg: FrontierConfig, seed: int = 0) -> dict[str, str]:
    """Run the real path on the null and canary variants of the same rows. `sound` is True when the random and shuffled controls yield
    NOT_FOUND and the leak canary does not (the instrument can see, and can refuse)."""
    res = {name: assess(fn(P, seed), now, cfg, code_hash="controls").answer.status.value
           for name, fn in (("random", random_control), ("shuffled", shuffled_outcome_control), ("canary", leak_canary))}
    res["sound"] = str(res["random"] == "NOT_FOUND" and res["shuffled"] == "NOT_FOUND" and res["canary"] != "NOT_FOUND")
    return res


# ==================================================================================================================
# frontier by group, sensitivity, persistence
# ==================================================================================================================
def frontier_by_group(P: Predictions, dimension: str, cfg: FrontierConfig, min_rows: int = 200) -> dict[str, Frontier]:
    """A full frontier inside each era / sector / regime. Each group's cells count toward the multiple-comparison price of the others."""
    if P.empty or not P.has(dimension):
        return {}
    groups = [g for g, n in P.frame[dimension].astype(str).value_counts().items() if n >= min_rows]
    extra = len(cfg.coverages) * max(len(groups) - 1, 0)
    return {g: compute_frontier(P.subset((P.frame[dimension].astype(str) == g).to_numpy()), cfg, extra_cells=extra, with_null=False)
            for g in sorted(groups)}


def gate_sensitivity(P: Predictions, now, cfg: FrontierConfig, gates: Sequence[float] = (0.70, 0.75, 0.80, 0.85, 0.90)) -> dict[float, str]:
    """Answer the same question at several accuracy targets: where does 'found' turn into 'not found'? Shows how much of the answer is
    the arbitrary threshold and how much is the data."""
    return {g: assess(P, now, dataclasses.replace(cfg, gate=g), code_hash="sensitivity").answer.status.value for g in gates}


def dump_report(r: FrontierReport, path: str | Path) -> str:
    """Write the identity-free summary plus hashes as JSON; returns the content hash. Nothing but numbers and verdict words is written."""
    body = {"record_id": r.record_id(), "config": r.config_digest, "data": r.data_digest, "code_hash": r.provenance.code_hash,
            "seed": r.provenance.seed, "summary": public_summary(r)}
    body["content_hash"] = stable_hash(body["summary"], 16)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, sort_keys=True, indent=1), encoding="utf-8")
    return body["content_hash"]


def load_report_summary(path: str | Path) -> dict[str, Any]:
    """Read a dumped summary and refuse it if its content hash no longer matches (tampering or a hand edit)."""
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if stable_hash(body["summary"], 16) != body.get("content_hash"):
        raise FirewallBreach(f"frontier report {path} fails its content hash")
    return body
