"""Winner/loser symmetry engine and the loss-risk knowledge bank (RESEARCH_BRAIN_CONTRACT C66 section 12; serves sections 5, 6, 34, 43, 44).

STATUS: IMPLEMENTED - NOT VALIDATED (C63: code and unit tests only; no real-data run has been made).

A pattern must not become trusted simply because it finds winners. For every important pattern this module measures BOTH
   "when the pattern predicts a winner"   and   "when the pattern predicts a loser":
true / false positives / negatives in winner space AND in loser space, the three-class (loser / neutral / winner) matrix, missed winners and
missed losers (silent misses, wrong-direction misses, near misses), and the pattern's behaviour on the cases that break patterns:
wrong-direction calls, large losses, regime transitions, external events, near-miss rows, high-volatility failures, low-liquidity failures
and (owner directive C67) what the mover did NEXT - consolidated, expanded, stopped, spiked or reversed. Each pattern then gets a
TrustVerdict that a winners-only record cannot earn.

It also builds a LOSS-RISK knowledge bank that is structurally separate from the opportunity bank: different item type, different id prefix,
different storage, mined only from loss definitions, queried only for what to AVOID or shrink, and audited for independence.

An always-on SWEEP (C67) walks (source family, era, cohort) units, always taking the least-covered unit next, is resumable from disk, and
accepts candidate precursors from another module as duck-typed inputs. Every hypothesis test it runs goes into ONE cumulative multiple-testing
ledger (engine.research.frontier.TestLedger), so as the search grows the bar for calling a small effect real rises with it.

Built on (imported, not copied):
  engine.pattern_stats            bh_qvalues (multiple testing), design-effect logic via engine.research.frontier
  engine.learning.calibration     wilson intervals
  engine.learning.core            Epistemic, Lifecycle, Provenance, DecisionEffect, FailureCause, stable_hash, require_past, FirewallBreach
  engine.research.core            MaturedRecord, Namespace, Knowability, MoveCategory (the section-4 category names for missed movers)
  engine.research.frontier        week_codes, design_effect, TestLedger, CoverageTracker (one ledger, one coverage tracker, shared)
  engine.missed_winners           WINNER threshold (the repo's one definition of a winner)

Time discipline (C56, rule 3): every reader takes `now` and sees only rows whose outcome matured strictly before it; the bank refuses items
dated at or after `now` and answers queries only from items matured before the query. Nothing identity-bearing (ticker, date, year) is
placed in a bank item's context or in the public summary; `assert_trader_safe` is applied to what leaves this module."""
from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine import missed_winners as base_missed
from engine import pattern_stats
from engine.learning import calibration as CAL
from engine.learning.core import (DecisionEffect, Epistemic, FailureCause, FirewallBreach, Lifecycle, Provenance, as_date,
                                  current_code_hash, require_past, stable_hash)
from engine.learning.knowledge import wall_stamp
from engine.learning.trader_view import assert_trader_safe
from engine.research.core import Knowability, MaturedRecord, MoveCategory, Namespace
from engine.research.frontier import CoverageTracker, TestLedger, design_effect, week_codes
from engine.research.loss_pipeline import LossRiskBank as _SharedLossBank, RiskEntry as _RiskEntry

NL = chr(10)
WINNER = float(base_missed.WINNER)
REQUIRED = ("pattern_id", "date", "ticker", "matured_at", "call", "fwd")
OPTIONAL_FLAGS = ("regime_transition", "external_event")
EPISODE_KINDS = ("consolidated", "expanded", "stopped", "spiked", "reversed")


# ==================================================================================================================
# vocabulary
# ==================================================================================================================
class Outcome(str, enum.Enum):
    TP = "TP"
    FP = "FP"
    TN = "TN"
    FN = "FN"


class Space(str, enum.Enum):
    WINNER = "WINNER"                 # "the pattern says this stock will be a winner"
    LOSER = "LOSER"                   # "the pattern says this stock will be a loser"


class CaseKind(str, enum.Enum):
    WRONG_DIRECTION = "WRONG_DIRECTION"
    LARGE_LOSS = "LARGE_LOSS"
    REGIME_TRANSITION = "REGIME_TRANSITION"
    EXTERNAL_EVENT = "EXTERNAL_EVENT"
    NEAR_MISS = "NEAR_MISS"
    HIGH_VOL_FAILURE = "HIGH_VOL_FAILURE"
    LOW_LIQUIDITY_FAILURE = "LOW_LIQUIDITY_FAILURE"
    EPISODE_CONSOLIDATED = "EPISODE_CONSOLIDATED"
    EPISODE_EXPANDED = "EPISODE_EXPANDED"
    EPISODE_STOPPED = "EPISODE_STOPPED"
    EPISODE_SPIKED = "EPISODE_SPIKED"
    EPISODE_REVERSED = "EPISODE_REVERSED"


OUTCOME_DEFINED = (CaseKind.WRONG_DIRECTION, CaseKind.LARGE_LOSS)


class SliceStatus(str, enum.Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"             # worse than the pattern's own average, but not provably so
    FAILS = "FAILS"                   # provably worse, or breaches the loss limits
    UNTESTED = "UNTESTED"             # too few calls: never read as OK


class Trust(str, enum.Enum):
    TRUSTED = "TRUSTED"
    NOT_TRUSTED = "NOT_TRUSTED"
    UNKNOWN = "UNKNOWN"               # the evidence cannot say


class MissKind(str, enum.Enum):
    SILENT = "SILENT"                 # pattern did not fire
    WRONG_WAY = "WRONG_WAY"           # pattern fired the opposite direction
    NEAR_MISS = "NEAR_MISS"           # pattern did not fire but was within the near-miss band of its threshold
    SUPPRESSED = "SUPPRESSED"         # not fired while a risk context (high vol / low liquidity / event) was present


class RiskKind(str, enum.Enum):
    WRONG_DIRECTION_CONTEXT = "WRONG_DIRECTION_CONTEXT"
    LARGE_LOSS_CONTEXT = "LARGE_LOSS_CONTEXT"
    GAP_RISK = "GAP_RISK"
    LIQUIDITY_TRAP = "LIQUIDITY_TRAP"
    EVENT_RISK = "EVENT_RISK"
    REGIME_SHIFT_RISK = "REGIME_SHIFT_RISK"
    VOL_BLOWUP = "VOL_BLOWUP"
    REVERSAL_RISK = "REVERSAL_RISK"


LOSS_KIND_CAUSE = {                               # what section-9 cause a loss context most often points at
    RiskKind.WRONG_DIRECTION_CONTEXT: FailureCause.WRONG_CONTEXT, RiskKind.LARGE_LOSS_CONTEXT: FailureCause.RISK_ERROR,
    RiskKind.GAP_RISK: FailureCause.RISK_ERROR, RiskKind.LIQUIDITY_TRAP: FailureCause.RISK_ERROR,
    RiskKind.EVENT_RISK: FailureCause.UNKNOWN, RiskKind.REGIME_SHIFT_RISK: FailureCause.REGIME_CHANGE,
    RiskKind.VOL_BLOWUP: FailureCause.RISK_ERROR, RiskKind.REVERSAL_RISK: FailureCause.REVERSAL}

CASE_LOSS_KIND = {CaseKind.WRONG_DIRECTION: RiskKind.WRONG_DIRECTION_CONTEXT, CaseKind.LARGE_LOSS: RiskKind.LARGE_LOSS_CONTEXT,
                  CaseKind.REGIME_TRANSITION: RiskKind.REGIME_SHIFT_RISK, CaseKind.EXTERNAL_EVENT: RiskKind.EVENT_RISK,
                  CaseKind.HIGH_VOL_FAILURE: RiskKind.VOL_BLOWUP, CaseKind.LOW_LIQUIDITY_FAILURE: RiskKind.LIQUIDITY_TRAP,
                  CaseKind.EPISODE_REVERSED: RiskKind.REVERSAL_RISK, CaseKind.NEAR_MISS: RiskKind.WRONG_DIRECTION_CONTEXT}


# ==================================================================================================================
# configuration
# ==================================================================================================================
@dataclass(frozen=True)
class SymmetryConfig:
    win_thr: float = WINNER                  # forward return >= win_thr is a winner
    loss_thr: float = 0.05                   # forward return <= -loss_thr is a loser (symmetric by default)
    large_loss: float = 0.15                 # a call losing more than this is a large loss
    high_vol_q: float = 0.80                 # trailing-vol quantile above which a row is "high volatility"
    low_liq_q: float = 0.20                  # dollar-volume quantile below which a row is "low liquidity"
    near_band: float = 0.10                  # |margin| within this of the firing threshold is a near miss
    min_n: int = 30                          # calls needed before a slice or a pattern is judged
    min_calls: int = 50                      # calls needed before a pattern can be TRUSTED
    max_large_loss_rate: float = 0.05        # tolerated share of calls losing more than `large_loss`
    max_wrong_direction_rate: float = 0.35   # tolerated share of decisive calls that went the opposite way
    adverse_tolerance: float = 1.25          # P(opposite mover | call) may exceed the base rate by at most this factor
    slice_drop: float = 0.05                 # hit-rate drop inside a slice beyond which it is DEGRADED
    fails_q: float = 0.10                    # BH q at which a worse slice is FAILS
    tol_return: float = 0.01                 # mean directional return tolerated below zero inside a slice
    level: float = 0.95
    horizon_days: float = 7.0
    seed: int = 0

    def validate(self) -> list[str]:
        errs = []
        if not (0 < self.win_thr < 1 and 0 < self.loss_thr < 1):
            errs.append("win_thr and loss_thr must lie in (0, 1)")
        if self.large_loss < self.loss_thr:
            errs.append("large_loss must be at least loss_thr")
        if not (0.5 <= self.high_vol_q < 1) or not (0 < self.low_liq_q <= 0.5):
            errs.append("high_vol_q in [0.5, 1) and low_liq_q in (0, 0.5]")
        if self.min_n < 5 or self.min_calls < self.min_n:
            errs.append("min_n >= 5 and min_calls >= min_n")
        if not (0 < self.max_large_loss_rate < 1 and 0 < self.max_wrong_direction_rate < 1):
            errs.append("loss-rate tolerances must lie in (0, 1)")
        if self.adverse_tolerance < 1:
            errs.append("adverse_tolerance must be >= 1")
        if not (0 < self.fails_q < 1):
            errs.append("fails_q must lie in (0, 1)")
        return errs

    def check(self) -> "SymmetryConfig":
        errs = self.validate()
        if errs:
            raise ValueError("invalid SymmetryConfig: " + "; ".join(errs))
        return self

    @property
    def z(self) -> float:
        return float(sps.norm.ppf(1 - (1 - self.level) / 2))

    def digest(self) -> str:
        return stable_hash(dataclasses.asdict(self), 12)


# ==================================================================================================================
# the observation frame
# ==================================================================================================================
class SymFrame:
    """Validated research-side table: one row per (pattern, date, ticker) OPPORTUNITY - fired or not. Without the unfired rows a missed
    winner cannot exist, so a frame containing only fired rows is refused when `require_universe` is set.

    Required: pattern_id, date, ticker, matured_at (> date), call (+1 predicts winner / -1 predicts loser / 0 silent), fwd (forward return).
    Optional context: lean (sign the pattern would call, also when silent), margin (score minus firing threshold; < 0 means unfired),
    vol (trailing volatility), dollar_volume, gap (overnight gap), regime, regime_transition (bool), external_event (bool),
    sector, mover_outcome (consolidated / expanded / stopped / spiked / reversed).
    Derived: side (call, else lean, else 0), winner, loser, dir_ret (side * fwd), hit, wrong_way, big_loss, high_vol, low_liq, near."""

    def __init__(self, frame: pd.DataFrame, cfg: SymmetryConfig | None = None, *, require_universe: bool = True):
        self.cfg = (cfg or SymmetryConfig()).check()
        f = frame.copy()
        for c in ("date", "matured_at"):
            if c in f:
                f[c] = pd.to_datetime(f[c])
        errs = self._problems(f, require_universe)
        if errs:
            raise ValueError("invalid SymFrame: " + "; ".join(errs))
        f = f.reset_index(drop=True)
        self.frame = self._derive(f)

    @staticmethod
    def _problems(f: pd.DataFrame, require_universe: bool) -> list[str]:
        errs = [f"missing column {c}" for c in REQUIRED if c not in f]
        if errs:
            return errs
        if len(f) == 0:
            return []
        if f[list(REQUIRED)].isna().any().any():
            errs.append("NaN in a required column")
            return errs
        if not f["call"].isin((-1, 0, 1)).all():
            errs.append("call must be -1, 0 or +1")
        if (f["matured_at"] <= f["date"]).any():
            errs.append("matured_at must be strictly after date")
        if f.duplicated(["pattern_id", "date", "ticker"]).any():
            errs.append("duplicate (pattern_id, date, ticker)")
        if not np.isfinite(f["fwd"].to_numpy(float)).all():
            errs.append("fwd must be finite")
        if require_universe:
            for pid, g in f.groupby("pattern_id"):
                if (g["call"] != 0).all():
                    errs.append(f"pattern {pid} has only fired rows: no unfired universe, so no missed winners or losers can be measured")
        return errs

    def _derive(self, f: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        if len(f) == 0:
            return f
        f["fwd"] = f["fwd"].astype(float)
        lean = f["lean"] if "lean" in f else pd.Series(0, index=f.index)
        f["side"] = np.where(f["call"] != 0, f["call"], lean.fillna(0).astype(int))
        f["winner"] = f["fwd"] >= c.win_thr
        f["loser"] = f["fwd"] <= -c.loss_thr
        f["dir_ret"] = f["side"] * f["fwd"]
        f["fired"] = f["call"] != 0
        f["hit"] = f["fired"] & (f["dir_ret"] > 0)
        f["wrong_way"] = f["fired"] & (((f["call"] > 0) & f["loser"]) | ((f["call"] < 0) & f["winner"]))
        f["big_loss"] = f["fired"] & (f["dir_ret"] <= -c.large_loss)
        for col in OPTIONAL_FLAGS:
            f[col] = f[col].fillna(False).astype(bool) if col in f else False
        f["high_vol"] = self._quantile_flag(f, "vol", c.high_vol_q, above=True)
        f["low_liq"] = self._quantile_flag(f, "dollar_volume", c.low_liq_q, above=False)
        f["near"] = (f["margin"].abs() <= c.near_band) if "margin" in f else False
        f["era"] = f["date"].dt.year.astype(str) if "era" not in f else f["era"].astype(str)
        f["wk"] = week_codes(f["date"])
        return f

    @staticmethod
    def _quantile_flag(f: pd.DataFrame, col: str, q: float, above: bool) -> pd.Series:
        """Row flag against a PER-PATTERN-DATE-INDEPENDENT quantile: computed on the whole frame's values so it cannot leak a pattern's fate."""
        if col not in f or f[col].notna().sum() < 10:
            return pd.Series(False, index=f.index)
        thr = float(f[col].quantile(q))
        return (f[col] >= thr) if above else (f[col] <= thr)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def empty(self) -> bool:
        return len(self.frame) == 0

    def patterns(self) -> list[str]:
        return sorted(self.frame["pattern_id"].unique()) if not self.empty else []

    def of(self, pattern_id: str) -> pd.DataFrame:
        return self.frame[self.frame["pattern_id"] == pattern_id]

    def matured_before(self, now) -> "SymFrame":
        """Only rows whose outcome was known strictly before `now`. Everything else is invisible, not merely ignored."""
        cut = pd.Timestamp(as_date(now))
        keep = self.frame[self.frame["matured_at"] < cut]
        return SymFrame(keep.drop(columns=self._derived_cols(), errors="ignore"), self.cfg, require_universe=False)

    def subset(self, mask) -> "SymFrame":
        keep = self.frame[np.asarray(mask, bool)]
        return SymFrame(keep.drop(columns=self._derived_cols(), errors="ignore"), self.cfg, require_universe=False)

    @staticmethod
    def _derived_cols() -> list[str]:
        return ["side", "winner", "loser", "dir_ret", "fired", "hit", "wrong_way", "big_loss", "high_vol", "low_liq", "near", "wk"]

    def has(self, col: str) -> bool:
        return col in self.frame and bool(self.frame[col].notna().any())

    def digest(self) -> str:
        if self.empty:
            return "empty"
        h = pd.util.hash_pandas_object(self.frame[["pattern_id", "date", "ticker", "call", "fwd"]], index=False).to_numpy()
        return stable_hash(h.tolist(), 16)

    @classmethod
    def concat(cls, parts: Sequence["SymFrame"], cfg: SymmetryConfig | None = None) -> "SymFrame":
        frames = [p.frame.drop(columns=cls._derived_cols(), errors="ignore") for p in parts if not p.empty]
        if not frames:
            return cls(pd.DataFrame({c: pd.Series(dtype=float) for c in REQUIRED}), cfg, require_universe=False)
        return cls(pd.concat(frames, ignore_index=True), cfg, require_universe=False)


# ==================================================================================================================
# the confusion matrix, in either space
# ==================================================================================================================
@dataclass(frozen=True)
class Confusion:
    """TP / FP / TN / FN for one space, with every rate that matters and Wilson intervals on the ones that get quoted."""
    space: Space
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def called(self) -> int:
        return self.tp + self.fp

    @property
    def actual_pos(self) -> int:
        return self.tp + self.fn

    @property
    def prevalence(self) -> float:
        return self.actual_pos / self.n if self.n else math.nan

    @property
    def precision(self) -> float:
        return self.tp / self.called if self.called else math.nan

    @property
    def recall(self) -> float:
        return self.tp / self.actual_pos if self.actual_pos else math.nan

    @property
    def specificity(self) -> float:
        neg = self.tn + self.fp
        return self.tn / neg if neg else math.nan

    @property
    def fpr(self) -> float:
        neg = self.tn + self.fp
        return self.fp / neg if neg else math.nan

    @property
    def fnr(self) -> float:
        return self.fn / self.actual_pos if self.actual_pos else math.nan

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if math.isfinite(p) and math.isfinite(r) and (p + r) > 0 else math.nan

    @property
    def lift(self) -> float:
        """precision / prevalence: how many times better than picking at random from the universe."""
        return self.precision / self.prevalence if self.called and self.prevalence and math.isfinite(self.prevalence) else math.nan

    @property
    def informedness(self) -> float:
        r, s = self.recall, self.specificity
        return r + s - 1.0 if math.isfinite(r) and math.isfinite(s) else math.nan

    @property
    def mcc(self) -> float:
        den = math.sqrt(float(self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        return (self.tp * self.tn - self.fp * self.fn) / den if den > 0 else math.nan

    def precision_ci(self, z: float = 1.96) -> tuple[float, float]:
        return CAL.wilson(self.tp, self.called, z) if self.called else (math.nan, math.nan)

    def recall_ci(self, z: float = 1.96) -> tuple[float, float]:
        return CAL.wilson(self.tp, self.actual_pos, z) if self.actual_pos else (math.nan, math.nan)

    def as_dict(self) -> dict[str, Any]:
        return dict(space=self.space.value, tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn, n=self.n, precision=self.precision, recall=self.recall,
                    specificity=self.specificity, fpr=self.fpr, f1=self.f1, lift=self.lift, mcc=self.mcc, informedness=self.informedness,
                    prevalence=self.prevalence)


def confusion_of(called: np.ndarray, actual: np.ndarray, space: Space) -> Confusion:
    """Build a Confusion from boolean arrays (pattern called positive in this space / the row really was positive in it)."""
    c, a = np.asarray(called, bool), np.asarray(actual, bool)
    if c.shape != a.shape:
        raise ValueError("called and actual must have equal shape")
    return Confusion(space, int((c & a).sum()), int((c & ~a).sum()), int((~c & ~a).sum()), int((~c & a).sum()))


def confusion_pair(f: pd.DataFrame) -> tuple[Confusion, Confusion]:
    """(winner-space, loser-space) confusions for one pattern's rows. Winner space: called = call +1, actual = winner. Loser space:
    called = call -1, actual = loser. The two are computed independently, so a pattern with a superb winner side and no loser side is visible."""
    win = confusion_of((f["call"] > 0).to_numpy(), f["winner"].to_numpy(), Space.WINNER)
    lose = confusion_of((f["call"] < 0).to_numpy(), f["loser"].to_numpy(), Space.LOSER)
    return win, lose


def three_class_matrix(f: pd.DataFrame) -> pd.DataFrame:
    """Rows = what the pattern said (loser call / silent / winner call), columns = what happened (loser / neutral / winner)."""
    say = np.select([f["call"] < 0, f["call"] > 0], ["said_loser", "said_winner"], "silent")
    got = np.select([f["loser"], f["winner"]], ["was_loser", "was_winner"], "was_neutral")
    m = pd.crosstab(pd.Categorical(say, ["said_loser", "silent", "said_winner"]), pd.Categorical(got, ["was_loser", "was_neutral", "was_winner"]),
                    dropna=False)
    m.index.name, m.columns.name = "said", "got"
    return m


# ==================================================================================================================
# statistics helpers
# ==================================================================================================================
def _wilson(k: float, n: float, z: float) -> tuple[float, float]:
    return CAL.wilson(k, n, z) if n > 0 else (math.nan, math.nan)


def effective_n(values: np.ndarray, week_ids: np.ndarray, n: int) -> float:
    """n divided by the week-cluster design effect of the 0/1 indicator: a slice full of one week's calls is not n independent tests.
    `week_ids` are the frame's precomputed week codes (SymFrame column `wk`)."""
    if n < 4:
        return float(n)
    return float(n / design_effect(np.asarray(values, float), np.asarray(week_ids)))


def two_prop_p(k1: float, n1: float, k2: float, n2: float, alternative: str = "less") -> float:
    """Pooled two-proportion z-test p-value on (possibly design-effect-reduced) counts. alternative: 'less' (group 1 rate lower),
    'greater' (higher), 'two-sided'. Returns 1.0 when either group is empty."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(max(pool * (1 - pool) * (1 / n1 + 1 / n2), 1e-12))
    z = (p1 - p2) / se
    if alternative == "less":
        return float(sps.norm.cdf(z))
    if alternative == "greater":
        return float(sps.norm.sf(z))
    return float(2 * sps.norm.sf(abs(z)))


def mean_diff_p(a: np.ndarray, b: np.ndarray, alternative: str = "less") -> float:
    """Welch t-test p-value for mean(a) vs mean(b); 1.0 when a group has fewer than 3 values."""
    if len(a) < 3 or len(b) < 3:
        return 1.0
    t = sps.ttest_ind(a, b, equal_var=False, alternative=alternative)
    return float(t.pvalue) if math.isfinite(t.pvalue) else 1.0


def bh(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """BH q-values (engine.pattern_stats.bh_qvalues) that tolerates an empty list and non-finite entries (treated as p = 1)."""
    arr = np.array([1.0 if not math.isfinite(x) else x for x in p], float)
    return pattern_stats.bh_qvalues(arr) if len(arr) else arr


# ==================================================================================================================
# missed winners and missed losers, per pattern
# ==================================================================================================================
@dataclass(frozen=True)
class MissBreakdown:
    """Movers the pattern did not catch, by why. `winner_*` counts rows that turned out winners, `loser_*` losers."""
    winners_total: int
    losers_total: int
    winner_caught: int
    loser_caught: int
    winner_silent: int
    loser_silent: int
    winner_wrong_way: int
    loser_wrong_way: int
    winner_near_miss: int
    loser_near_miss: int
    winner_suppressed: int
    loser_suppressed: int

    @property
    def winner_recall(self) -> float:
        return self.winner_caught / self.winners_total if self.winners_total else math.nan

    @property
    def loser_recall(self) -> float:
        return self.loser_caught / self.losers_total if self.losers_total else math.nan

    @property
    def missed_winners(self) -> int:
        return self.winners_total - self.winner_caught

    @property
    def missed_losers(self) -> int:
        return self.losers_total - self.loser_caught

    @property
    def recall_gap(self) -> float:
        """winner recall - loser recall: a pattern that only ever sees winners has a large positive gap."""
        return self.winner_recall - self.loser_recall if math.isfinite(self.winner_recall) and math.isfinite(self.loser_recall) else math.nan


def miss_kinds(f: pd.DataFrame) -> pd.Series:
    """Per-row MissKind (as strings) for rows that are movers the pattern did NOT catch in the right direction; '' elsewhere.
    A row is caught if the pattern called it in the direction it went. Order of explanation: wrong way, near miss, suppressed, silent."""
    caught = (f["winner"] & (f["call"] > 0)) | (f["loser"] & (f["call"] < 0))
    mover = f["winner"] | f["loser"]
    wrong = f["wrong_way"]
    near = (~f["fired"]) & f["near"]
    supp = (~f["fired"]) & (f["high_vol"] | f["low_liq"] | f["external_event"])
    kind = np.select([wrong, near, supp], [MissKind.WRONG_WAY.value, MissKind.NEAR_MISS.value, MissKind.SUPPRESSED.value], MissKind.SILENT.value)
    return pd.Series(np.where(mover & ~caught, kind, ""), index=f.index)


def miss_breakdown(f: pd.DataFrame) -> MissBreakdown:
    k = miss_kinds(f)
    w, l = f["winner"].to_numpy(), f["loser"].to_numpy()

    def cnt(kind: MissKind, mover: np.ndarray) -> int:
        return int(((k == kind.value).to_numpy() & mover).sum())
    return MissBreakdown(int(w.sum()), int(l.sum()), int((w & (f["call"] > 0).to_numpy()).sum()), int((l & (f["call"] < 0).to_numpy()).sum()),
                         cnt(MissKind.SILENT, w), cnt(MissKind.SILENT, l), cnt(MissKind.WRONG_WAY, w), cnt(MissKind.WRONG_WAY, l),
                         cnt(MissKind.NEAR_MISS, w), cnt(MissKind.NEAR_MISS, l), cnt(MissKind.SUPPRESSED, w), cnt(MissKind.SUPPRESSED, l))


def move_category(row: pd.Series) -> MoveCategory:
    """The section-4 category of one missed mover, so misses land in the same vocabulary the market-wide observer uses."""
    if row["wrong_way"]:
        return MoveCategory.FALSE_POSITIVE
    if row["near"] and not row["fired"]:
        return MoveCategory.NEAR_MISS
    if row["winner"]:
        return MoveCategory.EXTREME_UP if row["fwd"] >= 2 * WINNER else MoveCategory.WINNER
    return MoveCategory.EXTREME_DOWN if row["fwd"] <= -2 * WINNER else MoveCategory.LOSER


def top_missed(f: pd.DataFrame, n: int = 10, kind: str = "winner") -> pd.DataFrame:
    """The n largest movers of one sign that the pattern failed to call in that direction (research side: keeps ticker and date)."""
    k = miss_kinds(f)
    m = f[(k != "") & (f["winner"] if kind == "winner" else f["loser"])].copy()
    if m.empty:
        return m
    m["miss_kind"] = k[m.index]
    m["category"] = [move_category(r).value for _, r in m.iterrows()]
    return m.reindex(m["fwd"].abs().sort_values(ascending=False).index).head(n)[["pattern_id", "date", "ticker", "fwd", "call", "miss_kind", "category"]]


def orphan_movers(sf: SymFrame) -> dict[str, float]:
    """Movers that NO pattern called in the right direction on their (date, ticker): the part of the mover universe the whole pattern
    library is blind to. Their share is the ceiling on what more patterns of the same kind could still add."""
    f = sf.frame
    if f.empty:
        return {"movers": 0, "orphans": 0, "orphan_share": math.nan, "winner_orphan_share": math.nan, "loser_orphan_share": math.nan}
    g = f.assign(right=((f["winner"] & (f["call"] > 0)) | (f["loser"] & (f["call"] < 0)))).groupby(["date", "ticker"]).agg(
        winner=("winner", "max"), loser=("loser", "max"), right=("right", "max"))
    mov = g[g["winner"] | g["loser"]]
    orph = mov[~mov["right"]]

    def share(mask_mov: pd.Series, mask_or: pd.Series) -> float:
        return float(mask_or.sum() / mask_mov.sum()) if mask_mov.sum() else math.nan
    return {"movers": int(len(mov)), "orphans": int(len(orph)), "orphan_share": float(len(orph) / len(mov)) if len(mov) else math.nan,
            "winner_orphan_share": share(mov["winner"], orph["winner"]), "loser_orphan_share": share(mov["loser"], orph["loser"])}


def union_recall(sf: SymFrame) -> pd.DataFrame:
    """How much of the mover universe the library catches as patterns are added, best pattern first (greedy set cover on winners and
    losers separately): shows redundancy and the incremental value of each pattern's catches."""
    f = sf.frame
    rows = []
    if f.empty:
        return pd.DataFrame(columns=["pattern_id", "new_winners", "new_losers", "cum_winner_recall", "cum_loser_recall"])
    key = f["date"].astype(str) + "|" + f["ticker"].astype(str)
    wl, ll = set(key[f["winner"]]), set(key[f["loser"]])
    got_w = {p: set(key[(f["pattern_id"] == p) & f["winner"] & (f["call"] > 0)]) for p in sf.patterns()}
    got_l = {p: set(key[(f["pattern_id"] == p) & f["loser"] & (f["call"] < 0)]) for p in sf.patterns()}
    cov_w: set[str] = set()
    cov_l: set[str] = set()
    left = set(sf.patterns())
    while left:
        best = max(sorted(left), key=lambda p: (len(got_w[p] - cov_w) + len(got_l[p] - cov_l), p))
        nw, nl = got_w[best] - cov_w, got_l[best] - cov_l
        cov_w |= nw
        cov_l |= nl
        left.discard(best)
        rows.append(dict(pattern_id=best, new_winners=len(nw), new_losers=len(nl), cum_winner_recall=len(cov_w) / len(wl) if wl else math.nan,
                         cum_loser_recall=len(cov_l) / len(ll) if ll else math.nan))
    return pd.DataFrame(rows)


# ==================================================================================================================
# case slices: how a pattern behaves where patterns break
# ==================================================================================================================
def slice_mask(f: pd.DataFrame, kind: CaseKind) -> np.ndarray:
    """Rows in which a case kind applies. For the two outcome-defined kinds the mask is over FIRED rows and is itself the failure."""
    n = len(f)
    if kind == CaseKind.WRONG_DIRECTION:
        return f["wrong_way"].to_numpy()
    if kind == CaseKind.LARGE_LOSS:
        return f["big_loss"].to_numpy()
    if kind == CaseKind.REGIME_TRANSITION:
        return f["regime_transition"].to_numpy(bool)
    if kind == CaseKind.EXTERNAL_EVENT:
        return f["external_event"].to_numpy(bool)
    if kind == CaseKind.NEAR_MISS:
        return f["near"].to_numpy(bool)
    if kind == CaseKind.HIGH_VOL_FAILURE:
        return f["high_vol"].to_numpy(bool)
    if kind == CaseKind.LOW_LIQUIDITY_FAILURE:
        return f["low_liq"].to_numpy(bool)
    if kind.value.startswith("EPISODE_"):
        if "mover_outcome" not in f:
            return np.zeros(n, bool)
        return (f["mover_outcome"].astype(str).str.lower() == kind.value[len("EPISODE_"):].lower()).to_numpy()
    raise ValueError(f"unknown case kind {kind}")


def slice_available(f: pd.DataFrame, kind: CaseKind) -> bool:
    """A context that was never recorded is UNTESTED, not 'fine': the mask being all-False must not read as good behaviour."""
    need = {CaseKind.REGIME_TRANSITION: "regime_transition", CaseKind.EXTERNAL_EVENT: "external_event", CaseKind.NEAR_MISS: "margin",
            CaseKind.HIGH_VOL_FAILURE: "vol", CaseKind.LOW_LIQUIDITY_FAILURE: "dollar_volume"}
    if kind in OUTCOME_DEFINED:
        return True
    if kind.value.startswith("EPISODE_"):
        return "mover_outcome" in f and f["mover_outcome"].notna().any()
    col = need[kind]
    if kind in (CaseKind.REGIME_TRANSITION, CaseKind.EXTERNAL_EVENT):
        return bool(f[col].any())
    return col in f and f[col].notna().any()


@dataclass(frozen=True)
class SliceBehavior:
    """A pattern's behaviour on one case kind, compared with its behaviour everywhere else."""
    pattern_id: str
    kind: CaseKind
    n_rows: int
    n_calls: int
    hit_rate: float
    hit_lo: float
    hit_hi: float
    outside_hit_rate: float
    mean_dir_ret: float
    outside_mean_dir_ret: float
    large_loss_rate: float
    large_loss_lo: float
    worst: float
    wrong_way_rate: float
    p_worse: float                     # one-sided p that hit rate is lower inside than outside (design-effect adjusted)
    q: float = math.nan                # BH q over all (pattern, kind) tests in the run
    status: SliceStatus = SliceStatus.UNTESTED
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["kind"], d["status"] = self.kind.value, self.status.value
        return d


def outcome_slice(f: pd.DataFrame, kind: CaseKind, cfg: SymmetryConfig, pattern_id: str) -> SliceBehavior:
    """WRONG_DIRECTION / LARGE_LOSS: the rate of the failure among the pattern's decisive calls, with the limit the config allows."""
    calls = f[f["fired"]]
    dec = calls[calls["winner"] | calls["loser"]] if kind == CaseKind.WRONG_DIRECTION else calls
    n = len(dec)
    nan = math.nan
    if n == 0:
        return SliceBehavior(pattern_id, kind, len(f), 0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 1.0, nan, SliceStatus.UNTESTED, "no calls")
    fail = dec["wrong_way"] if kind == CaseKind.WRONG_DIRECTION else dec["big_loss"]
    k, rate = float(fail.sum()), float(fail.mean())
    lo, hi = _wilson(k, n, cfg.z)
    limit = cfg.max_wrong_direction_rate if kind == CaseKind.WRONG_DIRECTION else cfg.max_large_loss_rate
    ne = effective_n(fail.to_numpy(), dec["wk"].to_numpy(), n)
    p_excess = float(sps.binom.sf(int(round(rate * ne)) - 1, max(int(round(ne)), 1), limit)) if ne >= 1 else 1.0
    hit = float(dec["hit"].mean())
    if n < cfg.min_n:
        status, reason = SliceStatus.UNTESTED, f"only {n} calls"
    elif lo > limit:
        status, reason = SliceStatus.FAILS, f"{kind.value} rate {rate:.3f} (lower bound {lo:.3f}) exceeds the {limit:.3f} limit"
    elif rate > limit:
        status, reason = SliceStatus.DEGRADED, f"{kind.value} rate {rate:.3f} above the {limit:.3f} limit but not provably"
    else:
        status, reason = SliceStatus.OK, ""
    return SliceBehavior(pattern_id, kind, len(f), n, hit, lo if kind == CaseKind.WRONG_DIRECTION else nan, hi if kind == CaseKind.WRONG_DIRECTION else nan,
                         nan, float(dec["dir_ret"].mean()), nan, rate if kind == CaseKind.LARGE_LOSS else float(calls["big_loss"].mean()),
                         lo if kind == CaseKind.LARGE_LOSS else nan, float(dec["dir_ret"].min()),
                         rate if kind == CaseKind.WRONG_DIRECTION else float(calls["wrong_way"].sum() / max(len(calls), 1)), p_excess, nan, status, reason)


def context_slice(f: pd.DataFrame, kind: CaseKind, cfg: SymmetryConfig, pattern_id: str) -> SliceBehavior:
    """Context kinds: the pattern's fired rows INSIDE the context versus OUTSIDE it. Worse inside + significant = FAILS."""
    calls = f[f["fired"]]
    nan = math.nan
    if not slice_available(f, kind):
        return SliceBehavior(pattern_id, kind, len(f), 0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 1.0, nan, SliceStatus.UNTESTED,
                             "context never recorded")
    mask = slice_mask(calls, kind)
    ins, out = calls[mask], calls[~mask]
    n, n_out = len(ins), len(out)
    if n == 0:
        return SliceBehavior(pattern_id, kind, int(slice_mask(f, kind).sum()), 0, nan, nan, nan, float(out["hit"].mean()) if n_out else nan, nan, nan, nan, nan,
                             nan, nan, 1.0, nan, SliceStatus.UNTESTED, "pattern never fired here")
    hit, hit_out = float(ins["hit"].mean()), (float(out["hit"].mean()) if n_out else nan)
    lo, hi = _wilson(float(ins["hit"].sum()), n, cfg.z)
    ne_in = effective_n(ins["hit"].to_numpy(float), ins["wk"].to_numpy(), n)
    ne_out = effective_n(out["hit"].to_numpy(float), out["wk"].to_numpy(), n_out) if n_out else 0.0
    p_worse = two_prop_p(hit * ne_in, ne_in, (hit_out if n_out else 0) * ne_out, ne_out, "less") if n_out else 1.0
    p_ret = mean_diff_p(ins["dir_ret"].to_numpy(), out["dir_ret"].to_numpy(), "less") if n_out else 1.0
    p = min(1.0, 2 * min(p_worse, p_ret))                      # either lens can show harm; Bonferroni over the two
    ll = float(ins["big_loss"].mean())
    ll_lo = _wilson(float(ins["big_loss"].sum()), n, cfg.z)[0]
    return SliceBehavior(pattern_id, kind, int(slice_mask(f, kind).sum()), n, hit, lo, hi, hit_out, float(ins["dir_ret"].mean()),
                         float(out["dir_ret"].mean()) if n_out else nan, ll, ll_lo, float(ins["dir_ret"].min()), float(ins["wrong_way"].mean()), p)


def judge_context_slice(s: SliceBehavior, cfg: SymmetryConfig) -> SliceBehavior:
    """Attach the status once the run-wide q-value is known."""
    if s.status != SliceStatus.UNTESTED or s.n_calls == 0:
        return s
    if s.n_calls < cfg.min_n:
        return dataclasses.replace(s, status=SliceStatus.UNTESTED, reason=f"only {s.n_calls} calls in this context")
    worse_hit = math.isfinite(s.outside_hit_rate) and s.hit_rate < s.outside_hit_rate - cfg.slice_drop
    worse_ret = math.isfinite(s.outside_mean_dir_ret) and s.mean_dir_ret < s.outside_mean_dir_ret - cfg.tol_return
    loss_breach = math.isfinite(s.large_loss_lo) and s.large_loss_lo > cfg.max_large_loss_rate
    if loss_breach:
        return dataclasses.replace(s, status=SliceStatus.FAILS, reason=f"large-loss rate {s.large_loss_rate:.3f} (lower bound {s.large_loss_lo:.3f}) "
                                                                        f"breaches {cfg.max_large_loss_rate:.3f}")
    if (worse_hit or worse_ret) and math.isfinite(s.q) and s.q <= cfg.fails_q:
        return dataclasses.replace(s, status=SliceStatus.FAILS, reason=f"significantly worse inside (hit {s.hit_rate:.3f} vs {s.outside_hit_rate:.3f}, q {s.q:.3f})")
    if worse_hit or worse_ret or (s.mean_dir_ret < -cfg.tol_return):
        return dataclasses.replace(s, status=SliceStatus.DEGRADED, reason=f"worse inside (hit {s.hit_rate:.3f} vs {s.outside_hit_rate:.3f}) but not provable")
    return dataclasses.replace(s, status=SliceStatus.OK, reason="")


def pattern_slices(f: pd.DataFrame, cfg: SymmetryConfig, pattern_id: str, kinds: Sequence[CaseKind] | None = None) -> list[SliceBehavior]:
    """All case slices of one pattern; q-values are assigned over this pattern's own tests (the run-wide correction is applied later)."""
    kinds = list(kinds) if kinds is not None else list(CaseKind)
    raw = [outcome_slice(f, k, cfg, pattern_id) if k in OUTCOME_DEFINED else context_slice(f, k, cfg, pattern_id) for k in kinds]
    return finalize_slices(raw, cfg)


def finalize_slices(raw: Sequence[SliceBehavior], cfg: SymmetryConfig, q_values: Sequence[float] | None = None) -> list[SliceBehavior]:
    """Set q and status. Outcome-defined kinds already carry a status; context kinds are judged with q (given, or BH over `raw`)."""
    ctx_idx = [i for i, s in enumerate(raw) if s.kind not in OUTCOME_DEFINED]
    q = list(q_values) if q_values is not None else list(bh([raw[i].p_worse for i in ctx_idx]))
    out = list(raw)
    for j, i in enumerate(ctx_idx):
        out[i] = judge_context_slice(dataclasses.replace(raw[i], q=float(q[j])), cfg)
    return out


# ==================================================================================================================
# near misses: was the firing threshold in the right place?
# ==================================================================================================================
@dataclass(frozen=True)
class NearMissAnalysis:
    """Rows close to the firing threshold on both sides. If unfired near-misses would have done as well as the marginal fired calls, the
    threshold is too strict (winners are being thrown away); if far worse, it is well placed."""
    pattern_id: str
    n_unfired_near: int
    n_fired_near: int
    would_hit_unfired: float          # directional hit rate the unfired near-misses would have had, using their `lean`
    hit_fired_near: float
    hit_fired_far: float
    would_mean_ret: float
    fired_near_mean_ret: float
    p_equal: float                    # two-sided: unfired-near vs fired-near hit rate
    verdict: str                      # STRICT / WELL_PLACED / LOOSE / UNTESTED


def near_miss_analysis(f: pd.DataFrame, cfg: SymmetryConfig, pattern_id: str) -> NearMissAnalysis:
    nan = math.nan
    if "margin" not in f or "lean" not in f or not f["near"].any():
        return NearMissAnalysis(pattern_id, 0, 0, nan, nan, nan, nan, nan, 1.0, "UNTESTED")
    lean = f["lean"].fillna(0)
    near = f["near"].to_numpy(bool)
    un = f[near & ~f["fired"].to_numpy() & (lean.to_numpy() != 0)]
    fired_near = f[near & f["fired"].to_numpy()]
    fired_far = f[~near & f["fired"].to_numpy()]
    if len(un) < cfg.min_n or len(fired_near) < cfg.min_n:
        return NearMissAnalysis(pattern_id, len(un), len(fired_near), nan, nan, nan, nan, nan, 1.0, "UNTESTED")
    would = (un["lean"] * un["fwd"])
    w_hit, fn_hit = float((would > 0).mean()), float(fired_near["hit"].mean())
    far_hit = float(fired_far["hit"].mean()) if len(fired_far) else nan
    p = two_prop_p(w_hit * len(un), len(un), fn_hit * len(fired_near), len(fired_near), "two-sided")
    if p > 0.05:
        verdict = "STRICT" if w_hit >= 0.5 + (fn_hit - 0.5) * 0.5 else "WELL_PLACED"
    else:
        verdict = "STRICT" if w_hit > fn_hit else "LOOSE"
    return NearMissAnalysis(pattern_id, len(un), len(fired_near), w_hit, fn_hit, far_hit, float(would.mean()), float(fired_near["dir_ret"].mean()), p, verdict)


# ==================================================================================================================
# error symmetry: does finding winners come with picking losers?
# ==================================================================================================================
@dataclass(frozen=True)
class ErrorSymmetry:
    """Adverse-mover rates: P(the stock was a LOSER | pattern said winner) and P(WINNER | pattern said loser), against the universe base
    rates. A pattern that finds winners but also lands on disasters shows up here even if its precision looks fine."""
    long_calls: int
    short_calls: int
    long_adverse_rate: float              # P(loser | said winner)
    long_adverse_base: float              # P(loser) over the pattern's universe
    long_adverse_ratio: float
    long_adverse_p: float                 # one-sided: adverse rate > base
    short_adverse_rate: float
    short_adverse_base: float
    short_adverse_ratio: float
    short_adverse_p: float
    long_expectancy: float                # mean return of the long calls
    short_expectancy: float               # mean of side*fwd for short calls (positive = shorts earned)
    winner_loser_recall_gap: float

    def adverse_ok(self, cfg: SymmetryConfig) -> bool | None:
        checks = []
        for calls, ratio, p in ((self.long_calls, self.long_adverse_ratio, self.long_adverse_p), (self.short_calls, self.short_adverse_ratio, self.short_adverse_p)):
            if calls >= cfg.min_n and math.isfinite(ratio):
                checks.append(not (ratio > cfg.adverse_tolerance and p < 0.10))
        return all(checks) if checks else None


def error_symmetry(f: pd.DataFrame, cfg: SymmetryConfig) -> ErrorSymmetry:
    nan = math.nan
    longs, shorts = f[f["call"] > 0], f[f["call"] < 0]
    base_l, base_w = float(f["loser"].mean()) if len(f) else nan, float(f["winner"].mean()) if len(f) else nan

    def adverse(calls: pd.DataFrame, col: str, base: float) -> tuple[float, float, float]:
        if len(calls) == 0 or not math.isfinite(base) or base <= 0:
            return nan, nan, 1.0
        k = float(calls[col].sum())
        ne = effective_n(calls[col].to_numpy(float), calls["wk"].to_numpy(), len(calls))
        rate = k / len(calls)
        return rate, rate / base, float(sps.binom.sf(int(round(rate * ne)) - 1, max(int(round(ne)), 1), base))
    lr, lratio, lp = adverse(longs, "loser", base_l)
    sr, sratio, sp = adverse(shorts, "winner", base_w)
    mb = miss_breakdown(f)
    return ErrorSymmetry(len(longs), len(shorts), lr, base_l, lratio, lp, sr, base_w, sratio, sp, float(longs["fwd"].mean()) if len(longs) else nan,
                         float(shorts["dir_ret"].mean()) if len(shorts) else nan, mb.recall_gap)


# ==================================================================================================================
# the per-pattern symmetry report and the trust gate
# ==================================================================================================================
@dataclass(frozen=True)
class TrustDecision:
    verdict: Trust
    passed: tuple[str, ...]
    failed: tuple[str, ...]
    untested: tuple[str, ...]

    @property
    def reasons(self) -> str:
        return "; ".join(list(self.failed) + [f"{u} untested" for u in self.untested])


@dataclass(frozen=True)
class PatternSymmetry:
    pattern_id: str
    n_rows: int
    n_calls: int
    winner: Confusion
    loser: Confusion
    matrix: dict[str, dict[str, int]]
    misses: MissBreakdown
    errors: ErrorSymmetry
    slices: tuple[SliceBehavior, ...]
    near_miss: NearMissAnalysis
    hit_rate: float
    hit_lo: float
    edge_p: float                      # one-sided: precision in the pattern's called direction beats the universe rate
    trust: TrustDecision
    matured_through: str

    def slice_of(self, kind: CaseKind) -> SliceBehavior:
        for s in self.slices:
            if s.kind == kind:
                return s
        raise KeyError(kind)

    def failing_slices(self) -> list[SliceBehavior]:
        return [s for s in self.slices if s.status == SliceStatus.FAILS]


def _edge_p(f: pd.DataFrame) -> float:
    """One-sided p that the pattern's calls hit more often than a coin weighted by the universe's own direction split."""
    calls = f[f["fired"]]
    dec = calls[calls["winner"] | calls["loser"]]
    if len(dec) < 5:
        return 1.0
    long_share = float((dec["call"] > 0).mean())
    win_share = float(f[f["winner"] | f["loser"]]["winner"].mean()) if (f["winner"] | f["loser"]).any() else 0.5
    p0 = long_share * win_share + (1 - long_share) * (1 - win_share)          # accuracy of guessing with the pattern's own side mix
    ne = effective_n(dec["hit"].to_numpy(float), dec["wk"].to_numpy(), len(dec))
    return float(sps.binom.sf(int(round(float(dec["hit"].mean()) * ne)) - 1, max(int(round(ne)), 1), p0))


def trust_decision(n_calls: int, edge_p: float, hit_lo: float, errors: ErrorSymmetry, slices: Sequence[SliceBehavior], misses: MissBreakdown,
                   cfg: SymmetryConfig) -> TrustDecision:
    """A pattern is TRUSTED only when winners AND losers behaviour are acceptable. Winning precision alone can never earn it."""
    passed: list[str] = []
    failed: list[str] = []
    untested: list[str] = []

    def rec(name: str, ok: bool | None) -> None:
        (untested if ok is None else passed if ok else failed).append(name)
    if n_calls < cfg.min_calls:
        return TrustDecision(Trust.UNKNOWN, (), (), (f"only {n_calls} calls (need {cfg.min_calls})",))
    rec("finds_movers_better_than_chance", edge_p < 0.05 and hit_lo > 0.5)
    rec("adverse_movers_not_over_represented", errors.adverse_ok(cfg))
    for kind in (CaseKind.WRONG_DIRECTION, CaseKind.LARGE_LOSS):
        s = next(s for s in slices if s.kind == kind)
        rec(f"{kind.value.lower()}_within_limit", None if s.status == SliceStatus.UNTESTED else s.status != SliceStatus.FAILS and s.status != SliceStatus.DEGRADED)
    ctx = [s for s in slices if s.kind not in OUTCOME_DEFINED]
    for s in ctx:
        if s.status == SliceStatus.FAILS:
            failed.append(f"fails_in_{s.kind.value.lower()}")
    tested_ctx = [s for s in ctx if s.status != SliceStatus.UNTESTED]
    if len(tested_ctx) < 3:
        untested.append("fewer than three failure contexts could be tested")
    elif not any(s.status == SliceStatus.FAILS for s in ctx):
        passed.append("holds_in_failure_contexts")
    if errors.long_calls >= cfg.min_n and errors.short_calls >= cfg.min_n:
        rec("both_sides_positive_expectancy", errors.long_expectancy > 0 and errors.short_expectancy > 0)
    if failed:
        return TrustDecision(Trust.NOT_TRUSTED, tuple(passed), tuple(failed), tuple(untested))
    if untested:
        return TrustDecision(Trust.UNKNOWN, tuple(passed), (), tuple(untested))
    return TrustDecision(Trust.TRUSTED, tuple(passed), (), ())


def analyse_pattern(sf: SymFrame, pattern_id: str, kinds: Sequence[CaseKind] | None = None, *, slices: Sequence[SliceBehavior] | None = None) -> PatternSymmetry:
    """The full section-12 profile of one pattern. `slices` lets a caller supply run-wide-corrected slices (see analyse_library)."""
    cfg = sf.cfg
    f = sf.of(pattern_id)
    if f.empty:
        raise KeyError(f"no rows for pattern {pattern_id}")
    win, lose = confusion_pair(f)
    sl = tuple(slices) if slices is not None else tuple(pattern_slices(f, cfg, pattern_id, kinds))
    calls = f[f["fired"]]
    dec_hit = float(calls["hit"].mean()) if len(calls) else math.nan
    hit_lo = _wilson(float(calls["hit"].sum()), len(calls), cfg.z)[0] if len(calls) else math.nan
    errors = error_symmetry(f, cfg)
    misses = miss_breakdown(f)
    ep = _edge_p(f)
    td = trust_decision(len(calls), ep, hit_lo, errors, sl, misses, cfg)
    mat = three_class_matrix(f)
    return PatternSymmetry(pattern_id, len(f), len(calls), win, lose, {str(i): {str(c): int(mat.loc[i, c]) for c in mat.columns} for i in mat.index}, misses,
                           errors, sl, near_miss_analysis(f, cfg, pattern_id), dec_hit, hit_lo, ep, td, str(f["matured_at"].max().date()))


@dataclass(frozen=True)
class LibrarySymmetry:
    """Symmetry of every pattern, with run-wide multiple-testing control across all (pattern, case) tests."""
    patterns: tuple[PatternSymmetry, ...]
    n_tests: int
    orphans: dict[str, float]
    config_digest: str
    data_digest: str

    def get(self, pattern_id: str) -> PatternSymmetry:
        for p in self.patterns:
            if p.pattern_id == pattern_id:
                return p
        raise KeyError(pattern_id)

    def by_verdict(self, v: Trust) -> list[str]:
        return [p.pattern_id for p in self.patterns if p.trust.verdict == v]

    def table(self) -> pd.DataFrame:
        rows = []
        for p in self.patterns:
            rows.append(dict(pattern_id=p.pattern_id, calls=p.n_calls, hit=p.hit_rate, hit_lo=p.hit_lo, win_precision=p.winner.precision,
                             win_recall=p.winner.recall, lose_precision=p.loser.precision, lose_recall=p.loser.recall,
                             long_adverse_ratio=p.errors.long_adverse_ratio, short_adverse_ratio=p.errors.short_adverse_ratio,
                             failing="|".join(s.kind.value for s in p.failing_slices()), trust=p.trust.verdict.value))
        return pd.DataFrame(rows)


def analyse_library(sf: SymFrame, kinds: Sequence[CaseKind] | None = None, ledger: TestLedger | None = None, now=None) -> LibrarySymmetry:
    """Analyse every pattern. All context-slice p-values of ALL patterns are corrected together (BH), then each pattern is judged; if a
    ledger and `now` are given, each test is also written to the cumulative ledger so later sweeps pay for it."""
    cfg = sf.cfg
    kinds = list(kinds) if kinds is not None else list(CaseKind)
    per: dict[str, list[SliceBehavior]] = {}
    for pid in sf.patterns():
        f = sf.of(pid)
        per[pid] = [outcome_slice(f, k, cfg, pid) if k in OUTCOME_DEFINED else context_slice(f, k, cfg, pid) for k in kinds]
    flat = [(pid, i) for pid, ss in per.items() for i, s in enumerate(ss) if s.kind not in OUTCOME_DEFINED and s.n_calls > 0]
    qs = bh([per[pid][i].p_worse for pid, i in flat])
    qmap = {(pid, i): float(q) for (pid, i), q in zip(flat, qs)}
    out = []
    for pid, raw in per.items():
        fixed = []
        for i, s in enumerate(raw):
            if s.kind in OUTCOME_DEFINED:
                fixed.append(s)
            else:
                fixed.append(judge_context_slice(dataclasses.replace(s, q=qmap.get((pid, i), math.nan)), cfg) if s.n_calls > 0 else s)
        out.append(analyse_pattern(sf, pid, kinds, slices=fixed))
        if ledger is not None and now is not None:
            newest = sf.of(pid)["matured_at"].max()
            for s in fixed:
                if s.n_calls > 0 and math.isfinite(s.p_worse):
                    ledger.log(f"sym|{pid}|{s.kind.value}|{sf.digest()}", "symmetry", s.p_worse, newest, now)
    return LibrarySymmetry(tuple(out), len(flat), orphan_movers(sf), cfg.digest(), sf.digest())


def symmetry_by_era(sf: SymFrame, pattern_id: str) -> pd.DataFrame:
    """Winner and loser precision/recall per era for one pattern: does the symmetry hold every year, or only in the best one?"""
    f = sf.of(pattern_id)
    rows = []
    for era, g in f.groupby("era"):
        win, lose = confusion_pair(g)
        calls = g[g["fired"]]
        rows.append(dict(era=era, rows=len(g), calls=len(calls), win_precision=win.precision, win_recall=win.recall, lose_precision=lose.precision,
                         lose_recall=lose.recall, hit=float(calls["hit"].mean()) if len(calls) else math.nan,
                         big_loss_rate=float(calls["big_loss"].mean()) if len(calls) else math.nan))
    return pd.DataFrame(rows)


def era_consistency(sf: SymFrame, pattern_id: str, min_calls: int = 20) -> dict[str, float]:
    """How many eras with enough calls have a hit rate above 0.5 and a large-loss rate under the limit; the pattern's per-era hit spread."""
    t = symmetry_by_era(sf, pattern_id)
    t = t[t["calls"] >= min_calls]
    if t.empty:
        return {"eras": 0, "share_good_hit": math.nan, "share_loss_ok": math.nan, "hit_spread": math.nan}
    return {"eras": int(len(t)), "share_good_hit": float((t["hit"] > 0.5).mean()), "share_loss_ok": float((t["big_loss_rate"] <= sf.cfg.max_large_loss_rate).mean()),
            "hit_spread": float(t["hit"].max() - t["hit"].min())}


# ==================================================================================================================
# the LOSS-RISK knowledge bank (separate from the opportunity bank)
# ==================================================================================================================
LOSS_ALLOWED_EFFECTS = frozenset({DecisionEffect.POSITION_SIZE, DecisionEffect.ABSTENTION, DecisionEffect.STOP, DecisionEffect.EXIT,
                                  DecisionEffect.TIMING, DecisionEffect.CONFIDENCE, DecisionEffect.RESEARCH_PRIORITY, DecisionEffect.SELECTION})
LOSS_FORBIDDEN_EFFECTS = frozenset({DecisionEffect.RANKING, DecisionEffect.DIRECTION, DecisionEffect.PATTERN_WEIGHTING})
CONTEXT_DIMENSIONS = ("vol", "liquidity", "gap", "regime", "transition", "event", "sector", "episode", "margin_zone", "side")
EPISTEMIC_WEIGHT = {Epistemic.SUPPORTED: 1.0, Epistemic.CONDITIONAL: 0.7, Epistemic.HYPOTHESIS: 0.3, Epistemic.OBSERVED: 0.15}


@dataclass(frozen=True)
class LossRiskItem:
    """One learned way of losing. Polarity is fixed: this type cannot describe an opportunity, and it carries no direction or ranking effect.
    `context` is identity-free (bucket names only - never a ticker, date or year)."""
    risk_id: str
    pattern_id: str                          # "*" = market-wide (across every pattern)
    kind: RiskKind
    context: tuple[tuple[str, str], ...]
    measure: str                             # "large_loss" or "wrong_way"
    n: int
    losses: int
    loss_rate: float
    base_rate: float
    relative_risk: float
    ci_lo: float
    ci_hi: float
    mean_loss: float                         # mean directional return of the LOSING calls in this context (negative)
    p: float
    q: float
    oos_confirmed: bool | None
    oos_rate: float
    oos_n: int
    epistemic: Epistemic
    lifecycle: Lifecycle
    effects: tuple[DecisionEffect, ...]
    cause: FailureCause
    matured_at: str
    provenance: Provenance
    polarity: str = "LOSS"
    notes: str = ""

    def validate(self) -> list[str]:
        errs = []
        if not self.risk_id.startswith("LR"):
            errs.append("risk_id must start with LR (loss-risk namespace)")
        if self.polarity != "LOSS":
            errs.append("a loss-risk item must have polarity LOSS")
        if not self.context:
            errs.append("empty context")
        if any(k not in CONTEXT_DIMENSIONS for k, _ in self.context):
            errs.append("context dimension outside CONTEXT_DIMENSIONS")
        bad = [e.value for e in self.effects if e in LOSS_FORBIDDEN_EFFECTS or e not in LOSS_ALLOWED_EFFECTS]
        if bad:
            errs.append(f"decision effects not allowed for a loss-risk item: {bad}")
        if not self.effects:
            errs.append("a loss-risk item must state what decision it changes")
        if not (0 <= self.loss_rate <= 1 and 0 <= self.base_rate <= 1):
            errs.append("rates outside [0, 1]")
        if self.n < self.losses or self.n <= 0:
            errs.append("n must exceed zero and cover the losses")
        if self.oos_confirmed is True and not self.oos_n:
            errs.append("oos_confirmed without an out-of-sample count")
        errs += self.provenance.check()
        return errs

    def matches(self, ctx: Mapping[str, str], pattern_id: str | None = None) -> bool:
        if self.pattern_id != "*" and pattern_id is not None and self.pattern_id != pattern_id:
            return False
        return all(ctx.get(k) == v for k, v in self.context)

    def weight(self) -> float:
        """How hard this item should push a decision toward caution: excess relative risk, discounted by how well it is established."""
        if not math.isfinite(self.relative_risk) or self.relative_risk <= 1:
            return 0.0
        w = min(0.9, (self.relative_risk - 1.0) / (self.relative_risk + 1.0))
        return w * EPISTEMIC_WEIGHT.get(self.epistemic, 0.0) * (1.0 if self.oos_confirmed else 0.5 if self.oos_confirmed is None else 0.25)

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["kind"], d["epistemic"], d["lifecycle"], d["cause"] = self.kind.value, self.epistemic.value, self.lifecycle.value, self.cause.value
        d["effects"] = [e.value for e in self.effects]
        d["context"] = [list(c) for c in self.context]
        d["provenance"] = dataclasses.asdict(self.provenance)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LossRiskItem":
        d = dict(d)
        d["kind"], d["epistemic"], d["lifecycle"], d["cause"] = RiskKind(d["kind"]), Epistemic(d["epistemic"]), Lifecycle(d["lifecycle"]), FailureCause(d["cause"])
        d["effects"] = tuple(DecisionEffect(e) for e in d["effects"])
        d["context"] = tuple((str(a), str(b)) for a, b in d["context"])
        pv = dict(d["provenance"])
        pv["sealed_windows"], pv["parents"] = tuple(pv.get("sealed_windows", ())), tuple(pv.get("parents", ()))
        d["provenance"] = Provenance(**pv)
        return cls(**d)


def make_risk_id(pattern_id: str, context: Sequence[tuple[str, str]], measure: str) -> str:
    return "LR" + stable_hash({"p": pattern_id, "c": sorted(context), "m": measure}, 12)


class LossRiskBank(_SharedLossBank):
    """The ONE loss-risk bank: it IS engine.research.loss_pipeline.LossRiskBank (append-only RiskEntry versions, gated by maturity, disjointness
    check) extended for categorical loss CONTEXTS. Each item is stored as a RiskEntry of kind 'context' (so loss_pipeline's own risk_score, which
    reads only kind 'condition', is untouched) and the extra fields a RiskEntry has no slot for (context tuple, effects, lifecycle, holdout result)
    live in `_items`, keyed by the same id. There is no second store of entries. An optional hash-chained jsonl log makes it resumable.
    Every write needs `now` and refuses evidence not strictly older than `now`; every read is a point-in-time read."""

    def __init__(self, path: str | Path | None = None):
        super().__init__()
        self.path = Path(path) if path else None
        self._items: dict[str, LossRiskItem] = {}
        self._log: list[dict[str, Any]] = []
        self._prev = "GENESIS"
        if self.path and self.path.exists():
            self._load()

    def __contains__(self, risk_id: str) -> bool:
        return risk_id in self._items

    def get(self, risk_id: str) -> LossRiskItem:
        return self._items[risk_id]

    @staticmethod
    def as_entry(it: LossRiskItem) -> _RiskEntry:
        """Map an item onto the shared RiskEntry: signal = the context string, sign +1, n_ctrl = exposed calls in the context."""
        sig = it.pattern_id + "|" + "&".join(f"{k}={v}" for k, v in it.context)
        val = it.oos_rate / it.base_rate if it.oos_confirmed and it.base_rate > 0 and math.isfinite(it.oos_rate) else None
        return _RiskEntry(it.risk_id, "context", sig, 1, it.cause.value, it.losses, it.n, it.loss_rate, it.base_rate, it.relative_risk, it.p, it.q, 0,
                          it.mean_loss, val, Epistemic.RETIRED if it.lifecycle == Lifecycle.RETIRED else it.epistemic, 0, it.matured_at,
                          it.provenance.code_hash)

    def _store(self, it: LossRiskItem) -> None:
        self.upsert(self.as_entry(it))
        self._items[it.risk_id] = it

    def _append(self, op: str, item: LossRiskItem, now, extra: Mapping[str, Any] | None = None) -> None:
        rec: dict[str, Any] = {"op": op, "at": str(as_date(now)), "item": item.as_dict(), "extra": dict(extra or {}), "prev": self._prev}
        rec["hash"] = stable_hash({k: v for k, v in rec.items() if k != "hash"}, 16)
        self._prev = rec["hash"]
        self._log.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + NL)

    def add(self, item: LossRiskItem, now) -> str:
        """Add or supersede. Refuses (FirewallBreach) evidence not strictly before `now`; ValueError on a malformed item. Returns 'added',
        'superseded' (newer evidence for the same id) or 'stale' (older or equal evidence: ignored, never overwriting)."""
        errs = item.validate()
        if errs:
            raise ValueError("invalid LossRiskItem: " + "; ".join(errs))
        require_past(item.matured_at, now, f"loss-risk item {item.risk_id}")
        old = self._items.get(item.risk_id)
        if old is not None and as_date(item.matured_at) <= as_date(old.matured_at):
            return "stale"
        self._store(item)
        self._append("supersede" if old is not None else "add", item, now)
        return "superseded" if old is not None else "added"

    def items(self, now, lifecycle: Iterable[Lifecycle] | None = None, pattern_id: str | None = None) -> list[LossRiskItem]:
        """Point-in-time listing: only items whose evidence matured strictly before `now`."""
        if now is None:
            raise FirewallBreach("loss-risk bank read without `now`")
        want = set(lifecycle) if lifecycle is not None else None
        out = []
        for it in self._items.values():
            if as_date(it.matured_at) >= as_date(now):
                continue
            if want is not None and it.lifecycle not in want:
                continue
            if pattern_id is not None and it.pattern_id not in (pattern_id, "*"):
                continue
            out.append(it)
        return sorted(out, key=lambda i: (-i.weight(), i.risk_id))

    def query(self, now, context: Mapping[str, str], pattern_id: str | None = None, min_weight: float = 0.0) -> list[LossRiskItem]:
        """Items that apply to a situation described by bucket names, ordered by weight. ACTIVE / GROWTH / PEAK / BIRTH items only."""
        live = (Lifecycle.ACTIVE, Lifecycle.GROWTH, Lifecycle.PEAK, Lifecycle.BIRTH)
        return [i for i in self.items(now, live, pattern_id) if i.matches(context, pattern_id) and i.weight() >= min_weight]

    def retire(self, risk_id: str, reason: str, now) -> LossRiskItem:
        """Retire (never delete): the item stays readable for audit but no longer answers queries."""
        it = self._items[risk_id]
        new = dataclasses.replace(it, lifecycle=Lifecycle.RETIRED, epistemic=Epistemic.RETIRED, notes=(it.notes + " | " if it.notes else "") + f"retired: {reason}")
        self._store(new)
        self._append("retire", new, now, {"reason": reason})
        return new

    def audit_log(self, risk_id: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self._log if risk_id is None or r["item"]["risk_id"] == risk_id]

    def verify_chain(self) -> bool:
        prev = "GENESIS"
        for r in self._log:
            if r["prev"] != prev or stable_hash({k: v for k, v in r.items() if k != "hash"}, 16) != r["hash"]:
                return False
            prev = r["hash"]
        return True

    def _load(self) -> None:
        assert self.path is not None                   # only called when a path was given
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["prev"] != self._prev or stable_hash({k: v for k, v in rec.items() if k != "hash"}, 16) != rec["hash"]:
                raise FirewallBreach(f"loss-risk bank file {self.path} fails its hash chain")
            self._prev = rec["hash"]
            self._log.append(rec)
            self._store(LossRiskItem.from_dict(rec["item"]))

    def summary(self, now) -> dict[str, Any]:
        its = self.items(now)
        by_kind: dict[str, int] = {}
        for i in its:
            by_kind[i.kind.value] = by_kind.get(i.kind.value, 0) + 1
        return {"items": len(its), "confirmed": sum(1 for i in its if i.oos_confirmed), "by_kind": by_kind,
                "retired": sum(1 for i in its if i.lifecycle == Lifecycle.RETIRED)}


def independence_audit(bank: LossRiskBank, opportunity_ids: Iterable[str] = (), opportunity_path: str | Path | None = None) -> list[str]:
    """Problems that would make the loss bank not independent of the opportunity bank. Empty list = independent."""
    problems = []
    opp = set(opportunity_ids)
    try:
        bank.assert_disjoint(opp)                       # loss_pipeline's own disjointness check, on the shared storage
    except FirewallBreach as e:
        problems.append(str(e))
    shared = opp & set(bank._items)
    if shared:
        problems.append(f"{len(shared)} ids appear in both banks")
    if opportunity_path is not None and bank.path is not None and Path(opportunity_path).resolve() == bank.path.resolve():
        problems.append("both banks write the same file")
    for it in bank._items.values():
        if it.polarity != "LOSS":
            problems.append(f"{it.risk_id}: polarity {it.polarity}")
        if any(e in LOSS_FORBIDDEN_EFFECTS for e in it.effects):
            problems.append(f"{it.risk_id}: carries a direction/ranking effect, which only the opportunity bank may have")
        if any(p in opp for p in it.provenance.parents):
            problems.append(f"{it.risk_id}: derived from an opportunity-bank item")
    return problems


def risk_multiplier(bank: LossRiskBank, now, context: Mapping[str, str], pattern_id: str | None = None) -> tuple[float, tuple[str, ...]]:
    """Position-size multiplier in (0, 1] from every matching item (noisy-or of their weights). 1.0 = no known reason for caution.
    The multiplier can only shrink; a loss item never raises size or rank."""
    hits = bank.query(now, context, pattern_id)
    mult = 1.0
    for h in hits:
        mult *= 1.0 - h.weight()
    return float(max(mult, 0.05)), tuple(h.risk_id for h in hits)


# ==================================================================================================================
# describing a row by identity-free buckets
# ==================================================================================================================
def descriptors(f: pd.DataFrame, cfg: SymmetryConfig, gap_thr: float = 0.05) -> dict[str, pd.Series]:
    """Bucket-name series for each context dimension the frame supports. Values are words like 'high' or 'transition': nothing here can
    identify a ticker or a date, so a bank item built from them is safe to hand to the trader side."""
    out: dict[str, pd.Series] = {}
    if f["high_vol"].any():
        out["vol"] = pd.Series(np.where(f["high_vol"], "high", "normal"), index=f.index)
    if f["low_liq"].any():
        out["liquidity"] = pd.Series(np.where(f["low_liq"], "low", "normal"), index=f.index)
    if "gap" in f and f["gap"].notna().any():
        out["gap"] = pd.Series(np.where(f["gap"].abs() >= gap_thr, "big", "small"), index=f.index)
    if "regime" in f and f["regime"].notna().any():
        out["regime"] = f["regime"].astype(str)
    if f["regime_transition"].any():
        out["transition"] = pd.Series(np.where(f["regime_transition"], "transition", "steady"), index=f.index)
    if f["external_event"].any():
        out["event"] = pd.Series(np.where(f["external_event"], "event", "none"), index=f.index)
    if "sector" in f and f["sector"].notna().any():
        out["sector"] = f["sector"].astype(str)
    if "mover_outcome" in f and f["mover_outcome"].notna().any():
        out["episode"] = f["mover_outcome"].astype(str).str.lower()
    if "margin" in f and f["margin"].notna().any():
        out["margin_zone"] = pd.Series(np.where(f["near"], "near", "far"), index=f.index)
    out["side"] = pd.Series(np.where(f["side"] > 0, "long", np.where(f["side"] < 0, "short", "none")), index=f.index)
    return out


def kind_for_context(context: Sequence[tuple[str, str]], measure: str) -> RiskKind:
    d = dict(context)
    if d.get("vol") == "high":
        return RiskKind.VOL_BLOWUP
    if d.get("liquidity") == "low":
        return RiskKind.LIQUIDITY_TRAP
    if d.get("gap") == "big":
        return RiskKind.GAP_RISK
    if d.get("transition") == "transition":
        return RiskKind.REGIME_SHIFT_RISK
    if d.get("event") == "event":
        return RiskKind.EVENT_RISK
    if d.get("episode") == "reversed":
        return RiskKind.REVERSAL_RISK
    return RiskKind.WRONG_DIRECTION_CONTEXT if measure == "wrong_way" else RiskKind.LARGE_LOSS_CONTEXT


def effects_for_kind(kind: RiskKind) -> tuple[DecisionEffect, ...]:
    """What decision a loss context should change: liquidity and gap risk shrink size, event risk abstains, volatility widens stops."""
    return {RiskKind.LIQUIDITY_TRAP: (DecisionEffect.POSITION_SIZE, DecisionEffect.ABSTENTION), RiskKind.GAP_RISK: (DecisionEffect.POSITION_SIZE,),
            RiskKind.EVENT_RISK: (DecisionEffect.ABSTENTION, DecisionEffect.TIMING), RiskKind.VOL_BLOWUP: (DecisionEffect.POSITION_SIZE, DecisionEffect.STOP),
            RiskKind.REGIME_SHIFT_RISK: (DecisionEffect.POSITION_SIZE, DecisionEffect.CONFIDENCE),
            RiskKind.REVERSAL_RISK: (DecisionEffect.EXIT, DecisionEffect.TIMING)}.get(kind, (DecisionEffect.POSITION_SIZE, DecisionEffect.CONFIDENCE))


# ==================================================================================================================
# mining loss contexts
# ==================================================================================================================
@dataclass(frozen=True)
class MiningConfig:
    q_max: float = 0.10                  # BH q needed in the discovery period for an item to exist at all
    min_ctx_n: int = 40                  # exposed rows in a context
    min_losses: int = 6
    min_rr: float = 1.5                  # relative risk worth recording
    holdout_frac: float = 0.4            # latest share of weeks kept for confirmation
    confirm_p: float = 0.10
    confirm_rr: float = 1.2
    max_pair_dims: int = 2               # contexts of one or two descriptors
    measures: tuple[str, ...] = ("large_loss", "wrong_way")
    per_pattern: bool = True             # also mine within each pattern (else only market-wide '*')
    seed: int = 0

    def validate(self) -> list[str]:
        errs = []
        if not (0 < self.q_max < 1) or not (0 < self.holdout_frac < 0.9):
            errs.append("q_max in (0, 1) and holdout_frac in (0, 0.9)")
        if self.min_ctx_n < 10 or self.min_losses < 2 or self.min_rr <= 1:
            errs.append("min_ctx_n >= 10, min_losses >= 2, min_rr > 1")
        if self.max_pair_dims not in (1, 2):
            errs.append("max_pair_dims must be 1 or 2")
        if any(m not in ("large_loss", "wrong_way") for m in self.measures) or not self.measures:
            errs.append("measures must be a non-empty subset of large_loss / wrong_way")
        return errs

    def check(self) -> "MiningConfig":
        errs = self.validate()
        if errs:
            raise ValueError("invalid MiningConfig: " + "; ".join(errs))
        return self


@dataclass(frozen=True)
class Candidate:
    pattern_id: str
    context: tuple[tuple[str, str], ...]
    measure: str
    n: int
    k: int
    n_out: int
    k_out: int
    rate: float
    base: float
    rr: float
    p_raw: float
    p: float = 1.0
    q: float = 1.0


def _exposed(f: pd.DataFrame, measure: str, large_loss: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows where the pattern actually took a position (fired), and the loss indicator for the measure. Hypothetical unfired lean rows are
    never counted as losses: a loss bank is built from calls that were made."""
    e = f[f["fired"]]
    if measure == "wrong_way":
        e = e[e["winner"] | e["loser"]]
        loss = ((e["side"] > 0) & e["loser"]) | ((e["side"] < 0) & e["winner"])
    else:
        loss = e["dir_ret"] <= -large_loss
    return e, loss.to_numpy(bool)


def _context_masks(desc: Mapping[str, pd.Series], index: pd.Index, max_dims: int) -> dict[tuple[tuple[str, str], ...], np.ndarray]:
    """Boolean mask for every single (dimension, value) and, if allowed, every pair from two different dimensions. 'none' / 'steady' / 'normal'
    style baseline values are still candidates: a benign bucket can be the protective side of a comparison."""
    singles: dict[tuple[str, str], np.ndarray] = {}
    for dim, s in desc.items():
        s = s.reindex(index)
        for val in s.dropna().unique():
            singles[(dim, str(val))] = (s == val).to_numpy()
    out: dict[tuple[tuple[str, str], ...], np.ndarray] = {(k,): m for k, m in singles.items()}
    if max_dims >= 2:
        keys = sorted(singles)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if a[0] == b[0]:
                    continue
                m = singles[a] & singles[b]
                if m.sum() > 0:
                    out[(a, b)] = m
    return out


def _score_candidate(pattern_id: str, ctx: tuple[tuple[str, str], ...], measure: str, mask: np.ndarray, loss: np.ndarray, mc: MiningConfig) -> Candidate | None:
    n, n_out = int(mask.sum()), int((~mask).sum())
    if n < mc.min_ctx_n or n_out < mc.min_ctx_n:
        return None
    k, k_out = int(loss[mask].sum()), int(loss[~mask].sum())
    rate, base = k / n, k_out / n_out
    rr = rate / base if base > 0 else (math.inf if rate > 0 else 1.0)
    if k < mc.min_losses or rr < mc.min_rr:
        return None
    p_raw = two_prop_p(k, n, k_out, n_out, "greater")
    return Candidate(pattern_id, ctx, measure, n, k, n_out, k_out, rate, base, rr, p_raw)


def _refine_candidate(c: Candidate, mask: np.ndarray, loss: np.ndarray, wk: np.ndarray) -> Candidate:
    """Adjust the raw p for week clustering (the design effect of the loss indicator) once a candidate survived the cheap screen."""
    ne_in = effective_n(loss[mask].astype(float), wk[mask], c.n)
    ne_out = effective_n(loss[~mask].astype(float), wk[~mask], c.n_out)
    p = two_prop_p(c.rate * ne_in, ne_in, c.base * ne_out, ne_out, "greater")
    return dataclasses.replace(c, p=max(p, c.p_raw))


def split_dates(f: pd.DataFrame, holdout_frac: float) -> pd.Timestamp:
    """Boundary date: the latest `holdout_frac` of distinct weeks lie on/after it."""
    weeks = np.sort(f["date"].dt.to_period("W").dt.start_time.unique())
    if len(weeks) < 4:
        return pd.Timestamp.max
    return pd.Timestamp(weeks[int(len(weeks) * (1.0 - holdout_frac))])


def mine_loss_risks(sf: SymFrame, now, mc: MiningConfig | None = None, ledger: TestLedger | None = None, patterns: Sequence[str] | None = None,
                    code_hash: str | None = None) -> list[LossRiskItem]:
    """Find contexts in which losing is provably more common, using ONLY loss definitions (no winner label is read as a target).
    Discovery runs on the early weeks (rows whose outcome had not matured by the split are purged from it); every surviving context is then
    re-tested on the untouched holdout and marked oos_confirmed or not. BH is applied over ALL candidates of the run, plus each test goes into the
    cumulative ledger. Returns items (not added to any bank); rows matured at or after `now` are invisible."""
    mc = (mc or MiningConfig()).check()
    vis = sf.matured_before(now)
    if vis.empty:
        return []
    f_all = vis.frame
    split = split_dates(f_all, mc.holdout_frac)
    disc = f_all[(f_all["date"] < split) & (f_all["matured_at"] < split)]
    hold = f_all[f_all["date"] >= split]
    units = [("*", None)] + ([(p, p) for p in (patterns or vis.patterns())] if mc.per_pattern else [])
    cands: list[tuple[Candidate, np.ndarray, np.ndarray, pd.Series]] = []
    m_total = 0
    for pid, sel in units:
        d = disc if sel is None else disc[disc["pattern_id"] == sel]
        for measure in mc.measures:
            e, loss = _exposed(d, measure, vis.cfg.large_loss)
            if len(e) < 2 * mc.min_ctx_n:
                continue
            wk = e["wk"].to_numpy()
            masks = _context_masks(descriptors(e, vis.cfg), e.index, mc.max_pair_dims)
            m_total += len(masks)
            for ctx, mask in masks.items():
                c = _score_candidate(pid, ctx, measure, mask, loss, mc)
                if c is not None and c.p_raw <= 0.25:
                    cands.append((_refine_candidate(c, mask, loss, wk), mask, loss, e["date"]))
    if not cands:
        return []
    ps = np.ones(max(m_total, len(cands)))
    ps[:len(cands)] = [c.p for c, *_ in cands]
    qs = bh(ps)[:len(cands)]                                    # BH over EVERY context examined, the unscreened ones counting as p = 1
    newest = str(d["matured_at"].max().date()) if len(d) else str(as_date(now))
    if ledger is not None:
        ledger.log_null(f"lossnull|{vis.digest()}|{stable_hash(dataclasses.asdict(mc), 8)}", "loss_risk", max(m_total - len(cands), 0), newest, now)
    items: list[LossRiskItem] = []
    for (c, mask, loss, dates), q in zip(cands, qs):
        c = dataclasses.replace(c, q=float(q))
        if ledger is not None:
            ledger.log(f"loss|{c.pattern_id}|{c.measure}|{c.context}|{vis.digest()}", "loss_risk", c.p, newest, now)
        if c.q > mc.q_max:
            continue
        items.append(_confirm_and_build(c, hold, vis, mc, newest, now, code_hash))
    return sorted(items, key=lambda i: (i.q, -i.relative_risk, i.risk_id))


def _confirm_and_build(c: Candidate, hold: pd.DataFrame, vis: SymFrame, mc: MiningConfig, newest: str, now, code_hash: str | None) -> LossRiskItem:
    """Re-test a discovered context on the holdout weeks and build the bank item with an honest epistemic label."""
    h = hold if c.pattern_id == "*" else hold[hold["pattern_id"] == c.pattern_id]
    e, loss = _exposed(h, c.measure, vis.cfg.large_loss)
    oos_n, oos_rate, conf = 0, math.nan, None
    if len(e) >= 2 * mc.min_ctx_n:
        desc = descriptors(e, vis.cfg)
        if all(dim in desc for dim, _ in c.context):
            m = np.ones(len(e), bool)
            for dim, val in c.context:
                m &= (desc[dim].reindex(e.index) == val).to_numpy()
            oos_n = int(m.sum())
            if oos_n >= mc.min_ctx_n // 2 and (~m).sum() >= mc.min_ctx_n:
                k, k_out = int(loss[m].sum()), int(loss[~m].sum())
                oos_rate = k / oos_n
                base_h = k_out / int((~m).sum())
                p_h = two_prop_p(k, oos_n, k_out, int((~m).sum()), "greater")
                rr_h = oos_rate / base_h if base_h > 0 else (math.inf if k else 1.0)
                conf = bool(p_h <= mc.confirm_p and rr_h >= mc.confirm_rr)
    lo, hi = _wilson(c.k, c.n, 1.96)
    kind = kind_for_context(c.context, c.measure)
    mean_loss = math.nan
    ep = Epistemic.SUPPORTED if conf else Epistemic.HYPOTHESIS
    notes = "" if conf else ("holdout too small to confirm" if conf is None else "not confirmed on the holdout weeks")
    pv = Provenance(created_real=wall_stamp(), learned_at=newest, code_hash=code_hash or current_code_hash(),
                    data_hash=vis.digest(), config_hash=stable_hash(dataclasses.asdict(mc), 12), seed=mc.seed, outcomes_seen_through=newest)
    return LossRiskItem(make_risk_id(c.pattern_id, c.context, c.measure), c.pattern_id, kind, c.context, c.measure, c.n, c.k, c.rate, c.base, c.rr, lo, hi,
                        mean_loss, c.p, c.q, conf, oos_rate, oos_n, ep, Lifecycle.BIRTH if not conf else Lifecycle.ACTIVE, effects_for_kind(kind),
                        LOSS_KIND_CAUSE[kind], newest, pv, notes=notes)


def attach_mean_loss(items: Sequence[LossRiskItem], sf: SymFrame, now) -> list[LossRiskItem]:
    """Fill in `mean_loss` (average directional return of the losing calls inside each context) from the visible rows."""
    vis = sf.matured_before(now)
    out = []
    for it in items:
        f = vis.frame if it.pattern_id == "*" else vis.of(it.pattern_id)
        e, loss = _exposed(f, it.measure, vis.cfg.large_loss)
        desc = descriptors(e, vis.cfg)
        if not e.empty and all(d in desc for d, _ in it.context):
            m = np.ones(len(e), bool)
            for d, v in it.context:
                m &= (desc[d].reindex(e.index) == v).to_numpy()
            sel = e[m & loss]
            out.append(dataclasses.replace(it, mean_loss=float(sel["dir_ret"].mean()) if len(sel) else math.nan))
        else:
            out.append(it)
    return out


# ==================================================================================================================
# what avoiding a context costs and saves
# ==================================================================================================================
@dataclass(frozen=True)
class Avoidance:
    risk_id: str
    exposed_in_context: int
    losses_avoided: int
    loss_dollars_avoided: float           # sum of |directional return| over the large losses skipped
    winners_forgone: int
    gain_forgone: float                   # sum of positive directional returns skipped
    net_return_avoided: float             # -(sum of directional returns skipped): positive = skipping improved results
    mean_return_in_context: float
    p_negative: float                     # one-sided p that the mean directional return in the context is below zero
    loss_per_winner_forgone: float
    worth_avoiding: bool | None


def avoidance_tradeoff(sf: SymFrame, item: LossRiskItem, now) -> Avoidance:
    """Section 12 symmetry applied to the bank itself: a caution rule that throws away as many winners as it saves losses is not free."""
    vis = sf.matured_before(now)
    f = vis.frame if item.pattern_id == "*" else vis.of(item.pattern_id)
    e = f[f["side"] != 0]
    desc = descriptors(e, vis.cfg)
    if e.empty or not all(d in desc for d, _ in item.context):
        return Avoidance(item.risk_id, 0, 0, 0.0, 0, 0.0, 0.0, math.nan, 1.0, math.nan, None)
    m = np.ones(len(e), bool)
    for d, v in item.context:
        m &= (desc[d].reindex(e.index) == v).to_numpy()
    ins = e[m]
    if ins.empty:
        return Avoidance(item.risk_id, 0, 0, 0.0, 0, 0.0, 0.0, math.nan, 1.0, math.nan, None)
    r = ins["dir_ret"].to_numpy()
    big = r <= -vis.cfg.large_loss
    wins = r >= vis.cfg.win_thr
    p_neg = float(sps.ttest_1samp(r, 0.0, alternative="less").pvalue) if len(r) >= 5 and r.std() > 0 else 1.0
    lpw = float(-r[big].sum() / wins.sum()) if wins.sum() else math.inf
    worth = None if len(r) < vis.cfg.min_n else bool(-r.sum() > 0 and p_neg < 0.10)
    return Avoidance(item.risk_id, len(r), int(big.sum()), float(-r[big].sum()), int(wins.sum()), float(r[r > 0].sum()), float(-r.sum()),
                     float(r.mean()), p_neg, lpw, worth)


def revalidate_item(bank: LossRiskBank, risk_id: str, sf: SymFrame, now, mc: MiningConfig | None = None) -> LossRiskItem:
    """Re-test an existing bank item on rows that matured since it was written. Still elevated -> stays; no longer elevated -> DEGRADED, and
    twice in a row -> RETIRED. A risk that disappears is recorded as disappearing, never silently kept (section 14 health)."""
    mc = mc or MiningConfig()
    it = bank.get(risk_id)
    vis = sf.matured_before(now)
    fresh = vis.frame[vis.frame["matured_at"] > pd.Timestamp(as_date(it.matured_at))]
    f = fresh if it.pattern_id == "*" else fresh[fresh["pattern_id"] == it.pattern_id]
    e, loss = _exposed(f, it.measure, vis.cfg.large_loss)
    desc = descriptors(e, vis.cfg) if len(e) else {}
    if len(e) < 2 * mc.min_ctx_n or not all(d in desc for d, _ in it.context):
        return it
    m = np.ones(len(e), bool)
    for d, v in it.context:
        m &= (desc[d].reindex(e.index) == v).to_numpy()
    if m.sum() < mc.min_ctx_n // 2 or (~m).sum() < mc.min_ctx_n:
        return it
    k, k_out = int(loss[m].sum()), int(loss[~m].sum())
    p = two_prop_p(k, int(m.sum()), k_out, int((~m).sum()), "greater")
    newest = str(fresh["matured_at"].max().date())
    if p <= mc.confirm_p:
        new = dataclasses.replace(it, oos_confirmed=True, oos_rate=k / int(m.sum()), oos_n=int(m.sum()), matured_at=newest,
                                  epistemic=Epistemic.SUPPORTED, lifecycle=Lifecycle.ACTIVE, notes="revalidated: still elevated")
    elif it.lifecycle == Lifecycle.DEGRADED:
        bank.retire(risk_id, "no longer elevated on two consecutive fresh checks", now)
        return bank.get(risk_id)
    else:
        new = dataclasses.replace(it, oos_confirmed=False, oos_rate=k / int(m.sum()), oos_n=int(m.sum()), matured_at=newest,
                                  epistemic=Epistemic.DEGRADED, lifecycle=Lifecycle.DEGRADED, notes="fresh data no longer shows elevated risk")
    bank.add(new, now)
    return bank.get(risk_id)


# ==================================================================================================================
# controls: worlds with known truth
# ==================================================================================================================
def synthetic_symmetry_frame(seed: int = 0, kind: str = "planted", n_weeks: int = 70, per_week: int = 80, cfg: SymmetryConfig | None = None) -> SymFrame:
    """Deterministic universe with three patterns.
      good     a real, symmetric pattern: long calls hit winners, short calls hit losers, no special weakness.
      winners  finds winners (good long precision) but on LOW-LIQUIDITY rows suffers large losses - the classic winner-only pattern.
      noise    calls carry no information.
    kind='planted' plants: (a) `winners` weakness in low liquidity rows, (b) a market-wide loss context (regime transition rows have 4x the
    large-loss rate for every pattern). kind='null' plants nothing: every pattern is noise and no context is special."""
    rng = np.random.default_rng(seed)
    cfg = cfg or SymmetryConfig()
    weeks = pd.date_range("2016-01-04", periods=n_weeks, freq="7D")
    rows = []
    n = n_weeks * per_week
    date = np.repeat(weeks.to_numpy(), per_week)
    tick = np.array([f"T{int(i):03d}" for i in rng.integers(0, per_week * 2, n)])
    base = pd.DataFrame({"date": date, "ticker": tick}).drop_duplicates(["date", "ticker"]).reset_index(drop=True)
    n = len(base)
    vol = rng.lognormal(-3.0, 0.5, n)
    dvol = rng.lognormal(15, 1.0, n)
    transition = rng.random(n) < 0.12
    event = rng.random(n) < 0.08
    regime = np.where(pd.to_datetime(base["date"]).dt.year.to_numpy() % 2 == 0, "calm", "stress")
    low_liq = dvol <= np.quantile(dvol, cfg.low_liq_q)
    planted = kind == "planted"
    latent = rng.normal(0, 0.06, n) * (vol / vol.mean())
    big = rng.random(n) < (np.where(transition, 0.20, 0.05) if planted else 0.05)
    fwd = np.where(big, rng.choice([-1.0, 1.0], n) * rng.uniform(0.16, 0.35, n), latent)
    for pid in ("good", "winners", "noise"):
        margin = rng.normal(0, 0.15, n)
        if pid == "noise" or not planted:
            call = np.where(margin > 0.1, np.where(rng.random(n) < 0.5, 1, -1), 0)
        elif pid == "good":
            informed = np.where(np.abs(fwd) > 0.03, np.sign(fwd), 0) * (rng.random(n) < 0.72) + np.where(rng.random(n) < 0.28, np.where(rng.random(n) < 0.5, 1, -1), 0)
            call = np.where((np.abs(fwd) > 0.03) & (margin > 0), np.sign(informed).astype(int), 0)
        else:
            long_hit = (fwd > 0.03) & (rng.random(n) < 0.7)
            call = np.where((margin > 0) & (long_hit | (rng.random(n) < 0.05)), 1, 0)
        lean = np.where(margin > -cfg.near_band, np.where(call != 0, call, np.where(rng.random(n) < 0.5, 1, -1)), 0)
        f = pd.DataFrame({"pattern_id": pid, "date": base["date"], "ticker": base["ticker"], "call": call.astype(int), "lean": lean.astype(int), "margin": margin})
        f["fwd"] = fwd
        if pid == "winners" and planted:
            hit_bad = (call > 0) & low_liq & (rng.random(n) < 0.4)
            f["fwd"] = np.where(hit_bad, -rng.uniform(0.16, 0.3, n), fwd)
        f["vol"], f["dollar_volume"] = vol, dvol
        f["regime"], f["regime_transition"], f["external_event"] = regime, transition, event
        f["mover_outcome"] = np.array(EPISODE_KINDS)[rng.integers(0, len(EPISODE_KINDS), n)]
        f["matured_at"] = pd.to_datetime(f["date"]) + pd.Timedelta(days=7)
        rows.append(f)
    return SymFrame(pd.concat(rows, ignore_index=True), cfg)


# ==================================================================================================================
# public summaries and text reports
# ==================================================================================================================
def _num(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 6)


def _lg(x: float) -> float | None:
    """Counts leave this module as log10: a raw count near 1,900-2,100 would be read as a year by the blind-view scan."""
    return _num(math.log10(x)) if x and x > 0 else None


def public_summary(lib: LibrarySymmetry, bank: LossRiskBank | None = None, now=None) -> dict[str, Any]:
    """Identity-free numbers and verdict words: pattern ids are replaced by opaque tokens, no ticker or date leaves. Checked by assert_trader_safe."""
    tok = {p.pattern_id: "P" + stable_hash(p.pattern_id, 6) for p in lib.patterns}
    out: dict[str, Any] = {
        "patterns": [{"id": tok[p.pattern_id], "calls_log10": _lg(p.n_calls), "hit": _num(p.hit_rate), "hit_lo": _num(p.hit_lo), "trust": p.trust.verdict.value,
                      "winner_precision": _num(p.winner.precision), "winner_recall": _num(p.winner.recall), "loser_precision": _num(p.loser.precision),
                      "loser_recall": _num(p.loser.recall), "long_adverse_ratio": _num(p.errors.long_adverse_ratio),
                      "failing_cases": [s.kind.value for s in p.failing_slices()], "blockers": list(p.trust.failed), "untested_checks": len(p.trust.untested)}
                     for p in lib.patterns],
        "orphan_share": _num(lib.orphans.get("orphan_share", math.nan)), "tests_log10": _lg(lib.n_tests),
        "trusted": len(lib.by_verdict(Trust.TRUSTED)), "not_trusted": len(lib.by_verdict(Trust.NOT_TRUSTED)), "unknown": len(lib.by_verdict(Trust.UNKNOWN))}
    if bank is not None and now is not None:
        out["loss_bank"] = bank.summary(now)
    assert_trader_safe(out, "symmetry public summary")
    return out


def render_pattern(p: PatternSymmetry) -> str:
    L = [f"PATTERN {p.pattern_id}: {p.trust.verdict.value}  ({p.n_calls} calls, hit {p.hit_rate:.3f}, lower bound {p.hit_lo:.3f}, edge p {p.edge_p:.4f})"]
    for c in (p.winner, p.loser):
        L.append(f"  {c.space.value:6s} space  TP {c.tp} FP {c.fp} TN {c.tn} FN {c.fn}  precision {c.precision:.3f}  recall {c.recall:.3f}  "
                 f"lift {c.lift:.2f}  MCC {c.mcc:.3f}")
    m = p.misses
    L.append(f"  missed winners {m.missed_winners}/{m.winners_total} (silent {m.winner_silent}, wrong way {m.winner_wrong_way}, near {m.winner_near_miss}, "
             f"suppressed {m.winner_suppressed});  missed losers {m.missed_losers}/{m.losers_total} (silent {m.loser_silent}, wrong way {m.loser_wrong_way}, "
             f"near {m.loser_near_miss}, suppressed {m.loser_suppressed})")
    e = p.errors
    L.append(f"  adverse movers: long calls landing on losers x{e.long_adverse_ratio:.2f} of base (p {e.long_adverse_p:.3f}); "
             f"short calls landing on winners x{e.short_adverse_ratio:.2f} (p {e.short_adverse_p:.3f})")
    for s in p.slices:
        if s.status != SliceStatus.UNTESTED or s.n_calls:
            L.append(f"  {s.kind.value:24s} {s.status.value:9s} calls {s.n_calls:5d}  hit {s.hit_rate:.3f}  loss-rate {s.large_loss_rate:.3f}  {s.reason}")
    if p.trust.failed or p.trust.untested:
        L.append("  because: " + p.trust.reasons)
    return NL.join(L)


def render_library(lib: LibrarySymmetry) -> str:
    head = (f"WINNER/LOSER SYMMETRY  (IMPLEMENTED - NOT VALIDATED)  patterns {len(lib.patterns)}, tests {lib.n_tests}, orphan movers "
            f"{lib.orphans.get('orphan_share', math.nan):.3f}")
    return NL.join([head, ""] + [render_pattern(p) + NL for p in lib.patterns])


def render_bank(bank: LossRiskBank, now) -> str:
    L = [f"LOSS-RISK BANK  {len(bank)} items"]
    for it in bank.items(now):
        ctx = " & ".join(f"{k}={v}" for k, v in it.context)
        L.append(f"  {it.risk_id} [{it.pattern_id}] {it.kind.value} when {ctx}: loss rate {it.loss_rate:.3f} vs {it.base_rate:.3f} (x{it.relative_risk:.2f}), "
                 f"q {it.q:.4f}, {'CONFIRMED' if it.oos_confirmed else 'UNCONFIRMED'} {it.epistemic.value}/{it.lifecycle.value}")
    return NL.join(L)


# ==================================================================================================================
# the always-on sweep (C67): resumable, coverage-tracked, one cumulative ledger, precursor inputs
# ==================================================================================================================
@dataclass(frozen=True)
class SymmetryPrecursor:
    """Duck-typed candidate precursor from another module (R21): its `frame` must satisfy SymFrame (one pattern id = the precursor id)."""
    source_id: str
    family: str
    frame: pd.DataFrame

    @classmethod
    def coerce(cls, obj: Any) -> "SymmetryPrecursor":
        missing = [a for a in ("precursor_id", "source_family", "frame") if not hasattr(obj, a)]
        if missing:
            raise ValueError(f"precursor candidate lacks {missing}")
        f = obj.frame.copy()
        f["pattern_id"] = "precursor:" + str(obj.precursor_id)
        return cls(str(obj.precursor_id), str(obj.source_family), f)


@dataclass(frozen=True)
class SweepReport:
    unit: tuple[str, str, str] | None
    rows: int
    patterns_analysed: int
    items_found: int
    items_added: int
    ledger_size: int
    ledger_survivors: int
    evenness: float
    status: str


class SymmetrySweep:
    """Walks (source family, era, cohort) units, always the least-covered next, so a 24/7 run cannot keep re-mining one corner. Each visit runs the
    symmetry analysis and loss-context mining on that unit only, logs every test in the ONE cumulative ledger, and admits an item to the bank only
    if it survives the strict cumulative correction (BH over every test ever logged, with the Bonferroni guard). State is on disk: kill and resume."""

    COHORTS = ("all", "movers", "quiet")

    def __init__(self, tracker: CoverageTracker, ledger: TestLedger, bank: LossRiskBank, cfg: SymmetryConfig | None = None, mc: MiningConfig | None = None):
        self.tracker, self.ledger, self.bank = tracker, ledger, bank
        self.cfg, self.mc = (cfg or SymmetryConfig()).check(), (mc or MiningConfig()).check()
        self.sources: dict[str, SymFrame] = {}

    def add_source(self, family: str, sf: SymFrame) -> None:
        self.sources[family] = sf

    def add_precursors(self, candidates: Iterable[Any]) -> int:
        """Accept precursor candidates as sources. A malformed one raises (fail closed); a valid one becomes its own family."""
        n = 0
        for obj in candidates:
            s = SymmetryPrecursor.coerce(obj)
            self.sources[f"precursor:{s.family}:{s.source_id}"] = SymFrame(s.frame, self.cfg)
            n += 1
        return n

    def units(self) -> list[tuple[str, str, str]]:
        out = []
        for fam, sf in self.sources.items():
            out += [(fam, e, c) for e in sorted(sf.frame["era"].unique()) for c in self.COHORTS] if not sf.empty else []
        return out

    def _cohort_frame(self, sf: SymFrame, era: str, cohort: str, now) -> SymFrame:
        vis = sf.matured_before(now)
        keep = vis.frame["era"] == era
        if cohort == "movers":
            keep &= vis.frame["winner"] | vis.frame["loser"]
        elif cohort == "quiet":
            keep &= ~(vis.frame["winner"] | vis.frame["loser"])
        return SymFrame(vis.frame[keep].drop(columns=SymFrame._derived_cols(), errors="ignore"), self.cfg, require_universe=False)

    def run_once(self, now) -> SweepReport:
        unit = self.tracker.next_unit(self.units())
        if unit is None:
            return SweepReport(None, 0, 0, 0, 0, len(self.ledger), len(self.ledger.survivors()), 0.0, "no sources")
        fam, era, cohort = unit
        sub = self._cohort_frame(self.sources[fam], era, cohort, now)
        self.tracker.mark(fam, era, cohort, now, weight=max(len(sub), 1))
        if len(sub) < 4 * self.mc.min_ctx_n:
            self.tracker.save()
            return SweepReport(unit, len(sub), 0, 0, 0, len(self.ledger), len(self.ledger.survivors()), self.tracker.evenness(self.units()), "too few matured rows")
        analysed = 0
        items = mine_loss_risks(sub, now, self.mc, self.ledger)
        items = attach_mean_loss(items, sub, now)
        strict = set(self.ledger.survivors(self.mc.q_max, strict=True))
        added = 0
        for it in items:
            key = f"loss|{it.pattern_id}|{it.measure}|{it.context}|{sub.matured_before(now).digest()}"
            if key in strict and self.bank.add(it, now) in ("added", "superseded"):
                added += 1
        if not sub.empty and all((sub.of(p)["call"] == 0).any() for p in sub.patterns()):
            analyse_library(sub, ledger=self.ledger, now=now)
            analysed = len(sub.patterns())
        self.tracker.save()
        return SweepReport(unit, len(sub), analysed, len(items), added, len(self.ledger), len(self.ledger.survivors()), self.tracker.evenness(self.units()), "assessed")

    def run(self, now, steps: int) -> list[SweepReport]:
        return [self.run_once(now) for _ in range(steps)]


# ==================================================================================================================
# state and the single public entry the research loop calls
# ==================================================================================================================
class SymmetryState:
    """Research-side (MATURED_RESEARCH_STATE) container: observation batches, the loss-risk bank and the cumulative test ledger."""

    def __init__(self, cfg: SymmetryConfig | None = None, mc: MiningConfig | None = None, bank_path: str | Path | None = None,
                 ledger_path: str | Path | None = None):
        self.cfg, self.mc = (cfg or SymmetryConfig()).check(), (mc or MiningConfig()).check()
        self.batches: list[SymFrame] = []
        self.bank = LossRiskBank(bank_path)
        self.ledger = TestLedger(ledger_path)
        self.last: LibrarySymmetry | None = None

    def add(self, frame: pd.DataFrame | SymFrame) -> int:
        sf = frame if isinstance(frame, SymFrame) else SymFrame(frame, self.cfg)
        self.batches.append(sf)
        return len(sf)

    def all(self) -> SymFrame:
        return SymFrame.concat(self.batches, self.cfg)


@dataclass(frozen=True)
class SymmetryReport:
    now: str
    library: LibrarySymmetry
    new_items: tuple[LossRiskItem, ...]
    added: tuple[str, ...]
    independence_problems: tuple[str, ...]
    ledger_size: int
    ledger_survivors: int
    provenance: Provenance

    def record_id(self) -> str:
        return "SR" + stable_hash({"n": self.now, "c": self.library.config_digest, "d": self.library.data_digest}, 12)

    def to_matured_record(self, bank: LossRiskBank | None = None) -> MaturedRecord:
        """Trusted-side wrapper: reaches a decision only through record.gate(now)."""
        return MaturedRecord(self.record_id(), self.provenance.learned_at, public_summary(self.library, bank, self.now), self.provenance,
                             Namespace.MATURED_RESEARCH)


def step(state: SymmetryState, now, seed: int | None = None, *, code_hash: str | None = None, opportunity_ids: Iterable[str] = ()) -> SymmetryReport:
    """ONE public entry (CONTEXT rule 25): analyse symmetry for every pattern in `state` using rows matured strictly before `now`, mine loss
    contexts, admit only the items that survive the cumulative correction, audit bank independence, return the report."""
    mc = state.mc if seed is None else dataclasses.replace(state.mc, seed=int(seed))
    sf = state.all().matured_before(now)
    if sf.empty:
        prov = Provenance(wall_stamp(), str(as_date(now) - dt.timedelta(days=1)), code_hash or current_code_hash())
        empty = LibrarySymmetry((), 0, orphan_movers(sf), state.cfg.digest(), "empty")
        return SymmetryReport(str(as_date(now)), empty, (), (), tuple(independence_audit(state.bank, opportunity_ids)), len(state.ledger), 0, prov)
    lib = analyse_library(sf, ledger=state.ledger, now=now)
    items = attach_mean_loss(mine_loss_risks(sf, now, mc, state.ledger, code_hash=code_hash), sf, now)
    strict = set(state.ledger.survivors(mc.q_max, strict=True))
    added = []
    for it in items:
        key = f"loss|{it.pattern_id}|{it.measure}|{it.context}|{sf.digest()}"
        if key in strict and state.bank.add(it, now) in ("added", "superseded"):
            added.append(it.risk_id)
    newest = str(sf.frame["matured_at"].max().date())
    prov = Provenance(wall_stamp(), newest, code_hash or current_code_hash(), sf.digest(), state.cfg.digest(),
                      seed=mc.seed, outcomes_seen_through=newest)
    state.last = lib
    return SymmetryReport(str(as_date(now)), lib, tuple(items), tuple(added), tuple(independence_audit(state.bank, opportunity_ids)), len(state.ledger),
                          len(state.ledger.survivors()), prov)


# ==================================================================================================================
# uncertainty on the confusion rates
# ==================================================================================================================
@dataclass(frozen=True)
class RateCI:
    name: str
    value: float
    lo: float
    hi: float
    n_boot: int


def bootstrap_confusion(f: pd.DataFrame, rng: np.random.Generator, n_boot: int = 300, level: float = 0.95) -> dict[str, RateCI]:
    """Week-cluster bootstrap intervals for precision and recall in BOTH spaces. Whole weeks are resampled, so the interval reflects that
    a week of calls shares one market. Values are NaN (interval NaN) when a rate has no denominator in the full sample."""
    keys = ("win_precision", "win_recall", "lose_precision", "lose_recall")
    if f.empty:
        return {k: RateCI(k, math.nan, math.nan, math.nan, 0) for k in keys}
    wk = f["wk"].to_numpy()
    uniq, inv = np.unique(wk, return_inverse=True)
    G = len(uniq)
    cols = {"wc": (f["call"] > 0).to_numpy(), "wa": f["winner"].to_numpy(), "lc": (f["call"] < 0).to_numpy(), "la": f["loser"].to_numpy()}
    cnt = {"w_called": cols["wc"], "w_tp": cols["wc"] & cols["wa"], "w_act": cols["wa"], "l_called": cols["lc"], "l_tp": cols["lc"] & cols["la"], "l_act": cols["la"]}
    per = {k: np.bincount(inv, weights=v.astype(float), minlength=G) for k, v in cnt.items()}
    idx = rng.integers(0, G, size=(n_boot, G))
    tot = {k: v[idx].sum(1) for k, v in per.items()}
    full = {k: v.sum() for k, v in per.items()}
    out = {}
    a = (1 - level) / 2
    for name, num, den in (("win_precision", "w_tp", "w_called"), ("win_recall", "w_tp", "w_act"), ("lose_precision", "l_tp", "l_called"),
                           ("lose_recall", "l_tp", "l_act")):
        if full[den] == 0:
            out[name] = RateCI(name, math.nan, math.nan, math.nan, 0)
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            b = tot[num] / tot[den]
        b = b[np.isfinite(b)]
        lo, hi = (float(np.quantile(b, a)), float(np.quantile(b, 1 - a))) if len(b) > 20 else (math.nan, math.nan)
        out[name] = RateCI(name, float(full[num] / full[den]), lo, hi, len(b))
    return out


# ==================================================================================================================
# dose-response: does a stronger call mean a better call, on both sides?
# ==================================================================================================================
@dataclass(frozen=True)
class DoseResponse:
    side: str
    bins: int
    hit_by_bin: tuple[float, ...]
    n_by_bin: tuple[int, ...]
    spearman: float
    p: float
    monotone: bool | None


def dose_response(f: pd.DataFrame, side: str, n_bins: int = 4, min_n: int = 40) -> DoseResponse:
    """Hit rate by quantile of |margin| among the pattern's fired calls of one side ('long' / 'short'). A pattern whose most confident calls are
    not its best ones is mis-scored, and a side that fails to improve with margin has no ranking skill there."""
    if "margin" not in f:
        return DoseResponse(side, 0, (), (), math.nan, math.nan, None)
    c = f[(f["call"] > 0) if side == "long" else (f["call"] < 0)]
    c = c[c["margin"].notna()]
    if len(c) < min_n * 2:
        return DoseResponse(side, 0, (), (), math.nan, math.nan, None)
    q = pd.qcut(c["margin"].abs().rank(method="first"), n_bins, labels=False)
    g = c.groupby(q)["hit"].agg(["mean", "size"])
    rho = sps.spearmanr(c["margin"].abs(), c["hit"].astype(float))
    p_one = float(rho.pvalue / 2) if math.isfinite(rho.statistic) and rho.statistic > 0 else 1.0
    means = tuple(float(x) for x in g["mean"])
    return DoseResponse(side, len(g), means, tuple(int(x) for x in g["size"]), float(rho.statistic) if math.isfinite(rho.statistic) else math.nan, p_one,
                        bool(all(b >= a - 0.03 for a, b in zip(means[:-1], means[1:]))) if len(means) >= 3 else None)


# ==================================================================================================================
# what movers did NEXT (C67): consolidated, expanded, stopped, spiked, reversed
# ==================================================================================================================
@dataclass(frozen=True)
class EpisodeRow:
    episode: str
    n: int
    mean_fwd: float
    winner_rate: float
    loser_rate: float
    winner_lo: float
    loser_lo: float
    excess_winner: float               # winner_rate - overall winner rate
    excess_loser: float
    p_excess: float                    # two-sided, cluster adjusted, of the larger deviation
    q: float
    pattern_catch_winner: float        # share of this episode's winners the pattern library called long
    pattern_catch_loser: float


def episode_outcomes(sf: SymFrame, pattern_id: str | None = None) -> list[EpisodeRow]:
    """For each 'what the mover did next' kind, how often the following period was a winner or a loser, whether that differs from the base
    rates (BH-corrected across kinds), and how much of it a pattern (or the whole library) actually caught. Episodes are OUTCOME categories:
    they describe what happened and are used to study it on the research side only; they are never a feature for the trader."""
    f = sf.frame if pattern_id is None else sf.of(pattern_id)
    if f.empty or "mover_outcome" not in f or not f["mover_outcome"].notna().any():
        return []
    ep = f["mover_outcome"].astype(str).str.lower()
    base_w, base_l = float(f["winner"].mean()), float(f["loser"].mean())
    rows, ps = [], []
    for k in sorted(ep.unique()):
        g = f[ep == k]
        n = len(g)
        ne = effective_n(g["winner"].to_numpy(float), g["wk"].to_numpy(), n)
        pw = two_prop_p(g["winner"].mean() * ne, ne, base_w * ne, ne, "two-sided") if ne >= 2 else 1.0
        nel = effective_n(g["loser"].to_numpy(float), g["wk"].to_numpy(), n)
        pl = two_prop_p(g["loser"].mean() * nel, nel, base_l * nel, nel, "two-sided") if nel >= 2 else 1.0
        wl = _wilson(float(g["winner"].sum()), n, sf.cfg.z)[0]
        ll = _wilson(float(g["loser"].sum()), n, sf.cfg.z)[0]
        cw = float((g["winner"] & (g["call"] > 0)).sum() / g["winner"].sum()) if g["winner"].sum() else math.nan
        cl = float((g["loser"] & (g["call"] < 0)).sum() / g["loser"].sum()) if g["loser"].sum() else math.nan
        rows.append([k, n, float(g["fwd"].mean()), float(g["winner"].mean()), float(g["loser"].mean()), wl, ll, float(g["winner"].mean()) - base_w,
                     float(g["loser"].mean()) - base_l, min(1.0, 2 * min(pw, pl)), cw, cl])
        ps.append(min(1.0, 2 * min(pw, pl)))
    qs = bh(ps)
    return [EpisodeRow(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], float(q), r[10], r[11]) for r, q in zip(rows, qs)]


def episode_transition(sf: SymFrame) -> pd.DataFrame:
    """Row-normalised table: given a mover's episode kind, the share whose forward return was loser / neutral / winner. Kinds whose rows are
    mostly winners or mostly losers are the raw material for a precursor; kinds equal to the base split carry nothing."""
    f = sf.frame
    if f.empty or "mover_outcome" not in f or not f["mover_outcome"].notna().any():
        return pd.DataFrame()
    got = np.select([f["loser"], f["winner"]], ["loser", "winner"], "neutral")
    t = pd.crosstab(f["mover_outcome"].astype(str).str.lower(), pd.Categorical(got, ["loser", "neutral", "winner"]), normalize="index")
    return t


# ==================================================================================================================
# pruning and ranking the loss bank
# ==================================================================================================================
def dominated_items(items: Sequence[LossRiskItem], slack: float = 0.15) -> list[str]:
    """Ids of pair-context items that add nothing over a single-descriptor parent of the same pattern and measure: their relative risk is not
    more than (1 + slack) times the best parent's. Keeping them would count one fact several times in a noisy-or size multiplier."""
    singles: dict[tuple, float] = {}
    for it in items:
        if len(it.context) == 1:
            singles[(it.pattern_id, it.measure, it.context[0])] = it.relative_risk
    out = []
    for it in items:
        if len(it.context) < 2:
            continue
        parents = [singles[(it.pattern_id, it.measure, c)] for c in it.context if (it.pattern_id, it.measure, c) in singles]
        if parents and it.relative_risk <= max(parents) * (1 + slack):
            out.append(it.risk_id)
    return out


def prune_bank(bank: LossRiskBank, now, slack: float = 0.15) -> list[str]:
    """Retire dominated items (never delete). Returns the ids retired."""
    live = bank.items(now, (Lifecycle.ACTIVE, Lifecycle.GROWTH, Lifecycle.PEAK, Lifecycle.BIRTH))
    dead = dominated_items(live, slack)
    for rid in dead:
        bank.retire(rid, "dominated by a simpler parent context", now)
    return dead


def item_priority(it: LossRiskItem, exposure_share: float = 0.1) -> float:
    """Expected loss avoided per unit of caution: excess loss rate times the size of the typical loss times how often the context comes up,
    discounted by how established the item is. Used to order research and to break ties when the bank is queried."""
    excess = max(it.loss_rate - it.base_rate, 0.0)
    size = abs(it.mean_loss) if math.isfinite(it.mean_loss) else 0.15
    return float(excess * size * exposure_share * (EPISTEMIC_WEIGHT.get(it.epistemic, 0.0) + (0.25 if it.oos_confirmed else 0.0)))


def research_questions(bank: LossRiskBank, now, created_real: str, limit: int = 10):
    """Turn the highest-priority UNCONFIRMED bank items into identity-free ResearchQuestions (engine.research.core), so the research loop can
    decide whether to spend compute confirming or refuting them. Text contains bucket names and rates only."""
    from engine.research.core import Problem, ResearchQuestion
    cand = [i for i in bank.items(now) if i.oos_confirmed is not True and i.lifecycle != Lifecycle.RETIRED]
    cand.sort(key=lambda i: (-item_priority(i), i.risk_id))
    out = []
    for it in cand[:limit]:
        ctx = " & ".join(f"{k}={v}" for k, v in it.context)
        out.append(ResearchQuestion.make(f"Is {it.measure} risk really x{it.relative_risk:.1f} higher when {ctx}?", "loss", Problem.LOSS_AVOIDANCE, created_real,
                                         it.matured_at, "elevated on fresh weeks with q <= 0.10", "not elevated on fresh weeks",
                                         parents=(it.risk_id,)))
    return out


# ==================================================================================================================
# comparing symmetry across snapshots
# ==================================================================================================================
@dataclass(frozen=True)
class TrustChange:
    pattern_id: str
    before: Trust | None
    after: Trust | None
    newly_failing: tuple[str, ...]
    newly_passing: tuple[str, ...]
    hit_delta: float


def compare_libraries(before: LibrarySymmetry, after: LibrarySymmetry) -> list[TrustChange]:
    """What changed between two analyses: trust flips, case slices that started or stopped failing, hit-rate movement. A trust that improves
    only because a failing context is no longer observable would show as 'untested' upstream, not here (section 43)."""
    b = {p.pattern_id: p for p in before.patterns}
    a = {p.pattern_id: p for p in after.patterns}
    out = []
    for pid in sorted(set(a) | set(b)):
        pb, pa = b.get(pid), a.get(pid)
        fb = {s.kind.value for s in pb.failing_slices()} if pb else set()
        fa = {s.kind.value for s in pa.failing_slices()} if pa else set()
        dh = pa.hit_rate - pb.hit_rate if pa and pb and math.isfinite(pa.hit_rate) and math.isfinite(pb.hit_rate) else math.nan
        out.append(TrustChange(pid, pb.trust.verdict if pb else None, pa.trust.verdict if pa else None, tuple(sorted(fa - fb)), tuple(sorted(fb - fa)), dh))
    return [c for c in out if c.before != c.after or c.newly_failing or c.newly_passing]


def mover_blindness(sf: SymFrame) -> pd.DataFrame:
    """Per era: share of winners and of losers that NO pattern called correctly (orphans). Rising blindness in one era means the library's
    coverage of that period is thin, which is where new discovery should look first."""
    rows = []
    for era in sorted(sf.frame["era"].unique()) if not sf.empty else []:
        sub = SymFrame(sf.frame[sf.frame["era"] == era].drop(columns=SymFrame._derived_cols(), errors="ignore"), sf.cfg, require_universe=False)
        o = orphan_movers(sub)
        rows.append(dict(era=era, movers=o["movers"], orphan_share=o["orphan_share"], winner_orphan_share=o["winner_orphan_share"],
                         loser_orphan_share=o["loser_orphan_share"]))
    return pd.DataFrame(rows)


def loss_bank_coverage(bank: LossRiskBank, sf: SymFrame, now) -> dict[str, float]:
    """How much of the large-loss mass in the visible data falls inside SOME confirmed bank context: the share of big losses the bank could have
    warned about. The complement is loss the bank is blind to and should drive new loss research."""
    vis = sf.matured_before(now)
    e, loss = _exposed(vis.frame, "large_loss", vis.cfg.large_loss)
    if e.empty or not loss.any():
        return {"large_losses": 0, "warned_share": math.nan, "warned_confirmed_share": math.nan}
    desc = descriptors(e, vis.cfg)
    warned = np.zeros(len(e), bool)
    warned_conf = np.zeros(len(e), bool)
    for it in bank.items(now, (Lifecycle.ACTIVE, Lifecycle.GROWTH, Lifecycle.PEAK, Lifecycle.BIRTH)):
        if it.measure != "large_loss" or not all(d in desc for d, _ in it.context):
            continue
        m = np.ones(len(e), bool)
        for d, v in it.context:
            m &= (desc[d].reindex(e.index) == v).to_numpy()
        if it.pattern_id != "*":
            m &= (e["pattern_id"] == it.pattern_id).to_numpy()
        warned |= m
        if it.oos_confirmed:
            warned_conf |= m
    n = int(loss.sum())
    return {"large_losses": n, "warned_share": float((warned & loss).sum() / n), "warned_confirmed_share": float((warned_conf & loss).sum() / n)}


# ==================================================================================================================
# symmetry inside each context value
# ==================================================================================================================
@dataclass(frozen=True)
class ContextSymmetry:
    dimension: str
    value: str
    calls: int
    hit_rate: float
    long_hit: float
    short_hit: float
    large_loss_rate: float
    mean_dir_ret: float
    p_worse: float
    q: float
    verdict: SliceStatus


def context_symmetry(sf: SymFrame, pattern_id: str | None = None) -> list[ContextSymmetry]:
    """For every descriptor value (regime, sector, volatility bucket, episode kind ...) how do the pattern's long and short calls behave,
    compared with all its other calls? p-values are corrected across every (dimension, value) tested (BH). A context with fewer than
    min_n calls is UNTESTED, never OK."""
    cfg = sf.cfg
    f = sf.frame if pattern_id is None else sf.of(pattern_id)
    calls = f[f["fired"]]
    if calls.empty:
        return []
    desc = descriptors(calls, cfg)
    rows, ps = [], []
    for dim, s in desc.items():
        if dim == "side":
            continue
        for val in sorted(s.dropna().unique()):
            m = (s == val).to_numpy()
            ins, out = calls[m], calls[~m]
            if len(ins) == 0:
                continue
            longs, shorts = ins[ins["call"] > 0], ins[ins["call"] < 0]
            p = 1.0
            if len(ins) >= cfg.min_n and len(out) >= cfg.min_n:
                ni = effective_n(ins["hit"].to_numpy(float), ins["wk"].to_numpy(), len(ins))
                no = effective_n(out["hit"].to_numpy(float), out["wk"].to_numpy(), len(out))
                p = min(1.0, 2 * min(two_prop_p(ins["hit"].mean() * ni, ni, out["hit"].mean() * no, no, "less"),
                                     mean_diff_p(ins["dir_ret"].to_numpy(), out["dir_ret"].to_numpy(), "less")))
            rows.append((dim, str(val), len(ins), float(ins["hit"].mean()), float(longs["hit"].mean()) if len(longs) else math.nan,
                         float(shorts["hit"].mean()) if len(shorts) else math.nan, float(ins["big_loss"].mean()), float(ins["dir_ret"].mean()),
                         float(out["hit"].mean()) if len(out) else math.nan))
            ps.append(p)
    qs = bh(ps)
    out_rows = []
    for (dim, val, n, hit, lh, sh, ll, mr, hit_out), p, q in zip(rows, ps, qs):
        if n < cfg.min_n:
            v = SliceStatus.UNTESTED
        elif q <= cfg.fails_q and hit < hit_out:
            v = SliceStatus.FAILS
        elif hit < hit_out - cfg.slice_drop or mr < -cfg.tol_return:
            v = SliceStatus.DEGRADED
        else:
            v = SliceStatus.OK
        out_rows.append(ContextSymmetry(dim, val, n, hit, lh, sh, ll, mr, p, float(q), v))
    return out_rows


# ==================================================================================================================
# library redundancy: do two patterns catch the same movers?
# ==================================================================================================================
@dataclass(frozen=True)
class PairOverlap:
    a: str
    b: str
    winner_jaccard: float
    loser_jaccard: float
    joint_calls: int
    disagree_share: float              # share of jointly-called rows where the two patterns took opposite sides


def pattern_overlap(sf: SymFrame, min_calls: int = 30) -> list[PairOverlap]:
    """Pairwise overlap of caught winners and caught losers, and how often two patterns contradict each other on the same (date, ticker).
    Two patterns with Jaccard near 1 are one fact counted twice; two that often disagree cannot both be trusted on those rows."""
    f = sf.frame
    if f.empty:
        return []
    key = f["date"].astype(str) + "|" + f["ticker"].astype(str)
    pats = [p for p in sf.patterns() if (sf.of(p)["call"] != 0).sum() >= min_calls]
    caught_w = {p: set(key[(f["pattern_id"] == p) & f["winner"] & (f["call"] > 0)]) for p in pats}
    caught_l = {p: set(key[(f["pattern_id"] == p) & f["loser"] & (f["call"] < 0)]) for p in pats}
    side = {p: pd.Series(f.loc[(f["pattern_id"] == p) & f["fired"], "call"].to_numpy(), index=key[(f["pattern_id"] == p) & f["fired"]].to_numpy()) for p in pats}

    def jac(x: set, y: set) -> float:
        u = len(x | y)
        return len(x & y) / u if u else math.nan
    out = []
    for i, a in enumerate(pats):
        for b in pats[i + 1:]:
            sa, sb = side[a], side[b]
            common = sa.index.intersection(sb.index)
            dis = float((sa.reindex(common).to_numpy() != sb.reindex(common).to_numpy()).mean()) if len(common) else math.nan
            out.append(PairOverlap(a, b, jac(caught_w[a], caught_w[b]), jac(caught_l[a], caught_l[b]), int(len(common)), dis))
    return out


# ==================================================================================================================
# does the loss bank help? walk-forward evaluation of caution
# ==================================================================================================================
@dataclass(frozen=True)
class BankEvaluation:
    train_rows: int
    test_rows: int
    items_used: int
    exposed_calls: int
    scaled_calls: int                  # calls whose size the bank would have reduced
    mean_ret_unscaled: float
    mean_ret_scaled: float
    cvar5_unscaled: float
    cvar5_scaled: float
    big_loss_sum_unscaled: float
    big_loss_sum_scaled: float
    big_loss_reduction: float          # relative reduction in summed large-loss magnitude
    gain_given_up: float               # reduction in summed positive returns
    p_lower_mean_loss: float           # paired one-sided p that scaled cuts the mean size of losing calls
    helps: bool | None


def _size_vector(e: pd.DataFrame, items: Sequence[LossRiskItem], cfg: SymmetryConfig, floor: float = 0.05) -> np.ndarray:
    """Position-size multiplier per call from the given items (noisy-or of their weights, floor 0.05): the same rule risk_multiplier applies,
    computed for a whole table at once."""
    desc = descriptors(e, cfg)
    size = np.ones(len(e))
    for it in items:
        if not all(d in desc for d, _ in it.context):
            continue
        m = np.ones(len(e), bool)
        for d, v in it.context:
            m &= (desc[d].reindex(e.index) == v).to_numpy()
        if it.pattern_id != "*":
            m &= (e["pattern_id"] == it.pattern_id).to_numpy()
        size = np.where(m, size * (1.0 - it.weight()), size)
    return np.maximum(size, floor)


def evaluate_bank_walk_forward(sf: SymFrame, now, mc: MiningConfig | None = None, holdout_frac: float = 0.4, min_calls: int = 100) -> BankEvaluation:
    """Mine a bank on the early weeks (purged), then apply it to the later weeks it never saw, scaling each call by the bank's size multiplier.
    Reports what caution cost (gains given up) against what it saved (large-loss magnitude) on fresh data. Only ex-ante information about the
    later rows (their descriptors) is used to scale them; outcomes decide only the score."""
    mc = dataclasses.replace(mc or MiningConfig(), holdout_frac=holdout_frac)
    vis = sf.matured_before(now)
    nan = math.nan
    empty = BankEvaluation(0, 0, 0, 0, 0, nan, nan, nan, nan, nan, nan, nan, nan, 1.0, None)
    if vis.empty:
        return empty
    split = split_dates(vis.frame, holdout_frac)
    early = SymFrame(vis.frame[(vis.frame["date"] < split) & (vis.frame["matured_at"] < split)].drop(columns=SymFrame._derived_cols(), errors="ignore"),
                     vis.cfg, require_universe=False)
    late = vis.frame[vis.frame["date"] >= split]
    if early.empty or late.empty:
        return empty
    items = [i for i in mine_loss_risks(early, split, dataclasses.replace(mc, holdout_frac=0.3)) if i.oos_confirmed]
    e = late[late["fired"]]
    if len(e) < min_calls:
        return dataclasses.replace(empty, train_rows=len(early), test_rows=len(late), items_used=len(items))
    size = _size_vector(e, items, vis.cfg)
    r = e["dir_ret"].to_numpy()
    rs = r * size
    big = r <= -vis.cfg.large_loss
    k = max(int(math.ceil(0.05 * len(r))), 1)
    lose_u, lose_s = np.where(r < 0, -r, 0.0), np.where(r < 0, -rs, 0.0)
    p = float(sps.ttest_rel(lose_s, lose_u, alternative="less").pvalue) if (size < 1).any() and len(r) > 5 else 1.0
    bl_u, bl_s = float(-r[big].sum()), float(-rs[big].sum())
    red = (bl_u - bl_s) / bl_u if bl_u > 0 else nan
    gain = float(r[r > 0].sum() - rs[r > 0].sum())
    per_unit = float(rs.sum() / size.sum())                    # return per unit of capital actually deployed
    helps = None if not (size < 1).any() else bool(math.isfinite(red) and red > 0 and np.sort(rs / size)[:k].mean() >= np.sort(r)[:k].mean() - 1e-12
                                                   and per_unit >= float(r.mean()) - 0.005 and p < 0.10)
    return BankEvaluation(len(early), len(late), len(items), len(e), int((size < 1).sum()), float(r.mean()), float(rs.mean()), float(np.sort(r)[:k].mean()),
                          float(np.sort(rs)[:k].mean()), bl_u, bl_s, red, gain, p, helps)


# ==================================================================================================================
# hooks the research loop and the trusted-side curator can call
# ==================================================================================================================
def trader_caution(bank: LossRiskBank, now, context: Mapping[str, str], pattern_id: str | None = None) -> dict[str, Any]:
    """Identity-free caution payload for the curator to release: a size multiplier and the abstain flag, nothing else. The context must be
    bucket names (checked by assert_trader_safe), and the answer only uses items matured strictly before `now`."""
    assert_trader_safe(dict(context), "caution context")
    mult, ids = risk_multiplier(bank, now, context, pattern_id)
    kinds = sorted({bank.get(i).kind.value for i in ids})
    abstain = any(DecisionEffect.ABSTENTION in bank.get(i).effects and bank.get(i).weight() >= 0.35 for i in ids)
    out = {"size_multiplier": round(mult, 4), "abstain": abstain, "risk_kinds": kinds, "items_matched": len(ids)}
    assert_trader_safe(out, "caution payload")
    return out


def loss_first_report(lib: LibrarySymmetry, bank: LossRiskBank, now) -> str:
    """One page that puts losses before winners (section 5): failing patterns, the strongest live caution items, and the blind spots."""
    L = ["LOSS-FIRST SUMMARY (IMPLEMENTED - NOT VALIDATED)"]
    for p in lib.patterns:
        if p.trust.verdict != Trust.TRUSTED:
            L.append(f"  {p.pattern_id}: {p.trust.verdict.value} - {p.trust.reasons or 'no reason recorded'}")
    top = sorted(bank.items(now), key=lambda i: -item_priority(i))[:5]
    for it in top:
        L.append(f"  caution {it.risk_id} {it.kind.value} x{it.relative_risk:.2f} ({'confirmed' if it.oos_confirmed else 'unconfirmed'})")
    L.append(f"  orphan movers (no pattern called them right): {lib.orphans.get('orphan_share', math.nan):.3f}")
    return NL.join(L)


# ==================================================================================================================
# symmetry over time: a pattern whose loss side is getting worse loses trust
# ==================================================================================================================
@dataclass(frozen=True)
class WindowStat:
    index: int
    calls: int
    hit: float
    big_loss_rate: float
    wrong_way_rate: float
    mean_dir_ret: float
    long_hit: float
    short_hit: float


@dataclass(frozen=True)
class LossTrend:
    """Trend of the loss-side behaviour of one pattern across consecutive equal-week windows."""
    pattern_id: str
    windows: tuple[WindowStat, ...]
    tau_big_loss: float
    p_big_loss: float                  # one-sided p that the large-loss rate is RISING (Kendall)
    tau_wrong_way: float
    p_wrong_way: float
    tau_hit: float
    p_hit_falling: float               # one-sided p that the hit rate is FALLING
    early_late_big_loss: tuple[float, float]
    verdict: str                       # WORSENING / STABLE / IMPROVING / UNTESTED


def _kendall_one_sided(y: Sequence[float], rising: bool) -> tuple[float, float]:
    """Kendall tau of y against time and the one-sided p in the requested direction; (nan, 1.0) below 4 finite points."""
    yy = np.array([v for v in y if math.isfinite(v)], float)
    if len(yy) < 4 or np.ptp(yy) == 0:
        return math.nan, 1.0
    r = sps.kendalltau(np.arange(len(yy)), yy)
    tau, p2 = float(r.statistic), float(r.pvalue)
    p = p2 / 2 if (tau > 0) == rising else 1.0 - p2 / 2
    return tau, float(min(max(p, 0.0), 1.0))


def loss_trend(sf: SymFrame, pattern_id: str, n_windows: int = 6, min_calls: int = 20, alpha: float = 0.10) -> LossTrend:
    """Split the pattern's calls into `n_windows` blocks of equal week counts, in time order, and test whether large losses / wrong-way calls
    are rising or hit rate falling. WORSENING needs a significant trend in a loss measure AND the late windows worse than the early ones by
    a visible margin; anything with too few windows carrying `min_calls` is UNTESTED, never STABLE."""
    f = sf.of(pattern_id)
    calls = f[f["fired"]].sort_values("date")
    nan = math.nan
    if calls.empty:
        return LossTrend(pattern_id, (), nan, 1.0, nan, 1.0, nan, 1.0, (nan, nan), "UNTESTED")
    order = {w: i for i, w in enumerate(calls.groupby("wk")["date"].min().sort_values().index)}
    rank = calls["wk"].map(order).to_numpy()
    edges = np.linspace(0, len(order), n_windows + 1)
    win = np.clip(np.searchsorted(edges, rank, side="right") - 1, 0, n_windows - 1)
    stats: list[WindowStat] = []
    for i in range(n_windows):
        g = calls[win == i]
        if len(g) < min_calls:
            continue
        lg, sg = g[g["call"] > 0], g[g["call"] < 0]
        stats.append(WindowStat(i, len(g), float(g["hit"].mean()), float(g["big_loss"].mean()), float(g["wrong_way"].mean()), float(g["dir_ret"].mean()),
                                float(lg["hit"].mean()) if len(lg) else nan, float(sg["hit"].mean()) if len(sg) else nan))
    if len(stats) < 4:
        return LossTrend(pattern_id, tuple(stats), nan, 1.0, nan, 1.0, nan, 1.0, (nan, nan), "UNTESTED")
    tau_l, p_l = _kendall_one_sided([w.big_loss_rate for w in stats], True)
    tau_w, p_w = _kendall_one_sided([w.wrong_way_rate for w in stats], True)
    tau_h, p_h = _kendall_one_sided([w.hit for w in stats], False)
    half = len(stats) // 2
    early = float(np.mean([w.big_loss_rate for w in stats[:half]]))
    late = float(np.mean([w.big_loss_rate for w in stats[-half:]]))
    hit_drop = float(np.mean([w.hit for w in stats[:half]]) - np.mean([w.hit for w in stats[-half:]]))
    worse = (p_l <= alpha and late > early + 0.01) or (p_w <= alpha and hit_drop > 0.05) or (p_h <= alpha and hit_drop > 0.08)
    better = (p_l >= 1 - alpha and late < early - 0.01) or (p_h >= 1 - alpha and hit_drop < -0.05)
    return LossTrend(pattern_id, tuple(stats), tau_l, p_l, tau_w, p_w, tau_h, p_h, (early, late), "WORSENING" if worse else "IMPROVING" if better else "STABLE")


def trend_adjusted_trust(ps: PatternSymmetry, trend: LossTrend) -> TrustDecision:
    """Downgrade, never upgrade: a TRUSTED pattern whose loss side is WORSENING becomes NOT_TRUSTED; an UNTESTED trend on a TRUSTED pattern makes it
    UNKNOWN (trust that cannot be tracked is not trust). Patterns already NOT_TRUSTED or UNKNOWN keep their verdict and gain the reason."""
    td = ps.trust
    if trend.verdict == "WORSENING":
        note = f"loss side worsening over time (large-loss rate {trend.early_late_big_loss[0]:.3f} -> {trend.early_late_big_loss[1]:.3f})"
        return TrustDecision(Trust.NOT_TRUSTED, td.passed, td.failed + (note,), td.untested)
    if trend.verdict == "UNTESTED" and td.verdict == Trust.TRUSTED:
        return TrustDecision(Trust.UNKNOWN, td.passed, td.failed, td.untested + ("loss trend untested",))
    return td


@dataclass(frozen=True)
class SymmetryGateInput:
    """What the research quality gate (engine.research section 42) consumes for one pattern: a GateVerdict plus the reasons that produced it.
    PROMOTE is only possible for a TRUSTED pattern with a non-worsening loss side and no failing critical slice."""
    pattern_id: str
    verdict: Any                       # engine.research.core.GateVerdict
    trust: Trust
    critical_failures: tuple[str, ...]
    open_questions: tuple[str, ...]
    trend: str

    def as_dict(self) -> dict[str, Any]:
        return {"pattern_id": self.pattern_id, "verdict": str(self.verdict), "trust": self.trust.value, "critical_failures": list(self.critical_failures),
                "open_questions": list(self.open_questions), "trend": self.trend}


CRITICAL_SLICES = (CaseKind.WRONG_DIRECTION, CaseKind.LARGE_LOSS, CaseKind.REGIME_TRANSITION, CaseKind.EXTERNAL_EVENT, CaseKind.LOW_LIQUIDITY_FAILURE,
                   CaseKind.HIGH_VOL_FAILURE)


def gate_input(ps: PatternSymmetry, trend: LossTrend | None = None) -> SymmetryGateInput:
    """Symmetry-aware trust decision in the quality gate's own vocabulary (GateVerdict). Critical failures -> FAILED (or QUARANTINED when the only
    failure is a loss-limit breach the pattern could be sized around); UNKNOWN trust -> NEEDS_MORE_EVIDENCE; a worsening loss side -> QUARANTINED."""
    from engine.research.core import GateVerdict
    td = trend_adjusted_trust(ps, trend) if trend is not None else ps.trust
    crit = tuple(s.kind.value for s in ps.slices if s.kind in CRITICAL_SLICES and s.status == SliceStatus.FAILS)
    open_q = tuple(s.kind.value for s in ps.slices if s.kind in CRITICAL_SLICES and s.status == SliceStatus.UNTESTED) + td.untested
    tv = trend.verdict if trend is not None else "UNTESTED"
    if td.verdict == Trust.TRUSTED and not crit:
        v = GateVerdict.PROMOTE
    elif tv == "WORSENING":
        v = GateVerdict.QUARANTINED
    elif crit and set(crit) <= {CaseKind.LARGE_LOSS.value, CaseKind.LOW_LIQUIDITY_FAILURE.value, CaseKind.HIGH_VOL_FAILURE.value} and ps.hit_lo > 0.5:
        v = GateVerdict.QUARANTINED                    # works, but only if sized around a known loss context (the bank supplies the size)
    elif crit or td.verdict == Trust.NOT_TRUSTED:
        v = GateVerdict.FAILED
    else:
        v = GateVerdict.NEEDS_MORE_EVIDENCE
    return SymmetryGateInput(ps.pattern_id, v, td.verdict, crit, open_q, tv)


def gate_inputs(lib: LibrarySymmetry, sf: SymFrame, n_windows: int = 6) -> list[SymmetryGateInput]:
    """Gate inputs for every pattern in the library, each with its own loss trend."""
    return [gate_input(p, loss_trend(sf, p.pattern_id, n_windows)) for p in lib.patterns]


def symmetry_timeline(sf: SymFrame, pattern_id: str, n_windows: int = 6) -> pd.DataFrame:
    """Window-by-window table of a pattern's winner-side and loser-side behaviour (for reports and health checks)."""
    t = loss_trend(sf, pattern_id, n_windows)
    return pd.DataFrame([dataclasses.asdict(w) for w in t.windows])


# ==================================================================================================================
# regime-transition and external-event slices, in depth (section 12)
# ==================================================================================================================
@dataclass(frozen=True)
class EventSliceDetail:
    kind: CaseKind
    inside_calls: int
    outside_calls: int
    hit_inside: float
    hit_outside: float
    big_loss_inside: float
    big_loss_outside: float
    long_hit_inside: float
    short_hit_inside: float
    adverse_direction: str            # which side suffers more inside: LONG / SHORT / BOTH / NONE
    lift_in_loss_rate: float
    p_more_losses: float


def event_slice_detail(f: pd.DataFrame, kind: CaseKind, cfg: SymmetryConfig) -> EventSliceDetail:
    """Deeper look at the two context slices most patterns break in: regime transitions and external events. Reports which side (long or short)
    suffers, the lift in large-loss rate, and a one-sided test that losses are more common inside."""
    if kind not in (CaseKind.REGIME_TRANSITION, CaseKind.EXTERNAL_EVENT):
        raise ValueError("event_slice_detail handles REGIME_TRANSITION and EXTERNAL_EVENT only")
    nan = math.nan
    calls = f[f["fired"]]
    if not slice_available(f, kind) or calls.empty:
        return EventSliceDetail(kind, 0, len(calls), nan, nan, nan, nan, nan, nan, "NONE", nan, 1.0)
    m = slice_mask(calls, kind)
    ins, out = calls[m], calls[~m]
    if len(ins) == 0 or len(out) == 0:
        return EventSliceDetail(kind, len(ins), len(out), nan, nan, nan, nan, nan, nan, "NONE", nan, 1.0)
    lg, sg = ins[ins["call"] > 0], ins[ins["call"] < 0]
    bl_in, bl_out = float(ins["big_loss"].mean()), float(out["big_loss"].mean())
    ni = effective_n(ins["big_loss"].to_numpy(float), ins["wk"].to_numpy(), len(ins))
    no = effective_n(out["big_loss"].to_numpy(float), out["wk"].to_numpy(), len(out))
    p = two_prop_p(bl_in * ni, ni, bl_out * no, no, "greater")
    lh_out = float(out[out["call"] > 0]["hit"].mean()) if (out["call"] > 0).any() else nan
    sh_out = float(out[out["call"] < 0]["hit"].mean()) if (out["call"] < 0).any() else nan
    lh = float(lg["hit"].mean()) if len(lg) >= 5 else nan
    sh = float(sg["hit"].mean()) if len(sg) >= 5 else nan
    lose_long = math.isfinite(lh) and math.isfinite(lh_out) and lh < lh_out - cfg.slice_drop
    lose_short = math.isfinite(sh) and math.isfinite(sh_out) and sh < sh_out - cfg.slice_drop
    adverse = "BOTH" if lose_long and lose_short else "LONG" if lose_long else "SHORT" if lose_short else "NONE"
    return EventSliceDetail(kind, len(ins), len(out), float(ins["hit"].mean()), float(out["hit"].mean()), bl_in, bl_out, lh, sh, adverse,
                            bl_in / bl_out if bl_out > 0 else (math.inf if bl_in > 0 else 1.0), p)


def event_slice_report(sf: SymFrame) -> pd.DataFrame:
    """Regime-transition and external-event detail for every pattern, one row per (pattern, slice)."""
    rows = []
    for pid in sf.patterns():
        f = sf.of(pid)
        for k in (CaseKind.REGIME_TRANSITION, CaseKind.EXTERNAL_EVENT):
            d = event_slice_detail(f, k, sf.cfg)
            rows.append(dict(pattern_id=pid, slice=k.value, **{n: (v.value if isinstance(v, enum.Enum) else v) for n, v in dataclasses.asdict(d).items() if n != "kind"}))
    return pd.DataFrame(rows)


# ==================================================================================================================
# bank health: is each stored risk still showing up in fresh data?
# ==================================================================================================================
@dataclass(frozen=True)
class ItemHealth:
    risk_id: str
    stored_rate: float
    fresh_rate: float
    fresh_exposed: int
    fresh_base: float
    p_still_elevated: float
    status: str                       # HEALTHY / FADING / GONE / UNTESTED


def bank_health(bank: LossRiskBank, sf: SymFrame, now, min_exposed: int = 30) -> list[ItemHealth]:
    """For every live item, re-measure its loss rate on rows that matured AFTER the item was written (and before `now`). HEALTHY: still
    significantly above the fresh base rate. FADING: elevated but not provably. GONE: at or below base. UNTESTED: too little fresh exposure
    (silence is never health). Read-only: revalidate_item / retire act on the result."""
    vis = sf.matured_before(now)
    out = []
    for it in bank.items(now, (Lifecycle.ACTIVE, Lifecycle.GROWTH, Lifecycle.PEAK, Lifecycle.BIRTH)):
        fresh = vis.frame[vis.frame["matured_at"] > pd.Timestamp(as_date(it.matured_at))]
        f = fresh if it.pattern_id == "*" else fresh[fresh["pattern_id"] == it.pattern_id]
        e, loss = _exposed(f, it.measure, vis.cfg.large_loss)
        desc = descriptors(e, vis.cfg) if len(e) else {}
        if not len(e) or not all(d in desc for d, _ in it.context):
            out.append(ItemHealth(it.risk_id, it.loss_rate, math.nan, 0, math.nan, 1.0, "UNTESTED"))
            continue
        m = np.ones(len(e), bool)
        for d, v in it.context:
            m &= (desc[d].reindex(e.index) == v).to_numpy()
        n_in, n_out = int(m.sum()), int((~m).sum())
        if n_in < min_exposed or n_out < min_exposed:
            out.append(ItemHealth(it.risk_id, it.loss_rate, math.nan, n_in, math.nan, 1.0, "UNTESTED"))
            continue
        k, k_out = int(loss[m].sum()), int(loss[~m].sum())
        p = two_prop_p(k, n_in, k_out, n_out, "greater")
        rate, base = k / n_in, k_out / n_out
        status = "HEALTHY" if p <= 0.10 and rate > base else "FADING" if rate > base else "GONE"
        out.append(ItemHealth(it.risk_id, it.loss_rate, rate, n_in, base, p, status))
    return out


def health_counts(health: Sequence[ItemHealth]) -> dict[str, int]:
    """Status histogram of a bank_health result, with every status present so a missing key can never be read as zero risk of decay."""
    out = {k: 0 for k in ("HEALTHY", "FADING", "GONE", "UNTESTED")}
    for h in health:
        out[h.status] += 1
    return out


def stale_items(health: Sequence[ItemHealth]) -> list[str]:
    """Ids whose risk has gone (candidates for revalidate_item / retire); FADING and UNTESTED are kept, not guessed about."""
    return [h.risk_id for h in health if h.status == "GONE"]
