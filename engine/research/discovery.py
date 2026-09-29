"""Continual pattern discovery over every section-13 source family (contract C66 section 13; canons C35, C56, C57, C62, C63, C66).

Section 13: the research brain must keep searching history for relationships across price, volume, volatility, rank, structure,
events, filings, insiders, macro, breadth, correlation, dispersion, liquidity, regime and its own learned patterns - and every
finding must become a formal knowledge object with definition, provenance, discovery timestamp, training and validation periods,
evidence and independent-evidence counts, truth probability, reliability, transfer confidence, context dependence, failure
conditions, known counterexamples, complexity and decision impact. NOTHING becomes trusted because a miner found it.

What is built here, and what it extends instead of copying:
  * Panel / guards     rows dated at/after `now`, and rows whose outcome has not matured strictly before `now`, never enter; an
                       outcome supplied for an immature row is a FirewallBreach, not silently dropped. Discovery, embargo, validation
                       blocks and a reserved set of stocks are all defined on whole ISO weeks.
  * screening          singles -> pairs -> exceptions ("A and B unless C") on quantile levels, using engine.pattern_identity
                       expressions, engine.pattern_stats (local FDR from within-week outcome shuffles, Kish n_eff, prune_redundant)
                       and engine.candidates.ts_quintile_series. One weekly-sum kernel serves both the one-shot screen and the
                       year-by-year StreamingScreen, so a market-wide run never holds every name-day (mapping rule 27).
  * TrialLedger        cumulative multiple-testing accounting over EVERY discovery run ever made (hash-chained), plus alpha-investing
                       wealth; a pattern's q-value is judged against the whole search history, not its own run.
  * analyses           evidence counts, validation blocks pooled by engine.learning.knowledge.pool_effects, held-out stocks, context
                       profile, failure conditions, counterexamples, controls (ticker effect, concentration), complexity bar
                       (engine.learning.complexity), decision-impact estimate.
  * knowledge          each finding -> engine.learning.knowledge.KnowledgeObject through the existing PatternRecord bridge, promotion
                       RESEARCH, decision_effect NONE. Only fresh-data revalidation can raise HYPOTHESIS to SUPPORTED, and only the
                       promotion gate can give it a decision. MaturedRecord.gate(now) is the sole road to the trader.
  * DiscoveryEngine    `step(state, now, inputs)` is the ONE public entry the research loop calls: it schedules families so all
                       twenty-four are visited, screens, validates, dedupes against everything already known, revalidates the old.
Research filed under a real year is never released while that year is replayed in disguise (`release_filter`, mapping rule 27).
Status: IMPLEMENTED — NOT VALIDATED (C63: unit tests on planted and null worlds only; no real-data run)."""
from __future__ import annotations

import bisect
import dataclasses
import datetime as dt
import heapq
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine import pattern_identity as PI
from engine import pattern_stats as PS
from engine.learning import complexity as CX
from engine.learning import knowledge as KN
from engine.learning.core import (canonical_json, Confidence, DecisionEffect, Epistemic, FailureCause, Lifecycle, Promotion, Provenance,
                                  TemporalClass, current_code_hash)
from engine.learning.trader_view import string_reasons
from engine.research import discovery_sources as DS
from engine.research.core import (FirewallBreach, GateVerdict, MaturedRecord, Namespace, as_date, require_past, stable_hash)
from engine.research.discovery_sources import N_LEVELS, Quantiles, quantise_panel

NEVER_TRUSTED = (Promotion.RESEARCH,)          # a discovery is born in RESEARCH and stays there until the promotion gate says otherwise


class DiscoveryError(ValueError):
    """A panel, config or record the discovery engine cannot make sense of."""


# ------------------------------------------------------------------------------------------------------ configuration
@dataclasses.dataclass(frozen=True)
class DiscoveryConfig:
    horizon: int = 5                     # label horizon in sessions
    target: str = "excess"               # outcome explained: excess | abs_move | range_exp | continuation (discovery_sources.TARGETS)
    train_frac: float = 0.5              # share of usable weeks used for discovery
    n_val_blocks: int = 3                # validation periods after the discovery window
    holdout_frac: float = 0.2            # stocks reserved from discovery and time-validation, used only for transfer
    min_weeks: int = 40
    min_rows: int = 200                  # rows a candidate needs in the discovery window to be tested at all
    min_active_weeks: int = 12
    single_levels: tuple[int, ...] = (0, 4)
    top_singles: int = 30
    max_pairs: int = 1500
    max_unless: int = 200
    unless_top_pairs: int = 20
    random_pair_frac: float = 0.25
    null_reps: int = 5
    null: str = "both"                   # stock_shift (stock series rotated in time) | week_shuffle (shuffled inside each week) | both (union, conservative)
    hac_lags: int | None = None          # None = ceil(horizon / 5)
    shortlist_p: float = 0.02
    shortlist_q: float = 0.25
    max_shortlist: int = 100
    redundancy_overlap: float = 0.8
    suspicious_t: float = 10.0
    max_name_share: float = 0.35
    max_week_share: float = 0.25
    ticker_effect_min_ratio: float = 0.5
    min_independent: int = 30
    min_p_real: float = 0.5
    min_transfer: float = 0.4
    z_val: float = 1.28
    reliability_weeks: int = 26
    context_min_weeks: int = 12
    counterexamples_k: int = 5
    confirmations_needed: int = 2
    seed: int = 7

    def validate(self) -> list[str]:
        e = []
        if self.horizon < 1:
            e.append("horizon must be >= 1")
        if not 0.2 <= self.train_frac <= 0.8:
            e.append("train_frac must lie in [0.2, 0.8]")
        if self.n_val_blocks < 1 or self.null_reps < 1:
            e.append("n_val_blocks and null_reps must be >= 1")
        if not 0.0 <= self.holdout_frac < 0.6:
            e.append("holdout_frac must lie in [0, 0.6)")
        if any(not 0 <= lv < N_LEVELS for lv in self.single_levels) or not self.single_levels:
            e.append(f"single_levels must be non-empty levels in 0..{N_LEVELS - 1}")
        if not 0 < self.shortlist_p <= 0.2 or not 0 < self.shortlist_q <= 1:
            e.append("shortlist_p in (0, 0.2] and shortlist_q in (0, 1]")
        if self.null not in ("stock_shift", "week_shuffle", "both"):
            e.append("null must be stock_shift, week_shuffle or both")
        if self.target not in DS.TARGETS:
            e.append(f"target must be one of {DS.TARGETS}")
        if self.min_weeks < 20 or self.min_rows < 30:
            e.append("min_weeks >= 20 and min_rows >= 30 (a smaller test proves nothing)")
        if not 0 <= self.random_pair_frac <= 1 or self.max_shortlist < 1:
            e.append("random_pair_frac in [0,1] and max_shortlist >= 1")
        return e

    @property
    def tag(self) -> str:
        """Identity of the outcome being explained: part of every pattern id, so the same expression against a different outcome or
        horizon is a different pattern with its own history and its own trial."""
        return f"{self.target}_{self.horizon}d"

    @property
    def lags(self) -> int:
        return int(math.ceil(self.horizon / 5.0)) if self.hac_lags is None else int(self.hac_lags)


# ---------------------------------------------------------------------------------------------- weekly statistics kernel
def week_codes_of(dates: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """ISO-week code per row (calendar order) and the Monday of each week. One place, used by every window definition."""
    codes, n = PS.week_codes(dates)
    monday = (dates - pd.to_timedelta(dates.dayofweek, unit="D")).normalize()
    _, first = np.unique(codes, return_index=True)
    return codes, pd.DatetimeIndex(monday[first])


def weekly_sums(mask: np.ndarray | None, y: np.ndarray, wk: np.ndarray, n_wk: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-week sum of outcomes and count of rows under `mask` (all rows when None)."""
    if mask is None:
        return np.bincount(wk, weights=y, minlength=n_wk), np.bincount(wk, minlength=n_wk).astype(float)
    return np.bincount(wk[mask], weights=y[mask], minlength=n_wk), np.bincount(wk[mask], minlength=n_wk).astype(float)


@dataclasses.dataclass(frozen=True)
class WeeklyStat:
    mean: float
    t: float
    se: float
    n_eff: float
    n_weeks: int
    p: float

    @property
    def sign(self) -> int:
        return 0 if self.mean == 0 else (1 if self.mean > 0 else -1)


NO_STAT = WeeklyStat(0.0, 0.0, float("inf"), 0.0, 0, 1.0)


def weekly_stats_matrix(SY: np.ndarray, SW: np.ndarray, sel: np.ndarray, lags: int | None, min_weeks: int = 8
                        ) -> dict[str, np.ndarray]:
    """Week-clustered test for many candidates at once. SY, SW: (n_cand, n_wk) sums and counts; sel: weeks in the window.
    Each populated week is one observation weighted by its row count. lags=None reproduces pattern_stats.cluster_test exactly
    (weighted variance of week means / Kish n_eff); lags>=0 uses a Bartlett HAC variance of the weighted mean over the calendar
    week axis, which is what overlapping forward windows need. Candidates with fewer than `min_weeks` weeks get t = 0."""
    SY = np.atleast_2d(np.asarray(SY, dtype=float))
    SW = np.atleast_2d(np.asarray(SW, dtype=float))
    w = np.where(sel[None, :], SW, 0.0)
    tot = w.sum(axis=1)
    live = w > 0
    nweeks = live.sum(axis=1)
    x = np.divide(SY, SW, out=np.zeros_like(SY), where=SW > 0)
    p = np.divide(w, tot[:, None], out=np.zeros_like(w), where=tot[:, None] > 0)
    mean = (p * x).sum(axis=1)
    e = np.where(live, x - mean[:, None], 0.0)
    n_eff = np.divide(1.0, (p ** 2).sum(axis=1), out=np.zeros(len(tot)), where=(p ** 2).sum(axis=1) > 0)
    if lags is None:
        var = (p * e ** 2).sum(axis=1)
        se = np.sqrt(np.maximum(var, PS.VAR_FLOOR) / np.maximum(n_eff, 1.0))
    else:
        u = p * e
        var = (u ** 2).sum(axis=1)
        for l in range(1, int(lags) + 1):
            if u.shape[1] > l:
                var = var + 2.0 * (1.0 - l / (lags + 1.0)) * (u[:, :-l] * u[:, l:]).sum(axis=1)
        se = np.sqrt(np.maximum(var, PS.VAR_FLOOR))
    ok = nweeks >= min_weeks
    t = np.where(ok, mean / se, 0.0)
    df = np.maximum(nweeks - 1, 1)
    pv = np.where(ok, 2.0 * sps.t.sf(np.abs(t), df), 1.0)
    return {"mean": np.where(ok, mean, 0.0), "t": t, "se": np.where(ok, se, np.inf), "n_eff": np.where(ok, n_eff, nweeks.astype(float)),
            "n_weeks": nweeks, "p": pv}


def weekly_stat(sy: np.ndarray, sw: np.ndarray, sel: np.ndarray, lags: int | None, min_weeks: int = 8) -> WeeklyStat:
    r = weekly_stats_matrix(sy[None, :], sw[None, :], sel, lags, min_weeks)
    return WeeklyStat(float(r["mean"][0]), float(r["t"][0]), float(r["se"][0]), float(r["n_eff"][0]), int(r["n_weeks"][0]), float(r["p"][0]))


# ---------------------------------------------------------------------------------------- incremental time-series quintile
class TsQuantiler:
    """Expanding point-in-time quintile of a per-date series, resumable across year chunks: the sorted history is carried, so
    streaming the series in pieces gives exactly the levels of one pass (tested against candidates.ts_quintile_series)."""

    def __init__(self, min_history: int = 60):
        self.hist: list[float] = []
        self.min_history = min_history

    def push(self, values: pd.Series) -> pd.Series:
        out = np.full(len(values), -1, dtype=np.int8)
        for i, x in enumerate(values.astype(float).to_numpy()):
            if not np.isfinite(x):
                continue
            n_before = len(self.hist)
            bisect.insort(self.hist, x)
            if n_before + 1 < self.min_history:
                continue
            lo, hi = bisect.bisect_left(self.hist, x), bisect.bisect_right(self.hist, x)
            out[i] = min(int((lo + hi) / 2.0 / len(self.hist) * N_LEVELS), N_LEVELS - 1)
        return pd.Series(out, index=values.index)


# ------------------------------------------------------------------------------------------------------------- the panel
@dataclasses.dataclass
class Panel:
    """A guarded, week-indexed research panel. Built only by Panel.build; every window is a boolean mask over WEEKS."""
    X: pd.DataFrame
    y: np.ndarray
    dates: pd.DatetimeIndex
    tickers: np.ndarray
    wk: np.ndarray
    n_wk: int
    week_start: pd.DatetimeIndex
    fit_rows: np.ndarray                    # rows of non-reserved stocks (discovery and time-validation use only these)
    holdout_rows: np.ndarray
    sel_train: np.ndarray
    sel_blocks: list[np.ndarray]
    embargo_weeks: int
    now: pd.Timestamp
    outcomes_seen_through: pd.Timestamp
    cfg: DiscoveryConfig
    report: dict
    sector: np.ndarray | None = None
    _q: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def build(cls, X: pd.DataFrame, y: pd.Series, now, cfg: DiscoveryConfig, matured_at: pd.Series | None = None,
              sectors: Mapping[str, str] | None = None, on_immature: str = "raise") -> "Panel":
        errs = cfg.validate()
        if errs:
            raise DiscoveryError("; ".join(errs))
        if on_immature not in ("raise", "drop"):
            raise DiscoveryError("on_immature must be 'raise' or 'drop'")
        if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels != 2:
            raise DiscoveryError("X must be indexed by (date, ticker)")
        if not len(X) or not X.index.isin(y.index).any():
            raise DiscoveryError("y shares no (date, ticker) rows with X")
        if X.index.has_duplicates:
            raise DiscoveryError("duplicate (date, ticker) rows")
        now_ts = pd.Timestamp(as_date(now))
        rep: dict[str, Any] = {"rows_in": int(len(X))}
        d_all = pd.DatetimeIndex(X.index.get_level_values(0))
        future = d_all >= now_ts
        rep["rows_dated_at_or_after_now"] = int(future.sum())
        X = X[~future]
        yy = y.reindex(X.index)
        udates = np.sort(pd.DatetimeIndex(X.index.get_level_values(0)).unique())
        if matured_at is not None:
            m = matured_at.reindex(X.index)
            immature = (m.isna() | (pd.to_datetime(m) >= now_ts)).to_numpy()
        else:
            pos = pd.Series(np.arange(len(udates)), index=udates).reindex(X.index.get_level_values(0)).to_numpy()
            immature = pos + cfg.horizon >= len(udates)          # the exit session would be the last seen or beyond it
            # if the calendar's last session is itself before `now` the exit of pos+h needs a session that exists
        leaked = immature & yy.notna().to_numpy()
        rep["immature_rows"] = int(immature.sum())
        rep["immature_rows_with_outcome"] = int(leaked.sum())
        if leaked.any() and on_immature == "raise":
            raise FirewallBreach(f"{int(leaked.sum())} rows carry an outcome that has not matured strictly before now={now_ts.date()}")
        keep = ~immature & yy.notna().to_numpy()
        X, yy = X[keep], yy[keep]
        if len(X) == 0:
            raise DiscoveryError("no usable rows: nothing matured before now")
        order = np.lexsort((X.index.get_level_values(1), X.index.get_level_values(0)))
        X, yy = X.iloc[order], yy.iloc[order]
        dates = pd.DatetimeIndex(X.index.get_level_values(0))
        tick = np.asarray(X.index.get_level_values(1)).astype(str)
        wk, week_start = week_codes_of(dates)
        n_wk = len(week_start)
        if n_wk < cfg.min_weeks:
            raise DiscoveryError(f"only {n_wk} usable weeks; need {cfg.min_weeks}")
        emb = max(1, int(math.ceil(cfg.horizon / 5.0)))
        t_end = int(n_wk * cfg.train_frac)
        edges = np.linspace(t_end, n_wk, cfg.n_val_blocks + 1).astype(int)
        train = np.zeros(n_wk, bool)
        train[: max(t_end - emb, 0)] = True
        blocks = []
        for a, b in zip(edges[:-1], edges[1:]):
            s = np.zeros(n_wk, bool)
            s[a: max(b - emb, a)] = True
            blocks.append(s)
        hold = _holdout_mask(np.unique(tick), cfg.holdout_frac, cfg.seed)
        is_hold = np.isin(tick, np.array(sorted(hold), dtype=str)) if hold else np.zeros(len(tick), bool)
        last_idx = int(np.searchsorted(udates, dates.max()))
        seen = pd.Timestamp(udates[min(last_idx + cfg.horizon, len(udates) - 1)])
        sec = None
        if sectors:
            sec = np.array([sectors.get(t, "?") for t in tick])
        rep.update({"rows": int(len(X)), "weeks": n_wk, "reserved_stocks": len(hold), "embargo_weeks": emb,
                    "train_weeks": int(train.sum()), "block_weeks": [int(s.sum()) for s in blocks]})
        return cls(X, yy.to_numpy(dtype=float), dates, tick, wk, n_wk, week_start, ~is_hold, is_hold, train, blocks, emb, now_ts, seen,
                   cfg, rep, sec)

    def sel_val(self) -> np.ndarray:
        return np.any(self.sel_blocks, axis=0)

    def quantiles(self, cols: Sequence[str]) -> Quantiles:
        need = [c for c in cols if c not in self._q]
        if need:
            q = quantise_panel(self.X[need])
            for j, c in enumerate(q.features):
                self._q[c] = q.codes[:, j]
        return Quantiles(tuple(cols), np.column_stack([self._q[c] for c in cols]) if cols else np.empty((len(self.X), 0), np.int8))

    def window_rows(self, sel: np.ndarray, fit_only: bool = True) -> np.ndarray:
        rows = sel[self.wk]
        return rows & self.fit_rows if fit_only else rows

    def windows(self) -> dict[str, tuple[str, str]]:
        def span(sel):
            idx = np.flatnonzero(sel)
            return (str(self.week_start[idx[0]].date()), str((self.week_start[idx[-1]] + pd.Timedelta(days=6)).date())) if len(idx) else ("", "")
        out = {"train": span(self.sel_train)}
        for k, s in enumerate(self.sel_blocks):
            out[f"val{k}"] = span(s)
        return out


def _holdout_mask(tickers: np.ndarray, frac: float, seed: int) -> set[str]:
    """Reserved stocks by hash of (seed, name): independent of row order and of which other names exist, so the reserved set of a
    given name never changes as the universe grows (the name-order tie-break lesson)."""
    if frac <= 0:
        return set()
    cut = int(frac * 10_000)
    return {t for t in tickers if int(stable_hash({"s": seed, "t": str(t)}, 8), 16) % 10_000 < cut}


# ------------------------------------------------------------------------------------- cumulative multiple-testing account
@dataclasses.dataclass(frozen=True)
class RunEntry:
    run_id: str
    now: str
    n_trials: int
    n_new_keys: int
    by_family: tuple[tuple[str, int], ...]
    min_p: float
    p_digest: str
    prev_hash: str
    hash: str


class TrialLedger:
    """Every candidate ever tested by any discovery run, kept so that no pattern is judged against its own run alone.

    The pool holds one p-value per DISTINCT (pattern identity, data window) - re-running an identical search on identical data
    adds no new trial, but the same pattern on a different or longer window is a new look and counts. A new batch's q-value is
    Benjamini-Hochberg against the whole pool plus itself (a deliberately conservative ranking: the monotone step of BH would
    only lower q). Runs are hash-chained, so a history that was edited or reordered is detected by `verify`."""

    def __init__(self):
        self._pool = np.empty(0, dtype=float)
        self._keys: set[str] = set()
        self.runs: list[RunEntry] = []
        self.family_trials: dict[str, int] = {}
        self.family_confirmed: dict[str, int] = {}
        self.family_failed: dict[str, int] = {}
        self.times_tested: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._pool)

    @property
    def total_trials(self) -> int:
        return int(sum(r.n_trials for r in self.runs))

    @property
    def distinct_trials(self) -> int:
        return len(self._pool)

    def cumulative_q(self, p_new: Sequence[float]) -> np.ndarray:
        """q-value of each new p against pool + batch. Empty batch -> empty array."""
        p = np.clip(np.asarray(p_new, dtype=float), 0.0, 1.0)
        if p.size == 0:
            return p
        pool = np.sort(self._pool)
        batch = np.sort(p)
        m = len(pool) + len(batch)
        rank = np.searchsorted(pool, p, side="right") + np.searchsorted(batch, p, side="right")
        return np.minimum(1.0, p * m / np.maximum(rank, 1))

    def register(self, run_id: str, now, ids: Sequence[str], p: Sequence[float], families: Sequence[str], data_key: str) -> np.ndarray:
        """Account a batch and return its cumulative q-values. `ids`, `p`, `families` align; untestable candidates must be passed
        with p = 1.0 (a candidate that was generated but too thin to test is still a step of the search)."""
        if not (len(ids) == len(p) == len(families)):
            raise DiscoveryError("ledger batch arrays must align")
        if any(r.run_id == run_id for r in self.runs):
            raise DiscoveryError(f"run {run_id!r} already registered")
        parr = np.asarray(p, dtype=float)
        if parr.size and (np.isnan(parr).any() or (parr < 0).any() or (parr > 1).any()):
            raise DiscoveryError("p-values must lie in [0, 1]")
        q = self.cumulative_q(parr)
        add_p, new_keys = [], 0
        by_fam: dict[str, int] = {}
        for i, pi, f in zip(ids, parr, families):
            k = stable_hash({"id": i, "d": data_key}, 12)
            by_fam[f] = by_fam.get(f, 0) + 1
            self.times_tested[i] = self.times_tested.get(i, 0) + 1
            if k not in self._keys:
                self._keys.add(k)
                add_p.append(pi)
                new_keys += 1
        if add_p:
            self._pool = np.concatenate([self._pool, np.asarray(add_p)])
        for f, n in by_fam.items():
            self.family_trials[f] = self.family_trials.get(f, 0) + n
        prev = self.runs[-1].hash if self.runs else ""
        digest = stable_hash([round(float(x), 9) for x in parr], 12)
        body = {"run": run_id, "now": str(as_date(now)), "n": len(parr), "new": new_keys, "fam": sorted(by_fam.items()),
                "p": digest, "prev": prev}
        self.runs.append(RunEntry(run_id, str(as_date(now)), len(parr), new_keys, tuple(sorted(by_fam.items())),
                                  float(parr.min()) if parr.size else 1.0, digest, prev, stable_hash(body, 20)))
        return q

    def confirm(self, family: str, ok: bool) -> None:
        """A finding of this family later survived (or failed) fresh-data revalidation: the family's own false-discovery record."""
        d = self.family_confirmed if ok else self.family_failed
        d[family] = d.get(family, 0) + 1

    def family_fdr(self, family: str, prior_strength: float = 2.0) -> float:
        """Share of this family's revalidated findings that failed, shrunk toward one half so a family with two results is not 0 or 1."""
        ok, bad = self.family_confirmed.get(family, 0), self.family_failed.get(family, 0)
        return (bad + prior_strength / 2.0) / (ok + bad + prior_strength)

    def expected_best_null_t(self) -> float:
        """The |t| the best of all distinct trials would reach by luck alone (sqrt(2 ln m)); a single result has to beat it."""
        return float(math.sqrt(2.0 * math.log(max(self.distinct_trials, 2))))

    def verify(self) -> list[str]:
        errs, prev = [], ""
        for r in self.runs:
            body = {"run": r.run_id, "now": r.now, "n": r.n_trials, "new": r.n_new_keys, "fam": sorted(r.by_family), "p": r.p_digest,
                    "prev": prev}
            if r.prev_hash != prev:
                errs.append(f"run {r.run_id}: broken chain")
            if stable_hash(body, 20) != r.hash:
                errs.append(f"run {r.run_id}: content does not match its hash")
            prev = r.hash
        if len(self._pool) != sum(r.n_new_keys for r in self.runs):
            errs.append("pool size disagrees with the runs' new-key counts")
        return errs

    def to_dict(self) -> dict:
        return {"pool": [round(float(x), 9) for x in self._pool], "keys": sorted(self._keys),
                "runs": [dataclasses.asdict(r) for r in self.runs], "ft": self.family_trials, "fc": self.family_confirmed,
                "ff": self.family_failed, "tt": self.times_tested}

    @classmethod
    def from_dict(cls, d: Mapping) -> "TrialLedger":
        t = cls()
        t._pool = np.asarray(d["pool"], dtype=float)
        t._keys = set(d["keys"])
        t.runs = [RunEntry(**{**r, "by_family": tuple(tuple(x) for x in r["by_family"])}) for r in d["runs"]]
        t.family_trials, t.family_confirmed, t.family_failed, t.times_tested = d["ft"], d["fc"], d["ff"], d["tt"]
        bad = t.verify()
        if bad:
            raise DiscoveryError("ledger failed verification on load: " + "; ".join(bad))
        return t


class AlphaWealth:
    """Foster-Stine alpha-investing across runs. Each run may spend part of the wealth on its tests; a CONFIRMED discovery (one
    that survives out-of-sample validation, not one that merely screens well) pays wealth back, a run that finds nothing only
    spends. When wealth is gone the engine must stop widening its search until something is confirmed."""

    def __init__(self, wealth: float = 0.5, payout: float = 0.05, spend_frac: float = 0.1, floor: float = 0.005):
        self.wealth, self.payout, self.spend_frac, self.floor = float(wealth), float(payout), float(spend_frac), float(floor)
        self.history: list[tuple[str, float]] = []

    def per_test_alpha(self, n_planned: int) -> float:
        if n_planned <= 0 or self.wealth <= self.floor:
            return 0.0
        return min(0.05, self.wealth * self.spend_frac / n_planned)

    def max_trials(self, min_alpha: float = 1e-5) -> int:
        """How many tests the current wealth can afford at no less than `min_alpha` each."""
        return 0 if self.wealth <= self.floor else int(self.wealth * self.spend_frac / min_alpha)

    def settle(self, run_id: str, n_tests: int, alpha: float, n_confirmed: int) -> float:
        spent = n_tests * alpha / max(1.0 - alpha, 1e-9)
        self.wealth = max(self.wealth - spent, 0.0) + n_confirmed * self.payout
        self.history.append((run_id, self.wealth))
        return self.wealth


def shift_within_stocks(y: np.ndarray, tickers: np.ndarray, rng: np.random.Generator, min_shift: int = 20) -> np.ndarray:
    """Null outcomes that keep each stock's own history (drift, volatility clustering, autocorrelation) but sever its link to the features:
    every stock's outcome series is rotated in time by a random amount of at least `min_shift` rows. Shuffling inside a week does not do
    this - a feature that merely selects the same stocks for months would look 'significant' against a null with no stock persistence."""
    y = np.asarray(y, dtype=float)
    out = y.copy()
    order = np.argsort(tickers, kind="stable")
    cuts = np.flatnonzero(np.r_[True, tickers[order][1:] != tickers[order][:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        n = len(idx)
        if n < 2 * min_shift + 2:
            out[idx] = y[idx][rng.permutation(n)]
            continue
        out[idx] = np.roll(y[idx], int(rng.integers(min_shift, n - min_shift)))
    return out


def calibrated_p(p_param: np.ndarray, t: np.ndarray, null_t: np.ndarray) -> np.ndarray:
    """Never more optimistic than the null: the parametric p is raised to the empirical exceedance rate of |t| in the null distribution
    (add-one, so it can never be 0). This is what makes the p-values entering the cumulative ledger honest about persistence."""
    n = np.sort(np.asarray(null_t, dtype=float))
    if n.size < 2:
        return np.asarray(p_param, dtype=float)
    ge = n.size - np.searchsorted(n, np.abs(np.asarray(t, dtype=float)), side="left")
    return np.maximum(np.asarray(p_param, dtype=float), (1.0 + ge) / (1.0 + n.size))


# ------------------------------------------------------------------------------------------------ candidate screening
@dataclasses.dataclass(frozen=True)
class Cand:
    expr: PI.Expression
    origin: str                      # single | top_pair | ctx_pair | random_pair | unless
    families: tuple[str, ...]        # source families of the features involved

    @property
    def text(self) -> str:
        return self.expr.text

    @property
    def transform(self) -> str:
        return "ts_quintile" if any(f.startswith("m_") for f in self.expr.features) else "xs_quintile"

    def id(self, tag: str) -> str:
        return PI.pattern_id(self.expr, self.transform, tag)


@dataclasses.dataclass
class ScreenResult:
    cands: list[Cand]
    SY: np.ndarray
    SW: np.ndarray
    disc: dict[str, np.ndarray]
    tested: np.ndarray
    null_t: np.ndarray
    p_halluc: np.ndarray
    q_cum: np.ndarray
    trial_ids: list[str]
    fam_of_col: dict[str, str]
    n_generated: int

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"text": [c.text for c in self.cands], "origin": [c.origin for c in self.cands], "tested": self.tested,
                             "m_disc": self.disc["mean"], "t_disc": self.disc["t"], "p": self.disc["p"], "q_cum": self.q_cum,
                             "p_halluc": self.p_halluc, "n_eff": self.disc["n_eff"]})


class Screener:
    """Three-stage search on the discovery weeks only (validation weeks are never read here): every single tail-level condition,
    then pairs drawn from the strongest singles plus SEEDED random pairs (so an unfashionable feature can still be paired), then
    "A and B unless C" exceptions on the strongest pairs. Each stage's statistic is the weekly-sum kernel; the null re-runs the
    same candidate list on outcomes shuffled inside each week."""

    def __init__(self, panel: Panel, columns: Sequence[str], fam_of_col: Mapping[str, str], cfg: DiscoveryConfig):
        self.p, self.cfg = panel, cfg
        self.cols = sorted(columns)
        self.stock = [c for c in self.cols if not c.startswith("m_")]
        self.ctx = [c for c in self.cols if c.startswith("m_")]
        self.fam_of_col = dict(fam_of_col)
        q = panel.quantiles(self.cols)
        fit = panel.fit_rows
        self.codes = {c: q.codes[fit, j] for j, c in enumerate(q.features)}
        self.y = panel.y[fit]
        self.wk = panel.wk[fit]
        self.tk = panel.tickers[fit]
        self.rng = np.random.default_rng(cfg.seed)

    # -- evaluation of a candidate list -> weekly sums
    def evaluate(self, cands: Sequence[Cand], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_wk = self.p.n_wk
        SY = np.zeros((len(cands), n_wk))
        SW = np.zeros((len(cands), n_wk))
        by_col: dict[str, list[tuple[int, int]]] = {}
        by_pair: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
        rest: list[int] = []
        for i, c in enumerate(cands):
            b, u = c.expr.base, c.expr.unless
            if not u and len(b) == 1:
                by_col.setdefault(b[0].feature, []).append((i, b[0].level))
            elif not u and len(b) == 2:
                by_pair.setdefault((b[0].feature, b[1].feature), []).append((i, b[0].level, b[1].level))
            else:
                rest.append(i)
        for col, items in by_col.items():
            lv = self.codes[col]
            ok = lv >= 0
            idx = lv[ok].astype(np.int64) * n_wk + self.wk[ok]
            cy = np.bincount(idx, weights=y[ok], minlength=N_LEVELS * n_wk).reshape(N_LEVELS, n_wk)
            cw = np.bincount(idx, minlength=N_LEVELS * n_wk).reshape(N_LEVELS, n_wk)
            for i, l in items:
                SY[i], SW[i] = cy[l], cw[l]
        for (ca, cb), items in by_pair.items():
            la, lb = self.codes[ca], self.codes[cb]
            ok = (la >= 0) & (lb >= 0)
            idx = (la[ok].astype(np.int64) * N_LEVELS + lb[ok]) * n_wk + self.wk[ok]
            cells = N_LEVELS * N_LEVELS * n_wk
            cy = np.bincount(idx, weights=y[ok], minlength=cells).reshape(N_LEVELS * N_LEVELS, n_wk)
            cw = np.bincount(idx, minlength=cells).reshape(N_LEVELS * N_LEVELS, n_wk)
            for i, a, b in items:
                SY[i], SW[i] = cy[a * N_LEVELS + b], cw[a * N_LEVELS + b]
        for i in rest:
            m = self.row_mask(cands[i].expr)
            SY[i], SW[i] = weekly_sums(m, y, self.wk, n_wk)
        return SY, SW

    def row_mask(self, expr: PI.Expression) -> np.ndarray:
        m = np.ones(len(self.y), bool)
        for t in expr.base:
            m &= self.codes[t.feature] == t.level
        for e in expr.unless:
            c = self.codes[e.feature]
            m &= (c != e.level) & (c >= 0)
        return m

    def _stats(self, SY: np.ndarray, SW: np.ndarray) -> tuple[dict, np.ndarray]:
        sel = self.p.sel_train
        st = weekly_stats_matrix(SY, SW, sel, self.cfg.lags, self.cfg.min_active_weeks)
        rows = np.where(sel[None, :], SW, 0.0).sum(axis=1)
        tested = (rows >= self.cfg.min_rows) & (st["n_weeks"] >= self.cfg.min_active_weeks)
        for k in ("mean", "t"):
            st[k] = np.where(tested, st[k], 0.0)
        st["p"] = np.where(tested, st["p"], 1.0)
        return st, tested

    # -- candidate generation
    def _fams(self, expr: PI.Expression) -> tuple[str, ...]:
        return tuple(sorted({self.fam_of_col.get(f, "?") for f in expr.features}))

    def _mk(self, terms: Sequence[PI.Term], unless: Sequence[PI.Term], origin: str) -> Cand | None:
        try:
            e = PI.Expression.make(terms, unless)
        except PI.IdentityError:
            return None
        return Cand(e, origin, self._fams(e))

    def singles(self) -> list[Cand]:
        return [c for col in self.stock for lv in self.cfg.single_levels
                if (c := self._mk([PI.Term(col, lv)], [], "single")) is not None]

    def pairs(self, pool: Sequence[PI.Term], seen: set[str]) -> list[Cand]:
        cfg, out = self.cfg, []
        budget = cfg.max_pairs

        def add(c: Cand | None) -> None:
            if c is not None and c.text not in seen and len(out) < budget:
                seen.add(c.text)
                out.append(c)

        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                if a.feature != b.feature and not (a.feature.startswith("m_") and b.feature.startswith("m_")):
                    add(self._mk([a, b], [], "top_pair"))
        n_rand = int(cfg.max_pairs * cfg.random_pair_frac)
        n_ctx = int(cfg.max_pairs * 0.25) if self.ctx else 0
        stock_terms = [PI.Term(c, l) for c in self.stock for l in cfg.single_levels]
        if self.ctx and pool:
            ctx_terms = [PI.Term(c, l) for c in self.ctx for l in cfg.single_levels]
            for _ in range(n_ctx * 3):
                if sum(1 for c in out if c.origin == "ctx_pair") >= n_ctx:
                    break
                a = pool[int(self.rng.integers(len(pool)))]
                b = ctx_terms[int(self.rng.integers(len(ctx_terms)))]
                if not a.feature.startswith("m_"):
                    add(self._mk([a, b], [], "ctx_pair"))
        if len(stock_terms) > 1:
            for _ in range(n_rand * 4):
                if sum(1 for c in out if c.origin == "random_pair") >= n_rand:
                    break
                a, b = (stock_terms[int(k)] for k in self.rng.integers(len(stock_terms), size=2))
                if a.feature != b.feature:
                    add(self._mk([a, b], [], "random_pair"))
        return out

    def exceptions(self, ranked_pairs: Sequence[Cand], seen: set[str]) -> list[Cand]:
        cfg, out = self.cfg, []
        universe = [PI.Term(c, l) for c in self.cols for l in (0, N_LEVELS - 1)]
        if not universe:
            return out
        for base in ranked_pairs[: cfg.unless_top_pairs]:
            for _ in range(max(1, cfg.max_unless // max(cfg.unless_top_pairs, 1)) * 3):
                if len(out) >= cfg.max_unless:
                    return out
                e = universe[int(self.rng.integers(len(universe)))]
                if e.feature in base.expr.features:
                    continue
                c = self._mk(list(base.expr.base), [e], "unless")
                if c is not None and c.text not in seen:
                    seen.add(c.text)
                    out.append(c)
        return out

    def _search(self, y: np.ndarray) -> tuple[list[Cand], np.ndarray, np.ndarray, dict, np.ndarray]:
        """The whole staged search against outcome vector `y`: singles, pairs from the strongest singles, exceptions on the strongest
        pairs. Later stages are chosen from earlier stages' statistics, so the search itself manufactures large |t|; the null must
        repeat it, not merely re-test the final list."""
        cfg = self.cfg
        seen: set[str] = set()
        cands = self.singles()
        seen.update(c.text for c in cands)
        SY, SW = self.evaluate(cands, y)
        st, tested = self._stats(SY, SW)
        order = np.argsort(-np.abs(st["t"]) * tested, kind="mergesort")
        pool = [cands[int(i)].expr.base[0] for i in order[: cfg.top_singles] if tested[i]]
        pairs = self.pairs(pool, seen)
        if pairs:
            pSY, pSW = self.evaluate(pairs, y)
            pst, ptested = self._stats(pSY, pSW)
            cands, SY, SW = cands + pairs, np.vstack([SY, pSY]), np.vstack([SW, pSW])
            st = {k: np.concatenate([st[k], pst[k]]) for k in st}
            tested = np.concatenate([tested, ptested])
        two = [i for i, c in enumerate(cands) if len(c.expr.base) == 2]
        ranked = sorted(two, key=lambda i: (-abs(st["t"][i]) * tested[i], cands[i].text))
        exc = self.exceptions([cands[i] for i in ranked if tested[i]], seen)
        if exc:
            eSY, eSW = self.evaluate(exc, y)
            est, etested = self._stats(eSY, eSW)
            cands, SY, SW = cands + exc, np.vstack([SY, eSY]), np.vstack([SW, eSW])
            st = {k: np.concatenate([st[k], est[k]]) for k in st}
            tested = np.concatenate([tested, etested])
        return cands, SY, SW, st, tested

    def run(self) -> ScreenResult:
        cfg = self.cfg
        cands, SY, SW, st, tested = self._search(self.y)
        null_t = self._null()
        real_t = np.sort(np.abs(st["t"][tested])) if tested.any() else np.array([0.0])
        ph = np.array([PS.local_fdr(abs(t), null_t, real_t) if ok else 1.0 for t, ok in zip(st["t"], tested)])
        st["p"] = np.where(tested, calibrated_p(st["p"], st["t"], null_t), 1.0)
        ids = [c.id(cfg.tag) for c in cands]
        return ScreenResult(cands, SY, SW, st, tested, null_t, ph, np.ones(len(cands)), ids, self.fam_of_col, len(cands))

    def _null(self) -> np.ndarray:
        """|t| of every testable candidate when the WHOLE search is re-run on outcomes that no longer belong to these features: each stock's
        series rotated in time (keeps stock persistence) and shuffled inside each week (keeps week structure), unioned. Because the search
        is repeated, the null includes the selection effect of choosing pairs and exceptions by their own statistics."""
        out = []
        modes = ("stock_shift", "week_shuffle") if self.cfg.null == "both" else (self.cfg.null,)
        saved = self.rng
        try:
            for r in range(self.cfg.null_reps):
                for mode in modes:
                    rng = np.random.default_rng(self.cfg.seed * 1000 + r)
                    yp = (shift_within_stocks(self.y, self.tk, rng, max(self.cfg.horizon * 4, 20)) if mode == "stock_shift"
                          else PS.permute_within_clusters(self.y, self.wk, rng))
                    self.rng = np.random.default_rng(self.cfg.seed * 7919 + r)
                    _, _, _, st, tested = self._search(yp)
                    out.append(np.abs(st["t"][tested]))
        finally:
            self.rng = saved
        allt = np.concatenate(out) if out else np.array([])
        return np.sort(allt) if allt.size else np.array([0.0])


# ------------------------------------------------------------------------------------------------ streaming screen
class StreamingScreen:
    """The same weekly-sum kernel fed one year at a time. Holds only per-(candidate, week) sums plus a bounded heap of the worst
    counter-examples per candidate ("exception rows only", mapping rule 27), never the panel. Quantiles are cross-sectional per
    day (so a chunk is self-contained) or, for m_* columns, carried through TsQuantiler so the levels equal a one-pass build."""

    def __init__(self, cands: Sequence[Cand], n_weeks_hint: int = 4096, keep_worst: int = 5, min_names: int = 5):
        for c in cands:
            if c.expr.unless:
                raise DiscoveryError("StreamingScreen covers singles and pairs; exceptions are evaluated on the shortlist")
        self.cands = list(cands)
        self.cols = sorted({f for c in cands for f in c.expr.features})
        self.min_names = min_names
        self.sy: dict[int, float] = {}
        self.sw: dict[int, float] = {}
        self.ts = {c: TsQuantiler() for c in self.cols if c.startswith("m_")}
        self.worst_up: list[list[tuple[float, str]]] = [[] for _ in cands]      # largest positive outcomes seen per candidate
        self.worst_down: list[list[tuple[float, str]]] = [[] for _ in cands]    # most negative outcomes (stored negated)
        self.keep_worst = keep_worst
        self.last_end: pd.Timestamp | None = None
        self.week_sum_y = np.zeros((len(cands), 0))
        self.week_sum_w = np.zeros((len(cands), 0))
        self.week_keys: list[int] = []
        self.rows_seen = 0

    def _grow(self, keys: Sequence[int]) -> np.ndarray:
        pos = {k: i for i, k in enumerate(self.week_keys)}
        new = [k for k in sorted(set(keys)) if k not in pos]
        if new:
            merged = sorted(set(self.week_keys) | set(new))
            grown_y = np.zeros((len(self.cands), len(merged)))
            grown_w = np.zeros((len(self.cands), len(merged)))
            old_pos = {k: i for i, k in enumerate(merged)}
            for k, i in ((k, i) for i, k in enumerate(self.week_keys)):
                grown_y[:, old_pos[k]] = self.week_sum_y[:, i]
                grown_w[:, old_pos[k]] = self.week_sum_w[:, i]
            self.week_sum_y, self.week_sum_w, self.week_keys = grown_y, grown_w, merged
        return np.array([self.week_keys.index(k) for k in keys])

    def add_chunk(self, X: pd.DataFrame, y: pd.Series, now=None) -> int:
        """Accumulate one chunk (rows already trimmed to their year). Chunks must arrive in calendar order and, with `now`, must
        not contain rows dated at/after it. Returns the number of rows used."""
        d = pd.DatetimeIndex(X.index.get_level_values(0))
        if len(d) == 0:
            return 0
        if self.last_end is not None and d.min() <= self.last_end:
            raise DiscoveryError("chunks must be supplied in calendar order without overlap")
        if now is not None and d.max() >= pd.Timestamp(as_date(now)):
            raise FirewallBreach("a streamed chunk contains rows dated at/after now")
        self.last_end = d.max()
        ok = y.reindex(X.index).notna().to_numpy()
        Xc, yc = X[ok], y.reindex(X.index)[ok].to_numpy(dtype=float)
        dc = pd.DatetimeIndex(Xc.index.get_level_values(0))
        if len(Xc) == 0:
            return 0
        iso = dc.isocalendar()
        key = (iso["year"].astype(np.int64) * 100 + iso["week"].astype(np.int64)).to_numpy()
        uniq, inv = np.unique(key, return_inverse=True)
        cols = {}
        names = pd.Series(1, index=Xc.index).groupby(level=0).transform("sum").to_numpy()
        for c in self.cols:
            if c.startswith("m_"):
                per = Xc[c].groupby(level=0).first().sort_index()
                cols[c] = self.ts[c].push(per).reindex(dc).fillna(-1).to_numpy(dtype=np.int8)
            else:
                r = Xc[c].groupby(level=0).rank(pct=True).to_numpy()
                lv = np.minimum(np.nan_to_num(r, nan=-1.0) * N_LEVELS, N_LEVELS - 1).astype(np.int8)
                lv[~np.isfinite(r)] = -1
                lv[names < self.min_names] = -1
                cols[c] = lv
        wpos = self._grow(list(uniq))[inv]
        n_wk = len(self.week_keys)
        tick = np.asarray(Xc.index.get_level_values(1)).astype(str)
        for i, c in enumerate(self.cands):
            m = np.ones(len(yc), bool)
            for t in c.expr.base:
                m &= cols[t.feature] == t.level
            if not m.any():
                continue
            self.week_sum_y[i] += np.bincount(wpos[m], weights=yc[m], minlength=n_wk)
            self.week_sum_w[i] += np.bincount(wpos[m], minlength=n_wk)
            hit = np.flatnonzero(m)
            for heap, vals in ((self.worst_up[i], yc[hit]), (self.worst_down[i], -yc[hit])):
                for j in np.argsort(-vals)[: self.keep_worst]:
                    item = (float(vals[j]), f"{dc[hit[j]].date()}|{tick[hit[j]]}|{yc[hit[j]]:+.5f}")
                    if len(heap) < self.keep_worst:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
        self.rows_seen += len(yc)
        return len(yc)

    def stats(self, first_frac: float | None = None, lags: int | None = None, min_weeks: int = 8) -> dict[str, np.ndarray]:
        """Weekly statistics of the accumulated candidates over all weeks, or over the first `first_frac` of them."""
        n = len(self.week_keys)
        sel = np.ones(n, bool)
        if first_frac is not None:
            sel[int(n * first_frac):] = False
        return weekly_stats_matrix(self.week_sum_y, self.week_sum_w, sel, lags, min_weeks)

    def exceptions(self, i: int, direction: int) -> list[str]:
        """The worst outcomes AGAINST the pattern's direction seen for candidate i ('date|ticker|y'), most extreme first. These
        are the only rows a streamed run keeps."""
        heap = self.worst_down[i] if direction > 0 else self.worst_up[i]
        return [s for _, s in sorted(heap, reverse=True)]


# --------------------------------------------------------------------------------------------- section-13 record parts
@dataclasses.dataclass(frozen=True)
class EvidenceCounts:
    rows: int
    active_weeks: int
    episodes: int                # runs of consecutive active weeks: one regime of firing counts once
    names: int
    independent_weeks: int       # active weeks whose forward windows cannot overlap
    n_eff: float
    independent: int             # min(independent_weeks, names): the count a second study would have to reproduce


@dataclasses.dataclass(frozen=True)
class BlockStat:
    block: int
    mean: float
    t: float
    n_weeks: int


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    blocks: tuple[BlockStat, ...]
    pooled_mean: float | None        # signed, in excess-return units
    pooled_se: float | None
    pooled_t: float | None
    i2: float | None
    agree: int                       # blocks whose sign matches discovery
    n_blocks: int
    conf_factor: float               # Phi(t) when the pooled sign matches discovery, else 0


@dataclasses.dataclass(frozen=True)
class TransferProfile:
    cross_era: float | None
    cross_stock: float | None
    cross_sector: float | None
    cross_regime: float | None
    transfer: float | None           # 0 = no better than chance, 1 = holds everywhere it was tested
    tested: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ContextCell:
    feature: str
    bucket: str                      # low (levels 0-1) | mid (2) | high (3-4)
    levels: tuple[int, int]
    mean: float
    t: float
    n_weeks: int


@dataclasses.dataclass(frozen=True)
class FailureCondition:
    feature: str
    levels: tuple[int, int]
    kind: str                        # flip | vanish
    mean: float
    t: float
    n_weeks: int


@dataclasses.dataclass(frozen=True)
class Counterexample:
    date: str
    ticker: str
    outcome: float
    z: float


@dataclasses.dataclass(frozen=True)
class ControlResult:
    ticker_effect_ratio: float | None
    max_name_share: float
    max_week_share: float
    flags: tuple[str, ...]
    jackknife_min_t: float | None = None


@dataclasses.dataclass(frozen=True)
class ComplexityAudit:
    units: float
    bar_t: float
    t_used: float | None
    earns: bool | None


@dataclasses.dataclass(frozen=True)
class DecisionImpact:
    proposed_effects: tuple[str, ...]
    gain_per_week: float | None      # signed validation mean: what one position in the direction of the pattern earned per week
    gain_t: float | None
    redundant_with: tuple[str, ...] = ()
    status: str = "ESTIMATE_ONLY"    # a discovery never changes a decision itself


@dataclasses.dataclass(frozen=True)
class Dossier:
    """Everything section 13 requires about one discovered pattern, in MATURED_RESEARCH_STATE. It holds dates and tickers
    (counterexamples), so it never leaves the research side; the KnowledgeObject built from it is identity-free."""
    pattern_id: str
    text: str
    origin: str
    families: tuple[str, ...]
    run_id: str
    discovered_at: str
    outcomes_seen_through: str
    windows: dict[str, tuple[str, str]]
    direction: int
    t_disc: float
    m_disc: float
    p_raw: float
    p_halluc: float
    q_cum: float
    trials_so_far: int
    evidence: EvidenceCounts
    validation: ValidationResult
    holdout_t: float | None
    truth: float | None
    reliability: float | None
    transfer: TransferProfile
    context_dependence: float | None
    context_feature: str | None
    contexts: tuple[ContextCell, ...]
    failure_conditions: tuple[FailureCondition, ...]
    failed_periods: int
    counter_rate: float
    counter_excess: float
    counterexamples: tuple[Counterexample, ...]
    complexity: ComplexityAudit
    impact: DecisionImpact
    controls: ControlResult
    temporal: str
    verdict: GateVerdict
    reasons: tuple[str, ...]
    confirmations: int = 0
    namespace: Namespace = Namespace.MATURED_RESEARCH

    def digest(self) -> str:
        return stable_hash(self, 20)

    def to_dict(self) -> dict:
        return json.loads(canonical_json(self))


# --------------------------------------------------------------------------------------------------------- the analyzer
def _phi(z: float) -> float:
    return float(PS.norm_cdf(z))


def _episodes(active: np.ndarray) -> int:
    a = np.asarray(active, dtype=bool)
    return int(((~np.r_[False, a[:-1]]) & a).sum())


def _greedy_gap(active_idx: np.ndarray, gap: int) -> int:
    n, last = 0, -10 ** 9
    for i in active_idx:
        if i - last >= gap:
            n += 1
            last = int(i)
    return n


class Analyzer:
    """Section-13 analyses of one pattern mask on a Panel. Every method takes a boolean mask over ALL panel rows and returns a
    typed piece of the Dossier; none of them looks at rows the window definitions exclude (embargo weeks, immature rows)."""

    def __init__(self, panel: Panel, columns: Sequence[str], fam_of_col: Mapping[str, str], cfg: DiscoveryConfig):
        self.p, self.cfg = panel, cfg
        self.cols = sorted(columns)
        self.fam_of_col = dict(fam_of_col)
        q = panel.quantiles(self.cols)
        self.codes = {c: q.codes[:, j] for j, c in enumerate(q.features)}
        self.sel_all = panel.sel_train | panel.sel_val()
        self.rows_all = self.sel_all[panel.wk]
        self.ctx_cols = [c for c in self.cols if c.startswith("m_")]

    def mask(self, expr: PI.Expression) -> np.ndarray:
        m = np.ones(len(self.p.y), bool)
        for t in expr.base:
            m &= self.codes[t.feature] == t.level
        for e in expr.unless:
            c = self.codes[e.feature]
            m &= (c != e.level) & (c >= 0)
        return m

    def stat(self, m: np.ndarray, sel: np.ndarray, rows: np.ndarray | None = None, y: np.ndarray | None = None,
             min_weeks: int | None = None) -> WeeklyStat:
        rows = self.p.fit_rows if rows is None else rows
        sy, sw = weekly_sums(m & rows, self.p.y if y is None else y, self.p.wk, self.p.n_wk)
        return weekly_stat(sy, sw, sel, self.cfg.lags, min_weeks or self.cfg.min_active_weeks)

    # -- evidence
    def evidence(self, m: np.ndarray) -> EvidenceCounts:
        rows = m & self.rows_all & self.p.fit_rows
        sy, sw = weekly_sums(rows, self.p.y, self.p.wk, self.p.n_wk)
        active = sw > 0
        st = weekly_stat(sy, sw, self.sel_all, self.cfg.lags, 3)
        names = len(np.unique(self.p.tickers[rows]))
        indep = _greedy_gap(np.flatnonzero(active), self.p.embargo_weeks + 1)
        return EvidenceCounts(int(rows.sum()), int(active.sum()), _episodes(active), int(names), int(indep), float(st.n_eff),
                              int(min(indep, names)))

    # -- validation periods
    def validation(self, m: np.ndarray, direction: int) -> ValidationResult:
        blocks, effects = [], []
        for k, sel in enumerate(self.p.sel_blocks):
            s = self.stat(m, sel, min_weeks=self.cfg.context_min_weeks)
            blocks.append(BlockStat(k, s.mean, s.t, s.n_weeks))
            if s.n_weeks >= self.cfg.context_min_weeks and s.se not in (0.0, float("inf")) and s.mean != 0:
                effects.append(KN.Effect(s.sign, abs(s.mean), s.se, self.cfg.tag, self.cfg.horizon))
        agree = sum(1 for b in blocks if b.n_weeks >= self.cfg.context_min_weeks and b.mean * direction > 0)
        if not effects:
            return ValidationResult(tuple(blocks), None, None, None, None, agree, len(blocks), 0.0)
        pooled = KN.pool_effects(effects)
        signed = pooled.effect.signed
        t = signed / pooled.effect.uncertainty
        conf = _phi(t * direction) if t * direction > 0 else 0.0
        return ValidationResult(tuple(blocks), float(signed), float(pooled.effect.uncertainty), float(t), float(pooled.i2), agree,
                                len(blocks), float(conf))

    # -- transfer
    def _group_score(self, ts: Sequence[float], direction: int) -> float | None:
        ts = [t * direction for t in ts]
        return None if not ts else float(np.mean([_phi(t) for t in ts]))

    def _rescale(self, s: float | None) -> float | None:
        return None if s is None else float(np.clip((s - 0.5) * 2.0, 0.0, 1.0))

    def transfer(self, m: np.ndarray, direction: int, contexts: Sequence[ContextCell]) -> TransferProfile:
        n_wk = self.p.n_wk
        tr_idx = np.flatnonzero(self.p.sel_train)
        halves = []
        if len(tr_idx) >= 2 * self.cfg.context_min_weeks:
            for part in np.array_split(tr_idx, 2):
                s = np.zeros(n_wk, bool)
                s[part] = True
                halves.append(s)
        era_ts = []
        for sel in halves + list(self.p.sel_blocks):
            st = self.stat(m, sel, min_weeks=self.cfg.context_min_weeks)
            if st.n_weeks >= self.cfg.context_min_weeks:
                era_ts.append(st.t)
        cross_era = self._rescale(self._group_score(era_ts, direction))
        cross_stock = None
        if self.p.holdout_rows.any():
            st = self.stat(m, self.sel_all, rows=self.p.holdout_rows, min_weeks=self.cfg.context_min_weeks)
            if st.n_weeks >= self.cfg.context_min_weeks:
                cross_stock = self._rescale(_phi(st.t * direction))
        cross_sector = None
        if self.p.sector is not None:
            ts = []
            for s in np.unique(self.p.sector):
                rows = self.p.fit_rows & (self.p.sector == s)
                if rows.sum() < self.cfg.min_rows:
                    continue
                st = self.stat(m, self.sel_all, rows=rows, min_weeks=self.cfg.context_min_weeks)
                if st.n_weeks >= self.cfg.context_min_weeks:
                    ts.append(st.t)
            cross_sector = self._rescale(self._group_score(ts, direction))
        reg = [c.t for c in contexts if self.fam_of_col.get(c.feature) == "regime" and c.n_weeks >= self.cfg.context_min_weeks]
        cross_regime = self._rescale(self._group_score(reg, direction))
        avail = {k: v for k, v in (("era", cross_era), ("stock", cross_stock), ("sector", cross_sector), ("regime", cross_regime))
                 if v is not None}
        if not avail:
            return TransferProfile(cross_era, cross_stock, cross_sector, cross_regime, None, ())
        vals = list(avail.values())
        return TransferProfile(cross_era, cross_stock, cross_sector, cross_regime, float(0.5 * min(vals) + 0.5 * np.mean(vals)),
                               tuple(sorted(avail)))

    # -- context dependence and failure conditions
    def context_profile(self, m: np.ndarray, direction: int) -> tuple[tuple[ContextCell, ...], float | None, str | None]:
        cells: list[ContextCell] = []
        het: dict[str, tuple[float, float]] = {}
        buckets = (("low", (0, 1)), ("mid", (2, 2)), ("high", (3, 4)))
        for c in self.ctx_cols:
            lv = self.codes[c]
            effects, local = [], []
            for name, (lo, hi) in buckets:
                sel_rows = (lv >= lo) & (lv <= hi)
                s = self.stat(m & sel_rows, self.sel_all, min_weeks=self.cfg.context_min_weeks)
                local.append(ContextCell(c, name, (lo, hi), s.mean, s.t, s.n_weeks))
                if s.n_weeks >= self.cfg.context_min_weeks and math.isfinite(s.se) and s.se > 0:
                    effects.append((s.mean, s.se))
            cells.extend(local)
            if len(effects) >= 2:
                mu = np.array([e for e, _ in effects])
                w = np.array([1.0 / se ** 2 for _, se in effects])
                center = (w * mu).sum() / w.sum()
                q = float((w * (mu - center) ** 2).sum())
                df = len(effects) - 1
                i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
                het[c] = (i2, float(sps.chi2.sf(q, df)))
        if not het:
            return tuple(cells), None, None
        n_test = len(het)
        sig = {c: v for c, v in het.items() if min(1.0, v[1] * n_test) < 0.05}
        if not sig:
            return tuple(cells), 0.0, None
        best = max(sig, key=lambda c: (sig[c][0], c))
        return tuple(x for x in cells if x.feature == best), float(sig[best][0]), best

    def failure_conditions(self, cells: Sequence[ContextCell], direction: int, overall: float) -> tuple[FailureCondition, ...]:
        out = []
        for c in cells:
            if c.n_weeks < self.cfg.context_min_weeks:
                continue
            st = c.t * direction
            if st <= -1.5:
                out.append(FailureCondition(c.feature, c.levels, "flip", c.mean, c.t, c.n_weeks))
            elif abs(c.mean) < 0.25 * abs(overall) and st < 1.0:
                out.append(FailureCondition(c.feature, c.levels, "vanish", c.mean, c.t, c.n_weeks))
        return tuple(out)

    # -- counterexamples
    def counterexamples(self, m: np.ndarray, direction: int) -> tuple[float, float, tuple[Counterexample, ...]]:
        rows = m & self.rows_all & self.p.fit_rows
        if not rows.any():
            return 0.0, 0.0, ()
        y = self.p.y
        allrows = self.rows_all & self.p.fit_rows
        rate = float((direction * y[rows] < 0).mean())
        base = float((direction * y[allrows] < 0).mean())
        idx = np.flatnonzero(rows & (direction * y < 0))
        idx = idx[np.argsort(direction * y[idx], kind="mergesort")][: self.cfg.counterexamples_k]
        sd = float(y[allrows].std()) or 1.0
        ex = tuple(Counterexample(str(self.p.dates[i].date()), str(self.p.tickers[i]), float(y[i]), float(y[i] / sd)) for i in idx)
        return rate, rate - base, ex

    # -- controls
    def controls(self, m: np.ndarray, t_disc: float, direction: int) -> ControlResult:
        rows = m & self.rows_all & self.p.fit_rows
        flags: list[str] = []
        if not rows.any():
            return ControlResult(None, 1.0, 1.0, ("empty",))
        y = self.p.y
        train_rows = self.p.sel_train[self.p.wk] & self.p.fit_rows
        means = pd.Series(y[train_rows]).groupby(self.p.tickers[train_rows]).mean()
        dm = y - pd.Series(self.p.tickers).map(means).fillna(0.0).to_numpy()
        raw = self.stat(m, self.sel_all)
        adj = self.stat(m, self.sel_all, y=dm)
        ratio = float(adj.mean / raw.mean) if raw.mean != 0 else None
        cnt = pd.Series(self.p.tickers[rows]).value_counts()
        name_share = float(cnt.iloc[0] / cnt.sum())
        sy, sw = weekly_sums(rows, y, self.p.wk, self.p.n_wk)
        tot = np.abs(sy).sum()
        week_share = float(np.abs(sy).max() / tot) if tot > 0 else 1.0
        if ratio is not None and ratio < self.cfg.ticker_effect_min_ratio:
            flags.append("ticker_effect")
        if name_share > self.cfg.max_name_share and len(cnt) > 1:
            flags.append("name_concentration")
        if week_share > self.cfg.max_week_share:
            flags.append("week_concentration")
        if abs(t_disc) > self.cfg.suspicious_t:
            flags.append("implausible_t")
        jk = self.jackknife(m, direction, cnt.index[:3].tolist(), np.argsort(-np.abs(sy))[:3])
        if jk is not None and raw.t * direction >= 2.0 and jk <= 1.0:
            flags.append("fragile")
        return ControlResult(ratio, name_share, week_share, tuple(flags), jk)

    def jackknife(self, m: np.ndarray, direction: int, names: Sequence[str], weeks: Sequence[int]) -> float | None:
        """Smallest direction-signed t after deleting, one at a time, each of the most influential stocks and weeks. A pattern that is
        one stock or one week in disguise loses its t here even when its concentration shares look tolerable."""
        worst = []
        for n in names:
            s = self.stat(m & (self.p.tickers != n), self.sel_all)
            if s.n_weeks >= self.cfg.context_min_weeks:
                worst.append(s.t * direction)
        for w in weeks:
            sel = self.sel_all.copy()
            sel[int(w)] = False
            s = self.stat(m, sel)
            if s.n_weeks >= self.cfg.context_min_weeks:
                worst.append(s.t * direction)
        return float(min(worst)) if worst else None

    # -- reliability, temporal shape, complexity, impact
    def reliability(self, m: np.ndarray, direction: int) -> float | None:
        sy, sw = weekly_sums(m & self.p.fit_rows, self.p.y, self.p.wk, self.p.n_wk)
        idx = np.flatnonzero(self.sel_all & (sw > 0))[-self.cfg.reliability_weeks:]
        if len(idx) < 8:
            return None
        hits = int(((sy[idx] / sw[idx]) * direction > 0).sum())
        return float((1 + hits) / (2 + len(idx)))

    def temporal(self, val: ValidationResult, m_train: float, direction: int) -> str:
        ok = [b for b in val.blocks if b.n_weeks >= self.cfg.context_min_weeks]
        if len(ok) < 2 or m_train == 0:
            return TemporalClass.UNKNOWN.value
        signed = [b.mean * direction for b in ok]
        ratio = signed[-1] / abs(m_train)
        consistent = np.mean([s > 0 for s in signed])
        if consistent >= 0.99 and ratio >= 0.6:
            return TemporalClass.PERSISTENT.value
        if ratio < 0.3 and signed[0] > signed[-1]:
            return TemporalClass.SLOW_DECAY.value if consistent >= 0.5 else TemporalClass.FAST_DECAY.value
        if consistent < 0.5:
            return TemporalClass.EPISODIC.value
        return TemporalClass.UNKNOWN.value

    def complexity(self, text: str, t_used: float | None, direction: int) -> ComplexityAudit:
        spec = CX.spec_from_pattern_text(text)
        base = CX.RuleSpec("single", n_features=1, n_conditions=0, n_thresholds=1, tree_depth=1, n_free_params=1).units()
        units = spec.units()
        bar = float(CX.required_t(units - base))
        used = None if t_used is None else float(t_used * direction)
        return ComplexityAudit(float(units), bar, used, None if used is None else bool(used >= bar))

    def impact(self, families: Sequence[str], val: ValidationResult, direction: int) -> DecisionImpact:
        stock_side = {"volatility", "ranges", "relvol", "gaps", "liquidity", "correlations", "dispersion"}
        event_side = {"earnings", "filings", "insider", "eventtiming", "macro"}
        eff = ["RANKING"]
        if set(families) & stock_side:
            eff += ["SELECTION", "POSITION_SIZE"]
        if set(families) & event_side:
            eff += ["SELECTION", "TIMING"]
        eff = tuple(dict.fromkeys(eff))
        gain = None if val.pooled_mean is None else float(val.pooled_mean * direction)
        gt = None if val.pooled_t is None else float(val.pooled_t * direction)
        return DecisionImpact(eff, gain, gt)


# ----------------------------------------------------------------------------------------------------- verdict and record
def decide(cfg: DiscoveryConfig, t_disc: float, truth: float | None, val: ValidationResult, ev: EvidenceCounts, ctl: ControlResult,
           transfer: TransferProfile, cx: ComplexityAudit, direction: int, family_ok: bool) -> tuple[GateVerdict, tuple[str, ...]]:
    """Section 42: never PROMOTE from discovery. QUARANTINED beats FAILED beats NEEDS_MORE_EVIDENCE; UNKNOWN when validation is
    impossible (no populated validation block), which is different from failing it."""
    reasons: list[str] = []
    if not family_ok:
        return GateVerdict.QUARANTINED, ("source family failed the point-in-time audit",)
    if ctl.flags:
        reasons += [f"control: {f}" for f in ctl.flags]
        if {"implausible_t", "name_concentration", "week_concentration", "ticker_effect", "fragile"} & set(ctl.flags):
            return GateVerdict.QUARANTINED, tuple(reasons)
    if val.pooled_t is None:
        return GateVerdict.UNKNOWN, ("no validation block had enough weeks to test",)
    if val.conf_factor == 0.0 or val.pooled_t * direction <= 0:
        return GateVerdict.FAILED, tuple(reasons + ["validation periods contradict the discovery sign"])
    if truth is None or truth < cfg.min_p_real:
        reasons.append(f"truth probability below {cfg.min_p_real}")
        return GateVerdict.FAILED, tuple(reasons)
    if ev.independent < cfg.min_independent:
        reasons.append(f"independent evidence {ev.independent} < {cfg.min_independent}")
    if transfer.transfer is None:
        reasons.append("transfer untested")
    elif transfer.transfer < cfg.min_transfer:
        reasons.append(f"transfer {transfer.transfer:.2f} < {cfg.min_transfer}")
    if cx.earns is False:
        reasons.append("complexity has not earned its place")
    if val.i2 is not None and val.i2 > 0.75:
        reasons.append("validation periods disagree strongly (I2 > 0.75)")
    return GateVerdict.NEEDS_MORE_EVIDENCE, tuple(reasons or ["awaiting fresh-data confirmation"])


_VERDICT_STATE = {
    GateVerdict.FAILED: (Epistemic.CONTRADICTED, Lifecycle.FAILURE),
    GateVerdict.QUARANTINED: (Epistemic.GATED, Lifecycle.BIRTH),
    GateVerdict.UNKNOWN: (Epistemic.UNKNOWN, Lifecycle.BIRTH),
    GateVerdict.NEEDS_MORE_EVIDENCE: (Epistemic.HYPOTHESIS, Lifecycle.BIRTH),
    GateVerdict.PROMOTE: (Epistemic.HYPOTHESIS, Lifecycle.BIRTH),          # unreachable by construction; discovery cannot promote
}


def _band(levels: tuple[int, int]) -> tuple[float, float]:
    return float(levels[0]), float(levels[1])


def to_knowledge(d: Dossier, expr: PI.Expression, now, prov: Provenance, cfg: DiscoveryConfig) -> KN.KnowledgeObject:
    """Dossier -> KnowledgeObject through the existing PatternRecord bridge. The result is RESEARCH-promotion with decision effect
    NONE whatever its statistics; failures stay on the record (RETIRED is for things that once worked, not for things that never
    did). Identity-free: no ticker, no date in any text field (asserted)."""
    transform = "ts_quintile" if any(f.startswith("m_") for f in expr.features) else "xs_quintile"
    w = d.windows
    tr, vals = w["train"], [v for k, v in w.items() if k.startswith("val") and v[0]]
    rec = PI.new_record(expr, transform, cfg.tag, discovery=tr, validation=(vals[0][0], vals[-1][1]) if vals else None,
                        stats={"t_disc": d.t_disc, "t_conf": d.validation.pooled_t, "m_all": d.m_disc, "effect": d.m_disc,
                               "p_real": d.truth, "n_eff": d.evidence.n_eff, "p_hallucinated": d.p_halluc, "p_coincidence": d.q_cum},
                        state="candidate", family=expr.family)
    k = KN.from_pattern_record(rec, now, prov, decision_effect=(DecisionEffect.NONE,), n=d.evidence.rows)
    epi, life = _VERDICT_STATE[d.verdict]
    contexts, anti = list(k.contexts.conditions), list(k.anti_contexts.conditions)
    best = [c for c in d.contexts if c.t * d.direction >= 2.0 and c.n_weeks >= cfg.context_min_weeks]
    if d.context_feature and best and d.failure_conditions:
        b = max(best, key=lambda c: c.t * d.direction)
        contexts.append(KN.Condition("market", f"{b.feature}.q", "between", nums=_band(b.levels)))
    for f in d.failure_conditions:
        anti.append(KN.Condition("market", f"{f.feature}.q", "between", nums=_band(f.levels)))
    ctx_conf = None if d.context_dependence is None else float(np.clip(1.0 - d.context_dependence, 0.0, 1.0))
    cond = KN.ContextSet(tuple(contexts))
    antiset = KN.ContextSet(tuple(anti), any_of=True)
    v = d.validation
    if v.pooled_mean is not None and v.pooled_se:
        effect = KN.Effect(d.direction, abs(v.pooled_mean), float(v.pooled_se), cfg.tag, cfg.horizon)
    else:
        effect = KN.Effect(d.direction, abs(d.m_disc), None, cfg.tag, cfg.horizon)
    tp = d.transfer
    fail_rate = float(d.failed_periods / max(1, v.n_blocks + 2))
    risk = float(np.clip(0.5 * fail_rate + 0.5 * (1.0 - (d.reliability if d.reliability is not None else 0.5)), 0.0, 1.0))
    expl = ()
    if d.verdict == GateVerdict.FAILED:
        expl = (KN.FailureExplanation(FailureCause.FALSE_PATTERN, str(as_date(now)), None,
                                      "; ".join(d.reasons) or "validation contradicted discovery", d.pattern_id),)
    tags = ["mined_pattern", "discovery", *(f"source:{f}" for f in d.families), f"verdict:{d.verdict.value.lower()}"]
    tags += [f"control:{f}" for f in d.controls.flags]
    ev = KN.Evidence(d.evidence.rows, min(float(d.evidence.rows), d.evidence.n_eff), 1.0, None, d.outcomes_seen_through)
    new = dataclasses.replace(
        k, contexts=cond, anti_contexts=antiset, effect=effect,
        confidence=Confidence(truth=None if epi == Epistemic.UNKNOWN else d.truth, usefulness=None,
                              current_reliability=d.reliability, context=ctx_conf, transfer=tp.transfer, failure_risk=risk),
        evidence=ev, dynamics=KN.Dynamics(stability=None if v.n_blocks == 0 else float(v.agree / v.n_blocks),
                                          contradiction_rate=float(np.clip(d.counter_rate, 0.0, 1.0)), failure_rate=fail_rate),
        transfer=KN.TransferScores(tp.transfer, tp.cross_era, tp.cross_stock, tp.cross_sector, tp.cross_regime),
        temporal_class=TemporalClass(d.temporal), mechanism_tags=tuple(dict.fromkeys(tags)), failure_explanations=expl,
        epistemic=epi, lifecycle=life, promotion=Promotion.RESEARCH, decision_effect=(DecisionEffect.NONE,),
        source_experiments=(d.run_id,), source_runs=("dossier:" + d.digest(),),
        hypothesis=f"{expr.text} predicts {cfg.horizon}-session excess return (discovered, unconfirmed)")
    new.assert_valid()
    assert_identity_free(new)
    return new


TICKERISH = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")


def assert_identity_free(k: KN.KnowledgeObject, tickers: Iterable[str] = ()) -> None:
    """The trader-facing text of a knowledge object may not carry a date, a year or a stock name (canons C55, C58)."""
    banned = {str(t).upper() for t in tickers}
    for name in ("observation", "hypothesis", "interpretation"):
        s = getattr(k, name)
        why = string_reasons(s)
        if why:
            raise FirewallBreach(f"{k.knowledge_id}.{name} carries {why}")
        if banned and any(tok in banned for tok in TICKERISH.findall(s)):
            raise FirewallBreach(f"{k.knowledge_id}.{name} names a stock")
    for t in k.mechanism_tags:
        if string_reasons(t):
            raise FirewallBreach(f"{k.knowledge_id} tag {t!r} carries a date")


# ------------------------------------------------------------------------------------------------- state and scheduling
@dataclasses.dataclass
class FamilyRecord:
    visits: int = 0
    last_step: int = -1
    trials: int = 0
    survivors: int = 0          # findings still standing after validation (NEEDS_MORE_EVIDENCE or better)
    failed: int = 0             # findings that failed validation or were quarantined


@dataclasses.dataclass
class StepReport:
    step: int
    now: str
    families: tuple[str, ...]
    skipped: dict[str, str]
    pit_failed: dict[str, int]
    rows: int
    weeks: int
    generated: int
    tested: int
    shortlisted: int
    new: int
    retested: int
    verdicts: dict[str, int]
    revalidated: dict[str, int]
    total_trials: int
    distinct_trials: int
    best_null_t: float
    wealth: float
    halted: str = ""

    def render(self) -> str:
        v = ", ".join(f"{k}={n}" for k, n in sorted(self.verdicts.items())) or "none"
        r = ", ".join(f"{k}={n}" for k, n in sorted(self.revalidated.items())) or "none"
        lines = [f"discovery step {self.step}: {self.rows} rows over {self.weeks} weeks",
                 f"  families visited: {', '.join(self.families) or 'none'}",
                 f"  generated {self.generated}, testable {self.tested}, shortlisted {self.shortlisted}; new {self.new}, re-tested {self.retested}",
                 f"  verdicts: {v}", f"  fresh-data revalidation: {r}",
                 f"  cumulative trials {self.total_trials} ({self.distinct_trials} distinct); a lucky best of that many reaches |t| "
                 f"{self.best_null_t:.2f}", f"  alpha wealth {self.wealth:.4f}"]
        if self.skipped:
            lines.append("  skipped: " + "; ".join(f"{k} ({w})" for k, w in sorted(self.skipped.items())))
        if self.pit_failed:
            lines.append("  point-in-time audit FAILED for: " + ", ".join(sorted(self.pit_failed)))
        if self.halted:
            lines.append(f"  HALTED: {self.halted}")
        return "\n".join(lines)


@dataclasses.dataclass
class DiscoveryState:
    """Everything the discovery engine remembers between steps. MATURED_RESEARCH_STATE: never handed to the trader directly."""
    ledger: TrialLedger = dataclasses.field(default_factory=TrialLedger)
    wealth: AlphaWealth = dataclasses.field(default_factory=AlphaWealth)
    store: KN.KnowledgeStore = dataclasses.field(default_factory=KN.KnowledgeStore)
    dossiers: dict[str, Dossier] = dataclasses.field(default_factory=dict)
    families: dict[str, FamilyRecord] = dataclasses.field(default_factory=dict)
    pit: dict[str, list[str]] = dataclasses.field(default_factory=dict)         # family|fingerprint -> audit findings
    unit_evidence: dict[str, dict[str, list[float]]] = dataclasses.field(default_factory=dict)   # pattern -> unit -> [val mean, se, dir, t_disc]
    steps: int = 0
    history: list[StepReport] = dataclasses.field(default_factory=list)
    namespace: Namespace = Namespace.MATURED_RESEARCH

    def save(self, directory) -> None:
        """Atomic checkpoint: a kill costs one step, not the history. Knowledge is the store's own JSONL; dossiers use the
        knowledge module's strict typed codec, so a hand-edited dossier fails on load rather than being believed."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.store.dump(d / "knowledge.jsonl")
        blobs = {
            "ledger.json": self.ledger.to_dict(),
            "dossiers.json": {k: KN.encode(v) for k, v in self.dossiers.items()},
            "units.json": self.unit_evidence,
            "meta.json": {"steps": self.steps, "wealth": [self.wealth.wealth, self.wealth.payout, self.wealth.spend_frac, self.wealth.floor],
                          "families": {k: dataclasses.asdict(v) for k, v in self.families.items()}, "pit": self.pit},
        }
        for name, obj in blobs.items():
            tmp = d / (name + ".tmp")
            tmp.write_bytes(json.dumps(obj, sort_keys=True).encode("utf-8"))
            tmp.replace(d / name)

    @classmethod
    def load(cls, directory) -> "DiscoveryState":
        d = Path(directory)
        s = cls()
        s.store = KN.KnowledgeStore.load(d / "knowledge.jsonl")
        s.ledger = TrialLedger.from_dict(json.loads((d / "ledger.json").read_bytes()))
        if (d / "units.json").exists():
            s.unit_evidence = json.loads((d / "units.json").read_bytes())
        s.dossiers = {k: KN.decode(Dossier, v) for k, v in json.loads((d / "dossiers.json").read_bytes()).items()}
        meta = json.loads((d / "meta.json").read_bytes())
        s.steps = int(meta["steps"])
        s.wealth = AlphaWealth(*meta["wealth"])
        s.families = {k: FamilyRecord(**v) for k, v in meta["families"].items()}
        s.pit = meta["pit"]
        errs = s.store.verify()
        if errs:
            raise DiscoveryError("knowledge store failed verification on load: " + "; ".join(errs[:3]))
        return s


def choose_families(state: DiscoveryState, available: Sequence[str], k: int, max_staleness: int) -> list[str]:
    """Coverage first, yield second. A family never visited, or not visited for more than `max_staleness` steps, is forced in
    (oldest first); remaining slots go to the families with the best survivor rate per trial, discounted by the family's own
    false-discovery record. Deterministic: ties break on name."""
    recs = state.families
    stale = [f for f in available if f not in recs or state.steps - recs[f].last_step > max_staleness]
    stale.sort(key=lambda f: (recs[f].last_step if f in recs else -1, f))
    chosen = stale[:k]
    rest = [f for f in available if f not in chosen]

    def score(f: str) -> float:
        r = recs.get(f, FamilyRecord())
        return (r.survivors + 1.0) / (r.trials / 500.0 + 1.0) * (1.0 - state.ledger.family_fdr(f))

    rest.sort(key=lambda f: (-score(f), f))
    return chosen + rest[: max(k - len(chosen), 0)]


# ------------------------------------------------------------------------------------------------------ the miner wrapper
def mine_with_patternminer(X: pd.DataFrame, y: pd.Series, now, columns: Sequence[str], params: Mapping | None = None,
                           horizon: int = 5) -> tuple[list[Cand], pd.DataFrame, int]:
    """Run engine.patterns.PatternMiner on the given feature columns and return its patterns as extra candidates. The miner keeps
    its own statistics and statuses; here it is only a second GENERATOR - every pattern it proposes is re-measured by the same
    weekly kernel, judged against the same cumulative ledger and validated on the same held-out weeks as everything else."""
    from engine.patterns import PatternMiner
    ctx = [c for c in columns if c.startswith("m_")]
    miner = PatternMiner({"horizon": horizon, "target": f"excess_{horizon}d", **(params or {})})
    miner.fit(X[list(columns)], y, pd.Timestamp(as_date(now)), ctx_cols=ctx)
    table = miner.patterns
    out, seen = [], set()
    for text in table["key_named"].astype(str) if len(table) else []:
        try:
            e = PI.Expression.parse(text)
        except PI.IdentityError:
            continue
        if e.text not in seen:
            seen.add(e.text)
            out.append(Cand(e, "miner", ()))
    return out, table, int(miner.report.get("tested", len(table)))


# ---------------------------------------------------------------------------------------------- release and namespaces
def payload_of(k: KN.KnowledgeObject) -> dict[str, Any]:
    """The identity-free content of a knowledge object as a plain dict: expression, direction, size, confidence dimensions,
    scope. No provenance, no dates, no raw sample counts (a count of 2008 looks like a year to the trader-view gate)."""
    n_eff = max(k.evidence.effective_sample_size, 1.0)
    return {"pattern": k.observation, "direction": int(k.effect.direction), "effect": round(float(k.effect.size), 6),
            "truth": None if k.confidence.truth is None else round(float(k.confidence.truth), 4),
            "reliability": None if k.confidence.current_reliability is None else round(float(k.confidence.current_reliability), 4),
            "transfer": None if k.confidence.transfer is None else round(float(k.confidence.transfer), 4),
            "evidence_scale": round(math.log10(n_eff), 2), "epistemic": k.epistemic.value,
            "anti_context": [{"feature": c.feature, "levels": [float(x) for x in c.nums]} for c in k.anti_contexts.conditions],
            "context": [{"feature": c.feature, "levels": [float(x) for x in c.nums]} for c in k.contexts.conditions if c.op == "between"]}


def as_matured_record(k: KN.KnowledgeObject, d: Dossier) -> MaturedRecord:
    """A discovery as a research-world fact. It reaches a decision only through `.gate(now)` (outcome matured strictly before now)
    and only if the identity-free payload passes the trader-view scan."""
    from engine.learning.trader_view import find_violations
    payload = payload_of(k)
    bad = find_violations(payload)
    if bad:
        raise FirewallBreach(f"{k.knowledge_id} payload is not trader-safe: {bad[0]}")
    return MaturedRecord(k.knowledge_id, d.outcomes_seen_through, payload, k.provenance)


def release_filter(state: DiscoveryState, now, replay_windows: Sequence[tuple[Any, Any]] = ()) -> list[MaturedRecord]:
    """Research the trader may be handed at `now`: existed strictly before now, is not FAILED / QUARANTINED / UNKNOWN, and was NOT
    learned from a stretch of history that is currently being replayed in disguise (the same-year rerun leak, mapping rule 27):
    any dossier whose window or outcomes overlap a replayed window is withheld whole."""
    wins = [(pd.Timestamp(as_date(a)), pd.Timestamp(as_date(b))) for a, b in replay_windows]
    out = []
    for kid in state.store.ids():
        k = state.store.as_of(kid, now)
        if k is None or k.epistemic not in (Epistemic.SUPPORTED, Epistemic.CONDITIONAL):
            continue       # a mined HYPOTHESIS is never released: only what recurred or survived fresh data
        d = state.dossiers.get(kid[2:])
        if d is None or d.verdict not in (GateVerdict.NEEDS_MORE_EVIDENCE, GateVerdict.PROMOTE):
            continue
        spans = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in d.windows.values() if a] + \
                [(pd.Timestamp(d.outcomes_seen_through),) * 2]
        if any(a <= we and ws <= b for a, b in spans for ws, we in wins):
            continue
        rec = as_matured_record(k, d)
        try:
            rec.gate(now)
        except FirewallBreach:
            continue
        out.append(rec)
    return out


# --------------------------------------------------------------------------------------------------------- the engine
class DiscoveryEngine:
    def __init__(self, cfg: DiscoveryConfig = DiscoveryConfig(), source_cfg: DS.SourceConfig = DS.DEFAULT_SOURCE_CONFIG,
                 families_per_step: int = 6, max_staleness: int = 8, core_context: Sequence[str] = ("regime", "breadth"),
                 anchors: Sequence[str] = ("price", "volatility", "relvol"), audit: bool = True, audit_every: int = 20, use_miner: bool = False,
                 miner_params: Mapping | None = None, run_tag: str = "disc"):
        errs = cfg.validate() + source_cfg.validate()
        if errs:
            raise DiscoveryError("; ".join(errs))
        self.cfg, self.source_cfg = cfg, source_cfg
        self.families_per_step, self.max_staleness = families_per_step, max_staleness
        self.core_context, self.anchors = tuple(core_context), tuple(anchors)
        self.audit, self.audit_every, self.use_miner, self.miner_params, self.run_tag = audit, max(audit_every, 1), use_miner, miner_params, run_tag

    # -- point-in-time admissibility, cached per (family, inputs fingerprint)
    def admissible(self, state: DiscoveryState, inp: DS.SourceInputs) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
        ok, skipped, bad = [], {}, {}
        for name in DS.ALL_FAMILY_NAMES:
            if name in DS.DERIVED_FAMILIES:
                if inp.learned:
                    ok.append(name)
                else:
                    skipped[name] = "needs learned"
                continue
            why = DS.FAMILIES[name].usable(inp)
            if why:
                skipped[name] = why
                continue
            key = f"{name}|{stable_hash(DS.FAMILIES[name].fn.__code__.co_code.hex(), 8)}|{state.steps // self.audit_every}"
            if self.audit and key not in state.pit:
                state.pit[key] = [str(x) for x in DS.audit_pit(inp, [name], self.source_cfg)]
            found = state.pit.get(key, [])
            if found:
                bad[name] = found
            else:
                ok.append(name)
        return ok, skipped, bad

    def step(self, state: DiscoveryState, now, inputs: DS.SourceInputs, families: Sequence[str] | None = None,
             extra_cands: Sequence[Cand] = (), unit_id: str = "", cohort: str = "all", ordered: bool = True,
             label_hook: Callable[[pd.Series, pd.DataFrame], pd.Series] | None = None) -> StepReport:
        cfg = self.cfg
        now_ts = pd.Timestamp(as_date(now))
        if ordered and state.history and now_ts < pd.Timestamp(state.history[-1].now):
            raise FirewallBreach("discovery steps must be taken in non-decreasing time order")
        inp = inputs.before(now_ts)
        ok, skipped, bad = self.admissible(state, inp)
        avail = [f for f in ok if families is None or f in families]
        if families is not None:
            for f in families:
                if f in bad:
                    skipped[f] = "failed point-in-time audit"
        chosen = choose_families(state, avail, self.families_per_step, self.max_staleness) if families is None else list(avail)
        run_id = f"{self.run_tag}-{state.steps:05d}"
        halted = ""
        if state.wealth.wealth <= state.wealth.floor and state.steps > 0:
            halted = "alpha wealth exhausted: no widening until something is confirmed"
        cohort_need = ["price"] if cohort != "all" else []
        want = list(dict.fromkeys(chosen + cohort_need + [a for a in self.anchors if a in ok] + [c for c in self.core_context if c in ok]))
        if not chosen or halted:
            return self._close(state, now_ts, run_id, chosen, skipped, bad, 0, 0, 0, 0, 0, 0, 0, {}, {}, halted or "no admissible family")
        fb = DS.build_features(inp, want, self.source_cfg)
        keep = [c for c in fb.X.columns if fb.X[c].notna().mean() >= 0.3 and fb.X[c].nunique(dropna=True) > 1]
        y, matured = DS.target_labels(inp.bars, cfg.target, cfg.horizon, now_ts)
        if label_hook is not None:
            y = label_hook(y, fb.X[keep])
        panel = Panel.build(fb.X[keep], y, now_ts, cfg, matured, inp.sectors)
        if cohort != "all":
            panel = restrict_cohort(panel, cohort)
        an = Analyzer(panel, keep, fb.family_of, cfg)
        sc = Screener(panel, keep, fb.family_of, cfg)
        res = sc.run()
        extra: list[Cand] = []
        if self.use_miner:
            fitrows = panel.fit_rows
            extra, table, _ = mine_with_patternminer(panel.X[fitrows], pd.Series(panel.y[fitrows], index=panel.X.index[fitrows]), now_ts,
                                                     keep, self.miner_params, cfg.horizon)
            self._merge_extra(res, sc, extra)
        if extra_cands:
            for c in extra_cands:
                if any(f not in sc.codes for f in c.expr.features):
                    skipped[f"precursor:{c.text}"] = "feature not in the admissible panel"
            self._merge_extra(res, sc, list(extra_cands))
        prim = [self._primary(c, fb.family_of) for c in res.cands]
        data_key = stable_hash({"fp": inp.fingerprint(), "w": panel.windows()["train"], "cols": keep, "h": cfg.horizon, "s": cfg.seed})
        res.q_cum = state.ledger.register(run_id, now_ts, res.trial_ids, res.disc["p"], prim, data_key)
        alpha = state.wealth.per_test_alpha(len(res.cands))
        cand_idx = self._shortlist(res, sc)
        verdicts: dict[str, int] = {}
        new = retested = 0
        seen_thr = pd.Timestamp(panel.outcomes_seen_through)
        code_hash = current_code_hash() or "unknown"          # once per step: hashing the loaded modules is slow
        exprs = {res.cands[i].id(cfg.tag): res.cands[i].expr for i in cand_idx}
        overlaps = known_overlaps(an, state, exprs, overlap=cfg.redundancy_overlap) if state.dossiers else {}
        for i in cand_idx:
            d, expr = self._dossier(state, panel, an, res, i, run_id, now_ts, fb, seen_thr)
            verdicts[d.verdict.value] = verdicts.get(d.verdict.value, 0) + 1
            if unit_id and d.validation.pooled_mean is not None and d.validation.pooled_se:
                state.unit_evidence.setdefault(d.pattern_id, {})[unit_id] = [float(d.validation.pooled_mean), float(d.validation.pooled_se),
                                                                             float(d.direction), float(d.t_disc)]
            kid = "K-" + d.pattern_id
            prov = KN.make_provenance(seen_thr, data=inp.fingerprint(), config=dataclasses.asdict(cfg), experiment_id=run_id, run_id=run_id, code_hash=code_hash,
                                      seed=cfg.seed, outcomes_seen_through=seen_thr)
            red = overlaps.get(d.pattern_id, ())
            if red:
                d = dataclasses.replace(d, impact=dataclasses.replace(d.impact, redundant_with=red))
            k = to_knowledge(d, expr, now_ts, prov, cfg)
            if red:
                k = dataclasses.replace(k, relations=KN.Relations(redundant=red)).assert_valid()
            prev = state.store.latest(kid)
            if prev is None:
                state.store.add(k)
                new += 1
            else:
                retested += 1
                fresher = as_date(prov.learned_at) > as_date(prev.provenance.learned_at)
                try:
                    state.store.add(prev.new_version(max(now_ts, pd.Timestamp(prev.updated_at)), f"re-tested by {run_id}", learned_at=prov.learned_at if fresher else None,
                                                     **_carry(k)))
                except KN.SchemaError:
                    pass                                                        # identical evidence: an unchanged re-test is not a new version
            state.dossiers[d.pattern_id] = d
            for f in d.families:
                fr = state.families.setdefault(f, FamilyRecord())
                if d.verdict in (GateVerdict.NEEDS_MORE_EVIDENCE, GateVerdict.PROMOTE):
                    fr.survivors += 1
                elif d.verdict in (GateVerdict.FAILED, GateVerdict.QUARANTINED):
                    fr.failed += 1
        reval = self.revalidate(state, panel, an, now_ts)
        for f in chosen:
            fr = state.families.setdefault(f, FamilyRecord())
            fr.visits += 1
            fr.last_step = state.steps
        for c, p_ in zip(res.cands, prim):
            state.families.setdefault(p_, FamilyRecord()).trials += 1
        state.wealth.settle(run_id, len(res.cands), alpha, reval.get("confirmed", 0))
        return self._close(state, now_ts, run_id, chosen, skipped, bad, len(panel.y), panel.n_wk, len(res.cands), int(res.tested.sum()),
                           len(cand_idx), new, retested, verdicts, reval, halted)

    def _merge_extra(self, res: ScreenResult, sc: Screener, extra: Sequence[Cand]) -> int:
        """Append externally proposed candidates (PatternMiner, precursors) to a screen. They are measured by the same kernel, get the
        same shuffled-outcome null and enter the same ledger as everything generated here; duplicates of existing candidates are dropped."""
        cfg = self.cfg
        have = {x.text for x in res.cands}
        extra = [c for c in extra if c.text not in have and all(f in sc.codes for f in c.expr.features)]
        if not extra:
            return 0
        eSY, eSW = sc.evaluate(extra, sc.y)
        est, etest = sc._stats(eSY, eSW)
        res.cands, res.SY, res.SW = res.cands + extra, np.vstack([res.SY, eSY]), np.vstack([res.SW, eSW])
        est["p"] = np.where(etest, calibrated_p(est["p"], est["t"], res.null_t), 1.0)
        res.disc = {k: np.concatenate([res.disc[k], est[k]]) for k in res.disc}
        res.tested = np.concatenate([res.tested, etest])
        real_t = np.sort(np.abs(res.disc["t"][res.tested])) if res.tested.any() else np.array([0.0])
        res.p_halluc = np.concatenate([res.p_halluc, [PS.local_fdr(abs(t), res.null_t, real_t) if o else 1.0 for t, o in zip(est["t"], etest)]])
        res.trial_ids = res.trial_ids + [c.id(cfg.tag) for c in extra]
        res.q_cum = np.ones(len(res.cands))
        return len(extra)

    @staticmethod
    def _primary(c: Cand, fam_of_col: Mapping[str, str]) -> str:
        first = sorted(c.expr.base, key=lambda t: (t.feature.startswith("m_"), t.feature))[0].feature
        return fam_of_col.get(first, "miner" if c.origin == "miner" else "?")

    def _shortlist(self, res: ScreenResult, sc: Screener) -> list[int]:
        cfg = self.cfg
        ok = res.tested & (res.disc["p"] <= cfg.shortlist_p) & (res.q_cum <= cfg.shortlist_q)
        order = [int(i) for i in np.argsort(-np.abs(res.disc["t"]) * ok, kind="mergesort") if ok[i]][: cfg.max_shortlist]
        if len(order) < 2:
            return order
        rng = np.random.default_rng(cfg.seed)
        rows = np.sort(rng.choice(len(sc.y), min(len(sc.y), 200_000), replace=False))
        masks = [sc.row_mask(res.cands[i].expr)[rows] for i in order]
        kept, _ = PS.prune_redundant(list(range(len(order))), masks, cfg.redundancy_overlap)
        return [order[j] for j in kept]

    def _dossier(self, state: DiscoveryState, panel: Panel, an: Analyzer, res: ScreenResult, i: int, run_id: str, now: pd.Timestamp,
                 fb: DS.FeatureBuild, seen: pd.Timestamp) -> tuple[Dossier, PI.Expression]:
        cfg = self.cfg
        cand = res.cands[i]
        m = an.mask(cand.expr)
        t_disc, m_disc = float(res.disc["t"][i]), float(res.disc["mean"][i])
        direction = 1 if m_disc > 0 else -1
        ev = an.evidence(m)
        val = an.validation(m, direction)
        cells, dep, dep_feat = an.context_profile(m, direction)
        tp = an.transfer(m, direction, cells)
        hold = None
        if panel.holdout_rows.any():
            hs = an.stat(m, an.sel_all, rows=panel.holdout_rows, min_weeks=cfg.context_min_weeks)
            hold = hs.t if hs.n_weeks >= cfg.context_min_weeks else None
        p_h, q_c = float(res.p_halluc[i]), float(res.q_cum[i])
        truth = float(np.clip((1.0 - max(p_h, q_c)) * val.conf_factor, 0.0, 1.0)) if val.pooled_t is not None else None
        rate, excess, ex = an.counterexamples(m, direction)
        ctl = an.controls(m, t_disc, direction)
        overall = an.stat(m, an.sel_all).mean
        fails = an.failure_conditions(cells, direction, overall)
        fam = tuple(sorted({fb.family_of.get(f, "miner") for f in cand.expr.features}))
        cx = an.complexity(cand.expr.text, val.pooled_t, direction)
        verdict, reasons = decide(cfg, t_disc, truth, val, ev, ctl, tp, cx, direction, True)     # inadmissible families are never built
        failed_periods = sum(1 for b in val.blocks if b.n_weeks >= cfg.context_min_weeks and b.mean * direction <= 0)
        d = Dossier(
            pattern_id=cand.id(cfg.tag), text=cand.expr.text, origin=cand.origin, families=fam, run_id=run_id,
            discovered_at=str(now.date()), outcomes_seen_through=str(seen.date()), windows=panel.windows(), direction=direction,
            t_disc=t_disc, m_disc=m_disc, p_raw=float(res.disc["p"][i]), p_halluc=p_h, q_cum=q_c,
            trials_so_far=state.ledger.distinct_trials, evidence=ev, validation=val, holdout_t=hold, truth=truth,
            reliability=an.reliability(m, direction), transfer=tp, context_dependence=dep, context_feature=dep_feat, contexts=cells,
            failure_conditions=fails, failed_periods=failed_periods, counter_rate=rate, counter_excess=excess, counterexamples=ex,
            complexity=cx, impact=an.impact(fam, val, direction), controls=ctl,
            temporal=an.temporal(val, m_disc, direction), verdict=verdict, reasons=reasons)
        return d, cand.expr

    # -- fresh-data revalidation: the only road from HYPOTHESIS to SUPPORTED
    def revalidate(self, state: DiscoveryState, panel: Panel, an: Analyzer, now: pd.Timestamp) -> dict[str, int]:
        cfg = self.cfg
        counts = {"confirmed": 0, "contradicted": 0, "inconclusive": 0, "skipped": 0}
        for pid, d in sorted(state.dossiers.items()):
            k = state.store.latest("K-" + pid)
            if k is None or k.epistemic not in (Epistemic.HYPOTHESIS, Epistemic.SUPPORTED) or d.verdict != GateVerdict.NEEDS_MORE_EVIDENCE:
                continue
            expr = PI.Expression.parse(d.text)
            if any(f not in an.codes for f in expr.features):
                counts["skipped"] += 1
                continue
            cutoff = pd.Timestamp(d.outcomes_seen_through) + pd.Timedelta(days=7 * (panel.embargo_weeks + 1))
            fresh = np.asarray(panel.week_start > cutoff)
            if fresh.sum() < cfg.context_min_weeks:
                counts["skipped"] += 1
                continue
            st = an.stat(an.mask(expr), fresh, rows=np.ones(len(panel.y), bool), min_weeks=cfg.context_min_weeks)
            z = st.t * d.direction
            if z >= cfg.z_val:
                counts["confirmed"] += 1
                state.ledger.confirm(d.families[0], True)
                conf = d.confirmations + 1
                nd = dataclasses.replace(d, confirmations=conf)
                if conf >= cfg.confirmations_needed and k.epistemic == Epistemic.HYPOTHESIS:
                    k = k.new_version(now, f"fresh-data confirmation {conf}", epistemic=Epistemic.SUPPORTED)
                    state.store.add(k)
                state.dossiers[pid] = nd
            elif z <= -cfg.z_val:
                counts["contradicted"] += 1
                state.ledger.confirm(d.families[0], False)
                nk = k.new_version(now, "contradicted by fresh data", epistemic=Epistemic.CONTRADICTED, lifecycle=Lifecycle.FAILURE,
                                   failure_explanations=k.failure_explanations + (KN.FailureExplanation(
                                       FailureCause.FALSE_PATTERN, str(now.date()), None, "fresh data reversed the sign", pid),))
                state.store.add(nk)
                state.dossiers[pid] = dataclasses.replace(d, verdict=GateVerdict.FAILED, reasons=d.reasons + ("contradicted by fresh data",))
            else:
                counts["inconclusive"] += 1
        return counts

    def _close(self, state: DiscoveryState, now: pd.Timestamp, run_id: str, chosen, skipped, bad, rows, weeks, gen, tested, short, new,
               retested, verdicts, reval, halted) -> StepReport:
        rep = StepReport(state.steps, str(now.date()), tuple(chosen), dict(skipped), {k: len(v) for k, v in bad.items()}, rows, weeks, gen,
                         tested, short, new, retested, dict(verdicts), dict(reval), state.ledger.total_trials, state.ledger.distinct_trials,
                         state.ledger.expected_best_null_t(), state.wealth.wealth, halted)
        state.history.append(rep)
        state.steps += 1
        return rep


def _carry(k: KN.KnowledgeObject) -> dict[str, Any]:
    """The fields a re-test may change on an existing knowledge object (never identity, history or provenance)."""
    names = ("contexts", "anti_contexts", "effect", "confidence", "evidence", "dynamics", "transfer", "temporal_class", "mechanism_tags",
             "failure_explanations", "epistemic", "lifecycle")
    return {n: getattr(k, n) for n in names}


def step(state: DiscoveryState, now, inputs: DS.SourceInputs, engine: DiscoveryEngine | None = None, **kw) -> StepReport:
    """THE public entry (rule 25). One discovery step at `now`: schedule families, screen, validate, dedupe against everything
    known, revalidate the old on fresh data, account every trial. Returns a StepReport; `state` is updated in place."""
    return (engine or DiscoveryEngine(**kw)).step(state, now, inputs)


def run_families(state: DiscoveryState, now, inputs: DS.SourceInputs, engine: DiscoveryEngine, families: Sequence[str]) -> StepReport:
    """One step restricted to the named families (used by tests and by targeted research questions)."""
    return engine.step(state, now, inputs, families=families)


# ---------------------------------------------------------------------------------------------- streaming (year by year)
def stream_screen(loader: Callable[[pd.Timestamp, pd.Timestamp], DS.SourceInputs], years: Sequence[int], cands: Sequence[Cand],
                  horizon: int, families: Sequence[str] | None = None, source_cfg: DS.SourceConfig = DS.DEFAULT_SOURCE_CONFIG,
                  now=None, keep_worst: int = 5) -> tuple[StreamingScreen, pd.DataFrame]:
    """Screen a fixed candidate list over a market-wide history one calendar year at a time (mapping rule 27). Memory is one
    year of rows plus per-(candidate, week) sums and a handful of worst outcomes per candidate. Point-in-time is inherited from
    year_chunks: no year starting at/after `now`, no label maturing on/after it."""
    ss = StreamingScreen(cands, keep_worst=keep_worst)
    for _, X, y in DS.year_chunks(loader, years, horizon, families, source_cfg, now):
        ss.add_chunk(X, y.dropna(), now)
    st = ss.stats(lags=int(math.ceil(horizon / 5.0)))
    frame = pd.DataFrame({"text": [c.text for c in ss.cands], "mean": st["mean"], "t": st["t"], "p": st["p"], "n_weeks": st["n_weeks"],
                          "n_eff": st["n_eff"]})
    return ss, frame


# ----------------------------------------------------------------------------------------------- audits and reporting
def audit_state(state: DiscoveryState) -> list[str]:
    """Invariants that must hold for the discovery store at any time. An empty list means: the ledger chain and the knowledge
    chains verify, nothing here is promoted or changes a decision (only the promotion gate may do that, elsewhere), every text
    field is identity-free, and every knowledge object has a dossier."""
    errs = [f"ledger: {e}" for e in state.ledger.verify()] + [f"store: {e}" for e in state.store.verify()]
    for kid in state.store.ids():
        k = state.store.latest(kid)
        if k.promotion not in NEVER_TRUSTED and k.promotion != Promotion.RETIRED:
            errs.append(f"{kid}: promotion {k.promotion.value} set inside discovery")
        if k.decision_effect != (DecisionEffect.NONE,):
            errs.append(f"{kid}: decision effect set inside discovery")
        try:
            assert_identity_free(k)
        except FirewallBreach as e:
            errs.append(str(e))
        if kid[2:] not in state.dossiers:
            errs.append(f"{kid}: no dossier")
        d = state.dossiers.get(kid[2:])
        if d is not None and d.namespace != Namespace.MATURED_RESEARCH:
            errs.append(f"{kid}: dossier outside MATURED_RESEARCH_STATE")
    return errs


def coverage_report(state: DiscoveryState) -> dict[str, Any]:
    """Which section-13 families the search has actually visited, and which have produced anything: a family never visited is a
    hole in the search, not a family with nothing in it."""
    seen = {f for f, r in state.families.items() if r.visits > 0}
    fams = list(DS.ALL_FAMILY_NAMES)
    return {"visited": sorted(seen & set(fams)), "never_visited": sorted(set(fams) - seen),
            "with_survivors": sorted(f for f, r in state.families.items() if r.survivors > 0 and f in fams),
            "steps": state.steps, "false_discovery_by_family": {f: round(state.ledger.family_fdr(f), 3) for f in sorted(seen & set(fams))}}


def discovery_report(state: DiscoveryState, top: int = 15) -> str:
    """Plain-text account of what is known, how much was tried to find it, and how little of it is trusted."""
    tot = state.ledger.total_trials
    cov = coverage_report(state)
    lines = [f"DISCOVERY REPORT (MATURED_RESEARCH_STATE): {len(state.dossiers)} patterns, {tot} trials over {len(state.ledger.runs)} runs "
             f"({state.ledger.distinct_trials} distinct).",
             f"A lucky best of {max(state.ledger.distinct_trials, 2)} null candidates reaches |t| {state.ledger.expected_best_null_t():.2f}.",
             f"Families visited {len(cov['visited'])}/{len(DS.ALL_FAMILY_NAMES)}; never visited: {', '.join(cov['never_visited']) or 'none'}."]
    by_v: dict[str, int] = {}
    for d in state.dossiers.values():
        by_v[d.verdict.value] = by_v.get(d.verdict.value, 0) + 1
    lines.append("verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(by_v.items())))
    ranked = sorted(state.dossiers.values(), key=lambda d: (-(d.truth or 0.0), d.text))[:top]
    for d in ranked:
        t = "n/a" if d.truth is None else f"{d.truth:.2f}"
        tr = "n/a" if d.transfer.transfer is None else f"{d.transfer.transfer:.2f}"
        lines.append(f"  [{d.verdict.value}] {d.text}: dir {d.direction:+d}, t_disc {d.t_disc:+.2f}, truth {t}, transfer {tr}, "
                     f"independent {d.evidence.independent}, confirmations {d.confirmations}"
                     + (f"; {'; '.join(d.reasons)}" if d.reasons else ""))
    return "\n".join(lines)


# ======================================================================================================================
# C67: always-on sweep, mover cohorts, precursor intake, strict small-effect accounting, cross-unit recurrence
# ======================================================================================================================
COHORTS: dict[str, tuple[float, float, int]] = {          # name -> (min |ret|, max |ret|, side: 0 both, +1 up, -1 down) of the SIGNAL day
    "all": (0.0, math.inf, 0), "mover_5_10": (0.05, 0.10, 0), "mover_up_5_10": (0.05, 0.10, 1), "mover_down_5_10": (0.05, 0.10, -1),
    "mover_gt10": (0.10, math.inf, 0), "quiet": (0.0, 0.02, 0),
}


def cohort_mask(ret1: np.ndarray, name: str) -> np.ndarray:
    """Rows whose signal-day close-to-close return falls in the named band (canon C67: the 5-10% movers and what they did next). A
    missing return is in no cohort. The band is read from the day being decided, never from the outcome."""
    if name not in COHORTS:
        raise DiscoveryError(f"unknown cohort {name!r}; known: {sorted(COHORTS)}")
    lo, hi, side = COHORTS[name]
    r = np.asarray(ret1, dtype=float)
    ok = np.isfinite(r)
    a = np.abs(np.where(ok, r, 0.0))
    m = ok & (a >= lo) & (a < hi) if name != "quiet" else ok & (a < hi)
    if side:
        m &= (r * side) > 0
    return m


def restrict_cohort(panel: Panel, name: str, min_rows: int = 0) -> Panel:
    """The same panel with discovery, validation and hold-out rows limited to a cohort. Quantiles stay cross-sectional over the FULL
    universe of the day, so 'top volume quintile among movers' means top of the market, not top of the few movers."""
    col = "price__ret_1"
    if col not in panel.X.columns:
        raise DiscoveryError(f"cohort {name!r} needs the price family (column {col})")
    m = cohort_mask(panel.X[col].to_numpy(dtype=float), name)
    if m.sum() < max(min_rows, 1):
        raise DiscoveryError(f"cohort {name!r} has only {int(m.sum())} rows")
    rep = {**panel.report, "cohort": name, "cohort_rows": int(m.sum()), "cohort_share": float(m.mean())}
    return dataclasses.replace(panel, fit_rows=panel.fit_rows & m, holdout_rows=panel.holdout_rows & m, report=rep)


# ----------------------------------------------------------------------------- strict accounting for small effects
def bh_threshold(pool: Sequence[float], fdr: float = 0.05) -> float:
    """Largest raw p that Benjamini-Hochberg would still reject across the WHOLE pool of distinct trials (0 if none)."""
    p = np.sort(np.asarray(pool, dtype=float))
    if p.size == 0:
        return 0.0
    ok = np.flatnonzero(p <= fdr * (np.arange(p.size) + 1) / p.size)
    return float(p[ok[-1]]) if ok.size else 0.0


def storey_pi0(pool: Sequence[float], lam: float = 0.5) -> float:
    """Estimated share of true nulls among the trials: share of p above lam, rescaled. 1.0 (all null) is the safe default for a
    small or empty pool; the estimate is floored so a lucky pool cannot declare that nothing is chance."""
    p = np.asarray(pool, dtype=float)
    if p.size < 50:
        return 1.0
    return float(np.clip((p > lam).mean() / (1.0 - lam), 0.1, 1.0))


def expected_false_hits(ledger: TrialLedger, p_threshold: float) -> float:
    """How many of the distinct trials would pass p <= threshold if every one were null: pi0 * m * threshold."""
    return storey_pi0(ledger._pool) * ledger.distinct_trials * float(p_threshold)


def distinct_patterns(ledger: TrialLedger) -> int:
    """Number of different pattern identities ever tested (the multiplicity that applies to any cross-window combination)."""
    return len(ledger.times_tested)


def z_for_p(p: float) -> float:
    return float(sps.norm.isf(min(max(p, 1e-300), 0.999999) / 2.0))


def minimum_detectable_effect(se: float, p_threshold: float, power: float = 0.8) -> float:
    """Smallest true effect a test with standard error `se` detects with probability `power` at raw threshold `p_threshold`. With a huge
    ledger the threshold is tiny, so this is the honest floor on 'small real effects' a search can claim."""
    if not (se > 0 and math.isfinite(se)):
        return float("inf")
    return float((z_for_p(p_threshold) + sps.norm.ppf(power)) * se)


def power_at(effect: float, se: float, p_threshold: float) -> float:
    if not (se > 0 and math.isfinite(se)):
        return 0.0
    z = z_for_p(p_threshold)
    d = abs(effect) / se
    return float(sps.norm.sf(z - d) + sps.norm.cdf(-z - d))


@dataclasses.dataclass(frozen=True)
class SmallEffectPolicy:
    """What the ledger demands before a small effect may be called real, from the ledger itself."""
    fdr: float
    p_threshold: float
    z_required: float
    pi0: float
    distinct_trials: int
    expected_false: float
    mde_at_se: Mapping[str, float]

    def admits(self, p: float) -> bool:
        return p <= self.p_threshold


def small_effect_policy(ledger: TrialLedger, fdr: float = 0.05, ses: Mapping[str, float] | None = None) -> SmallEffectPolicy:
    """Strictness scales with the search: the threshold is BH over the pooled trials and never looser than Bonferroni-at-fdr divided by
    the number of DISTINCT patterns tried when the pool is thin. `ses` maps a label to a typical weekly-cluster standard error so the
    report can say which effect sizes are still detectable."""
    m = max(ledger.distinct_trials, 1)
    thr = bh_threshold(ledger._pool, fdr) if m >= 20 else fdr / m
    thr = min(thr, fdr) if thr > 0 else fdr / m
    thr = max(thr, 1e-12)
    mde = {k: minimum_detectable_effect(v, thr) for k, v in (ses or {}).items()}
    return SmallEffectPolicy(fdr, thr, z_for_p(thr), storey_pi0(ledger._pool), ledger.distinct_trials, expected_false_hits(ledger, thr), mde)


# ------------------------------------------------------------------------------------------------- precursor intake
@dataclasses.dataclass(frozen=True)
class PrecursorIntake:
    """A candidate precursor offered by another research module (R21 precursors, questions, break research). Only an expression over
    columns the panel has, and the cohort of episodes it is about: discovery measures it like any other candidate, so being proposed
    by a sister module earns no trust and no exemption from the ledger."""
    expression: str
    cohort: str = "all"
    episode_type: str = ""
    source: str = "external"


def as_intake(item: Any) -> PrecursorIntake:
    """Accept a string, a mapping, or any object with .expression / .text (duck-typed so R21's record class is not imported)."""
    if isinstance(item, PrecursorIntake):
        return item
    if isinstance(item, str):
        return PrecursorIntake(item)
    get = (lambda k, d="": item.get(k, d)) if isinstance(item, Mapping) else (lambda k, d="": getattr(item, k, d))
    expr = get("expression") or get("text") or get("key_named")
    if not expr:
        raise DiscoveryError(f"precursor {item!r} has no expression")
    return PrecursorIntake(str(expr), str(get("cohort", "all") or "all"), str(get("episode_type", "")), str(get("source", "external")))


def precursor_candidates(items: Iterable[Any], fam_of_col: Mapping[str, str] | None = None, horizon: int = 5
                         ) -> tuple[dict[str, list[Cand]], dict[str, str]]:
    """Group intake by cohort into Cands (origin 'precursor'). Unparseable or cohort-unknown items are returned with the reason and are
    NOT silently dropped. Duplicate expressions within a cohort collapse to one."""
    out: dict[str, list[Cand]] = {}
    bad: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    for raw in items:
        try:
            it = as_intake(raw)
            e = PI.Expression.parse(it.expression)
        except (PI.IdentityError, DiscoveryError) as ex:
            bad[str(raw)[:80]] = str(ex)[:120]
            continue
        if it.cohort not in COHORTS:
            bad[it.expression] = f"unknown cohort {it.cohort!r}"
            continue
        if (it.cohort, e.text) in seen:
            continue
        seen.add((it.cohort, e.text))
        fams = tuple(sorted({(fam_of_col or {}).get(f, "?") for f in e.features}))
        out.setdefault(it.cohort, []).append(Cand(e, "precursor", fams))
    return out, bad


# ------------------------------------------------------------------------------- recurrence across units (years, slices)
@dataclasses.dataclass(frozen=True)
class Recurrence:
    pattern_id: str
    n_units: int
    n_years: int
    n_eras: int
    agree_share: float
    pooled: float | None
    se: float | None
    z: float | None
    p: float
    q: float
    i2: float | None
    m_patterns: int
    verdict: str
    eras: tuple[int, ...] = ()


def _unit_year(uid: str) -> int:
    return int(uid.split("|")[0])


def recurrence(state: DiscoveryState, pid: str, m_patterns: int, era_edges: Sequence[int] | None = None) -> Recurrence | None:
    """Combine the out-of-sample (validation) estimates a pattern earned in different units. Units of the same year share market weeks, so
    they are averaged with the LARGEST of their standard errors (fully dependent, the conservative end) and only years are treated as
    independent; the years are pooled with random effects so disagreement widens the interval. Sign is taken from each unit's own
    discovery direction; a unit that found the opposite sign counts against."""
    ev = state.unit_evidence.get(pid)
    if not ev:
        return None
    from engine.research.episodes import ERA_EDGES
    edges = tuple(era_edges or ERA_EDGES)
    d = state.dossiers.get(pid)
    ref = d.direction if d is not None else 1
    by_year: dict[int, list[tuple[float, float, float]]] = {}
    for uid, (mean, se, direction, _) in ev.items():
        by_year.setdefault(_unit_year(uid), []).append((mean * ref, se, direction))
    effects, agree = [], 0
    for yr, rows in sorted(by_year.items()):
        mu = float(np.mean([r[0] for r in rows]))
        se = float(max(r[1] for r in rows))
        agree += int(mu > 0)
        if se > 0 and mu != 0:
            effects.append(KN.Effect(1 if mu > 0 else -1, abs(mu), se, "excess", 5))
    eras = tuple(sorted({int(np.searchsorted(np.asarray(edges), y, side="right")) for y in by_year}))
    if not effects:
        return Recurrence(pid, len(ev), len(by_year), len(eras), 0.0, None, None, None, 1.0, 1.0, None, m_patterns, "NO_ESTIMATE", eras)
    pooled = KN.pool_effects(effects)
    z = pooled.effect.signed / pooled.effect.uncertainty
    p = float(2.0 * sps.norm.sf(abs(z)))
    return Recurrence(pid, len(ev), len(by_year), len(eras), agree / len(by_year), float(pooled.effect.signed),
                      float(pooled.effect.uncertainty), float(z), p, min(1.0, p * m_patterns), float(pooled.i2), m_patterns, "PENDING", eras)


def recurrence_table(state: DiscoveryState, era_edges: Sequence[int] | None = None) -> list[Recurrence]:
    """Every pattern with unit evidence, with BH q-values over ALL patterns ever tested (not only those that recur): a pattern that
    recurs in 3 years out of the thousands tried still has to clear the multiplicity of the whole search."""
    m = max(distinct_patterns(state.ledger), len(state.unit_evidence), 1)
    rows = [r for pid in sorted(state.unit_evidence) if (r := recurrence(state, pid, m, era_edges)) is not None]
    if not rows:
        return []
    order = sorted(range(len(rows)), key=lambda i: rows[i].p)
    q = np.ones(len(rows))
    run = 1.0
    for rank in range(len(rows), 0, -1):
        i = order[rank - 1]
        run = min(run, rows[i].p * m / rank)
        q[i] = min(run, 1.0)
    return [dataclasses.replace(r, q=float(q[i])) for i, r in enumerate(rows)]


def judge_recurrence(r: Recurrence, cfg: DiscoveryConfig, policy: SmallEffectPolicy | None = None, q_max: float = 0.05,
                     min_years: int = 3, min_eras: int = 2, min_agree: float = 0.75, max_i2: float = 0.75) -> Recurrence:
    """RECURS only if it holds in several independent years across more than one era with the same sign, survives the whole-search q,
    and (when a policy is given) the ledger's own p threshold. Anything short of that stays PENDING; a significant OPPOSITE pooled sign is
    REVERSED. Nothing here promotes: it can only license 'SUPPORTED' research knowledge."""
    if r.z is None:
        return r
    if r.z < 0 and r.q <= q_max and r.n_years >= min_years:
        return dataclasses.replace(r, verdict="REVERSED")
    ok = (r.n_years >= min_years and r.n_eras >= min_eras and r.agree_share >= min_agree and r.q <= q_max and r.z > 0
          and (r.i2 is None or r.i2 <= max_i2) and (policy is None or policy.admits(r.p)))
    return dataclasses.replace(r, verdict="RECURS" if ok else "PENDING")


def apply_recurrence(state: DiscoveryState, now, cfg: DiscoveryConfig, policy: SmallEffectPolicy | None = None,
                     era_edges: Sequence[int] | None = None) -> dict[str, int]:
    """Write recurrence verdicts into the knowledge store as new versions: RECURS -> SUPPORTED, REVERSED -> CONTRADICTED. Promotion stays
    RESEARCH and the decision effect stays NONE. Idempotent: an object already in the target state is left alone."""
    now_ts = pd.Timestamp(as_date(now))
    out = {"supported": 0, "contradicted": 0, "unchanged": 0}
    for r in recurrence_table(state, era_edges):
        r = judge_recurrence(r, cfg, policy)
        k = state.store.latest("K-" + r.pattern_id)
        if k is None or r.verdict not in ("RECURS", "REVERSED"):
            out["unchanged"] += 1
            continue
        stamp = max(now_ts, pd.Timestamp(k.updated_at))
        if r.verdict == "RECURS" and k.epistemic in (Epistemic.HYPOTHESIS, Epistemic.UNKNOWN):
            state.store.add(k.new_version(stamp, f"recurred in {r.n_years} years across {r.n_eras} eras (q={r.q:.3g})",
                                          epistemic=Epistemic.SUPPORTED, confidence=dataclasses.replace(k.confidence, truth=float(np.clip(1.0 - r.q, 0.0, 1.0)))))
            out["supported"] += 1
        elif r.verdict == "REVERSED" and k.epistemic not in (Epistemic.CONTRADICTED, Epistemic.RETIRED):
            expl = k.failure_explanations + (KN.FailureExplanation(FailureCause.REVERSAL, str(stamp.date()), None,
                                                                  f"pooled across {r.n_years} years the sign is opposite", r.pattern_id),)
            state.store.add(k.new_version(stamp, "reversed across years", epistemic=Epistemic.CONTRADICTED, lifecycle=Lifecycle.FAILURE,
                                          failure_explanations=expl))
            out["contradicted"] += 1
        else:
            out["unchanged"] += 1
    return out


# --------------------------------------------------------------------------------------------------------- always-on sweep
def restrict_inputs(inp: DS.SourceInputs, tickers: Iterable[str]) -> DS.SourceInputs:
    """The same inputs limited to a set of stocks (every per-ticker table), leaving date structure untouched."""
    keep = {str(t) for t in tickers}

    def cut(t: pd.DataFrame | None) -> pd.DataFrame | None:
        return None if t is None else t[t["ticker"].astype(str).isin(keep)]
    b = inp.bars[inp.bars["ticker"].astype(str).isin(keep)]
    sec = None if inp.sectors is None else {k: v for k, v in inp.sectors.items() if k in keep}
    return dataclasses.replace(inp, bars=b, sectors=sec, earnings=cut(inp.earnings), filings=cut(inp.filings), insiders=cut(inp.insiders))


@dataclasses.dataclass(frozen=True)
class SweepConfig:
    years: tuple[int, ...]
    n_slices: int = 4
    families: tuple[str, ...] = ()               # empty = every family in DS.ALL_FAMILY_NAMES
    cohorts: tuple[str, ...] = ("all",)
    salt: int = 0
    warmup_days: int = 420
    cushion_days: int = 14
    min_names: int = 12
    checkpoint_every: int = 1

    def validate(self) -> list[str]:
        e = []
        if not self.years or self.n_slices < 1 or self.min_names < 3 or self.checkpoint_every < 1:
            e.append("years, n_slices >= 1, min_names >= 3, checkpoint_every >= 1")
        e += [f"unknown cohort {c!r}" for c in self.cohorts if c not in COHORTS]
        e += [f"unknown family {f!r}" for f in self.families if f not in DS.ALL_FAMILY_NAMES and f not in DS.FAMILIES]
        return e

    def lenses(self) -> tuple[str, ...]:
        fams = self.families or tuple(DS.ALL_FAMILY_NAMES)
        return tuple(f"{f}@{c}" for f in fams for c in self.cohorts)


class DiscoverySweep:
    """Resumable, always-on driver: keeps taking the least-covered (year, universe slice, source family x cohort) unit, runs one discovery
    step on it with hindsight allowed (research side, mapping rule 27), records it in the shared CoverageBook and checkpoints, so a kill
    costs at most the unit in flight. Coverage is EXTENDED from episodes.CoverageBook (lens = 'family@cohort'), never a second book."""

    def __init__(self, cfg: SweepConfig, engine: "DiscoveryEngine", directory=None):
        from engine.research.episodes import CoverageBook
        errs = cfg.validate()
        if errs:
            raise DiscoveryError("; ".join(errs))
        self.cfg, self.engine = cfg, engine
        self.dir = None if directory is None else Path(directory)
        self.book = CoverageBook(cfg.years, cfg.n_slices, cfg.lenses())
        self.tag = "disc:" + stable_hash({"d": dataclasses.asdict(engine.cfg), "s": dataclasses.asdict(engine.source_cfg), "salt": cfg.salt}, 10)
        self.units_run = 0

    # -- persistence
    @classmethod
    def resume(cls, cfg: SweepConfig, engine: "DiscoveryEngine", directory) -> tuple["DiscoverySweep", DiscoveryState]:
        """Reload book and state if a checkpoint exists (a partial or edited one is refused), else start fresh."""
        from engine.research.episodes import CoverageBook
        sw = cls(cfg, engine, directory)
        d = Path(directory)
        if (d / "coverage.json").exists():
            book = CoverageBook.load(d / "coverage.json")
            if book.n_slices != cfg.n_slices or book.lenses != cfg.lenses():
                raise DiscoveryError("checkpoint was written for a different sweep shape")
            book.extend_years(cfg.years)
            sw.book = book
        state = DiscoveryState.load(d / "state") if (d / "state" / "meta.json").exists() else DiscoveryState()
        return sw, state

    def checkpoint(self, state: DiscoveryState) -> None:
        if self.dir is None:
            return
        state.save(self.dir / "state")
        self.book.save(self.dir / "coverage.json")          # book last: a kill in between re-runs a unit, never skips one

    # -- what to do next
    def data_through(self, last_date) -> dict[int, str]:
        last = pd.Timestamp(as_date(last_date))
        return {y: str(min(pd.Timestamp(year=y, month=12, day=31), last).date()) for y in self.cfg.years if pd.Timestamp(year=y, month=1, day=1) <= last}

    def pending(self, last_date, limit: int | None = None):
        return self.book.pending(features=(self.tag,), data_through=self.data_through(last_date), limit=limit)

    def next_family(self, last_date) -> str | None:
        """Family of the least-covered pending unit (for callers that only want the direction of travel)."""
        p = self.pending(last_date, 1)
        return p[0].lens.split("@")[0] if p else None

    # -- one unit
    def run_unit(self, state: DiscoveryState, unit, loader: Callable[[pd.Timestamp, pd.Timestamp], DS.SourceInputs],
                 precursors: Iterable[Any] = ()) -> StepReport | None:
        from engine.research.episodes import UnitRecord, slice_tickers
        fam, cohort = unit.lens.split("@")
        start, end = pd.Timestamp(year=unit.year, month=1, day=1), pd.Timestamp(year=unit.year, month=12, day=31)
        inp = loader(start - pd.Timedelta(days=self.cfg.warmup_days), end + pd.Timedelta(days=self.cfg.cushion_days))
        names = slice_tickers(inp.tickers(), unit.slice_id, self.cfg.n_slices, self.cfg.salt)
        dates = pd.to_datetime(inp.bars["date"])
        last = dates.max() if len(dates) else start
        through = min(pd.Timestamp(last), end)
        complete = bool(pd.Timestamp(last) >= end)
        rec = dict(uid=unit.uid, through=str(through.date()), complete=complete, n_days=int(dates[(dates >= start) & (dates <= end)].nunique()),
                   n_names=len(names), features_done=(self.tag,), config_digest=self.tag[5:], code_hash="")
        if len(names) < self.cfg.min_names:
            self.book.mark(UnitRecord(n_episodes=0, n_suspect=0, band_counts={"thin_universe": 1}, **rec))
            return None
        sub = restrict_inputs(inp, names)
        groups, bad = precursor_candidates(precursors, None, self.engine.cfg.horizon)
        now = end + pd.Timedelta(days=self.cfg.cushion_days)
        try:
            rep = self.engine.step(state, now, sub, families=[fam], extra_cands=groups.get(cohort, []), unit_id=unit.uid, cohort=cohort,
                                   ordered=False)
        except (DiscoveryError, DS.SourceError):
            self.book.mark(UnitRecord(n_episodes=0, n_suspect=0, band_counts={"unusable": 1}, **rec))
            return None
        for k, why in bad.items():
            rep.skipped[f"precursor:{k}"] = why
        self.book.mark(UnitRecord(n_episodes=rep.shortlisted, n_suspect=rep.verdicts.get("FAILED", 0) + rep.verdicts.get("QUARANTINED", 0),
                                  band_counts={k: int(v) for k, v in rep.verdicts.items()}, **rec))
        self.units_run += 1
        return rep

    def run(self, state: DiscoveryState, loader, last_date, max_units: int | None = None, stop: Callable[[], bool] = lambda: False,
            precursors: Iterable[Any] = ()) -> list[StepReport]:
        """Take pending units, least-covered first, until none remain, `max_units` is reached or `stop()` says so (for a 24/7 service the
        caller loops this as new sessions arrive: an in-progress year is re-opened when its data grows)."""
        out: list[StepReport] = []
        n = 0
        while not stop() and (max_units is None or n < max_units):
            todo = self.pending(last_date, 1)
            if not todo:
                break
            rep = self.run_unit(state, todo[0], loader, precursors)
            n += 1
            if rep is not None:
                out.append(rep)
            if n % self.cfg.checkpoint_every == 0:
                self.checkpoint(state)
        self.checkpoint(state)
        return out

    # -- reporting
    def coverage(self, last_date=None) -> dict[str, Any]:
        """Where the search has and has not been: by family, cohort, year and era, plus thin/unusable units and the pending count."""
        from engine.research.episodes import Unit, era_of
        dt_ = self.data_through(last_date) if last_date is not None else None
        rep = self.book.report((self.tag,))
        by_f: dict[str, list[int]] = {}
        by_c: dict[str, list[int]] = {}
        by_era: dict[int, list[int]] = {}
        thin = unusable = 0
        for u in self.book.all_units():
            if dt_ is not None and u.year not in dt_:
                continue
            done = int(self.book.is_done(u, (self.tag,), dt_))
            f, c = u.lens.split("@")
            for tab, key in ((by_f, f), (by_c, c), (by_era, int(era_of(pd.DatetimeIndex([pd.Timestamp(year=u.year, month=6, day=1)]))[0]))):
                tab.setdefault(key, [0, 0])
                tab[key][0] += done
                tab[key][1] += 1
        for r in self.book.records.values():
            thin += r.band_counts.get("thin_universe", 0)
            unusable += r.band_counts.get("unusable", 0)
        frac = lambda t: {k: round(v[0] / v[1], 3) for k, v in sorted(t.items()) if v[1]}
        pending = self.pending(last_date) if last_date is not None else []
        return {"fraction_done": rep["fraction_done"], "by_family": frac(by_f), "by_cohort": frac(by_c), "by_era": frac(by_era),
                "thin_units": thin, "unusable_units": unusable, "pending": len(pending),
                "least_covered_family": min(by_f, key=lambda k: (by_f[k][0] / max(by_f[k][1], 1), k)) if by_f else None}


def sweep_report(sweep: DiscoverySweep, state: DiscoveryState, last_date) -> str:
    cov = sweep.coverage(last_date)
    pol = small_effect_policy(state.ledger)
    rec = [judge_recurrence(r, sweep.engine.cfg, pol) for r in recurrence_table(state)]
    n_rec = sum(r.verdict == "RECURS" for r in rec)
    lines = [f"SWEEP: {cov['fraction_done']:.1%} of units done, {cov['pending']} pending; least-covered family {cov['least_covered_family']}",
             "  by family: " + ", ".join(f"{k}={v:.0%}" for k, v in cov["by_family"].items()),
             "  by era:    " + ", ".join(f"{k}={v:.0%}" for k, v in cov["by_era"].items()),
             f"  thin universe units {cov['thin_units']}, unusable units {cov['unusable_units']}",
             f"  ledger: {state.ledger.distinct_trials} distinct trials, BH p threshold {pol.p_threshold:.2e} (|z| >= {pol.z_required:.2f}), "
             f"pi0 {pol.pi0:.2f}, expected chance passes {pol.expected_false:.2f}",
             f"  recurrence: {n_rec} of {len(rec)} patterns recur across years and eras"]
    return "\n".join(lines)


# ======================================================================================================================
# Self-checks of the discovery machinery: does it find planted truth, and does it stay quiet on nothing?
# ======================================================================================================================
def known_overlaps(an: "Analyzer", state: DiscoveryState, exprs: Mapping[str, PI.Expression], max_known: int = 400, overlap: float = 0.8
                   ) -> dict[str, tuple[str, ...]]:
    """Cross-run redundancy: which already-known patterns fire on nearly the same rows as each new one. Within-run pruning cannot see
    last month's findings; without this a rediscovery under a different name would be counted as independent evidence twice. Only known
    patterns whose features exist in this panel can be compared; the rest are silently non-comparable (they cannot overlap what is absent)."""
    known = [(pid, d) for pid, d in sorted(state.dossiers.items()) if pid not in exprs and d.verdict != GateVerdict.FAILED][:max_known]
    masks: dict[str, np.ndarray] = {}
    for pid, d in known:
        try:
            e = PI.Expression.parse(d.text)
        except PI.IdentityError:
            continue
        if all(f in an.codes for f in e.features):
            masks[pid] = an.mask(e)
    out: dict[str, tuple[str, ...]] = {}
    for pid, e in exprs.items():
        m = an.mask(e)
        hit = tuple(sorted("K-" + k for k, km in masks.items() if PS.jaccard(m, km) > overlap))
        if hit:
            out[pid] = hit
    return out


def shuffle_within_dates(y: pd.Series, seed: int) -> pd.Series:
    """Outcomes permuted across stocks inside each date: every feature-outcome link is destroyed while each day's own outcome
    distribution (and so every market-wide effect) is kept. The null the discovery machinery must find nothing in."""
    rng = np.random.default_rng(seed)
    dates = np.asarray(y.index.get_level_values(0))
    order = np.lexsort((rng.random(len(y)), dates))
    home = np.argsort(dates, kind="stable")
    out = np.empty(len(y))
    out[home] = y.to_numpy()[order]
    return pd.Series(out, index=y.index, name=y.name)


def plant_effect(expr_text: str, effect: float, seed: int = 0) -> Callable[[pd.Series, pd.DataFrame], pd.Series]:
    """Label hook that adds `effect` to the outcome of every row where the expression holds (levels from the same quantiser discovery
    uses). The truth the machinery must recover; `effect` may be negative."""
    expr = PI.Expression.parse(expr_text)

    def hook(y: pd.Series, X: pd.DataFrame) -> pd.Series:
        q = quantise_panel(X[list(expr.features)])
        m = expr.mask(q.codes, list(q.features))
        add = pd.Series(np.where(m, effect, 0.0), index=X.index).reindex(y.index).fillna(0.0)
        return y + add
    return hook


@dataclasses.dataclass(frozen=True)
class NullCalibration:
    runs: int
    families: tuple[str, ...]
    with_survivor: int                 # runs in which any finding reached NEEDS_MORE_EVIDENCE
    with_trusted: int                  # ... and carried truth probability >= min_p_real
    survivors: int
    trusted: int
    tested: int
    generated: int

    @property
    def false_discovery_rate(self) -> float:
        return self.with_survivor / self.runs if self.runs else float("nan")

    @property
    def trusted_rate(self) -> float:
        return self.with_trusted / self.runs if self.runs else float("nan")


def null_calibration(engine: "DiscoveryEngine", inputs: DS.SourceInputs, now, families: Sequence[str], n_runs: int = 3,
                     seed: int = 0, trust_at: float = 0.9) -> NullCalibration:
    """Run the whole pipeline on outcome-shuffled labels, n_runs times, each on a fresh state. The rate at which it still reports a
    surviving finding is the engine's own false-discovery rate on this data - the number the research brain's health check needs and
    that no amount of reading the code can supply."""
    surv = trusted = nsurv = ntr = tested = gen = 0
    for r in range(n_runs):
        st = DiscoveryState()
        rep = engine.step(st, now, inputs, families=list(families), label_hook=lambda y, X, r=r: shuffle_within_dates(y, seed * 1000 + r))
        ok = [d for d in st.dossiers.values() if d.verdict == GateVerdict.NEEDS_MORE_EVIDENCE]
        tr = [d for d in ok if (d.truth or 0.0) >= trust_at]
        surv += bool(ok)
        trusted += bool(tr)
        nsurv += len(ok)
        ntr += len(tr)
        tested += rep.tested
        gen += rep.generated
    return NullCalibration(n_runs, tuple(families), surv, trusted, nsurv, ntr, tested, gen)


@dataclasses.dataclass(frozen=True)
class PowerPoint:
    effect: float
    found: bool
    truth: float | None
    t_disc: float | None
    verdict: str


def power_curve(engine: "DiscoveryEngine", inputs: DS.SourceInputs, now, families: Sequence[str], expr_text: str,
                effects: Sequence[float], seed: int = 0) -> list[PowerPoint]:
    """Plant effects of increasing size on a known expression and record whether discovery recovers it with the right sign and does not
    reject it. The smallest planted effect that is found bounds what 'small patterns' the search can actually see."""
    out = []
    want_terms = set(str(t) for t in PI.Expression.parse(expr_text).base)
    for e in effects:
        st = DiscoveryState()
        engine.step(st, now, inputs, families=list(families), label_hook=plant_effect(expr_text, e, seed))
        want = 1 if e > 0 else -1
        # a stronger conjunction that contains the planted condition is the same discovery, better expressed
        hits = [d for d in st.dossiers.values() if want_terms <= set(str(t) for t in PI.Expression.parse(d.text).base) and d.direction == want
                and not PI.Expression.parse(d.text).unless]
        ok = [d for d in hits if d.verdict in (GateVerdict.NEEDS_MORE_EVIDENCE, GateVerdict.UNKNOWN)]
        best = max(ok or hits, key=lambda d: abs(d.t_disc), default=None)
        out.append(PowerPoint(float(e), bool(ok), None if best is None else best.truth, None if best is None else best.t_disc,
                              "not_shortlisted" if best is None else best.verdict.value))
    return out


def seed_stability(make_engine: Callable[[int], "DiscoveryEngine"], inputs: DS.SourceInputs, now, families: Sequence[str],
                   seeds: Sequence[int] = (1, 2, 3)) -> dict[str, Any]:
    """Re-run the search under different seeds (random pair draws, exception bases, shuffled nulls). Findings that appear under every
    seed are properties of the data; findings that appear under one are properties of the search. Returns the pairwise Jaccard of the
    shortlisted identities and the ids common to all seeds."""
    sets = []
    for sd in seeds:
        st = DiscoveryState()
        make_engine(sd).step(st, now, inputs, families=list(families))
        sets.append({pid for pid, d in st.dossiers.items() if d.verdict != GateVerdict.FAILED})
    pairs = [len(a & b) / len(a | b) if (a | b) else 1.0 for i, a in enumerate(sets) for b in sets[i + 1:]]
    common = set.intersection(*sets) if sets else set()
    return {"seeds": list(seeds), "sizes": [len(s) for s in sets], "mean_jaccard": float(np.mean(pairs)) if pairs else 1.0,
            "min_jaccard": float(np.min(pairs)) if pairs else 1.0, "common": sorted(common)}


def sweep_horizons(engine: "DiscoveryEngine", state: DiscoveryState, now, inputs: DS.SourceInputs, horizons: Sequence[int],
                   families: Sequence[str] | None = None, targets: Sequence[str] | None = None) -> list[StepReport]:
    """One step per (target, horizon) on the same state and ledger: 'what did it do next' at 1, 3, 5 sessions, and for direction and for
    range separately. Each pair is its own outcome tag, so its patterns are distinct trials and cost ledger wealth like any other."""
    out = []
    for tg in targets or [engine.cfg.target]:
        for h in horizons:
            e = DiscoveryEngine(dataclasses.replace(engine.cfg, horizon=int(h), target=tg), engine.source_cfg, engine.families_per_step,
                                engine.max_staleness, engine.core_context, engine.anchors, engine.audit, engine.audit_every, engine.use_miner,
                                engine.miner_params, engine.run_tag)
            out.append(e.step(state, now, inputs, families=families, ordered=False))
    return out


# --------------------------------------------------------------------------------------------------------- tabulations
def era_breakdown(state: DiscoveryState, era_edges: Sequence[int] | None = None) -> pd.DataFrame:
    """Findings per discovery era and verdict, with the median truth probability: is discovery only ever succeeding in one era?"""
    from engine.research.episodes import ERA_EDGES
    edges = np.asarray(era_edges or ERA_EDGES)
    rows = []
    for d in state.dossiers.values():
        yr = int(d.windows["train"][0][:4]) if d.windows.get("train") and d.windows["train"][0] else 0
        rows.append({"era": int(np.searchsorted(edges, yr, side="right")), "verdict": d.verdict.value, "truth": d.truth})
    if not rows:
        return pd.DataFrame(columns=["era", "verdict", "n", "median_truth"])
    df = pd.DataFrame(rows)
    return df.groupby(["era", "verdict"]).agg(n=("verdict", "size"), median_truth=("truth", "median")).reset_index()


def family_yield_table(state: DiscoveryState) -> pd.DataFrame:
    """Per source family: trials, survivors, failures, revalidated results and the smoothed failure rate. The families that only ever
    produce chance findings are visible here, and the scheduler already discounts them."""
    rows = []
    for f in sorted(set(state.families) | set(state.ledger.family_trials)):
        r = state.families.get(f, FamilyRecord())
        rows.append({"family": f, "visits": r.visits, "trials": state.ledger.family_trials.get(f, r.trials), "survivors": r.survivors,
                     "failed": r.failed, "confirmed": state.ledger.family_confirmed.get(f, 0),
                     "contradicted": state.ledger.family_failed.get(f, 0), "fdr": round(state.ledger.family_fdr(f), 3)})
    return pd.DataFrame(rows)


def explain(state: DiscoveryState, pid: str) -> str:
    """Everything known about one pattern in words: what it says, what it cost to find, how it validated, where it fails, and what it is
    NOT (never a decision). The counterexamples name stocks and dates, so this text is research-side only."""
    d = state.dossiers.get(pid)
    if d is None:
        return f"{pid}: unknown"
    v, ev, tp = d.validation, d.evidence, d.transfer
    fmt = lambda x, f="{:.3f}": "not measured" if x is None else f.format(x)
    lines = [f"{d.text}  [{d.verdict.value}]  ({', '.join(d.families)}; origin {d.origin})",
             f"  discovered {d.discovered_at} (run {d.run_id}) on {d.windows['train'][0]}..{d.windows['train'][1]}; outcomes seen through {d.outcomes_seen_through}",
             f"  direction {d.direction:+d}, discovery t {d.t_disc:+.2f}, mean {d.m_disc:+.5f}; raw p {d.p_raw:.2e}, cumulative q {d.q_cum:.3f} "
             f"after {d.trials_so_far} distinct trials",
             f"  evidence: {ev.rows} rows, {ev.active_weeks} active weeks in {ev.episodes} episodes, {ev.names} stocks; independent {ev.independent}; n_eff {ev.n_eff:.1f}",
             f"  validation: pooled t {fmt(v.pooled_t)}, {v.agree}/{v.n_blocks} blocks agree; held-out stocks t {fmt(d.holdout_t)}",
             f"  truth {fmt(d.truth)}, reliability {fmt(d.reliability)}, transfer {fmt(tp.transfer)} (era {fmt(tp.cross_era)}, stock {fmt(tp.cross_stock)}, "
             f"sector {fmt(tp.cross_sector)}, regime {fmt(tp.cross_regime)})",
             f"  context dependence {fmt(d.context_dependence)}" + (f" on {d.context_feature}" if d.context_feature else ""),
             f"  complexity {d.complexity.units:.1f} units, needs t {d.complexity.bar_t:.2f}, earns its place: {d.complexity.earns}",
             f"  counter-examples: {d.counter_rate:.1%} of firings go the other way ({d.counter_excess:+.1%} vs baseline); temporal shape {d.temporal}",
             f"  decision impact: {', '.join(d.impact.proposed_effects)} (estimate only, gain/week {fmt(d.impact.gain_per_week, '{:+.5f}')})"]
    if d.failure_conditions:
        lines.append("  fails when: " + "; ".join(f"{f.feature} in levels {f.levels[0]}-{f.levels[1]} ({f.kind})" for f in d.failure_conditions))
    if d.reasons:
        lines.append("  not trusted because: " + "; ".join(d.reasons))
    lines += [f"  e.g. {c.date} {c.ticker}: {c.outcome:+.4f} ({c.z:+.1f} sd)" for c in d.counterexamples[:3]]
    return "\n".join(lines)
