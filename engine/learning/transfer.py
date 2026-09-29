"""Unseen-year / cross-context transfer evaluation (contract C62 section 26; checklist E08, E09, E14-E16, J12-J14).

STATUS: IMPLEMENTED - NOT VALIDATED (C63: unit-tested on planted synthetic data only, never run on the real archive).

Same-year improvement is not enough. A learned rule is judged on situations it was NOT trained in, along six axes:
    YEAR        a calendar year the learner never saw
    REGIME      a market regime it never saw (trend x volatility state, computed point-in-time)
    STOCK       tickers it never saw
    SECTOR      sectors it never saw
    VOLATILITY  a stock-volatility bucket it never saw
    STOCK_TYPE  a beta/size type it never saw
For each axis the module reports the per-unit gain on familiar contexts and on novel contexts, and feeds both to
transfer_score (ratio, over-specialisation, memorisation gap, stability). The four things that make the numbers honest:

  1. Forward only. The headline cross-context gain uses units dated AFTER the learner's `learned_through` date. Novel units dated
     earlier are counted as anachronistic (the lesson came from their future) and reported, never used in the headline (C54/C56).
  2. Like against like. The same-context comparison group is also dated after `learned_through`, so time itself does not
     explain a ratio. Familiar units dated before it are REPLAYS; replay-minus-forward on familiar contexts is the memorisation gap.
  3. Fail closed. Any unit whose outcome had not matured before `now`, or a scope learned at/after `now`, or a unit that is both
     'replayed from training' and 'novel', raises FirewallBreach instead of being filtered away.
  4. UNKNOWN is not novel. A unit whose context could not be labelled is excluded from that axis; it is counted as untested.

A per-unit gain is learned_outcome - baseline_outcome for the same decision unit (a pick, a week, a ticker-date): the learner
never sees both outcomes, the harness does. Everything is seeded; nothing reads a clock or a file."""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import (Confidence, FirewallBreach, KnowledgeLike, ValidationLabel, _StrEnum, as_date, canonical_json, clip01,
                   require_past, stable_hash)
from .. import blind_gates as BG
from .. import learning_delta as LD
from . import transfer_score as TS

UNKNOWN_LABEL = "UNKNOWN"
REQUIRED = ("date", "mature", "ticker", "base", "learned")
CLUSTER_FORMATS = {"week": "%G-W%V", "month": "%Y-%m", "year": "%Y"}


class Axis(_StrEnum):
    YEAR = "YEAR"
    REGIME = "REGIME"
    STOCK = "STOCK"
    SECTOR = "SECTOR"
    VOLATILITY = "VOLATILITY"
    STOCK_TYPE = "STOCK_TYPE"


ALL_AXES = tuple(Axis)
AXIS_COL = {Axis.YEAR: "year", Axis.REGIME: "regime", Axis.STOCK: "ticker", Axis.SECTOR: "sector",
            Axis.VOLATILITY: "vol_bucket", Axis.STOCK_TYPE: "stock_type"}
HEADLINE_NAME = {Axis.YEAR: "cross_year_gain", Axis.REGIME: "cross_regime_gain", Axis.STOCK: "cross_stock_gain",
                 Axis.SECTOR: "cross_sector_gain", Axis.VOLATILITY: "cross_volatility_gain", Axis.STOCK_TYPE: "cross_stock_type_gain"}
# worst first: the overall verdict of a report is the worst verdict over the axes that were asked for
VERDICT_ORDER = (TS.TransferVerdictLabel.IDENTITY_DEPENDENT, TS.TransferVerdictLabel.OVER_SPECIALISED, TS.TransferVerdictLabel.HARMFUL,
                 TS.TransferVerdictLabel.UNSTABLE, TS.TransferVerdictLabel.INSUFFICIENT_EVIDENCE, TS.TransferVerdictLabel.NO_LEARNING,
                 TS.TransferVerdictLabel.TRANSFER_ONLY, TS.TransferVerdictLabel.GENERALISES)


# ---------------------------------------------------------------------------------------------------------------
# point-in-time context labellers (decision at the close of `date`; nothing after it is read)
# ---------------------------------------------------------------------------------------------------------------
def label_regimes(market_returns: pd.Series, now, *, lookback: int = 60, vol_lookback: int = 20, trend_thr: float = 0.03) -> pd.Series:
    """Regime label per period from the market return series alone: trend (trailing cumulative return over `lookback` periods:
    BULL above +trend_thr, BEAR below -trend_thr, else FLAT) x volatility (trailing sd over `vol_lookback` periods above the
    expanding median of its own past: STRESS, else CALM). Rolling and expanding windows end AT the label's own date, so a label
    never depends on a later return. Periods without a full lookback are UNKNOWN. Raises FirewallBreach if the series reaches `now`."""
    r = pd.Series(market_returns, dtype=float).sort_index()
    if len(r) and pd.Timestamp(r.index.max()) >= pd.Timestamp(as_date(now)):
        raise FirewallBreach(f"market series runs to {pd.Timestamp(r.index.max()).date()}, not strictly before now={as_date(now)}")
    if not np.isfinite(r.to_numpy()).all():
        raise ValueError("market returns contain NaN/inf")
    trend = np.expm1(np.log1p(r).rolling(lookback, min_periods=lookback).sum())
    vol = r.rolling(vol_lookback, min_periods=vol_lookback).std()
    ref = vol.expanding(min_periods=vol_lookback).median()
    t = pd.Series(np.where(trend > trend_thr, "BULL", np.where(trend < -trend_thr, "BEAR", "FLAT")), index=r.index)
    v = pd.Series(np.where(vol > ref, "STRESS", "CALM"), index=r.index)
    out = (t + "_" + v).where(trend.notna() & vol.notna() & ref.notna(), UNKNOWN_LABEL)
    out.name = "regime"
    return out


def regime_for_dates(labels: pd.Series, dates: Iterable) -> pd.Series:
    """As-of lookup: each date gets the label of the last regime date on or before it (never after); earlier dates are UNKNOWN."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    lab = labels.sort_index()
    out = lab.reindex(idx.union(lab.index)).ffill().reindex(idx)
    return out.fillna(UNKNOWN_LABEL).astype(str)


def label_volatility(returns: pd.DataFrame, *, lookback: int = 60, n_buckets: int = 3) -> pd.Series:
    """Per (date, ticker): volatility bucket V1..Vn from the cross-sectional rank of each stock's trailing return sd (window ends
    at the date). Rows without a full window are absent (attach_context turns them into UNKNOWN)."""
    if n_buckets < 2:
        raise ValueError("need at least two buckets")
    sd = returns.astype(float).rolling(lookback, min_periods=lookback).std()
    frac = (sd.rank(axis=1, method="average") - 1).div(sd.notna().sum(axis=1), axis=0)      # 0 .. (n-1)/n: the lowest name is bucket 1
    b = np.minimum(np.floor(frac.to_numpy() * n_buckets) + 1, n_buckets)
    lab = pd.DataFrame(b, index=sd.index, columns=sd.columns)
    s = lab.stack().dropna()                      # rows without a full window are absent, whatever the pandas stack default is
    s = ("V" + s.astype(int).astype(str))
    s.index.names = ["date", "ticker"]
    s.name = "vol_bucket"
    return s


def label_stock_type(beta: Mapping[str, float], dollar_volume: Mapping[str, float], data_through, now) -> dict:
    """Static ticker -> type from beta (median split: HB/LB) and dollar volume (median split: LARGE/SMALL), measured on data that
    ends strictly before `now`. Tickers missing either measure are absent."""
    require_past(data_through, now, "stock-type measurement window")
    common = sorted(set(beta) & set(dollar_volume))
    if not common:
        return {}
    b = pd.Series({t: float(beta[t]) for t in common})
    v = pd.Series({t: float(dollar_volume[t]) for t in common})
    b, v = b[np.isfinite(b)], v[np.isfinite(v)]
    common = sorted(set(b.index) & set(v.index))
    mb, mv = b[common].median(), v[common].median()
    return {t: ("HB" if b[t] >= mb else "LB") + "_" + ("LARGE" if v[t] >= mv else "SMALL") for t in common}


def attach_context(units: pd.DataFrame, *, regimes: pd.Series | None = None, vol: pd.Series | None = None,
                   sectors: Mapping[str, str] | None = None, stock_types: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Add regime / vol_bucket / sector / stock_type columns from the labellers above. Anything not labelled stays UNKNOWN."""
    d = units.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    if regimes is not None:
        d["regime"] = regime_for_dates(regimes, d["date"]).to_numpy()
    if vol is not None:
        key = pd.MultiIndex.from_arrays([d["date"], d["ticker"].astype(str)], names=["date", "ticker"])
        d["vol_bucket"] = vol.reindex(key).fillna(UNKNOWN_LABEL).astype(str).to_numpy()
    if sectors is not None:
        d["sector"] = d["ticker"].astype(str).map(sectors).fillna(UNKNOWN_LABEL)
    if stock_types is not None:
        d["stock_type"] = d["ticker"].astype(str).map(stock_types).fillna(UNKNOWN_LABEL)
    return d


# ---------------------------------------------------------------------------------------------------------------
# unit table
# ---------------------------------------------------------------------------------------------------------------
def prepare_units(df: pd.DataFrame, now, *, cluster_by: str = "month") -> pd.DataFrame:
    """Validate and normalise a table of decision units. Columns: date (decision close), mature (date the outcome was known),
    ticker, base (outcome without the lesson), learned (outcome with it); optional year/regime/sector/vol_bucket/stock_type/
    learned_disguised. Adds gain, key, cluster and string labels. Raises FirewallBreach for any outcome not matured before `now`,
    ValueError for malformed rows. The empty table is valid (every report on it is INSUFFICIENT_EVIDENCE)."""
    if cluster_by not in CLUSTER_FORMATS:
        raise ValueError(f"cluster_by must be one of {sorted(CLUSTER_FORMATS)}")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"unit table missing columns {missing}")
    d = df.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["mature"] = pd.to_datetime(d["mature"]).dt.normalize()
    d["ticker"] = d["ticker"].astype(str)
    if len(d):
        if (d["mature"] < d["date"]).any():
            raise ValueError("a unit matured before it was decided")
        late = d["mature"] >= pd.Timestamp(as_date(now))
        if late.any():
            raise FirewallBreach(f"{int(late.sum())} unit outcomes mature at/after now={as_date(now)} (first {d.loc[late, 'mature'].min().date()}): "
                                 "an unfinished outcome is not evidence")
        for c in ("base", "learned") + (("learned_disguised",) if "learned_disguised" in d else ()):
            if not np.isfinite(d[c].to_numpy(float)).all():
                raise ValueError(f"column {c} contains NaN/inf")
        if d.duplicated(["ticker", "date"]).any():
            raise ValueError("duplicate (ticker, date) units: each decision unit must appear once")
    if "year" not in d:
        d["year"] = d["date"].dt.year
    d["year"] = d["year"].astype(int).astype(str) if len(d) else d["year"].astype(str)
    for col in ("regime", "sector", "vol_bucket", "stock_type"):
        d[col] = d[col].fillna(UNKNOWN_LABEL).astype(str) if col in d else UNKNOWN_LABEL
    d["gain"] = d["learned"].astype(float) - d["base"].astype(float)
    d["key"] = d["ticker"] + "|" + d["date"].dt.strftime("%Y-%m-%d")
    d["cluster"] = d["date"].dt.strftime(CLUSTER_FORMATS[cluster_by]) if len(d) else ""
    return d


def drop_immature(df: pd.DataFrame, now) -> tuple[pd.DataFrame, int]:
    """The EXPLICIT alternative to the firewall's refusal: drop units not matured before `now`, and say how many. Callers that
    filter must own the count; prepare_units itself never filters silently."""
    m = pd.to_datetime(df["mature"]).dt.normalize() < pd.Timestamp(as_date(now))
    return df[m].copy(), int((~m).sum())


@dataclass(frozen=True)
class TrainingScope:
    """What the learner was trained on, per axis, and until when. Built from the training units, never typed by hand."""
    learned_through: dt.date
    years: frozenset = frozenset()
    regimes: frozenset = frozenset()
    tickers: frozenset = frozenset()
    sectors: frozenset = frozenset()
    vol_buckets: frozenset = frozenset()
    stock_types: frozenset = frozenset()
    keys: frozenset = frozenset()          # (ticker|date) of every training unit: replays are detected against these

    @classmethod
    def from_units(cls, train: pd.DataFrame, learned_through) -> "TrainingScope":
        lt = as_date(learned_through)
        if len(train) and (pd.to_datetime(train["mature"]).dt.date > lt).any():
            raise FirewallBreach("a training unit matured after learned_through: the scope would contain outcomes the learner "
                                 "could not have used")
        known = lambda col: frozenset(x for x in train[col].astype(str).unique() if x != UNKNOWN_LABEL) if len(train) else frozenset()
        return cls(lt, known("year"), known("regime"), known("ticker"), known("sector"), known("vol_bucket"), known("stock_type"),
                   frozenset(train["key"]) if len(train) and "key" in train else frozenset())

    def labels(self, axis: Axis) -> frozenset:
        return {Axis.YEAR: self.years, Axis.REGIME: self.regimes, Axis.STOCK: self.tickers, Axis.SECTOR: self.sectors,
                Axis.VOLATILITY: self.vol_buckets, Axis.STOCK_TYPE: self.stock_types}[axis]

    def validate(self, now) -> None:
        require_past(self.learned_through, now, "training scope learned_through")

    def fingerprint(self) -> str:
        return stable_hash({"lt": self.learned_through.isoformat(), "y": sorted(self.years), "r": sorted(self.regimes),
                            "s": sorted(self.sectors), "v": sorted(self.vol_buckets), "t": sorted(self.stock_types),
                            "n_tickers": len(self.tickers), "n_keys": len(self.keys)})


def scope_from_knowledge(items: Sequence[KnowledgeLike], train: pd.DataFrame, now) -> TrainingScope:
    """Training scope of a set of knowledge items: the union of the contexts they were learned in (item.contexts keys 'year',
    'regime', 'sector', 'vol_bucket', 'stock_type', 'ticker'; values a label or an iterable of labels), with learned_through the
    latest provenance.learned_at. An item that could not have existed at `now` raises FirewallBreach."""
    if not items:
        raise ValueError("no knowledge items")
    lt = None
    bag: dict[str, set[str]] = {c: set() for c in ("year", "regime", "sector", "vol_bucket", "stock_type", "ticker")}
    for it in items:
        bad = KnowledgeLike.conforms(it)
        if bad:
            raise ValueError(f"not a knowledge item: {bad}")
        if not it.provenance.could_exist_at(now):
            raise FirewallBreach(f"knowledge {it.knowledge_id} (learned {it.provenance.learned_at}, saw outcomes through "
                                 f"{it.provenance.outcomes_seen_through or it.provenance.learned_at}) could not exist at now={as_date(now)}")
        la = as_date(it.provenance.learned_at)
        lt = la if lt is None or la > lt else lt
        for dim, cond in it.contexts.items():
            if dim in bag:
                bag[dim].update(str(x) for x in ([cond] if isinstance(cond, (str, int, float)) else cond))
    base = TrainingScope.from_units(train, lt)
    pick = lambda name, fallback: frozenset(bag[name]) if bag[name] else fallback
    return dataclasses.replace(base, years=pick("year", base.years), regimes=pick("regime", base.regimes),
                               sectors=pick("sector", base.sectors), vol_buckets=pick("vol_bucket", base.vol_buckets),
                               stock_types=pick("stock_type", base.stock_types), tickers=pick("ticker", base.tickers))


# ---------------------------------------------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class _Masks:
    label: dict            # axis -> Series of str labels
    known: dict            # axis -> bool Series (label is not UNKNOWN)
    fam: dict              # axis -> bool Series (known and in training scope)
    nov: dict              # axis -> bool Series (known and NOT in training scope)
    fwd: pd.Series         # date strictly after learned_through
    replay: pd.Series      # unit key was in training


def _masks(d: pd.DataFrame, scope: TrainingScope) -> _Masks:
    lt = pd.Timestamp(scope.learned_through)
    label = {ax: d[AXIS_COL[ax]].astype(str) for ax in ALL_AXES}
    known = {ax: label[ax] != UNKNOWN_LABEL for ax in ALL_AXES}
    fam = {ax: known[ax] & label[ax].isin(scope.labels(ax)) for ax in ALL_AXES}
    nov = {ax: known[ax] & ~label[ax].isin(scope.labels(ax)) for ax in ALL_AXES}
    fwd = d["date"] > lt
    replay = d["key"].isin(scope.keys) if scope.keys else pd.Series(False, index=d.index)
    anynov = np.zeros(len(d), bool)
    for ax in ALL_AXES:
        anynov |= nov[ax].to_numpy()
    if (replay & (anynov | fwd)).any():
        raise FirewallBreach("scope inconsistent: units flagged as training replays are also novel or dated after learned_through "
                             f"({int((replay & (pd.Series(anynov, index=d.index) | fwd)).sum())} units)")
    return _Masks(label, known, fam, nov, fwd, replay)


def _select(m: _Masks, axis: Axis, mode: str):
    """(same_forward, same_replay, cross_forward, cross_past) boolean arrays for one axis."""
    others = [o for o in ALL_AXES if o != axis]
    if mode == "isolated":
        all_fam = np.ones(len(m.fwd), bool)
        for o in others:
            all_fam &= m.fam[o].to_numpy()
        fam = m.fam[axis].to_numpy() & all_fam
        nov = m.nov[axis].to_numpy() & all_fam
    elif mode == "marginal":
        fam, nov = m.fam[axis].to_numpy(), m.nov[axis].to_numpy()
    else:
        raise ValueError("mode must be 'marginal' or 'isolated'")
    fwd, rep = m.fwd.to_numpy(), m.replay.to_numpy()
    return fam & fwd & ~rep, fam & ~fwd, nov & fwd, nov & ~fwd


def _group_labels(d: pd.DataFrame, axis: Axis, mask: np.ndarray, n_stock_groups: int) -> pd.Series:
    lab = d.loc[mask, AXIS_COL[axis]].astype(str)
    if axis == Axis.STOCK:
        lab = lab.map(lambda t: f"tickers-g{int(stable_hash(t, 8), 16) % n_stock_groups}")
    return lab


# ---------------------------------------------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AxisResult:
    axis: Axis
    mode: str
    same: TS.BootMean                      # familiar, dated after learned_through (or all familiar if too few)
    cross: TS.BootMean                     # novel, dated after learned_through: THE cross-context gain
    replay: TS.BootMean                    # familiar, dated up to learned_through (the learner's own training situations)
    ratio: TS.RatioResult
    ratio_ci: TS.RatioCI | None
    specialisation: TS.Specialisation
    memorization: TS.GapResult | None      # replay - same_forward on familiar units
    stability: TS.TransferStability
    groups: tuple                          # ((label, n, gain), ...) for the novel groups
    verdict: TS.TransferVerdict
    n_anachronistic: int                   # novel units dated at/before learned_through (excluded from the headline)
    n_unknown: int                         # units whose label on this axis was UNKNOWN (untested, not novel)
    same_includes_replay: bool

    @property
    def tested(self) -> bool:
        return self.cross.n > 0

    def as_dict(self) -> dict:
        return {"axis": self.axis.value, "mode": self.mode, "same": self.same.as_dict(), "cross": self.cross.as_dict(),
                "replay": self.replay.as_dict(), "ratio": self.ratio.value, "ratio_status": self.ratio.status.value,
                "ratio_lo": self.ratio_ci.lo if self.ratio_ci else None, "ratio_hi": self.ratio_ci.hi if self.ratio_ci else None,
                "ratio_bounded": self.ratio_ci.bounded if self.ratio_ci else None,
                "specialisation": self.specialisation.flag, "memorization_gap": self.memorization.as_dict() if self.memorization else None,
                "stability": dataclasses.asdict(self.stability), "groups": [list(g) for g in self.groups],
                "verdict": self.verdict.label.value, "validation": self.verdict.validation.value, "reasons": list(self.verdict.reasons),
                "n_anachronistic": self.n_anachronistic, "n_unknown": self.n_unknown, "same_includes_replay": self.same_includes_replay}


@dataclass(frozen=True)
class TransferReport:
    now: dt.date
    scope_fingerprint: str
    n_units: int
    axes: Mapping[Axis, AxisResult]
    depth: pd.DataFrame                    # gain by novelty depth (number of novel axes)
    year_gap: pd.DataFrame                 # gain by years away from the nearest trained year
    identity: TS.GapResult | None
    overall: TS.TransferVerdictLabel
    untested_axes: tuple
    label: ValidationLabel = ValidationLabel.NOT_VALIDATED

    def gain(self, axis: Axis) -> float:
        r = self.axes.get(axis)
        return float("nan") if r is None or not r.tested else r.cross.mean

    @property
    def same_year_gain(self) -> float:
        r = self.axes.get(Axis.YEAR)
        return float("nan") if r is None else r.same.mean

    @property
    def cross_year_gain(self) -> float:
        return self.gain(Axis.YEAR)

    @property
    def cross_regime_gain(self) -> float:
        return self.gain(Axis.REGIME)

    @property
    def cross_stock_gain(self) -> float:
        return self.gain(Axis.STOCK)

    @property
    def cross_sector_gain(self) -> float:
        return self.gain(Axis.SECTOR)

    @property
    def cross_volatility_gain(self) -> float:
        return self.gain(Axis.VOLATILITY)

    def headline(self) -> dict:
        """The tracked gains named by section 26 (NaN = untested, which is different from zero)."""
        out = {"same_year_gain": self.same_year_gain}
        for ax in ALL_AXES:
            out[HEADLINE_NAME[ax]] = self.gain(ax)
        return out

    def transfer_ratio(self, axis: Axis = Axis.YEAR) -> TS.RatioResult | None:
        r = self.axes.get(axis)
        return None if r is None else r.ratio

    def as_record(self) -> dict:
        rec = {"now": self.now.isoformat(), "scope": self.scope_fingerprint, "n_units": self.n_units, "overall": self.overall.value,
               "label": self.label.value, "untested_axes": [a.value for a in self.untested_axes], "headline": self.headline(),
               "axes": {a.value: r.as_dict() for a, r in self.axes.items()},
               "identity": self.identity.as_dict() if self.identity else None,
               "depth": self.depth.to_dict("records"), "year_gap": self.year_gap.to_dict("records")}
        rec["report_id"] = stable_hash(rec)
        return rec


# ---------------------------------------------------------------------------------------------------------------
# the evaluation
# ---------------------------------------------------------------------------------------------------------------
def _bm(d, mask, n_boot, seed):
    return TS.cluster_bootstrap_mean(d.loc[mask, "gain"].to_numpy(), d.loc[mask, "cluster"].to_numpy(), n_boot=n_boot, seed=seed)


def _axis_result(d, m, scope, axis, mode, n_boot, seed, min_units, min_group_units, n_stock_groups, ident, ratio_boot) -> AxisResult:
    fam_fwd, fam_rep, nov_fwd, nov_past = _select(m, axis, mode)
    s0 = seed * 7919 + int(stable_hash(axis.value, 4), 16) % 9973
    same_incl_replay = int(fam_fwd.sum()) < min_units <= int((fam_fwd | fam_rep).sum())
    same_mask = (fam_fwd | fam_rep) if same_incl_replay else fam_fwd
    same, cross, rep = _bm(d, same_mask, n_boot, s0 + 1), _bm(d, nov_fwd, n_boot, s0 + 2), _bm(d, fam_rep, n_boot, s0 + 3)
    ratio = TS.transfer_ratio(cross.mean, same.mean, n_cross=cross.n, n_same=same.n, min_n=min_units)
    rci = None
    if ratio_boot and cross.n >= min_units and same.n >= min_units:
        rci = TS.ratio_ci(d.loc[nov_fwd, "gain"], d.loc[same_mask, "gain"], d.loc[nov_fwd, "cluster"], d.loc[same_mask, "cluster"],
                          n_boot=n_boot, seed=s0 + 4)
    spec = TS.specialisation(same, cross, ratio)
    mem = None
    if int(fam_rep.sum()) >= min_units and int(fam_fwd.sum()) >= min_units:
        mem = TS.memorization_gap(d.loc[fam_rep, "gain"], d.loc[fam_fwd, "gain"], d.loc[fam_rep, "cluster"], d.loc[fam_fwd, "cluster"],
                                  n_boot=n_boot, seed=s0 + 5)
    gl = _group_labels(d, axis, nov_fwd, n_stock_groups)
    groups = []
    for lab, sub in d.loc[nov_fwd].groupby(gl.to_numpy()):
        if len(sub) >= min_group_units:
            groups.append((str(lab), int(len(sub)), float(sub["gain"].mean())))
    stab = TS.transfer_stability([g[2] for g in groups])
    thin = cross.n < min_units or same.n < min_units
    if thin:
        verdict = TS.TransferVerdict(TS.TransferVerdictLabel.INSUFFICIENT_EVIDENCE, ValidationLabel.INSUFFICIENT_EVIDENCE,
                                     (f"fewer than {min_units} units (same={same.n}, cross={cross.n})",), ratio)
    else:
        verdict = TS.classify_transfer(same, cross, ratio, spec, mem=mem, ident=ident, stability=stab)
    n_unknown = int((~m.known[axis]).sum())
    return AxisResult(axis, mode, same, cross, rep, ratio, rci, spec, mem, stab, tuple(groups), verdict,
                      int(nov_past.sum()), n_unknown, same_incl_replay)


def depth_table(d: pd.DataFrame, m: _Masks, *, seed: int = 0, n_boot: int = 400, min_units: int = 30) -> pd.DataFrame:
    """Gain against novelty depth (how many of the six axes are novel at once) on forward, non-replayed units. Depth 0 is the
    fully familiar situation. A learner that has learned a rule keeps its gain as depth grows; a memoriser's gain falls to zero."""
    fwd = (m.fwd & ~m.replay).to_numpy()
    depth = np.zeros(len(d), int)
    for ax in ALL_AXES:
        depth += m.nov[ax].to_numpy().astype(int)
    rows = []
    for k in sorted(set(depth[fwd].tolist())):
        sel = fwd & (depth == k)
        bm = _bm(d, sel, n_boot, seed + 101 * (k + 1))
        rows.append({"depth": int(k), "n": bm.n, "gain": bm.mean, "lo": bm.lo, "hi": bm.hi, "enough": bm.n >= min_units})
    return pd.DataFrame(rows, columns=["depth", "n", "gain", "lo", "hi", "enough"])


def year_gap_table(d: pd.DataFrame, m: _Masks, scope: TrainingScope, *, seed: int = 0, n_boot: int = 400, min_units: int = 30) -> pd.DataFrame:
    """Gain against the distance (in years) from the nearest trained year: does transfer decay the further away the test year is?"""
    trained = np.array(sorted(int(y) for y in scope.years if str(y).lstrip("-").isdigit()), int)
    if len(trained) == 0:
        return pd.DataFrame(columns=["gap", "n", "gain", "lo", "hi", "enough"])
    yr = d["year"].astype(int).to_numpy()
    gap = np.abs(yr[:, None] - trained[None, :]).min(axis=1)
    sel0 = (m.nov[Axis.YEAR] & m.fwd).to_numpy()
    rows = []
    for g in sorted(set(gap[sel0].tolist())):
        bm = _bm(d, sel0 & (gap == g), n_boot, seed + 211 * (g + 1))
        rows.append({"gap": int(g), "n": bm.n, "gain": bm.mean, "lo": bm.lo, "hi": bm.hi, "enough": bm.n >= min_units})
    return pd.DataFrame(rows, columns=["gap", "n", "gain", "lo", "hi", "enough"])


def evaluate_transfer(units: pd.DataFrame, scope: TrainingScope, now, *, axes: Sequence[Axis] = ALL_AXES, mode: str = "marginal",
                      cluster_by: str = "month", n_boot: int = 600, seed: int = 0, min_units: int = 30, min_group_units: int = 10,
                      n_stock_groups: int = 8, ratio_boot: bool = True) -> TransferReport:
    """Cross-context transfer report for one learned state. `units` holds decision units with base and learned outcomes (see
    prepare_units); `scope` says what the learner was trained on. Optional column learned_disguised (the same learner's outcome
    on the same unit after identities were scrambled) adds the identity gap. mode 'isolated' compares units that are novel on
    ONE axis and familiar on all others with units familiar on all axes, so an axis is not credited for another axis's novelty."""
    d = prepare_units(units, now, cluster_by=cluster_by)
    scope.validate(now)
    if len(d) == 0:
        empty = pd.DataFrame(columns=["depth", "n", "gain", "lo", "hi", "enough"])
        return TransferReport(as_date(now), scope.fingerprint(), 0, {}, empty, pd.DataFrame(columns=["gap", "n", "gain", "lo", "hi", "enough"]),
                              None, TS.TransferVerdictLabel.INSUFFICIENT_EVIDENCE, tuple(axes), ValidationLabel.INSUFFICIENT_EVIDENCE)
    m = _masks(d, scope)
    ident = None
    if "learned_disguised" in d:
        sel = (m.fwd & ~m.replay).to_numpy()
        if int(sel.sum()) >= min_units:
            ident = TS.identity_gap(d.loc[sel, "learned"] - d.loc[sel, "base"], d.loc[sel, "learned_disguised"] - d.loc[sel, "base"],
                                    d.loc[sel, "cluster"], n_boot=n_boot, seed=seed + 17)
    res, untested = {}, []
    for ax in axes:
        r = _axis_result(d, m, scope, ax, mode, n_boot, seed, min_units, min_group_units, n_stock_groups, ident, ratio_boot)
        res[ax] = r
        if not r.tested:
            untested.append(ax)
    order = {v: i for i, v in enumerate(VERDICT_ORDER)}
    overall = min((r.verdict.label for r in res.values()), key=lambda v: order[v], default=TS.TransferVerdictLabel.INSUFFICIENT_EVIDENCE)
    lab = ValidationLabel.NOT_VALIDATED if overall in (TS.TransferVerdictLabel.GENERALISES, TS.TransferVerdictLabel.TRANSFER_ONLY,
                                                        TS.TransferVerdictLabel.NO_LEARNING) else (
        ValidationLabel.INSUFFICIENT_EVIDENCE if overall == TS.TransferVerdictLabel.INSUFFICIENT_EVIDENCE else ValidationLabel.FAILED_VALIDATION)
    return TransferReport(as_date(now), scope.fingerprint(), len(d), res, depth_table(d, m, seed=seed, n_boot=max(200, n_boot // 2), min_units=min_units),
                          year_gap_table(d, m, scope, seed=seed, n_boot=max(200, n_boot // 2), min_units=min_units), ident, overall,
                          tuple(untested), lab)


def transfer_confidence(report: TransferReport) -> float | None:
    """Value for Confidence.transfer (section 11): None (UNTESTED) when no axis was testable; 0 when any tested axis is
    over-specialised or identity dependent; otherwise the mean over tested axes of clip01(ratio) scaled by that axis's stability
    score, with TRANSFER_ONLY axes counting 0.5. Deliberately conservative: an axis that failed to bound its ratio counts 0."""
    tested = [r for r in report.axes.values() if r.tested]
    if not tested:
        return None
    if any(r.verdict.label in (TS.TransferVerdictLabel.OVER_SPECIALISED, TS.TransferVerdictLabel.IDENTITY_DEPENDENT,
                               TS.TransferVerdictLabel.HARMFUL) for r in tested):
        return 0.0
    vals = []
    for r in tested:
        if r.ratio.status == TS.RatioStatus.OK and r.ratio.value is not None:
            bounded = r.ratio_ci is None or r.ratio_ci.bounded
            vals.append(clip01(r.ratio.value) * max(r.stability.score, 0.2 if r.stability.n_groups < 3 else 0.0) if bounded else 0.0)
        elif r.ratio.status == TS.RatioStatus.TRANSFER_ONLY:
            vals.append(0.5)
        else:
            vals.append(0.0)
    return clip01(float(np.mean(vals)))


def knowledge_confidence(report: TransferReport, base: Confidence | None = None) -> Confidence:
    """A Confidence with only the `transfer` dimension filled (other dimensions untouched or None)."""
    t = transfer_confidence(report)
    return dataclasses.replace(base if base is not None else Confidence(), transfer=t)


def render_report(report: TransferReport) -> str:
    """Plain-text report. Always prints the validation label; never the word 'improved'."""
    L = [f"TRANSFER REPORT  now={report.now}  units={report.n_units}  scope={report.scope_fingerprint}  [{report.label.value}]",
         f"overall: {report.overall.value}   untested axes: {', '.join(a.value for a in report.untested_axes) or 'none'}"]
    for ax, r in report.axes.items():
        L.append(f"- {ax.value:<10} same {r.same.mean:+.5f} (n={r.same.n})  cross {r.cross.mean:+.5f} "
                 f"[{r.cross.lo:+.4f},{r.cross.hi:+.4f}] (n={r.cross.n})  ratio "
                 f"{'n/a' if r.ratio.value is None else f'{r.ratio.value:+.2f}'} ({r.ratio.status.value})  spec={r.specialisation.flag}  "
                 f"stability={r.stability.score:.2f}  -> {r.verdict.label.value}")
        if r.n_anachronistic:
            L.append(f"    {r.n_anachronistic} novel units dated before learned_through were excluded (anachronistic)")
        if r.same_includes_replay:
            L.append("    same-context group includes replays (too few forward units): ratio is conservative")
        if r.n_unknown:
            L.append(f"    {r.n_unknown} units unlabelled on this axis (untested, not novel)")
        for reason in r.verdict.reasons:
            L.append(f"    why: {reason}")
    if report.identity is not None:
        L.append(f"identity gap {report.identity.gap:+.5f} [{report.identity.lo:+.4f},{report.identity.hi:+.4f}] flagged={report.identity.flagged}")
    if len(report.depth):
        L.append("gain by novelty depth: " + ", ".join(f"d{int(r.depth)}={r.gain:+.4f}(n={int(r.n)})" for r in report.depth.itertuples()))
    if len(report.year_gap):
        L.append("gain by years from nearest trained year: " + ", ".join(f"{int(r.gap)}y={r.gain:+.4f}(n={int(r.n)})" for r in report.year_gap.itertuples()))
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------
# re-fitting harness: leave-one-context-out evaluation of a learner that can be retrained
# ---------------------------------------------------------------------------------------------------------------
Fit = Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]]


@dataclass(frozen=True)
class Fold:
    axis: Axis
    test_label: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    forward: bool              # every test unit is dated after every training unit matured
    note: str = ""


def make_folds(d: pd.DataFrame, axis: Axis, *, seed: int = 0, n_stock_folds: int = 5, min_train_years: int = 2) -> list[Fold]:
    """Held-out-context folds on a PREPARED unit table.
    YEAR        expanding origin: test year y trains on years before y whose outcomes matured before y began (forward, purged).
    STOCK       tickers hashed into n_stock_folds groups (seeded); a test group is never in training (contemporaneous dates).
    others      leave one label out; training contains the other labels at all dates (not forward: reported as such)."""
    folds: list[Fold] = []
    if axis == Axis.YEAR:
        years = sorted(d["year"].astype(int).unique())
        for y in years:
            start = d.loc[d["year"].astype(int) == y, "date"].min()
            tr = np.flatnonzero(((d["year"].astype(int) < y) & (d["mature"] < start)).to_numpy())
            te = np.flatnonzero((d["year"].astype(int) == y).to_numpy())
            if len(set(d.iloc[tr]["year"])) >= min_train_years and len(te):
                folds.append(Fold(axis, str(y), tr, te, True, "expanding origin, purged"))
    elif axis == Axis.STOCK:
        g = d["ticker"].map(lambda t: int(stable_hash([t, seed], 8), 16) % n_stock_folds).to_numpy()
        for k in range(n_stock_folds):
            tr, te = np.flatnonzero(g != k), np.flatnonzero(g == k)
            if len(tr) and len(te):
                folds.append(Fold(axis, f"tickers-g{k}", tr, te, False, "contemporaneous: dates overlap training"))
    else:
        col = AXIS_COL[axis]
        lab = d[col].astype(str)
        for L in sorted(set(lab) - {UNKNOWN_LABEL}):
            tr = np.flatnonzero(((lab != L) & (lab != UNKNOWN_LABEL)).to_numpy())
            te = np.flatnonzero((lab == L).to_numpy())
            if len(tr) and len(te):
                folds.append(Fold(axis, L, tr, te, False, "leave-one-label-out: training includes other labels at all dates"))
    return folds


@dataclass(frozen=True)
class FoldResult:
    fold: Fold
    gain_test: np.ndarray      # per test unit
    gain_replay: np.ndarray    # per training unit, predicted by the model that trained on them
    cluster_test: np.ndarray
    cluster_replay: np.ndarray


def _apply(d: pd.DataFrame, predict, idx: np.ndarray) -> np.ndarray:
    sub = d.iloc[idx]
    w = np.asarray(predict(sub), float).ravel()
    if len(w) != len(sub):
        raise ValueError(f"learner returned {len(w)} weights for {len(sub)} units")
    if not np.isfinite(w).all() or (w < -1e-12).any() or (w > 1 + 1e-12).any():
        raise ValueError("learner weights must be finite and within [0, 1]")
    return np.clip(w, 0.0, 1.0) * (sub["alt"].to_numpy(float) - sub["base"].to_numpy(float))


def run_folds(units: pd.DataFrame, fit: Fit, folds: Sequence[Fold], now, *, cluster_by: str = "month") -> list[FoldResult]:
    """Retrain `fit` on each fold's training units and score the held-out units. `units` needs base and alt (the outcome had the
    lesson-altered action been taken); learned = base + w * (alt - base) where w in [0,1] is the learner's weight for the altered
    action. The learner sees only training rows (with both outcomes) and, at prediction time, only the test row's features."""
    d = prepare_units(units.assign(learned=units["base"]), now, cluster_by=cluster_by) if "learned" not in units else prepare_units(units, now, cluster_by=cluster_by)
    if "alt" not in d:
        raise ValueError("run_folds needs an 'alt' column")
    out = []
    for f in folds:
        if f.forward:                                    # training outcomes must be known before the test period starts
            if d.iloc[f.train_idx]["mature"].max() >= d.iloc[f.test_idx]["date"].min():
                raise FirewallBreach(f"fold {f.test_label}: a training outcome matured at/after the first test decision")
        if set(f.train_idx) & set(f.test_idx):
            raise FirewallBreach(f"fold {f.test_label}: a unit is in both training and test")
        predict = fit(d.iloc[f.train_idx].copy())
        out.append(FoldResult(f, _apply(d, predict, f.test_idx), _apply(d, predict, f.train_idx),
                              d.iloc[f.test_idx]["cluster"].to_numpy(), d.iloc[f.train_idx]["cluster"].to_numpy()))
    return out


def axis_result_from_folds(axis: Axis, results: Sequence[FoldResult], *, n_boot: int = 600, seed: int = 0, min_units: int = 30) -> AxisResult:
    """Pool fold results: same-context = the trained model replayed on its own training units, cross-context = held-out gains."""
    if not results:
        z = TS.cluster_bootstrap_mean([], None)
        ratio = TS.transfer_ratio(float("nan"), float("nan"))
        return AxisResult(axis, "folds", z, z, z, ratio, None, TS.specialisation(z, z, ratio), None, TS.transfer_stability([]), (),
                          TS.TransferVerdict(TS.TransferVerdictLabel.INSUFFICIENT_EVIDENCE, ValidationLabel.INSUFFICIENT_EVIDENCE, ("no folds",)),
                          0, 0, False)
    te = np.concatenate([r.gain_test for r in results])
    tc = np.concatenate([r.cluster_test for r in results])
    rp = np.concatenate([r.gain_replay for r in results])
    rc = np.concatenate([r.cluster_replay for r in results])
    same, cross = TS.cluster_bootstrap_mean(rp, rc, n_boot=n_boot, seed=seed + 1), TS.cluster_bootstrap_mean(te, tc, n_boot=n_boot, seed=seed + 2)
    ratio = TS.transfer_ratio(cross.mean, same.mean, n_cross=cross.n, n_same=same.n, min_n=min_units)
    rci = TS.ratio_ci(te, rp, tc, rc, n_boot=n_boot, seed=seed + 3) if min(cross.n, same.n) >= min_units else None
    spec = TS.specialisation(same, cross, ratio)
    groups = tuple((r.fold.test_label, int(len(r.gain_test)), float(r.gain_test.mean())) for r in results if len(r.gain_test))
    stab = TS.transfer_stability([g[2] for g in groups])
    verdict = TS.classify_transfer(same, cross, ratio, spec, stability=stab)
    return AxisResult(axis, "folds", same, cross, same, ratio, rci, spec, None, stab, groups, verdict, 0, 0, False)


def cross_validate_transfer(units: pd.DataFrame, fit: Fit, now, *, axes: Sequence[Axis] = (Axis.YEAR, Axis.STOCK), seed: int = 0,
                            n_boot: int = 600, min_units: int = 30, **fold_kw) -> dict:
    """Retrain-and-hold-out transfer per axis. Returns {axis: AxisResult}."""
    d = prepare_units(units.assign(learned=units["base"]) if "learned" not in units else units, now)
    return {ax: axis_result_from_folds(ax, run_folds(units, fit, make_folds(d, ax, seed=seed, **fold_kw), now), n_boot=n_boot, seed=seed,
                                       min_units=min_units) for ax in axes}


# ---------------------------------------------------------------------------------------------------------------
# planted data and reference learners (the harness's own controls; C63 planted-defect calibration)
# ---------------------------------------------------------------------------------------------------------------
def synthetic_transfer_units(seed: int = 0, n_years: int = 6, n_tickers: int = 40, weeks_per_year: int = 26, beta: float = 0.02,
                             ticker_effect_sd: float = 0.02, noise_sd: float = 0.03, n_sectors: int = 4, first_year: int = 2010,
                             beta_flips_in: str | None = None) -> pd.DataFrame:
    """Planted world. Unit outcome under the altered action: alt = base + beta*f1 + ticker_effect + noise, base ~ N(0, noise_sd).
    beta*f1 is a TRANSFERABLE rule (same in every year, stock and sector); ticker_effect is a persistent per-ticker offset that
    can only be memorised. `beta_flips_in='BEAR'` flips the rule's sign in years whose regime contains that string (a
    regime-bound rule). Regimes alternate by year (BULL_CALM / BEAR_STRESS); volatility buckets and types are per ticker."""
    rng = np.random.default_rng(seed)
    tick = [f"T{i:03d}" for i in range(n_tickers)]
    eff = dict(zip(tick, rng.normal(0, ticker_effect_sd, n_tickers)))
    sector = {t: f"S{i % n_sectors}" for i, t in enumerate(tick)}
    vol = {t: f"V{1 + (i * 3) // n_tickers}" for i, t in enumerate(tick)}
    stype = {t: ("HB" if i % 2 else "LB") + "_" + ("LARGE" if (i // 2) % 2 else "SMALL") for i, t in enumerate(tick)}
    rows = []
    for yi in range(n_years):
        year = first_year + yi
        regime = "BULL_CALM" if yi % 2 == 0 else "BEAR_STRESS"
        b = -beta if (beta_flips_in and beta_flips_in in regime) else beta
        for w in range(weeks_per_year):
            date = pd.Timestamp(year=year, month=1, day=4) + pd.Timedelta(weeks=w)
            f = rng.normal(0, 1, n_tickers)
            base = rng.normal(0, noise_sd, n_tickers)
            alt = base + b * f + np.array([eff[t] for t in tick]) + rng.normal(0, noise_sd / 2, n_tickers)
            for i, t in enumerate(tick):
                rows.append((date, date + pd.Timedelta(days=7), t, float(base[i]), float(alt[i]), float(f[i]), sector[t], regime, vol[t], stype[t]))
    return pd.DataFrame(rows, columns=["date", "mature", "ticker", "base", "alt", "f1", "sector", "regime", "vol_bucket", "stock_type"])


def rule_learner_fit(train: pd.DataFrame):
    """A legitimate learner: fits the pooled slope of (alt - base) on f1 and switches to the altered action when the fitted
    effect is positive. Knows nothing about tickers or dates."""
    x, y = train["f1"].to_numpy(float), (train["alt"] - train["base"]).to_numpy(float)
    var = float(x.var())
    b = float(np.cov(x, y, ddof=0)[0, 1] / var) if var > 0 else 0.0
    return lambda t: (b * t["f1"].to_numpy(float) > 0).astype(float)


def identity_memoriser_fit(train: pd.DataFrame):
    """CONTROL C: stores the realised (alt - base) of every (ticker, date) it has seen and replays the sign. Unseen keys get 0."""
    tbl = dict(zip(train["ticker"].astype(str) + "|" + pd.to_datetime(train["date"]).dt.strftime("%Y-%m-%d"),
                   (train["alt"] - train["base"]).to_numpy(float)))
    return lambda t: np.array([1.0 if tbl.get(k, 0.0) > 0 else 0.0 for k in
                               (t["ticker"].astype(str) + "|" + pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d"))])


def ticker_memoriser_fit(train: pd.DataFrame):
    """Stores each ticker's mean (alt - base): a real gain on tickers it has seen, none on new ones."""
    m = (train["alt"] - train["base"]).groupby(train["ticker"].astype(str)).mean()
    return lambda t: (t["ticker"].astype(str).map(m).fillna(0.0).to_numpy(float) > 0).astype(float)


def random_learner_fit(seed: int = 0):
    """CONTROL D: a coin-flip weight per unit. Its expected gain is half the mean of (alt - base); it sets the luck floor."""
    def fit(train):
        def predict(t):
            r = np.random.default_rng(int(stable_hash([seed, len(t), str(t["ticker"].iloc[0]) if len(t) else ""], 8), 16) % (2 ** 32))
            return (r.random(len(t)) < 0.5).astype(float)
        return predict
    return fit


def harness_selfcheck(seed: int = 0, n_boot: int = 300) -> dict:
    """Can the harness tell a rule from a lookup table? Returns the overall verdict of each reference learner over YEAR and STOCK
    axes on the planted world. A harness where the memoriser is not flagged is void."""
    u = synthetic_transfer_units(seed=seed, n_years=6, n_tickers=30, weeks_per_year=20)
    now = pd.Timestamp("2030-01-01")
    out = {}
    for name, fit in (("rule", rule_learner_fit), ("identity_memoriser", identity_memoriser_fit), ("ticker_memoriser", ticker_memoriser_fit)):
        res = cross_validate_transfer(u, fit, now, axes=(Axis.YEAR, Axis.STOCK), seed=seed, n_boot=n_boot, min_units=30)
        out[name] = {ax.value: (r.verdict.label.value, r.ratio.value, r.cross.mean, r.same.mean) for ax, r in res.items()}
    return out


# ---------------------------------------------------------------------------------------------------------------
# interaction between axes: where exactly does the lesson stop working?
# ---------------------------------------------------------------------------------------------------------------
def interaction_table(units: pd.DataFrame, scope: TrainingScope, now, ax1: Axis, ax2: Axis, *, seed: int = 0, n_boot: int = 400,
                      min_units: int = 30, cluster_by: str = "month") -> pd.DataFrame:
    """Gain on forward, non-replayed units in the four cells familiar/novel on ax1 x familiar/novel on ax2. Shows interactions the
    marginal report hides: 'transfers to new years in the same regime, fails to new regimes in any year'. Units unlabelled on
    either axis are excluded (untested, not novel). Cells below min_units are reported with enough=False."""
    if ax1 == ax2:
        raise ValueError("choose two different axes")
    d = prepare_units(units, now, cluster_by=cluster_by)
    scope.validate(now)
    m = _masks(d, scope) if len(d) else None
    rows = []
    if m is not None:
        base = (m.fwd & ~m.replay).to_numpy()
        for f1 in (True, False):
            for f2 in (True, False):
                a = (m.fam[ax1] if f1 else m.nov[ax1]).to_numpy()
                b = (m.fam[ax2] if f2 else m.nov[ax2]).to_numpy()
                bm = _bm(d, base & a & b, n_boot, seed + 31 * (int(f1) * 2 + int(f2) + 1))
                rows.append({ax1.value: "familiar" if f1 else "novel", ax2.value: "familiar" if f2 else "novel", "n": bm.n, "gain": bm.mean,
                             "lo": bm.lo, "hi": bm.hi, "enough": bm.n >= min_units})
    return pd.DataFrame(rows, columns=[ax1.value, ax2.value, "n", "gain", "lo", "hi", "enough"])


# ---------------------------------------------------------------------------------------------------------------
# distance and time decay
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DecayTable:
    kind: str                       # 'feature-distance' or 'time-since-learning'
    frame: pd.DataFrame
    spearman: float                 # rank correlation of per-unit gain with distance / age
    half_life: float | None         # time decay only: days for the gain to halve (None: no decay found)
    decays: bool

    def statement(self) -> str:
        if self.kind == "time-since-learning":
            if not self.decays:
                return "no decay of the gain with time since learning detected"
            return "lesson decays" + (f" with half-life {self.half_life:.0f} days" if self.half_life else "")
        return ("gain falls as the situation moves away from the training conditions" if self.decays
                else "gain does not fall with distance from the training conditions")


_DECAY_COLS = {"feature-distance": ["bin", "lo_dist", "hi_dist", "n", "gain", "lo", "hi", "enough"],
               "time-since-learning": ["bin", "days_from", "days_to", "n", "gain", "lo", "hi", "enough"]}


def distance_decay(units: pd.DataFrame, train: pd.DataFrame, scope: TrainingScope, now, feature_cols: Sequence[str], *, n_bins: int = 4,
                   seed: int = 0, n_boot: int = 400, min_units: int = 30, cluster_by: str = "month") -> DecayTable:
    """Gain against how far a situation's numeric context (e.g. market volatility, trailing return, breadth) lies from the
    training conditions. Distance = root-mean-square z-score of the features, centred and scaled on the TRAINING units only, so
    'different market conditions' is measured on a continuum, not only by labels. Forward, non-replayed units only."""
    from scipy.stats import spearmanr
    d = prepare_units(units, now, cluster_by=cluster_by)
    t = prepare_units(train, now, cluster_by=cluster_by)
    scope.validate(now)
    for c in feature_cols:
        if c not in d or c not in t:
            raise ValueError(f"feature column {c!r} missing from units or train")
    empty = DecayTable("feature-distance", pd.DataFrame(columns=_DECAY_COLS["feature-distance"]), float("nan"), None, False)
    if len(d) == 0 or len(t) < 2:
        return empty
    cols = list(feature_cols)
    mu, sd = t[cols].mean(), t[cols].std(ddof=0).replace(0, 1.0)
    dist = np.sqrt((((d[cols] - mu) / sd) ** 2).mean(axis=1)).to_numpy()
    m = _masks(d, scope)
    sel = (m.fwd & ~m.replay).to_numpy() & np.isfinite(dist)
    if int(sel.sum()) < max(min_units, 2 * n_bins):
        return empty
    edges = np.unique(np.quantile(dist[sel], np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        bm = _bm(d, sel & (idx == b), n_boot, seed + 53 * (b + 1))
        rows.append({"bin": b, "lo_dist": float(edges[b]), "hi_dist": float(edges[b + 1]), "n": bm.n, "gain": bm.mean, "lo": bm.lo,
                     "hi": bm.hi, "enough": bm.n >= min_units})
    fr = pd.DataFrame(rows)
    g = d.loc[sel, "gain"]
    rho = float(spearmanr(dist[sel], g.to_numpy())[0]) if g.std() > 0 else float("nan")
    ok = fr[fr["enough"]]
    decays = bool(len(ok) >= 2 and math.isfinite(rho) and rho < -0.03 and ok["gain"].iloc[-1] < ok["gain"].iloc[0] and ok["lo"].iloc[0] > 0)
    return DecayTable("feature-distance", fr, rho, None, decays)


def time_decay(units: pd.DataFrame, scope: TrainingScope, now, *, bin_days: int = 91, seed: int = 0, n_boot: int = 400, min_units: int = 30,
               cluster_by: str = "month") -> DecayTable:
    """Gain against time since the learner stopped learning (`learned_through`). Half-life: exponential fit through the bins with
    a positive gain (at least three); None when the gain does not fall. A short half-life means the lesson is regime- or
    period-bound and must be re-validated on a schedule rather than trusted forever (section 14 time-aware memory)."""
    from scipy.stats import spearmanr
    d = prepare_units(units, now, cluster_by=cluster_by)
    scope.validate(now)
    kind = "time-since-learning"
    if len(d) == 0:
        return DecayTable(kind, pd.DataFrame(columns=_DECAY_COLS[kind]), float("nan"), None, False)
    age = (d["date"] - pd.Timestamp(scope.learned_through)).dt.days.to_numpy()
    m = _masks(d, scope)
    sel = (m.fwd & ~m.replay).to_numpy()
    rows = []
    for b in sorted(set((age[sel] // bin_days).tolist())):
        bm = _bm(d, sel & (age // bin_days == b), n_boot, seed + 71 * (int(b) + 1))
        rows.append({"bin": int(b), "days_from": int(b * bin_days), "days_to": int((b + 1) * bin_days), "n": bm.n, "gain": bm.mean,
                     "lo": bm.lo, "hi": bm.hi, "enough": bm.n >= min_units})
    fr = pd.DataFrame(rows, columns=_DECAY_COLS[kind])
    ok = fr[fr["enough"] & (fr["gain"] > 0)]
    half, decays = None, False
    if len(ok) >= 3:
        tmid = (ok["days_from"] + ok["days_to"]).to_numpy() / 2.0
        slope = np.polyfit(tmid, np.log(ok["gain"].to_numpy()), 1)[0]
        separated = ok["hi"].iloc[-1] < ok["lo"].iloc[0]           # the last bin's interval lies wholly below the first's
        if slope < 0 and separated:
            half, decays = float(math.log(2) / -slope), True
    g = d.loc[sel, "gain"]
    rho = float(spearmanr(age[sel], g.to_numpy())[0]) if sel.sum() > 3 and g.std() > 0 else float("nan")
    if decays and math.isfinite(rho) and rho > -0.02:
        half, decays = None, False                    # the bins fell but the unit-level trend does not: do not claim decay
    return DecayTable(kind, fr, rho, half, decays)


# ---------------------------------------------------------------------------------------------------------------
# walk-forward: a fresh lesson at every origin
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WalkForward:
    table: pd.DataFrame            # one row per origin: cut, n_train, n_test, gain
    pooled: TS.BootMean
    stability: TS.TransferStability
    rolling: TS.RollingStability
    p_signflip: float

    def gains(self) -> np.ndarray:
        return self.table["gain"].to_numpy(float)


def walk_forward_transfer(units: pd.DataFrame, fit: Fit, now, *, cuts: Sequence | None = None, horizon_days: int = 91, purge_days: int = 0,
                          min_train_units: int = 200, n_cuts: int = 8, cluster_by: str = "month", n_boot: int = 500, seed: int = 0) -> WalkForward:
    """Rolling-origin evaluation. At each cut the learner is retrained on units whose outcomes matured before cut - purge_days,
    then scored on the units decided in [cut, cut + horizon_days). Every test unit is later than everything the learner used, so
    each origin is a genuine unseen-future test; the series of origin gains is the transfer track record (stability, rolling
    persistence). `cuts` default to n_cuts evenly spaced decision dates after the first min_train_units matured units."""
    d = prepare_units(units.assign(learned=units["base"]) if "learned" not in units else units, now, cluster_by=cluster_by)
    cols = ["cut", "n_train", "n_test", "gain"]
    empty = WalkForward(pd.DataFrame(columns=cols), TS.cluster_bootstrap_mean([], None), TS.transfer_stability([]), TS.rolling_stability([]), float("nan"))
    if len(d) == 0:
        return empty
    if "alt" not in d:
        raise ValueError("walk_forward_transfer needs an 'alt' column")
    if cuts is None:
        dates = np.array(sorted(d["date"].unique()))
        first = d.sort_values("mature")["mature"].iloc[min(min_train_units, len(d) - 1)]
        pool = dates[dates > np.datetime64(first)]
        cuts = list(pd.to_datetime(pool[np.linspace(0, len(pool) - 1, n_cuts).astype(int)])) if len(pool) else []
    rows, gains, cls = [], [], []
    for c in sorted(set(pd.Timestamp(x) for x in cuts)):
        tr = np.flatnonzero((d["mature"] < c - pd.Timedelta(days=purge_days)).to_numpy())
        te = np.flatnonzero(((d["date"] >= c) & (d["date"] < c + pd.Timedelta(days=horizon_days))).to_numpy())
        if len(tr) < min_train_units or len(te) == 0:
            continue
        if d.iloc[tr]["mature"].max() >= d.iloc[te]["date"].min():
            raise FirewallBreach(f"walk-forward cut {c.date()}: a training outcome matured at/after the first test decision")
        g = _apply(d, fit(d.iloc[tr].copy()), te)
        rows.append({"cut": c, "n_train": int(len(tr)), "n_test": int(len(te)), "gain": float(g.mean())})
        gains.append(g)
        cls.append(d.iloc[te]["cluster"].to_numpy())
    if not rows:
        return empty
    tab = pd.DataFrame(rows, columns=cols)
    allg, allc = np.concatenate(gains), np.concatenate(cls)
    return WalkForward(tab, TS.cluster_bootstrap_mean(allg, allc, n_boot=n_boot, seed=seed), TS.transfer_stability(tab["gain"]),
                       TS.rolling_stability(tab["gain"], window=min(4, max(1, len(tab)))), TS.cluster_signflip_p(allg, allc, seed=seed))


# ---------------------------------------------------------------------------------------------------------------
# per-era and per-item breakdowns, report comparison, persistence
# ---------------------------------------------------------------------------------------------------------------
def evaluate_by_era(units: pd.DataFrame, scope: TrainingScope, now, **kw) -> dict:
    """evaluate_transfer separately per cost/regime era of the unit's date (blind_gates.era_of, reused). An era with too few units
    is still reported, as INSUFFICIENT_EVIDENCE, never dropped."""
    d = prepare_units(units, now, cluster_by=kw.get("cluster_by", "month"))
    era = d["date"].map(BG.era_of) if len(d) else pd.Series([], dtype=str)
    return {e: evaluate_transfer(d[era == e], scope, now, **kw) for e in sorted(set(era))}


def item_transfer_table(units_by_item: Mapping[str, pd.DataFrame], scopes: Mapping[str, TrainingScope], now, *, seed: int = 0, n_boot: int = 300,
                        min_units: int = 30, method: str = "bh") -> pd.DataFrame:
    """One row per knowledge item: its unseen-year gain, interval, cluster sign-flip p-value, multiple-testing adjusted q across
    ALL items tested (a hundred items will show a few 'transferring' by luck), verdict and Confidence.transfer. Items sorted by id."""
    rows = []
    for kid in sorted(units_by_item):
        if kid not in scopes:
            raise KeyError(f"no training scope for item {kid}")
        d = prepare_units(units_by_item[kid], now)
        rep = evaluate_transfer(d, scopes[kid], now, axes=(Axis.YEAR,), seed=seed, n_boot=n_boot, min_units=min_units, ratio_boot=False)
        r = rep.axes.get(Axis.YEAR)
        if r is None or not r.tested:
            rows.append({"item": kid, "n": 0, "gain": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"),
                         "verdict": "INSUFFICIENT_EVIDENCE", "confidence_transfer": None})
            continue
        sel = _select(_masks(d, scopes[kid]), Axis.YEAR, "marginal")[2]
        rows.append({"item": kid, "n": r.cross.n, "gain": r.cross.mean, "lo": r.cross.lo, "hi": r.cross.hi,
                     "p": TS.cluster_signflip_p(d.loc[sel, "gain"], d.loc[sel, "cluster"], seed=seed), "verdict": r.verdict.label.value,
                     "confidence_transfer": transfer_confidence(rep)})
    fr = pd.DataFrame(rows, columns=["item", "n", "gain", "lo", "hi", "p", "verdict", "confidence_transfer"])
    fr["q"] = TS.adjust_many(fr["p"].to_numpy(float), method) if len(fr) else []
    return fr


def compare_reports(old: TransferReport, new: TransferReport) -> dict:
    """Per axis: did the unseen-context gain separate (non-overlapping intervals) upward or downward between two learner versions?
    Untested axes on either side are listed, not compared."""
    out: dict[str, list[str]] = {"better": [], "worse": [], "same": [], "untested": []}
    for ax in sorted(set(old.axes) | set(new.axes), key=lambda a: a.value):
        a, b = old.axes.get(ax), new.axes.get(ax)
        if a is None or b is None or not a.tested or not b.tested or not (math.isfinite(a.cross.lo) and math.isfinite(b.cross.lo)):
            out["untested"].append(ax.value)
        elif b.cross.lo > a.cross.hi:
            out["better"].append(ax.value)
        elif b.cross.hi < a.cross.lo:
            out["worse"].append(ax.value)
        else:
            out["same"].append(ax.value)
    return out


def json_load(path) -> dict:
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_report(report: TransferReport, out_dir, *, cfg=None, seed=None, name: str | None = None) -> dict:
    """Write <name>.json (record + provenance stamp) and <name>.txt (render) atomically (temp file then replace) and return the
    paths. Never overwrites a different record under the same name: a report id is content-addressed and part of the default name."""
    from .. import provenance
    rec = report.as_record()
    rec["provenance"] = provenance.stamp(cfg, seed)
    base = name or f"transfer_{report.now.isoformat()}_{rec['report_id']}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / f"{base}.json", "txt": out / f"{base}.txt"}
    if paths["json"].exists() and json_load(paths["json"]).get("report_id") != rec["report_id"]:
        raise FileExistsError(f"{paths['json']} holds a different report; choose another name")
    for key, text in (("json", canonical_json(rec)), ("txt", render_report(report))):
        tmp = paths[key].with_suffix(paths[key].suffix + ".tmp")
        tmp.write_bytes(text.encode("utf-8"))                # bytes: no CRLF rewrite on Windows
        os.replace(tmp, paths[key])
    return {k: str(v) for k, v in paths.items()}


# ---------------------------------------------------------------------------------------------------------------
# identity scrambling for the unit-table harness (full windows use learning_delta.make_presentation)
# ---------------------------------------------------------------------------------------------------------------
def disguised_presentation(window, seed: int):
    """The audited whole-window disguise: learning_delta.make_presentation with a random (not order-preserving) relabel and a fresh
    whole-week date shift. A thin pointer so this package does not grow a second disguise."""
    return LD.make_presentation(window, seed, order_preserving=False)


def scrambled_frame(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Unit-table analogue of the disguise: tickers permuted among themselves, dates moved by one common whole-week shift; every
    feature and outcome untouched. A learner that reads only features is unaffected; one that keys on (ticker, date) is blind."""
    rng = np.random.default_rng(seed)
    tick = np.array(sorted(df["ticker"].astype(str).unique()))
    perm = dict(zip(tick, rng.permutation(tick)))
    shift = pd.Timedelta(weeks=int(rng.integers(-300, 300)) or 1)
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).map(perm)
    out["date"] = pd.to_datetime(out["date"]) + shift
    if "mature" in out:
        out["mature"] = pd.to_datetime(out["mature"]) + shift
    return out


def identity_probe_units(units: pd.DataFrame, fit: Fit, fold: Fold, now, *, seed: int = 0, on: str = "test") -> pd.DataFrame:
    """Units with learned (original identities) and learned_disguised (identities scrambled) so evaluate_transfer / identity_gap can
    report the identity gap. Training is done once on the fold's training units. on='test' scores the held-out units (a memoriser
    has nothing to recall there, so its gap is ~0 by construction); on='train' scores the situations the learner trained on,
    which is where identity recall shows: original identities replay the answer, scrambled ones do not."""
    if on not in ("test", "train"):
        raise ValueError("on must be 'test' or 'train'")
    d = prepare_units(units.assign(learned=units["base"]) if "learned" not in units else units, now)
    predict = fit(d.iloc[fold.train_idx].copy())
    te = d.iloc[fold.test_idx if on == "test" else fold.train_idx].copy()
    delta = te["alt"].to_numpy(float) - te["base"].to_numpy(float)
    w0 = np.asarray(predict(te), float)
    w1 = np.asarray(predict(scrambled_frame(te, seed)), float)
    te["learned"] = te["base"] + np.clip(w0, 0, 1) * delta
    te["learned_disguised"] = te["base"] + np.clip(w1, 0, 1) * delta
    return te


# ---------------------------------------------------------------------------------------------------------------
# a declarative protocol, so a transfer test is reproducible and versioned
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TransferProtocol:
    """Everything that shapes a transfer evaluation except the data. Frozen and hashed: two reports are comparable only if their
    protocol fingerprints match, and a threshold cannot be quietly loosened between learner versions."""
    axes: tuple = ALL_AXES
    mode: str = "marginal"
    cluster_by: str = "month"
    n_boot: int = 600
    min_units: int = 30
    min_group_units: int = 10
    n_stock_groups: int = 8
    ratio_boot: bool = True
    seed: int = 0

    def validate(self) -> list[str]:
        errs = []
        if self.mode not in ("marginal", "isolated"):
            errs.append(f"mode {self.mode!r}")
        if self.cluster_by not in CLUSTER_FORMATS:
            errs.append(f"cluster_by {self.cluster_by!r}")
        if not self.axes or any(not isinstance(a, Axis) for a in self.axes):
            errs.append("axes must be a non-empty tuple of Axis")
        if self.n_boot < 100:
            errs.append("n_boot < 100 gives unusable intervals")
        if self.min_units < 5 or self.min_group_units < 2 or self.n_stock_groups < 2:
            errs.append("minimum sizes too small to test anything")
        return errs

    def fingerprint(self) -> str:
        return stable_hash({**dataclasses.asdict(self), "axes": [a.value for a in self.axes]})

    def run(self, units: pd.DataFrame, scope: TrainingScope, now) -> TransferReport:
        errs = self.validate()
        if errs:
            raise ValueError("invalid protocol: " + "; ".join(errs))
        return evaluate_transfer(units, scope, now, axes=self.axes, mode=self.mode, cluster_by=self.cluster_by, n_boot=self.n_boot, seed=self.seed,
                                 min_units=self.min_units, min_group_units=self.min_group_units, n_stock_groups=self.n_stock_groups, ratio_boot=self.ratio_boot)


# ---------------------------------------------------------------------------------------------------------------
# the transfer matrix: which contexts teach which
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TransferMatrix:
    axis: Axis
    gains: pd.DataFrame            # rows = context trained on, columns = context tested on
    counts: pd.DataFrame
    forward: pd.DataFrame          # True where the test context lies wholly after the training context (a legitimate future test)
    diagonal_note: str

    def summary(self) -> dict:
        """Diagonal (same context, out of time) against off-diagonal (other context) and the asymmetry of pairs."""
        g = self.gains
        labs = list(g.index)
        diag = np.array([g.loc[a, a] for a in labs if a in g.columns], float)
        off = [(a, b, g.loc[a, b]) for a in labs for b in g.columns if a != b and np.isfinite(g.loc[a, b])]
        offv = np.array([v for _, _, v in off], float)
        fwd = np.array([g.loc[a, b] for a in labs for b in g.columns if a != b and bool(self.forward.loc[a, b]) and np.isfinite(g.loc[a, b])], float)
        asym = [abs(g.loc[a, b] - g.loc[b, a]) for a in labs for b in labs if a < b and np.isfinite(g.loc[a, b]) and np.isfinite(g.loc[b, a])]
        worst = min(off, key=lambda t: t[2]) if off else None
        dm, om = (float(np.nanmean(diag)) if len(diag) and np.isfinite(diag).any() else float("nan")), (float(offv.mean()) if len(offv) else float("nan"))
        return {"diagonal_mean": dm, "offdiagonal_mean": om, "forward_offdiagonal_mean": float(fwd.mean()) if len(fwd) else float("nan"),
                "ratio": TS.transfer_ratio(om, dm) if math.isfinite(dm) and math.isfinite(om) else None,
                "mean_asymmetry": float(np.mean(asym)) if asym else float("nan"),
                "worst_pair": None if worst is None else {"trained_on": worst[0], "tested_on": worst[1], "gain": float(worst[2])},
                "share_offdiagonal_positive": float((offv > 0).mean()) if len(offv) else float("nan")}


def transfer_matrix(units: pd.DataFrame, fit: Fit, now, axis: Axis = Axis.YEAR, *, min_units: int = 30, max_labels: int = 12,
                    cluster_by: str = "month") -> TransferMatrix:
    """Train on ONE context at a time and test on every context. The diagonal is same-context but out of time (train on the earlier
    half of the context's units, test on the later half); off-diagonal cells train on the whole of context i and test on the
    whole of context j. For YEAR, cell (i, j) is a legitimate future test only when j > i and i's outcomes matured before j
    began (marked in `forward`; others are reported but flagged). Cells with fewer than min_units test units are NaN."""
    d = prepare_units(units.assign(learned=units["base"]) if "learned" not in units else units, now, cluster_by=cluster_by)
    if "alt" not in d:
        raise ValueError("transfer_matrix needs an 'alt' column")
    col = AXIS_COL[axis]
    labs = sorted(x for x in d[col].astype(str).unique() if x != UNKNOWN_LABEL)[:max_labels]
    g = pd.DataFrame(np.nan, index=labs, columns=labs)
    n = pd.DataFrame(0, index=labs, columns=labs)
    fw = pd.DataFrame(False, index=labs, columns=labs)
    for a in labs:
        ia = np.flatnonzero((d[col].astype(str) == a).to_numpy())
        if len(ia) < 2 * min_units:
            continue
        ia = ia[np.argsort(d.iloc[ia]["date"].to_numpy(), kind="mergesort")]
        half = len(ia) // 2
        model_full = fit(d.iloc[ia].copy())
        model_half = fit(d.iloc[ia[:half]].copy())
        for b in labs:
            if a == b:
                te = ia[half:]
                te = te[(d.iloc[te]["date"] > d.iloc[ia[:half]]["mature"].max()).to_numpy()]      # purge: nothing the model trained on overlaps
                if len(te) >= min_units:
                    g.loc[a, b] = float(_apply(d, model_half, te).mean())
                    n.loc[a, b] = len(te)
                    fw.loc[a, b] = True
                continue
            ib = np.flatnonzero((d[col].astype(str) == b).to_numpy())
            if len(ib) < min_units:
                continue
            g.loc[a, b] = float(_apply(d, model_full, ib).mean())
            n.loc[a, b] = len(ib)
            fw.loc[a, b] = bool(axis == Axis.YEAR and d.iloc[ia]["mature"].max() < d.iloc[ib]["date"].min())
    note = "diagonal: trained on the earlier half of the context, tested on the later half (same context, out of time)"
    return TransferMatrix(axis, g, n, fw, note)


# ---------------------------------------------------------------------------------------------------------------
# is the drop from same-context to cross-context bigger than chance?
# ---------------------------------------------------------------------------------------------------------------
def specialisation_permutation_test(units: pd.DataFrame, scope: TrainingScope, now, axis: Axis = Axis.YEAR, *, n_perm: int = 2000, seed: int = 0,
                                    cluster_by: str = "month", min_clusters: int = 4) -> dict:
    """Permutation test of 'the lesson works better in familiar contexts than in novel ones'. Statistic = mean gain on forward
    familiar units minus mean gain on forward novel units. Under the null the familiar/novel label carries no information, so whole
    clusters (months) are reassigned between the two groups at random, keeping group sizes. Small p = real specialisation.
    Untestable (returns p=NaN and a reason) with fewer than min_clusters clusters on either side."""
    d = prepare_units(units, now, cluster_by=cluster_by)
    scope.validate(now)
    if len(d) == 0:
        return {"stat": float("nan"), "p": float("nan"), "n_perm": 0, "reason": "no units"}
    fam_fwd, _, nov_fwd, _ = _select(_masks(d, scope), axis, "marginal")
    sel = fam_fwd | nov_fwd
    sub = d.loc[sel, ["cluster", "gain"]].assign(fam=fam_fwd[sel])
    cl = sub.groupby("cluster").agg(s=("gain", "sum"), n=("gain", "size"), fam=("fam", "mean"))
    is_fam = (cl["fam"] > 0.5).to_numpy()
    if is_fam.sum() < min_clusters or (~is_fam).sum() < min_clusters:
        return {"stat": float("nan"), "p": float("nan"), "n_perm": 0, "reason": f"fewer than {min_clusters} clusters on one side"}
    s, n = cl["s"].to_numpy(), cl["n"].to_numpy(float)
    stat_of = lambda mask: float(s[mask].sum() / n[mask].sum() - s[~mask].sum() / n[~mask].sum())
    obs = stat_of(is_fam)
    rng = np.random.default_rng(seed)
    k = int(is_fam.sum())
    hits = 0
    for _ in range(n_perm):
        m = np.zeros(len(s), bool)
        m[rng.choice(len(s), k, replace=False)] = True
        hits += stat_of(m) >= obs - 1e-15
    return {"stat": obs, "p": float((hits + 1) / (n_perm + 1)), "n_perm": int(n_perm), "reason": ""}


# ---------------------------------------------------------------------------------------------------------------
# transfer track record across learner versions
# ---------------------------------------------------------------------------------------------------------------
class TransferHistory:
    """Append-only record of transfer reports per learner version, in time order. Detects a newer version that transfers worse than
    an older one (a regression) and refuses out-of-order or duplicate versions or a changed protocol without saying so."""

    def __init__(self):
        self._rows: list[tuple] = []          # (version, now, protocol fingerprint, report)

    def __len__(self):
        return len(self._rows)

    def add(self, version: str, report: TransferReport, protocol: TransferProtocol | None = None) -> None:
        if any(v == version for v, *_ in self._rows):
            raise ValueError(f"version {version} already recorded: history is never overwritten")
        if self._rows and report.now < self._rows[-1][1]:
            raise FirewallBreach(f"report dated {report.now} precedes the previous one ({self._rows[-1][1]}): history must be in time order")
        self._rows.append((version, report.now, protocol.fingerprint() if protocol else "", report))

    def trajectory(self, axis: Axis) -> list[tuple]:
        """[(version, cross-context gain or NaN)] for one axis."""
        return [(v, r.gain(axis)) for v, _, _, r in self._rows]

    def protocol_changes(self) -> list[str]:
        out = []
        for (v0, _, p0, _), (v1, _, p1, _) in zip(self._rows, self._rows[1:]):
            if p0 and p1 and p0 != p1:
                out.append(f"{v0} -> {v1}: protocol changed ({p0} -> {p1}); their gains are not comparable")
        return out

    def regressions(self) -> list[str]:
        """Consecutive versions where the newer one's unseen-context gain on some axis is significantly lower than the older one's."""
        out = []
        for (v0, _, _, r0), (v1, _, _, r1) in zip(self._rows, self._rows[1:]):
            for ax in r1.axes:
                cmp_ = compare_reports(r0, r1)
                if ax.value in cmp_["worse"]:
                    out.append(f"{v1} transfers worse than {v0} on {ax.value}")
        return sorted(set(out))


# ---------------------------------------------------------------------------------------------------------------
# tables for reports
# ---------------------------------------------------------------------------------------------------------------
def axis_table(report: TransferReport) -> pd.DataFrame:
    """One row per axis, in the order asked for: same-context and cross-context gain, ratio (NaN when undefined), specialisation
    flag, stability score, verdict. Untested axes appear with NaN gains."""
    rows = []
    for ax, r in report.axes.items():
        rows.append({"axis": ax.value, "same": r.same.mean, "cross": r.cross.mean if r.tested else float("nan"),
                     "ratio": r.ratio.value if r.ratio.value is not None else float("nan"), "specialisation": r.specialisation.flag,
                     "stability": r.stability.score, "label": r.verdict.label.value})
    return TS.verdict_table(rows)


def group_table(report: TransferReport, axis: Axis) -> pd.DataFrame:
    """Gain in each novel group of one axis (year, regime, sector, ticker group ...) with its unit count, worst group first."""
    r = report.axes.get(axis)
    rows = [{"group": g[0], "n": g[1], "gain": g[2]} for g in (r.groups if r else ())]
    return pd.DataFrame(rows, columns=["group", "n", "gain"]).sort_values("gain", kind="mergesort").reset_index(drop=True)


def planning_note(report: TransferReport, axis: Axis = Axis.YEAR, target_gain: float = 0.005) -> str:
    """What it would take to settle an inconclusive axis: the minimum detectable gain of the test as run, and the number of monthly
    clusters needed to see `target_gain`. Empty string when the axis was not tested or its interval is unbounded."""
    r = report.axes.get(axis)
    if r is None or not r.tested or r.cross.n_clusters < 3:
        return ""
    se = (r.cross.hi - r.cross.lo) / 3.92 if math.isfinite(r.cross.lo) and math.isfinite(r.cross.hi) else float("nan")
    if not math.isfinite(se) or se <= 0:
        return ""
    sd = se * math.sqrt(r.cross.n_clusters)
    return (f"{axis.value}: {r.cross.n_clusters} clusters give a detectable gain of about {2.487 * se:.4f}; "
            f"seeing {target_gain:.4f} would take about {TS.clusters_needed(target_gain, sd)} clusters")


# ---------------------------------------------------------------------------------------------------------------
# section 26 completion: gains per context with intervals, market conditions, per-rule ledgers
# ---------------------------------------------------------------------------------------------------------------
CONTEXT_GAIN_COLS = ["axis", "group", "familiar", "forward", "n", "gain", "lo", "hi", "p_signflip"]


def context_gain_table(units: pd.DataFrame, scope: TrainingScope, now, axis: Axis, *, cluster_by: str = "month", n_boot: int = 400, seed: int = 0,
                       min_units: int = 30, n_stock_groups: int = 8) -> pd.DataFrame:
    """Gain with a cluster-bootstrap interval and a sign-flip p-value for EVERY context group on one axis (each regime, sector,
    volatility bucket, stock type, year, ticker group), not only for their pooled average. `familiar` says whether the group was in
    training; `forward` says the row is restricted to units decided after learned_through (the only rows that can prove transfer).
    Groups below min_units are kept with NaN gain so a thin context is visible rather than absent."""
    d = prepare_units(units, now, cluster_by=cluster_by)
    scope.validate(now)
    rows = []
    if len(d):
        m = _masks(d, scope)
        use = (m.fwd & ~m.replay).to_numpy() & m.known[axis].to_numpy()
        grp = _group_labels(d, axis, use, n_stock_groups)
        for g, sub in d.loc[use].groupby(grp.to_numpy()):
            fam = bool(m.fam[axis].loc[sub.index].all()) if axis != Axis.STOCK else bool(m.fam[axis].loc[sub.index].mean() > 0.5)
            if len(sub) < min_units:
                rows.append({"axis": axis.value, "group": str(g), "familiar": fam, "forward": True, "n": len(sub), "gain": float("nan"),
                             "lo": float("nan"), "hi": float("nan"), "p_signflip": float("nan")})
                continue
            bm = TS.cluster_bootstrap_mean(sub["gain"], sub["cluster"], n_boot=n_boot, seed=seed + int(stable_hash(str(g), 4), 16) % 997)
            rows.append({"axis": axis.value, "group": str(g), "familiar": fam, "forward": True, "n": bm.n, "gain": bm.mean, "lo": bm.lo, "hi": bm.hi,
                         "p_signflip": TS.cluster_signflip_p(sub["gain"], sub["cluster"], seed=seed)})
    return pd.DataFrame(rows, columns=CONTEXT_GAIN_COLS)


def market_condition_gains(units: pd.DataFrame, train: pd.DataFrame, scope: TrainingScope, now, cond_cols: Sequence[str], *, n_bins: int = 3,
                           cluster_by: str = "month", n_boot: int = 400, seed: int = 0, min_units: int = 30) -> pd.DataFrame:
    """Cross-market-condition gains: for each numeric market-condition column (trailing market return, index volatility, breadth ...)
    the forward units are cut at the TRAINING distribution's quantiles and the gain is reported per band with an interval. A band
    the training data never reached (below its minimum or above its maximum) is marked outside_training - the strictest novelty.
    Bands are decided from `train` only, so the test data cannot move the cut points."""
    d = prepare_units(units, now, cluster_by=cluster_by)
    t = prepare_units(train, now, cluster_by=cluster_by)
    scope.validate(now)
    cols = ["condition", "band", "outside_training", "n", "gain", "lo", "hi"]
    rows = []
    if len(d) and len(t):
        m = _masks(d, scope)
        use = (m.fwd & ~m.replay).to_numpy()
        for c in cond_cols:
            if c not in d or c not in t:
                raise ValueError(f"condition column {c!r} missing")
            edges = np.quantile(t[c].to_numpy(float), np.linspace(0, 1, n_bins + 1))
            lo_t, hi_t = edges[0], edges[-1]
            band = np.clip(np.searchsorted(edges[1:-1], d[c].to_numpy(float), side="right"), 0, n_bins - 1)
            outside = (d[c].to_numpy(float) < lo_t) | (d[c].to_numpy(float) > hi_t)
            for b in range(n_bins):
                for out in (False, True):
                    sel = use & (band == b) & (outside == out)
                    if not sel.any():
                        continue
                    bm = _bm(d, sel, n_boot, seed + 13 * b + int(out))
                    rows.append({"condition": c, "band": b, "outside_training": out, "n": bm.n, "gain": bm.mean if bm.n >= min_units else float("nan"),
                                 "lo": bm.lo if bm.n >= min_units else float("nan"), "hi": bm.hi if bm.n >= min_units else float("nan")})
    return pd.DataFrame(rows, columns=cols)


REQUIRED_TRANSFER_AXES = ("YEAR", "REGIME", "STOCK", "SECTOR", "VOLATILITY", "MARKET_CONDITION")


class RuleTransferLedger:
    """Per-rule record of where a learned rule has been tested for transfer. Section 26: every meaningful learned rule must eventually
    be tested across years, regimes, market conditions, sectors, stock types and volatility states. The ledger is append-only and
    time-ordered; a rule's status is derived from its tests, never set by hand:
      UNTESTED   no test on any required axis
      PARTIAL    some required axes tested, some outstanding
      COMPLETE   every required axis tested and none failed
      FAILED     a tested axis showed the rule harmful or over-specialised there"""

    def __init__(self, required: Sequence[str] = REQUIRED_TRANSFER_AXES):
        self.required = tuple(required)
        self._tests: dict[str, list[dict]] = {}

    def rules(self) -> list[str]:
        return sorted(self._tests)

    def register(self, rule_id: str) -> None:
        self._tests.setdefault(rule_id, [])

    def record(self, rule_id: str, axis: str, tested_on, gain: TS.BootMean, verdict: TS.TransferVerdictLabel, *, n_groups: int = 0) -> None:
        if axis not in self.required:
            raise ValueError(f"axis {axis!r} is not one of the required axes {self.required}")
        rows = self._tests.setdefault(rule_id, [])
        when = as_date(tested_on)
        if rows and when < rows[-1]["when"]:
            raise FirewallBreach(f"test dated {when} precedes the rule's previous test ({rows[-1]['when']}): the ledger is time ordered")
        rows.append({"axis": axis, "when": when, "gain": gain.mean, "lo": gain.lo, "hi": gain.hi, "n": gain.n, "verdict": verdict.value, "n_groups": n_groups})

    def record_report(self, rule_id: str, report: "TransferReport", tested_on=None) -> int:
        """Log every tested axis of a TransferReport for the rule; returns how many axes were logged."""
        n = 0
        for ax, r in report.axes.items():
            if r.tested and ax.value in self.required:
                self.record(rule_id, ax.value, tested_on or report.now, r.cross, r.verdict.label, n_groups=r.stability.n_groups)
                n += 1
        if not n:
            self.register(rule_id)
        return n

    def latest(self, rule_id: str) -> dict:
        """Most recent test per axis."""
        out = {}
        for t in self._tests.get(rule_id, []):
            out[t["axis"]] = t
        return out

    def outstanding(self, rule_id: str) -> list[str]:
        done = self.latest(rule_id)
        return [a for a in self.required if a not in done]

    def status(self, rule_id: str) -> str:
        done = self.latest(rule_id)
        bad = (TS.TransferVerdictLabel.HARMFUL.value, TS.TransferVerdictLabel.OVER_SPECIALISED.value, TS.TransferVerdictLabel.IDENTITY_DEPENDENT.value)
        if any(t["verdict"] in bad for t in done.values()):
            return "FAILED"
        if not done:
            return "UNTESTED"
        return "COMPLETE" if not self.outstanding(rule_id) else "PARTIAL"

    def overdue(self, now, max_age_days: int = 365) -> list[str]:
        """Rules whose most recent test on some axis is older than max_age_days, or which still have outstanding axes."""
        out = []
        for r in self.rules():
            latest = self.latest(r)
            stale = any((as_date(now) - t["when"]).days > max_age_days for t in latest.values())
            if stale or self.outstanding(r):
                out.append(r)
        return out

    def summary(self) -> pd.DataFrame:
        rows = [{"rule": r, "status": self.status(r), "tested": len(self.latest(r)), "outstanding": ",".join(self.outstanding(r))} for r in self.rules()]
        return pd.DataFrame(rows, columns=["rule", "status", "tested", "outstanding"])


def full_context_report(units: pd.DataFrame, scope: TrainingScope, now, *, axes: Sequence[Axis] = ALL_AXES, train: pd.DataFrame | None = None,
                        cond_cols: Sequence[str] = (), n_boot: int = 300, seed: int = 0, min_units: int = 30, cluster_by: str = "month") -> dict:
    """Section 26 in one call: the gain in every context group of every axis (with intervals) and, when `train` and `cond_cols` are
    given, per market-condition band. Also a coverage summary per axis: how many contexts were tested, how many gained significantly,
    lost significantly, or were too thin, and an exact binomial sign test of 'more contexts gained than lost'."""
    from scipy.stats import binomtest
    tables = {ax.value: context_gain_table(units, scope, now, ax, cluster_by=cluster_by, n_boot=n_boot, seed=seed, min_units=min_units) for ax in axes}
    if train is not None and len(cond_cols):
        tables["MARKET_CONDITION"] = market_condition_gains(units, train, scope, now, cond_cols, n_boot=n_boot, seed=seed, min_units=min_units,
                                                             cluster_by=cluster_by)
    cover = {}
    for name, tb in tables.items():
        if name == "MARKET_CONDITION":
            tb = tb.rename(columns={"band": "group"})
        ok = tb[np.isfinite(tb["gain"])] if len(tb) else tb
        up, down = int((ok["lo"] > 0).sum()), int((ok["hi"] < 0).sum())
        decided = up + down
        p = float(binomtest(up, decided, 0.5, alternative="greater").pvalue) if decided else float("nan")
        cover[name] = {"contexts": int(len(tb)), "thin": int(len(tb) - len(ok)), "gained": up, "lost": down, "undecided": int(len(ok) - decided), "p_more_gain_than_loss": p}
    return {"tables": tables, "coverage": cover}


def ledger_records(ledger: RuleTransferLedger) -> list[dict]:
    """Serialisable copy of every test in a ledger (dates as ISO strings)."""
    return [{"rule": r, **{**t, "when": t["when"].isoformat()}} for r in ledger.rules() for t in ledger._tests[r]]


def ledger_from_records(records: Sequence[Mapping], required: Sequence[str] = REQUIRED_TRANSFER_AXES) -> RuleTransferLedger:
    """Rebuild a ledger from stored records, re-applying every check: an unknown axis or an out-of-order test is refused."""
    led = RuleTransferLedger(required)
    for r in records:
        g = TS.BootMean(r["gain"], r["lo"], r["hi"], r["n"], max(2, r.get("n_groups", 0)))
        led.record(r["rule"], r["axis"], r["when"], g, TS.TransferVerdictLabel(r["verdict"]), n_groups=r.get("n_groups", 0))
    return led


TRACKED_AXES = {"cross_year_gain": Axis.YEAR, "cross_regime_gain": Axis.REGIME, "cross_stock_gain": Axis.STOCK, "cross_sector_gain": Axis.SECTOR,
                "cross_volatility_gain": Axis.VOLATILITY, "cross_stock_type_gain": Axis.STOCK_TYPE}


def tracked_gains_from_folds(units: pd.DataFrame, fit: Fit, now, *, n_boot: int = 400, seed: int = 0, min_units: int = 30) -> dict:
    """The section-26 tracked quantities for a RE-TRAINABLE learner, each with an interval: same_year_gain (the trained model replayed
    on its own years) and cross_year / cross_regime / cross_stock / cross_sector / cross_volatility / cross_stock_type gain from
    hold-one-context-out folds (forward-purged for years). An axis with no valid folds (one label only, too little data) is reported
    as None: untested, not zero. Also returns the ratio for each tested axis, with the bootstrap over-specialisation verdict."""
    d = prepare_units(units.assign(learned=units["base"]) if "learned" not in units else units, now)
    out: dict = {"same_year_gain": None}
    for name, ax in TRACKED_AXES.items():
        folds = make_folds(d, ax, seed=seed)
        out[name] = None
        if not folds:
            continue
        fr = run_folds(units, fit, folds, now)
        res = axis_result_from_folds(ax, fr, n_boot=n_boot, seed=seed, min_units=min_units)
        te = np.concatenate([r.gain_test for r in fr])
        tc = np.concatenate([r.cluster_test for r in fr])
        rp = np.concatenate([r.gain_replay for r in fr])
        rc = np.concatenate([r.cluster_replay for r in fr])
        spec = TS.over_specialisation_verdict(te, rp, tc, rc, n_boot=n_boot, seed=seed)
        out[name] = {"gain": res.cross, "same": res.same, "ratio": res.ratio, "verdict": res.verdict.label.value, "specialisation": spec}
        if ax == Axis.YEAR:
            out["same_year_gain"] = res.same
    return out


def record_tracked(ledger: RuleTransferLedger, rule_id: str, tracked: Mapping, tested_on) -> int:
    """Write the axes of a tracked_gains_from_folds result into a rule's ledger (year, regime, stock, sector, volatility). Axes that were
    not testable (None) are skipped, so they stay outstanding. Returns the number of axes logged."""
    to_axis = {"cross_year_gain": "YEAR", "cross_regime_gain": "REGIME", "cross_stock_gain": "STOCK", "cross_sector_gain": "SECTOR", "cross_volatility_gain": "VOLATILITY"}
    ledger.register(rule_id)
    n = 0
    for key, axis in to_axis.items():
        t = tracked.get(key)
        if t is None or axis not in ledger.required:
            continue
        ledger.record(rule_id, axis, tested_on, t["gain"], TS.TransferVerdictLabel(t["verdict"]), n_groups=t["gain"].n_clusters)
        n += 1
    return n


def record_market_conditions(ledger: RuleTransferLedger, rule_id: str, table: pd.DataFrame, tested_on) -> bool:
    """Log the MARKET_CONDITION axis from a market_condition_gains table: the rule counts as tested there once at least two bands have a
    measured gain; the pooled gain over measured bands (unit-weighted) is stored with a conservative verdict (HARMFUL if any measured band
    is significantly negative, else GENERALISES if every measured band gained, else NO_LEARNING)."""
    ok = table[np.isfinite(table["gain"])] if len(table) else table
    if len(ok) < 2:
        return False
    w = ok["n"].to_numpy(float)
    pooled = TS.BootMean(float((ok["gain"] * w).sum() / w.sum()), float(ok["lo"].min()), float(ok["hi"].max()), int(w.sum()), int(len(ok)))
    if (ok["hi"] < 0).any():
        v = TS.TransferVerdictLabel.HARMFUL
    elif (ok["lo"] > 0).all():
        v = TS.TransferVerdictLabel.GENERALISES
    else:
        v = TS.TransferVerdictLabel.NO_LEARNING
    ledger.record(rule_id, "MARKET_CONDITION", tested_on, pooled, v, n_groups=len(ok))
    return True
