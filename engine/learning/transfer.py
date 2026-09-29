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
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import (Confidence, FirewallBreach, KnowledgeLike, ValidationLabel, _StrEnum, as_date, clip01, require_past,
                   stable_hash)
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
    pct = sd.rank(axis=1, pct=True, method="average")
    b = np.minimum((pct.to_numpy() * n_buckets).astype(float), n_buckets - 1e-9)
    lab = pd.DataFrame(np.where(np.isfinite(b), np.floor(b) + 1, np.nan), index=sd.index, columns=sd.columns)
    s = lab.stack()
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
    bag = {c: set() for c in ("year", "regime", "sector", "vol_bucket", "stock_type", "ticker")}
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
