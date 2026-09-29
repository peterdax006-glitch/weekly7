"""Predictive value vs portfolio value (contract C62 section 34; checklist J12, L06-L07 support).

STATUS: IMPLEMENTED - NOT VALIDATED (unit-tested on planted synthetic panels only).

A statistically predictive signal may not improve the portfolio; a better portfolio may owe nothing to prediction. So the eight
kinds of value are measured SEPARATELY and never blended into one number:

    predictive effect     rank IC of the expected-return score against the realised forward return, plus the IC of what is new
                          once the baseline score is removed (incremental IC)
    movement prediction   IC of the movement score against |forward return| and AUC for 'moves at least mover_thr'
    ranking value         NDCG@k of the ordering against realised in-band moves (the tier-1 target)
    selection value       the picks actually made: mover / in-band hit rate against the universe and against random k-of-n
    direction value       accuracy and Brier skill of P(up) on the movers (tier 3; reported, never allowed to dominate)
    timing value          Cov(exposure, portfolio return): does the learner deploy more when the picks pay?
    risk value            tier-2 risk of the simulated weekly portfolio (worst-5% week, drawdown, catastrophic weeks, overshoot)
    portfolio value       the existing tiered objective (engine/objective.py): lexicographic, with gaming flags

Each component is base vs new, with a cluster-bootstrap interval on the paired difference. The existing tiered objective stays
authoritative: direction is tier 3 and cannot buy back damage to tier 1 or 2, and a translation diagnosis says when a
prediction gain did not reach the portfolio (or a portfolio gain has no prediction behind it).

Panel convention: DataFrame indexed by MultiIndex (date, ticker); `fwd` is the return over the holding period that starts at the
NEXT session's open after the decision at `date`'s close; `mature` is the date `fwd` is known. Dates must be spaced at least one
holding period apart so weekly portfolio returns do not overlap (use subsample_dates)."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .. import objective as O
from .core import FirewallBreach, ValidationLabel, _StrEnum, as_date, stable_hash
from . import transfer_score as TS

STATUS_MEASURED, STATUS_UNTESTED, STATUS_INSUFFICIENT = "MEASURED", "UNTESTED", "INSUFFICIENT"
COMPONENTS = ("predictive_effect", "movement_prediction", "ranking_value", "selection_value", "direction_value", "timing_value",
              "risk_value", "portfolio_value")
TIER_OF = {"predictive_effect": None, "movement_prediction": 1, "ranking_value": 1, "selection_value": 1, "direction_value": 3,
           "timing_value": 2, "risk_value": 2, "portfolio_value": None}


class Diagnosis(_StrEnum):
    PREDICTIVE_NOT_PORTFOLIO = "PREDICTIVE_NOT_PORTFOLIO"          # the signal is better but the portfolio is not
    PORTFOLIO_WITHOUT_PREDICTIVE = "PORTFOLIO_WITHOUT_PREDICTIVE"  # the portfolio is better with no prediction behind it
    DIRECTION_CANNOT_BUY_BACK = "DIRECTION_CANNOT_BUY_BACK"        # a tier-3 gain alongside tier-1/2 damage
    RISK_BOUGHT_BY_HIDING = "RISK_BOUGHT_BY_HIDING"                # objective gaming flag: cash hiding / no edge / one window
    TIER_CONFLICT = "TIER_CONFLICT"                                # tiers disagree in sign
    CLEAN = "CLEAN"


@dataclass(frozen=True)
class ValueSpec:
    """Which columns hold what. Each pair is (baseline column, new column)."""
    fwd: str = "fwd"
    mature: str | None = "mature"
    horizon_days: int = 7
    score: tuple | None = ("score_base", "score_new")          # expected signed return
    move: tuple | None = ("move_base", "move_new")             # expected movement magnitude / mover probability
    direction: tuple | None = ("dir_base", "dir_new")          # P(up)
    exposure: tuple | None = None                             # per-date deployed fraction in [0, 1], constant within a date
    pick: str = "move"                                       # which signal ranks the picks: 'move', 'score' or 'both' (per-date rank average)
    weighting: str = "equal"                                 # 'equal', 'rank' (linear in pick rank) or 'signal' (proportional to the z-scored signal)
    direction_mode: str = "none"                             # 'none' | 'filter' (skip names with P(up) < 0.5) | 'sign' (short names with P(up) < 0.5)
    cost_mode: str = "round_trip"                            # 'round_trip' (all capital re-traded weekly) | 'turnover' (only names that changed)

    def pair(self, name: str):
        return getattr(self, name)

    def fingerprint(self) -> str:
        return stable_hash(dataclasses.asdict(self))


@dataclass(frozen=True)
class ComponentValue:
    name: str
    tier: int | None
    base: float
    new: float
    delta: float
    lo: float
    hi: float
    n: int                         # dates (or weeks) behind the estimate
    status: str = STATUS_MEASURED
    note: str = ""

    @property
    def significant_gain(self) -> bool:
        return self.status == STATUS_MEASURED and self.lo > 0

    @property
    def significant_loss(self) -> bool:
        return self.status == STATUS_MEASURED and self.hi < 0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _untested(name, why, status=STATUS_UNTESTED) -> ComponentValue:
    nan = float("nan")
    return ComponentValue(name, TIER_OF[name], nan, nan, nan, nan, nan, 0, status, why)


@dataclass(frozen=True)
class ValueDecomposition:
    now: pd.Timestamp
    spec_id: str
    k: int
    n_dates: int
    components: Mapping[str, ComponentValue]
    portfolio_accept: bool | None
    portfolio_reason: str
    gaming_new: tuple
    gaming_base: tuple
    decisive_tier: int | None
    decisive_lower_bound: float
    diagnoses: tuple
    label: ValidationLabel = ValidationLabel.NOT_VALIDATED

    def __getitem__(self, name: str) -> ComponentValue:
        return self.components[name]

    def deltas(self) -> dict:
        """One delta per component. Deliberately NOT summed or averaged: they live on different scales and tiers."""
        return {n: c.delta for n, c in self.components.items()}

    def translation(self) -> dict:
        """Which measured prediction gains reached the portfolio. `reached` is True only when the objective accepted the change."""
        pred = [n for n in ("predictive_effect", "movement_prediction", "ranking_value", "selection_value") if self.components[n].significant_gain]
        return {"predictive_gains": pred, "reached": bool(self.portfolio_accept), "n_predictive_gains": len(pred)}

    def as_record(self) -> dict:
        rec = {"now": str(self.now.date()), "spec": self.spec_id, "k": self.k, "n_dates": self.n_dates, "label": self.label.value,
               "components": {n: c.as_dict() for n, c in self.components.items()}, "portfolio_accept": self.portfolio_accept,
               "portfolio_reason": self.portfolio_reason, "gaming_new": list(self.gaming_new), "gaming_base": list(self.gaming_base),
               "decisive_tier": self.decisive_tier, "decisive_lower_bound": self.decisive_lower_bound,
               "diagnoses": [d.value for d in self.diagnoses]}
        rec["id"] = stable_hash(rec)
        return rec

    def render(self) -> str:
        L = [f"VALUE DECOMPOSITION now={self.now.date()} k={self.k} dates={self.n_dates} [{self.label.value}]"]
        for n in COMPONENTS:
            c = self.components[n]
            if c.status != STATUS_MEASURED:
                L.append(f"- {n:<20} {c.status}: {c.note}")
            else:
                L.append(f"- {n:<20} tier={c.tier}  base {c.base:+.4f} new {c.new:+.4f}  delta {c.delta:+.4f} [{c.lo:+.4f},{c.hi:+.4f}] n={c.n}")
        L.append(f"portfolio: {'NOT RUN' if self.portfolio_accept is None else 'ACCEPT' if self.portfolio_accept else 'REJECT'} ({self.portfolio_reason}); gaming new={list(self.gaming_new) or '-'}")
        L.append("diagnosis: " + ", ".join(d.value for d in self.diagnoses))
        return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------
# panel checks
# ---------------------------------------------------------------------------------------------------------------
def check_panel(panel: pd.DataFrame, spec: ValueSpec, now) -> pd.DataFrame:
    """Validate a panel and return a copy sorted by (date, ticker). Fails closed (FirewallBreach) when any forward return had not
    matured strictly before `now`; raises ValueError for a non-(date, ticker) index, duplicate rows, or NaN/inf returns."""
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("panel must be indexed by MultiIndex (date, ticker)")
    p = panel.copy()
    p.index = pd.MultiIndex.from_arrays([pd.to_datetime(p.index.get_level_values(0)).normalize(), p.index.get_level_values(1).astype(str)],
                                        names=["date", "ticker"])
    if p.index.duplicated().any():
        raise ValueError("duplicate (date, ticker) rows")
    if spec.fwd not in p:
        raise ValueError(f"missing forward-return column {spec.fwd!r}")
    if len(p) and not np.isfinite(p[spec.fwd].to_numpy(float)).all():
        raise ValueError("forward returns contain NaN/inf")
    now_ts = pd.Timestamp(as_date(now))
    dates = p.index.get_level_values(0)
    if spec.mature and spec.mature in p:
        mat = pd.to_datetime(p[spec.mature]).dt.normalize() if len(p) else pd.Series([], dtype="datetime64[ns]")
    else:
        mat = pd.Series(dates + pd.Timedelta(days=spec.horizon_days), index=p.index)
    if len(p) and (mat.to_numpy() >= np.datetime64(now_ts)).any():
        raise FirewallBreach(f"{int((mat.to_numpy() >= np.datetime64(now_ts)).sum())} rows have a forward return that matures at/after "
                             f"now={now_ts.date()}")
    return p.sort_index()


def subsample_dates(panel: pd.DataFrame, every: int) -> pd.DataFrame:
    """Keep every `every`-th decision date so holding periods do not overlap (weekly returns must be independent periods)."""
    if every < 1:
        raise ValueError("every must be >= 1")
    ds = sorted(panel.index.get_level_values(0).unique())[::every]
    return panel[panel.index.get_level_values(0).isin(ds)]


def _check_spacing(p: pd.DataFrame, spec: ValueSpec) -> None:
    ds = pd.DatetimeIndex(sorted(p.index.get_level_values(0).unique()))
    if len(ds) > 1 and np.median(np.diff(ds.values).astype("timedelta64[D]").astype(float)) < 0.8 * spec.horizon_days:
        raise ValueError("decision dates are closer together than the holding period: weekly portfolio returns would overlap; "
                         "use subsample_dates")


# ---------------------------------------------------------------------------------------------------------------
# per-date statistics (vectorised)
# ---------------------------------------------------------------------------------------------------------------
def _rank(s: pd.Series) -> pd.Series:
    return s.groupby(level=0).rank(method="average")


def per_date_rank_corr(x: pd.Series, y: pd.Series, min_names: int = 8) -> pd.Series:
    """Spearman correlation of two aligned series within each date (NaN where fewer than min_names names or a constant side)."""
    xr, yr = _rank(x), _rank(y)
    g = lambda s: s.groupby(level=0)
    xc, yc = xr - g(xr).transform("mean"), yr - g(yr).transform("mean")
    num, dx, dy, n = g(xc * yc).sum(), g(xc * xc).sum(), g(yc * yc).sum(), g(xr).size()
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = num / np.sqrt(dx * dy)
    ic[(n < min_names) | (dx <= 0) | (dy <= 0)] = np.nan
    return ic


def per_date_auc(score: pd.Series, positive: pd.Series, min_names: int = 8) -> pd.Series:
    """Within-date AUC of `score` for the binary label (rank-sum form, ties averaged); NaN unless both classes are present."""
    r = _rank(score)
    y = positive.astype(float)
    g = lambda s: s.groupby(level=0)
    npos, n = g(y).sum(), g(y).size()
    nneg = n - npos
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (g(r * y).sum() - npos * (npos + 1) / 2) / (npos * nneg)
    auc[(npos < 1) | (nneg < 1) | (n < min_names)] = np.nan
    return auc


def _jitter(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n)


def top_k_picks(score: pd.Series, k: int, rng: np.random.Generator, min_names: int = 8) -> pd.Series:
    """Boolean Series (same index): the top-k names per date by score. Ties are broken by a seeded random draw - never by name
    order, which would leak identity into the ordering. Dates with fewer than min_names names pick nothing."""
    out = np.zeros(len(score), bool)
    dates = score.index.get_level_values(0)
    vals = score.to_numpy(float)
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]]) if len(score) else np.array([], int)
    ends = np.r_[starts[1:], len(score)]
    for a, b in zip(starts, ends):
        if b - a < min_names or b - a < k:
            continue
        v = np.where(np.isfinite(vals[a:b]), vals[a:b], -np.inf)
        order = np.lexsort((_jitter(b - a, rng), -v))
        out[a + order[:k]] = True
    return pd.Series(out, index=score.index)


def relevance(fwd: pd.Series, band=O.BAND) -> pd.Series:
    """1 when the name moved into the 5-10% band (either direction), else 0: the tier-1 target."""
    a = fwd.abs()
    return ((a >= band[0]) & (a <= band[1])).astype(float)


def per_date_ndcg(score: pd.Series, rel: pd.Series, k: int, rng: np.random.Generator, min_names: int = 8) -> pd.Series:
    """NDCG@k of the ordering by `score` against binary relevance, per date. Dates with no relevant name (ideal DCG 0) are NaN."""
    dates = score.index.get_level_values(0)
    vals, rl = score.to_numpy(float), rel.to_numpy(float)
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]]) if len(score) else np.array([], int)
    ends = np.r_[starts[1:], len(score)]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    res, idx = [], []
    for a, b in zip(starts, ends):
        n = b - a
        if n < min_names or n < k:
            res.append(np.nan)
        else:
            v = np.where(np.isfinite(vals[a:b]), vals[a:b], -np.inf)
            order = np.lexsort((_jitter(n, rng), -v))[:k]
            ideal = np.sort(rl[a:b])[::-1][:k]
            idcg = float((ideal * disc[:len(ideal)]).sum())
            res.append(float((rl[a:b][order] * disc[:len(order)]).sum() / idcg) if idcg > 0 else np.nan)
        idx.append(dates[a])
    return pd.Series(res, index=pd.DatetimeIndex(idx, name="date"), dtype=float)


def incremental_ic(base: pd.Series, new: pd.Series, target: pd.Series, min_names: int = 8) -> pd.Series:
    """Per-date IC (with `target`) of the part of `new` not explained by `base`: rank-regress new on base within each date, take the
    residual. Zero when `new` only re-ranks what `base` already says."""
    rb, rn = _rank(base), _rank(new)
    g = lambda s: s.groupby(level=0)
    bc, nc = rb - g(rb).transform("mean"), rn - g(rn).transform("mean")
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = g(bc * nc).sum() / g(bc * bc).sum()
    resid = nc - bc * beta.reindex(bc.index.get_level_values(0)).to_numpy()
    return per_date_rank_corr(resid, target, min_names)


# ---------------------------------------------------------------------------------------------------------------
# comparing two arms over dates
# ---------------------------------------------------------------------------------------------------------------
def _paired(name: str, base: pd.Series, new: pd.Series, n_boot: int, seed: int, min_dates: int, note: str = "", cluster: str = "%Y-%m") -> ComponentValue:
    """Component from per-date values of the two arms: means, and a cluster (month) bootstrap on the per-date paired difference."""
    j = pd.concat([base.rename("b"), new.rename("n")], axis=1).dropna()
    if len(j) < min_dates:
        return _untested(name, f"only {len(j)} usable dates (< {min_dates})", STATUS_INSUFFICIENT)
    cl = pd.DatetimeIndex(j.index).strftime(cluster).to_numpy()
    bm = TS.cluster_bootstrap_mean((j["n"] - j["b"]).to_numpy(), cl, n_boot=n_boot, seed=seed)
    return ComponentValue(name, TIER_OF[name], float(j["b"].mean()), float(j["n"].mean()), bm.mean, bm.lo, bm.hi, int(len(j)), note=note)


def _lift(p: pd.DataFrame, picks: pd.Series, flag: pd.Series) -> pd.Series:
    """Per date: hit rate among picks minus hit rate in the whole cross-section (the random-pick expectation)."""
    hit = flag.astype(float)
    pr = hit[picks].groupby(level=0).mean()
    un = hit.groupby(level=0).mean()
    return (pr - un.reindex(pr.index)).rename("lift")


# ---------------------------------------------------------------------------------------------------------------
# simulated portfolio
# ---------------------------------------------------------------------------------------------------------------
def pick_columns(spec: ValueSpec) -> tuple | None:
    """The (base, new) column pair(s) that rank picks, or None when the spec has no usable signal."""
    if spec.pick == "both":
        return (spec.move + spec.score) if spec.move and spec.score else None
    return getattr(spec, spec.pick)


def pick_signal(p: pd.DataFrame, spec: ValueSpec, arm: int) -> pd.Series:
    """The ranking signal of one arm. 'both' averages the within-date percentile ranks of the movement and score signals, so
    neither scale dominates."""
    need = (spec.move + spec.score) if spec.pick == "both" and spec.move and spec.score else (getattr(spec, spec.pick) or ())
    gone = [c for c in need if c not in p]
    if gone:
        raise ValueError(f"panel lacks pick-signal columns {gone}")
    if spec.pick == "both":
        if not (spec.move and spec.score):
            raise ValueError("pick='both' needs both move and score columns")
        return (_rank(p[spec.move[arm]]) / p[spec.move[arm]].groupby(level=0).transform("size") +
                _rank(p[spec.score[arm]]) / p[spec.score[arm]].groupby(level=0).transform("size")) / 2
    col = getattr(spec, spec.pick)
    if col is None:
        raise ValueError(f"spec has no {spec.pick!r} columns to rank picks by")
    return p[col[arm]]


def pick_weights(sig: pd.Series, picks: pd.Series, scheme: str) -> pd.Series:
    """Weights of the picked names, summing to 1 within each date. equal: 1/k. rank: proportional to rank position (best pick
    largest). signal: proportional to the z-scored signal shifted to be positive (best pick largest, never negative)."""
    chosen = sig[picks]
    if scheme == "equal":
        w = pd.Series(1.0, index=chosen.index)
    elif scheme == "rank":
        w = chosen.groupby(level=0).rank(method="first")
    elif scheme == "signal":
        z = (chosen - chosen.groupby(level=0).transform("mean")) / chosen.groupby(level=0).transform("std").replace(0, np.nan)
        w = (z.fillna(0.0) - z.fillna(0.0).groupby(level=0).transform("min") + 0.25)
    else:
        raise ValueError("weighting must be 'equal', 'rank' or 'signal'")
    return w / w.groupby(level=0).transform("sum")


def simulate_weekly(p: pd.DataFrame, spec: ValueSpec, arm: int, k: int, rng: np.random.Generator, *, cost_bps: float = 5.0,
                    min_names: int = 8) -> tuple[pd.Series, pd.Series]:
    """(weekly return per date, picks mask) of one arm: top-k by the arm's pick signal, weighted per spec.weighting, long-only
    unless spec.direction_mode='sign'. direction_mode 'filter' leaves the capital of a name with P(up) < 0.5 idle; 'sign' shorts it
    (a research option: the paper account may not short). Deployed fraction comes from the exposure column when given (else 1).
    Costs: 'round_trip' charges 2 x cost_bps on the deployed fraction every date; 'turnover' only on names that changed since the
    previous date. Decisions are made at the date's close and `fwd` is the return from the next open, so no same-bar look-ahead."""
    sig = pick_signal(p, spec, arm)
    picks = top_k_picks(sig, k, rng, min_names)
    w = pick_weights(sig, picks, spec.weighting)
    side = pd.Series(1.0, index=w.index)
    if spec.direction_mode != "none":
        if not spec.direction or spec.direction[arm] not in p:
            raise ValueError("direction_mode needs direction columns present in the panel")
        pu = p.loc[picks, spec.direction[arm]]
        side = pd.Series(np.where(pu >= 0.5, 1.0, 0.0 if spec.direction_mode == "filter" else -1.0), index=w.index)
    contrib = w * side * p.loc[picks, spec.fwd]
    gross = contrib.groupby(level=0).sum()
    deployed = (w * side.abs()).groupby(level=0).sum()
    e = p[spec.exposure[arm]].groupby(level=0).first().reindex(gross.index).clip(0, 1) if spec.exposure else pd.Series(1.0, index=gross.index)
    if spec.cost_mode == "turnover":
        names = picks[picks].groupby(level=0).apply(lambda s: set(s.index.get_level_values(1)))
        prev, churn = set(), []
        for d in gross.index:
            cur = names.get(d, set())
            churn.append(1.0 if not prev else 1 - len(cur & prev) / max(1, len(cur)))
            prev = cur
        cost = pd.Series(churn, index=gross.index) * 2 * cost_bps / 1e4
    elif spec.cost_mode == "round_trip":
        cost = deployed * 2 * cost_bps / 1e4
    else:
        raise ValueError("cost_mode must be 'round_trip' or 'turnover'")
    return (e * gross - e * cost).rename("week"), picks


def weeks_to_rows(weeks: np.ndarray, window: int) -> list[dict]:
    """Objective rows: consecutive blocks of `window` weeks (a remainder shorter than half a block joins the previous block)."""
    w = np.asarray(weeks, float)
    if len(w) == 0:
        return []
    nb = max(1, len(w) // window)
    cuts = np.linspace(0, len(w), nb + 1).astype(int)
    return [O.week_row(w[a:b]) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def window_risk(row: dict) -> float:
    """Per-window tier-2 quantity, the same terms as objective.evaluate's risk (higher is better)."""
    return row["worst5"] + row["max_dd"] / 4 - row.get("cat_rate", 0.0) - O.OVER_RISK_W * row["over_band"]


def timing_cov(exposure: pd.Series, weekly_unit: pd.Series) -> float:
    """Population covariance between deployed fraction and the pre-cost weekly return of a fully invested portfolio: the value of
    deploying more when the picks pay, separate from the level of exposure."""
    j = pd.concat([exposure.rename("e"), weekly_unit.rename("r")], axis=1).dropna()
    if len(j) < 3:
        return float("nan")
    return float(((j["e"] - j["e"].mean()) * (j["r"] - j["r"].mean())).mean())


# ---------------------------------------------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------------------------------------------
def decompose_value(panel: pd.DataFrame, spec: ValueSpec, now, *, k: int = 5, mover_thr: float = 0.05, band=O.BAND, cost_bps: float = 5.0,
                    min_names: int = 8, min_dates: int = 20, window_weeks: int = 26, seed: int = 0, n_boot: int = 500) -> ValueDecomposition:
    """Measure the eight kinds of value of `new` against `base` on a panel. See the module docstring. Every component that cannot
    be measured (missing columns, too few dates) is reported UNTESTED / INSUFFICIENT, never as zero."""
    p = check_panel(panel, spec, now)
    _check_spacing(p, spec)
    fwd = p[spec.fwd]
    absf = fwd.abs()
    mover = absf >= mover_thr
    rel = relevance(fwd, band)
    seeds = np.random.default_rng(seed)
    comps: dict[str, ComponentValue] = {}
    have = lambda pair: pair is not None and all(c in p for c in pair)
    sd = lambda: int(seeds.integers(0, 2 ** 31 - 1))

    # 1 predictive effect + incremental IC
    if have(spec.score):
        b, n = spec.score
        ic_b, ic_n = per_date_rank_corr(p[b], fwd, min_names), per_date_rank_corr(p[n], fwd, min_names)
        inc = incremental_ic(p[b], p[n], fwd, min_names)
        c = _paired("predictive_effect", ic_b, ic_n, n_boot, sd(), min_dates, note=f"rank IC vs forward return; incremental IC of new over base {float(inc.mean()):+.4f}")
        comps["predictive_effect"] = c
    else:
        comps["predictive_effect"] = _untested("predictive_effect", "no score columns")

    # 2 movement prediction: IC to |fwd| (the AUC for movers is folded into the note)
    if have(spec.move):
        b, n = spec.move
        ic_b, ic_n = per_date_rank_corr(p[b], absf, min_names), per_date_rank_corr(p[n], absf, min_names)
        auc_b, auc_n = per_date_auc(p[b], mover, min_names), per_date_auc(p[n], mover, min_names)
        comps["movement_prediction"] = _paired("movement_prediction", ic_b, ic_n, n_boot, sd(), min_dates,
                                               note=f"IC vs |fwd|; mover AUC base {float(auc_b.mean()):.3f} new {float(auc_n.mean()):.3f}")
    else:
        comps["movement_prediction"] = _untested("movement_prediction", "no movement columns")

    # portfolio arms (need a pick signal)
    pick_pair = pick_columns(spec)
    weeks = picks = None
    if pick_pair is not None and have(pick_pair):
        wk_b, pk_b = simulate_weekly(p, spec, 0, k, np.random.default_rng(seed + 11), cost_bps=cost_bps, min_names=min_names)
        wk_n, pk_n = simulate_weekly(p, spec, 1, k, np.random.default_rng(seed + 11), cost_bps=cost_bps, min_names=min_names)
        common = wk_b.index.intersection(wk_n.index)
        weeks, picks = (wk_b.loc[common], wk_n.loc[common]), (pk_b, pk_n)
        # 3 ranking value (NDCG@k on the pick signal)
        nd_b = per_date_ndcg(pick_signal(p, spec, 0), rel, k, np.random.default_rng(seed + 21), min_names)
        nd_n = per_date_ndcg(pick_signal(p, spec, 1), rel, k, np.random.default_rng(seed + 21), min_names)
        comps["ranking_value"] = _paired("ranking_value", nd_b, nd_n, n_boot, sd(), min_dates, note=f"NDCG@{k} against in-band moves")
        # 4 selection value: in-band hit rate of the picks minus the universe rate
        lb, ln = _lift(p, pk_b, rel > 0), _lift(p, pk_n, rel > 0)
        comps["selection_value"] = _paired("selection_value", lb, ln, n_boot, sd(), min_dates, note=f"picks' in-band rate minus universe rate (k={k})")
    else:
        comps["ranking_value"] = _untested("ranking_value", "no pick signal columns")
        comps["selection_value"] = _untested("selection_value", "no pick signal columns")

    # 5 direction value: Brier skill of P(up) on the movers, against the movers' own base rate (tier 3: reported, not dominant)
    if have(spec.direction):
        m = p[mover]
        if len(m) >= min_dates * min_names / 2:
            up = (m[spec.fwd] > 0).astype(float)
            base_rate = up.groupby(level=0).transform("mean")
            b0 = (base_rate - up) ** 2
            skill = lambda col: 1 - ((m[col].clip(0, 1) - up) ** 2).groupby(level=0).mean() / b0.groupby(level=0).mean().replace(0, np.nan)
            sb, sn = skill(spec.direction[0]), skill(spec.direction[1])
            comps["direction_value"] = _paired("direction_value", sb, sn, n_boot, sd(), min_dates,
                                               note=f"Brier skill vs the movers' base rate ({float(up.mean()):.3f} up)")
        else:
            comps["direction_value"] = _untested("direction_value", "too few movers", STATUS_INSUFFICIENT)
    else:
        comps["direction_value"] = _untested("direction_value", "no direction columns")

    # 6 timing value
    if spec.exposure and have(spec.exposure) and weeks is not None:
        full = p.loc[picks[1], spec.fwd].groupby(level=0).mean().reindex(common)
        eb = p[spec.exposure[0]].groupby(level=0).first().reindex(common)
        en = p[spec.exposure[1]].groupby(level=0).first().reindex(common)
        if len(common) >= min_dates:
            cb, cn = timing_cov(eb, full), timing_cov(en, full)
            mo = pd.DatetimeIndex(common).strftime("%Y-%m").to_numpy()
            contrib = lambda e: ((e - e.mean()) * (full - full.mean()))
            bm = TS.cluster_bootstrap_mean((contrib(en) - contrib(eb)).to_numpy(), mo, n_boot=n_boot, seed=sd())
            comps["timing_value"] = ComponentValue("timing_value", 2, cb, cn, cn - cb, bm.lo, bm.hi, int(len(common)), note="Cov(exposure, pick return)")
        else:
            comps["timing_value"] = _untested("timing_value", "too few dates", STATUS_INSUFFICIENT)
    else:
        comps["timing_value"] = _untested("timing_value", "no exposure columns: nothing to time")

    # 7 risk value + 8 portfolio value through the tiered objective
    accept, reason, gam_n, gam_b, dtier, dlb = None, "no portfolio (missing pick signal)", (), (), None, float("nan")
    if weeks is not None and len(common) >= min_dates:
        rows_b, rows_n = weeks_to_rows(weeks[0].to_numpy(), window_weeks), weeks_to_rows(weeks[1].to_numpy(), window_weeks)
        sb, sn = O.evaluate(rows_b), O.evaluate(rows_n)
        accept, reason = O.firewall(sb, sn, rows_b, rows_n)
        gam_b, gam_n = tuple(O.gaming_flags(rows_b)), tuple(O.gaming_flags(rows_n))
        if len(rows_b) == len(rows_n) and len(rows_b) >= 2:
            dtier, diffs = O.decisive_diffs(rows_b, rows_n)
            dlb = O.paired_bootstrap(diffs, n_boot=n_boot, seed=sd()) if dtier is not None else float("nan")
            rd = np.array([window_risk(b) - window_risk(a) for a, b in zip(rows_b, rows_n)])
            rb_ = np.array([window_risk(a) for a in rows_b])
            bm = TS.cluster_bootstrap_mean(rd, None, n_boot=n_boot, seed=sd())
            comps["risk_value"] = ComponentValue("risk_value", 2, float(rb_.mean()), float(rb_.mean() + rd.mean()), bm.mean, bm.lo, bm.hi, len(rd),
                                                 note="per-window worst-5% + drawdown/4 - catastrophic - overshoot (higher is better)")
        else:
            comps["risk_value"] = ComponentValue("risk_value", 2, sb.risk, sn.risk, sn.risk - sb.risk, float("-inf"), float("inf"), len(rows_b),
                                                 STATUS_INSUFFICIENT, "fewer than two windows: no interval")
        mo = pd.DatetimeIndex(common).strftime("%Y-%m").to_numpy()
        wd = TS.cluster_bootstrap_mean(np.abs(weeks[1].to_numpy()) - np.abs(weeks[0].to_numpy()), mo, n_boot=n_boot, seed=sd())
        cmp_ = int(O.compare(rows_b, rows_n))
        comps["portfolio_value"] = ComponentValue(
            "portfolio_value", None, float(sb.soft), float(sn.soft), float(cmp_), dlb if math.isfinite(dlb) else float("-inf"),
            float("inf") if accept else 0.0, len(rows_n),
            note=f"delta is the lexicographic comparison (+1 better, -1 worse, 0 tied on the tier grid); base/new are the smooth legacy scores; "
                 f"keys {list(sb.key)} -> {list(sn.key)}; firewall: {reason}; mean |week| {float(np.abs(weeks[0]).mean()):.4f} -> "
                 f"{float(np.abs(weeks[1]).mean()):.4f} (paired CI {wd.lo:+.4f}..{wd.hi:+.4f})")
    else:
        comps["risk_value"] = _untested("risk_value", "no portfolio simulated", STATUS_INSUFFICIENT)
        comps["portfolio_value"] = _untested("portfolio_value", "no portfolio simulated", STATUS_INSUFFICIENT)
    diags = diagnose(comps, accept, gam_n, gam_b)
    label = ValidationLabel.NOT_VALIDATED if any(c.status == STATUS_MEASURED for c in comps.values()) else ValidationLabel.INSUFFICIENT_EVIDENCE
    return ValueDecomposition(pd.Timestamp(as_date(now)), spec.fingerprint(), k, int(p.index.get_level_values(0).nunique()), comps, accept,
                              reason, gam_n, gam_b, dtier, dlb, diags, label)


def diagnose(comps: Mapping[str, ComponentValue], accept: bool | None, gaming_new: Sequence[str], gaming_base: Sequence[str]) -> tuple:
    """Where value was lost or claimed without support. Several diagnoses can hold at once."""
    out = []
    pred = [n for n in ("predictive_effect", "movement_prediction", "ranking_value", "selection_value") if comps[n].significant_gain]
    if accept is not None and pred and not accept:
        out.append(Diagnosis.PREDICTIVE_NOT_PORTFOLIO)
    if accept and not pred:
        out.append(Diagnosis.PORTFOLIO_WITHOUT_PREDICTIVE)
    tier12_loss = [n for n in ("movement_prediction", "ranking_value", "selection_value", "risk_value", "timing_value") if comps[n].significant_loss]
    if comps["direction_value"].significant_gain and tier12_loss:
        out.append(Diagnosis.DIRECTION_CANNOT_BUY_BACK)
    if any(g not in gaming_base for g in gaming_new):
        out.append(Diagnosis.RISK_BOUGHT_BY_HIDING)
    signs = {int(np.sign(comps[n].delta)) for n in ("movement_prediction", "ranking_value", "selection_value", "risk_value")
             if comps[n].status == STATUS_MEASURED and (comps[n].significant_gain or comps[n].significant_loss)}
    if signs == {1, -1}:
        out.append(Diagnosis.TIER_CONFLICT)
    return tuple(out) if out else (Diagnosis.CLEAN,)


def decompose_by(panel: pd.DataFrame, spec: ValueSpec, labels: pd.Series, now, *, min_dates: int = 20, **kw) -> dict:
    """Run the decomposition separately for each label of a per-date series (era, regime, year). Groups with fewer than
    `min_dates` dates are reported as None (thin), never silently dropped."""
    p = check_panel(panel, spec, now)
    d = p.index.get_level_values(0)
    lab = labels.reindex(d).astype(str).to_numpy()
    out = {}
    for L in sorted(set(lab)):
        sub = p[lab == L]
        out[L] = decompose_value(sub, spec, now, min_dates=min_dates, **kw) if sub.index.get_level_values(0).nunique() >= min_dates else None
    return out


# ---------------------------------------------------------------------------------------------------------------
# planted panel for the tests and the self-check
# ---------------------------------------------------------------------------------------------------------------
def synthetic_value_panel(seed: int = 0, n_dates: int = 80, n_names: int = 40, *, move_skill: float = 0.0, score_skill: float = 0.0,
                          dir_skill: float = 0.0, start: str = "2015-01-05", spacing_days: int = 7, timing_skill: float = 0.0) -> pd.DataFrame:
    """A world with planted skill. Each name has a hidden movement size m (lognormal around 4%) and a hidden sign; the realised
    forward return is sign*m + noise. The *_base scores are pure noise. `move_skill` in [0,1] mixes the true (log) m into
    move_new, `score_skill` the true sign into score_new, `dir_skill` the true sign into dir_new (P(up)); `timing_skill` makes
    exp_new higher in dates where the cross-section will move more (exp_base is constant 1). skill=0 -> new is as blind as base."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    frames = []
    for d in range(n_dates):
        date = t0 + pd.Timedelta(days=spacing_days * d)
        m = np.exp(rng.normal(np.log(0.04), 0.5, n_names))
        sgn = rng.choice([-1.0, 1.0], n_names)
        z = lambda: rng.normal(0, 1, n_names)
        mix = lambda skill, truth: skill * truth + math.sqrt(max(0.0, 1 - skill ** 2)) * z()
        wave = float(m.mean() / 0.04)
        frames.append(pd.DataFrame({
            "date": date, "ticker": [f"N{i:03d}" for i in range(n_names)], "mature": date + pd.Timedelta(days=spacing_days),
            "fwd": sgn * m + rng.normal(0, 0.005, n_names),
            "score_base": z(), "score_new": mix(score_skill, sgn),
            "move_base": z(), "move_new": mix(move_skill, (np.log(m) - np.log(0.04)) / 0.5),
            "dir_base": 1 / (1 + np.exp(-z())), "dir_new": 1 / (1 + np.exp(-(dir_skill * 3.0 * sgn + (1 - 0.8 * dir_skill) * z()))),
            "exp_base": 1.0, "exp_new": float(np.clip(0.5 + timing_skill * (wave - 1.0) * 2.0, 0, 1)) if timing_skill else 1.0}))
    return pd.concat(frames).set_index(["date", "ticker"]).sort_index()


# ---------------------------------------------------------------------------------------------------------------
# arm summaries, sensitivity sweeps, luck floor, signal attribution, consistency
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmSummary:
    """Plain statistics of one simulated weekly series, in the objective's own vocabulary."""
    n_weeks: int
    mean_week: float
    sd_week: float
    mean_abs_week: float
    in_band: float
    over_band: float
    under_band: float
    worst5: float
    max_dd: float
    pos_share: float
    cat_rate: float
    flat_share: float
    tier_key: tuple

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["tier_key"] = list(self.tier_key)
        return d


def arm_summary(weeks) -> ArmSummary:
    w = np.asarray(weeks, float)
    r = O.week_row(w)
    ev = O.evaluate([r]) if len(w) else O.evaluate([])
    return ArmSummary(int(len(w)), r["mean_week"], r["sd_week"], r["abs_mean"], r["in_band"], r["over_band"], r["under_band"], r["worst5"], r["max_dd"],
                      float((w > 0).mean()) if len(w) else 0.0, r["cat_rate"], r["flat_share"], tuple(ev.key))


def run_arms(panel: pd.DataFrame, spec: ValueSpec, now, *, k: int = 5, cost_bps: float = 5.0, min_names: int = 8, seed: int = 0) -> dict:
    """Simulate both arms on the common dates and return their weekly series and summaries: {'base': (weeks, ArmSummary), 'new': ...}.
    Both arms use the same seeded tie-break stream, so a difference is never a tie-break accident."""
    p = check_panel(panel, spec, now)
    _check_spacing(p, spec)
    w0, _ = simulate_weekly(p, spec, 0, k, np.random.default_rng(seed + 11), cost_bps=cost_bps, min_names=min_names)
    w1, _ = simulate_weekly(p, spec, 1, k, np.random.default_rng(seed + 11), cost_bps=cost_bps, min_names=min_names)
    common = w0.index.intersection(w1.index)
    w0, w1 = w0.loc[common], w1.loc[common]
    return {"base": (w0, arm_summary(w0.to_numpy())), "new": (w1, arm_summary(w1.to_numpy()))}


def k_sweep(panel: pd.DataFrame, spec: ValueSpec, now, ks: Sequence[int] = (3, 5, 8, 12), *, n_boot: int = 200, seed: int = 0, **kw) -> pd.DataFrame:
    """How the value depends on portfolio size: selection and ranking deltas, in-band share and objective verdict at each k. A
    signal that only helps at k=3 is fragile; one that helps at every k is not an artefact of the pick count."""
    rows = []
    for k in ks:
        d = decompose_value(panel, spec, now, k=k, n_boot=n_boot, seed=seed, **kw)
        arms = run_arms(panel, spec, now, k=k, seed=seed, cost_bps=kw.get("cost_bps", 5.0), min_names=kw.get("min_names", 8))
        c = d["selection_value"]
        rows.append({"k": k, "selection_delta": c.delta, "sel_lo": c.lo, "sel_hi": c.hi, "ndcg_delta": d["ranking_value"].delta,
                     "in_band_base": arms["base"][1].in_band, "in_band_new": arms["new"][1].in_band,
                     "abs_week_base": arms["base"][1].mean_abs_week, "abs_week_new": arms["new"][1].mean_abs_week,
                     "accepted": d.portfolio_accept, "reason": d.portfolio_reason})
    return pd.DataFrame(rows)


def cost_sweep(panel: pd.DataFrame, spec: ValueSpec, now, costs: Sequence[float] = (0, 5, 10, 20, 40), *, k: int = 5, seed: int = 0, window_weeks: int = 26,
               **kw) -> tuple[pd.DataFrame, float | None]:
    """Objective comparison of new vs base at several round-trip costs (bps per side). Returns the table and the break-even cost:
    the first cost at which the new arm no longer beats the base arm on the tier grid (None if it always does). A gain that dies at
    10 bps is not a gain."""
    rows, breakeven = [], None
    for c in costs:
        arms = run_arms(panel, spec, now, k=k, cost_bps=float(c), seed=seed, **kw)
        rb, rn = weeks_to_rows(arms["base"][0].to_numpy(), window_weeks), weeks_to_rows(arms["new"][0].to_numpy(), window_weeks)
        acc, why = O.firewall(O.evaluate(rb), O.evaluate(rn), rb, rn) if rb and rn else (False, "no weeks")
        cmp_ = int(O.compare(rb, rn)) if rb and rn else 0
        rows.append({"cost_bps": float(c), "compare": cmp_, "accepted": bool(acc), "reason": why, "mean_week_new": arms["new"][1].mean_week,
                     "mean_week_base": arms["base"][1].mean_week})
        if breakeven is None and not acc:
            breakeven = float(c)
    return pd.DataFrame(rows), breakeven


@dataclass(frozen=True)
class LuckFloor:
    arm_lift: float
    random_mean: float
    random_sd: float
    percentile: float            # share of random pickers the arm beats
    z: float
    beats_luck: bool             # above the 95th percentile of random pickers


def random_floor(panel: pd.DataFrame, spec: ValueSpec, now, *, arm: int = 1, k: int = 5, n_sims: int = 200, min_names: int = 8, seed: int = 0) -> LuckFloor:
    """Selection lift (in-band hit rate of the picks minus the universe rate) of the arm against `n_sims` random k-of-n pickers on the
    same dates. A lift that a random picker beats one time in five is luck, however significant its bootstrap looks."""
    p = check_panel(panel, spec, now)
    rel = relevance(p[spec.fwd])
    sig = pick_signal(p, spec, arm)
    lift = lambda pk: float(_lift(p, pk, rel > 0).mean())
    actual = lift(top_k_picks(sig, k, np.random.default_rng(seed), min_names))
    rng = np.random.default_rng(seed + 1)
    sims = np.array([lift(top_k_picks(pd.Series(rng.random(len(p)), index=p.index), k, rng, min_names)) for _ in range(n_sims)])
    sd = float(sims.std(ddof=1)) if n_sims > 1 else float("nan")
    return LuckFloor(actual, float(sims.mean()), sd, float((sims < actual).mean()), (actual - float(sims.mean())) / sd if sd and sd > 0 else float("nan"),
                     bool((sims < actual).mean() >= 0.95))


def attribute_signals(panel: pd.DataFrame, spec: ValueSpec, now, *, k: int = 5, cost_bps: float = 5.0, min_names: int = 8, seed: int = 0,
                      window_weeks: int = 26) -> pd.DataFrame:
    """Which of the two new signals earned the portfolio change? Four pick rules, each the per-date rank average of one movement
    source (base/new) and one score source (base/new): bb, nb, bn, nn. Rows give each variant's in-band share, mean |week|, selection lift and
    verdict against bb; the last rows are the Shapley-style marginals (mean of the two orders) for movement, score and their
    interaction, on the in-band share. Needs both a movement and a score pair."""
    if not (spec.move and spec.score) or any(c not in panel for c in spec.move + spec.score):
        raise ValueError("attribute_signals needs both movement and score columns in the panel")
    p = check_panel(panel, spec, now)
    _check_spacing(p, spec)
    rk = lambda col: _rank(p[col]) / p[col].groupby(level=0).transform("size")
    p = p.copy()
    src = {"b": 0, "n": 1}
    for name, (m, s_) in {"bb": ("b", "b"), "nb": ("n", "b"), "bn": ("b", "n"), "nn": ("n", "n")}.items():
        p["_sig_" + name] = (rk(spec.move[src[m]]) + rk(spec.score[src[s_]])) / 2
    rel = relevance(p[spec.fwd])
    out, weeks_by, rows_by = {}, {}, {}
    for name in ("bb", "nb", "bn", "nn"):
        sp = dataclasses.replace(spec, pick="score", score=("_sig_" + name, "_sig_" + name), exposure=None, direction_mode="none")
        wk, pk = simulate_weekly(p, sp, 0, k, np.random.default_rng(seed + 11), cost_bps=cost_bps, min_names=min_names)
        weeks_by[name] = wk
        out[name] = (arm_summary(wk.to_numpy()), float(_lift(p, pk, rel > 0).mean()))
    base_rows = weeks_to_rows(weeks_by["bb"].to_numpy(), window_weeks)
    table = []
    for name in ("bb", "nb", "bn", "nn"):
        rows = weeks_to_rows(weeks_by[name].to_numpy(), window_weeks)
        acc, why = (True, "baseline") if name == "bb" else (O.firewall(O.evaluate(base_rows), O.evaluate(rows), base_rows, rows) if rows and base_rows else (False, "no weeks"))
        table.append({"variant": name, "movement_source": "new" if name[0] == "n" else "base", "score_source": "new" if name[1] == "n" else "base",
                      "in_band": out[name][0].in_band, "mean_abs_week": out[name][0].mean_abs_week, "lift": out[name][1], "accepted": bool(acc), "reason": why})
    v = {r["variant"]: r["in_band"] for r in table}
    lf = {r["variant"]: r["lift"] for r in table}
    for label, fn in (("movement", lambda d: ((d["nb"] - d["bb"]) + (d["nn"] - d["bn"])) / 2),
                      ("score", lambda d: ((d["bn"] - d["bb"]) + (d["nn"] - d["nb"])) / 2),
                      ("interaction", lambda d: d["nn"] - d["nb"] - d["bn"] + d["bb"])):
        table.append({"variant": f"marginal:{label}", "movement_source": "", "score_source": "", "in_band": fn(v), "mean_abs_week": float("nan"),
                      "lift": fn(lf), "accepted": None, "reason": "Shapley-style marginal (in_band, lift)"})
    return pd.DataFrame(table)


def split_half_consistency(panel: pd.DataFrame, spec: ValueSpec, now, *, min_dates: int = 20, **kw) -> dict:
    """Decompose the first and second half of the dates separately. A component 'replicates' if its delta has the same sign in both
    halves and is measured in both. Returns {'first': dec, 'second': dec, 'agree': [...], 'disagree': [...], 'unmeasured': [...]}."""
    p = check_panel(panel, spec, now)
    dates = sorted(p.index.get_level_values(0).unique())
    if len(dates) < 2 * min_dates:
        return {"first": None, "second": None, "agree": [], "disagree": [], "unmeasured": list(COMPONENTS)}
    cut = dates[len(dates) // 2]
    d = p.index.get_level_values(0)
    a, b = decompose_value(p[d < cut], spec, now, min_dates=min_dates, **kw), decompose_value(p[d >= cut], spec, now, min_dates=min_dates, **kw)
    agree, disagree, unm = [], [], []
    for n in COMPONENTS:
        ca, cb = a[n], b[n]
        if ca.status != STATUS_MEASURED or cb.status != STATUS_MEASURED or not (math.isfinite(ca.delta) and math.isfinite(cb.delta)):
            unm.append(n)
        elif np.sign(ca.delta) == np.sign(cb.delta) and ca.delta != 0:
            agree.append(n)
        else:
            disagree.append(n)
    return {"first": a, "second": b, "agree": agree, "disagree": disagree, "unmeasured": unm}


# ---------------------------------------------------------------------------------------------------------------
# building a panel from separate prediction and outcome tables; plain-language explanation
# ---------------------------------------------------------------------------------------------------------------
def panel_from_predictions(preds: pd.DataFrame, outcomes: pd.DataFrame, spec: ValueSpec, now) -> tuple[pd.DataFrame, dict]:
    """Join model outputs (indexed (date, ticker), no outcome columns) with realised outcomes (fwd, mature) into a checked panel.
    Refuses predictions that already carry the outcome column (an answer column in the input is a leak), refuses outcomes that
    mature at/after `now`, and reports how many prediction rows found no outcome and vice versa."""
    for bad in (spec.fwd, spec.mature):
        if bad and bad in preds.columns:
            raise FirewallBreach(f"prediction table contains outcome column {bad!r}")
    j = preds.join(outcomes[[c for c in (spec.fwd, spec.mature) if c and c in outcomes.columns]], how="inner")
    info = {"n_predictions": int(len(preds)), "n_outcomes": int(len(outcomes)), "n_joined": int(len(j)),
            "predictions_without_outcome": int(len(preds) - len(j)), "outcomes_without_prediction": int(len(outcomes) - len(j))}
    return check_panel(j, spec, now), info


def explain(dec: ValueDecomposition) -> list[str]:
    """Plain-language reading of a decomposition against the tiered objective. Says what improved, what did not, and why a
    prediction gain may not be a portfolio gain; contains no verdict on 'learning'."""
    L = []
    for n in COMPONENTS:
        c = dec[n]
        if c.status != STATUS_MEASURED:
            L.append(f"{n}: {c.status.lower()} ({c.note}).")
        elif c.significant_gain:
            L.append(f"{n}: better by {c.delta:+.4f} (CI {c.lo:+.4f}..{c.hi:+.4f})" + (f", tier {c.tier}" if c.tier else "") + ".")
        elif c.significant_loss:
            L.append(f"{n}: WORSE by {c.delta:+.4f} (CI {c.lo:+.4f}..{c.hi:+.4f})" + (f", tier {c.tier}" if c.tier else "") + ".")
        else:
            L.append(f"{n}: no reliable change ({c.delta:+.4f}, CI {c.lo:+.4f}..{c.hi:+.4f}).")
    for d in dec.diagnoses:
        L.append({Diagnosis.PREDICTIVE_NOT_PORTFOLIO: "A prediction gain did not become a portfolio gain: the objective rejected the change (e.g. overshoot past 10% counts as risk).",
                  Diagnosis.PORTFOLIO_WITHOUT_PREDICTIVE: "The portfolio improved with no measured prediction gain behind it: treat as luck or sizing until explained.",
                  Diagnosis.DIRECTION_CANNOT_BUY_BACK: "Direction improved while a higher tier got worse; tier 3 cannot buy back damage to tier 1 or 2.",
                  Diagnosis.RISK_BOUGHT_BY_HIDING: "The objective's gaming guard fired: the tier scores were reached by hiding in cash, leverage on noise, or one lucky window.",
                  Diagnosis.TIER_CONFLICT: "Tiers disagree in sign; the lexicographic order, not their sum, decides.",
                  Diagnosis.CLEAN: "No translation problem detected."}[d])
    return L


# ---------------------------------------------------------------------------------------------------------------
# is it edge or is it leverage?  (the 7% band can be reached by scaling noise)
# ---------------------------------------------------------------------------------------------------------------
def scale_to_target(weeks: pd.Series, *, target: float = O.TARGET, lookback: int = 8, lo: float = 0.25, hi: float = 4.0) -> pd.Series:
    """Volatility-target a weekly series using ONLY the past: the scale applied to week t is target / (mean |week| over the
    `lookback` weeks before t), clipped to [lo, hi]. Until `lookback` weeks exist the scale is 1. This is what a leveraged
    strategy without any edge would do to sit in the band; it is used to ask whether an arm still wins once volatility is matched."""
    w = weeks.astype(float)
    trailing = w.abs().rolling(lookback, min_periods=lookback).mean().shift(1)
    scale = (target / trailing).clip(lo, hi).fillna(1.0)
    return (w * scale).rename(weeks.name)


def vol_matched_comparison(panel: pd.DataFrame, spec: ValueSpec, now, *, k: int = 5, cost_bps: float = 5.0, lookback: int = 8, window_weeks: int = 26,
                           min_names: int = 8, seed: int = 0) -> dict:
    """Both arms volatility-targeted to 7% with past-only scaling, then compared on the tiered objective. If the new arm's advantage
    disappears (or reverses) once volatility is matched, its raw band share came from taking more risk per week, not from picking better
    names. Returns each arm's raw and matched summaries, the matched firewall decision, and the verdict string."""
    arms = run_arms(panel, spec, now, k=k, cost_bps=cost_bps, min_names=min_names, seed=seed)
    (wb, sb), (wn, sn) = arms["base"], arms["new"]
    mb, mn = scale_to_target(wb, lookback=lookback), scale_to_target(wn, lookback=lookback)
    rb, rn = weeks_to_rows(mb.to_numpy(), window_weeks), weeks_to_rows(mn.to_numpy(), window_weeks)
    if not rb or not rn:
        return {"raw": {"base": sb, "new": sn}, "matched": None, "accepted": None, "reason": "no weeks", "verdict": "INSUFFICIENT"}
    acc, why = O.firewall(O.evaluate(rb), O.evaluate(rn), rb, rn)
    raw_acc = O.compare(weeks_to_rows(wb.to_numpy(), window_weeks), weeks_to_rows(wn.to_numpy(), window_weeks))
    matched = {"base": arm_summary(mb.to_numpy()), "new": arm_summary(mn.to_numpy())}
    if acc:
        verdict = "EDGE_SURVIVES_VOL_MATCHING"
    elif raw_acc > 0:
        verdict = "ADVANTAGE_IS_LEVERAGE"                 # better raw, not better once volatility is matched
    else:
        verdict = "NO_ADVANTAGE"
    return {"raw": {"base": sb, "new": sn}, "matched": matched, "accepted": bool(acc), "reason": why, "raw_compare": int(raw_acc), "verdict": verdict}


# ---------------------------------------------------------------------------------------------------------------
# diagnostics of the signals themselves
# ---------------------------------------------------------------------------------------------------------------
def decile_table(panel: pd.DataFrame, spec: ValueSpec, now, col: str, *, n_bins: int = 10, min_names: int = 8, mover_thr: float = 0.05,
                 n_boot: int = 300, seed: int = 0) -> dict:
    """Realised outcomes by within-date score decile: mean forward return, mean |return|, share of in-band moves and share of
    movers. A monotone rise with the decile is a signal that ranks well everywhere; a signal that only lifts the top decile ranks
    only at the extreme (still fine for top-k picking, useless for sizing). Also the top-minus-bottom spread of |fwd| per date with a
    month-cluster bootstrap interval, and the Spearman correlation of decile with outcome."""
    from scipy.stats import spearmanr
    p = check_panel(panel, spec, now)
    if col not in p:
        raise ValueError(f"no column {col!r}")
    n = p[col].groupby(level=0).transform("size")
    pct = (_rank(p[col]) - 1) / (n - 1).replace(0, np.nan)
    dec = np.minimum((pct * n_bins).astype(float), n_bins - 1e-9).map(lambda v: int(v) if np.isfinite(v) else -1)
    ok = (n >= min_names) & (dec >= 0)
    q = p[ok].assign(_dec=dec[ok], _abs=p[spec.fwd].abs()[ok], _band=relevance(p[spec.fwd])[ok], _mov=(p[spec.fwd].abs() >= mover_thr)[ok].astype(float))
    if len(q) == 0:
        return {"table": pd.DataFrame(columns=["decile", "n", "mean_fwd", "mean_abs", "band_rate", "mover_rate"]), "spearman": float("nan"), "spread": None}
    tab = q.groupby("_dec").agg(n=(spec.fwd, "size"), mean_fwd=(spec.fwd, "mean"), mean_abs=("_abs", "mean"), band_rate=("_band", "mean"),
                                mover_rate=("_mov", "mean")).reset_index().rename(columns={"_dec": "decile"})
    rho = float(spearmanr(tab["decile"], tab["mean_abs"])[0]) if len(tab) > 2 and tab["mean_abs"].std() > 0 else float("nan")
    top, bot = q[q["_dec"] == n_bins - 1]["_abs"].groupby(level=0).mean(), q[q["_dec"] == 0]["_abs"].groupby(level=0).mean()
    spread = (top - bot.reindex(top.index)).dropna()
    bm = TS.cluster_bootstrap_mean(spread.to_numpy(), pd.DatetimeIndex(spread.index).strftime("%Y-%m").to_numpy(), n_boot=n_boot, seed=seed) if len(spread) else None
    return {"table": tab, "spearman": rho, "spread": bm, "monotone": bool(np.isfinite(rho) and rho > 0.8)}


def decile_compare(panel: pd.DataFrame, spec: ValueSpec, now, which: str = "move", **kw) -> dict:
    """decile_table for the base and new columns of one signal ('move' or 'score')."""
    pair = getattr(spec, which)
    if pair is None:
        raise ValueError(f"spec has no {which!r} columns")
    return {"base": decile_table(panel, spec, now, pair[0], **kw), "new": decile_table(panel, spec, now, pair[1], **kw)}


def mover_reliability(panel: pd.DataFrame, spec: ValueSpec, now, *, arm: int = 1, mover_thr: float = 0.05, n_bins: int = 10) -> dict:
    """Pooled equal-mass bins of the movement score against the realised mover rate (|fwd| >= mover_thr). If the score lives in
    [0, 1] it is treated as a probability and the expected calibration error is reported; otherwise only the ranking (monotonicity,
    top-bin lift over the base rate) is."""
    from scipy.stats import spearmanr
    if spec.move is None:
        raise ValueError("spec has no movement columns")
    p = check_panel(panel, spec, now)
    s = p[spec.move[arm]].to_numpy(float)
    y = (p[spec.fwd].abs() >= mover_thr).to_numpy(float)
    if len(s) < n_bins * 3:
        return {"table": pd.DataFrame(), "spearman": float("nan"), "top_lift": float("nan"), "ece": float("nan"), "base_rate": float("nan")}
    order = np.argsort(s, kind="mergesort")
    rows = []
    for i, ch in enumerate(np.array_split(order, n_bins)):
        rows.append({"bin": i, "n": int(len(ch)), "mean_score": float(s[ch].mean()), "mover_rate": float(y[ch].mean())})
    tab = pd.DataFrame(rows)
    prob_like = bool(((s >= 0) & (s <= 1)).all())
    ece = float(sum(r.n / len(s) * abs(r.mean_score - r.mover_rate) for r in tab.itertuples())) if prob_like else float("nan")
    rho = float(spearmanr(tab["bin"], tab["mover_rate"])[0]) if tab["mover_rate"].std() > 0 else float("nan")
    return {"table": tab, "spearman": rho, "top_lift": float(tab["mover_rate"].iloc[-1] - y.mean()), "ece": ece, "base_rate": float(y.mean())}


def direction_by_conviction(panel: pd.DataFrame, spec: ValueSpec, now, *, arm: int = 1, mover_thr: float = 0.05, n_bins: int = 5, n_boot: int = 300,
                            seed: int = 0) -> pd.DataFrame:
    """Direction accuracy on movers by conviction |P(up) - 0.5|. A direction model that knows something is right more often when it
    is more sure; flat accuracy across conviction bins (and near 0.5 overall) is the signature of no direction signal - reported so
    the tier-3 component is not credited for what conviction cannot back up. Bins are equal-mass; intervals cluster by month."""
    if spec.direction is None:
        raise ValueError("spec has no direction columns")
    p = check_panel(panel, spec, now)
    m = p[p[spec.fwd].abs() >= mover_thr]
    cols = ["bin", "n", "conviction", "accuracy", "lo", "hi"]
    if len(m) < n_bins * 10:
        return pd.DataFrame(columns=cols)
    pu = m[spec.direction[arm]].to_numpy(float)
    conv = np.abs(pu - 0.5)
    correct = ((pu >= 0.5) == (m[spec.fwd].to_numpy() > 0)).astype(float)
    mo = pd.DatetimeIndex(m.index.get_level_values(0)).strftime("%Y-%m").to_numpy()
    rows = []
    for i, ch in enumerate(np.array_split(np.argsort(conv, kind="mergesort"), n_bins)):
        bm = TS.cluster_bootstrap_mean(correct[ch], mo[ch], n_boot=n_boot, seed=seed + i)
        rows.append({"bin": i, "n": int(len(ch)), "conviction": float(conv[ch].mean()), "accuracy": bm.mean, "lo": bm.lo, "hi": bm.hi})
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------------------------------------------
# tables, comparison of two decompositions, tail attribution
# ---------------------------------------------------------------------------------------------------------------
def component_table(dec: ValueDecomposition) -> pd.DataFrame:
    """The eight components as rows, in the contract's order, with tier, base, new, delta, interval, n and status."""
    rows = [{"component": n, "tier": dec[n].tier, "base": dec[n].base, "new": dec[n].new, "delta": dec[n].delta, "lo": dec[n].lo, "hi": dec[n].hi,
             "n": dec[n].n, "status": dec[n].status} for n in COMPONENTS]
    return pd.DataFrame(rows)


def compare_decompositions(a: ValueDecomposition, b: ValueDecomposition) -> dict:
    """Component by component: did the delta of `b` separate from the delta of `a` (non-overlapping intervals)? Used to compare two
    learner versions on the same panel. Unmeasured components on either side are listed, never compared."""
    out = {"better": [], "worse": [], "same": [], "unmeasured": []}
    for n in COMPONENTS:
        ca, cb = a[n], b[n]
        if ca.status != STATUS_MEASURED or cb.status != STATUS_MEASURED or not all(math.isfinite(x) for x in (ca.lo, ca.hi, cb.lo, cb.hi)):
            out["unmeasured"].append(n)
        elif cb.lo > ca.hi:
            out["better"].append(n)
        elif cb.hi < ca.lo:
            out["worse"].append(n)
        else:
            out["same"].append(n)
    return out


def tail_attribution(panel: pd.DataFrame, spec: ValueSpec, now, *, k: int = 5, tail: float = 0.10, cost_bps: float = 5.0, min_names: int = 8,
                     seed: int = 0) -> dict:
    """What the new arm did in the base arm's worst weeks. The `tail` share of weeks with the lowest base return: mean base return,
    mean new return, and in how many of them the new arm was better. A change that lifts the average by giving up tail protection
    (or that fixes the tail and costs the average) shows here, not in the mean."""
    arms = run_arms(panel, spec, now, k=k, cost_bps=cost_bps, min_names=min_names, seed=seed)
    wb, wn = arms["base"][0], arms["new"][0]
    if len(wb) < 10:
        return {"n_tail": 0, "base_mean": float("nan"), "new_mean": float("nan"), "share_better": float("nan"), "average_gain": float("nan")}
    n_t = max(1, int(round(tail * len(wb))))
    idx = wb.nsmallest(n_t).index
    return {"n_tail": int(n_t), "base_mean": float(wb.loc[idx].mean()), "new_mean": float(wn.loc[idx].mean()),
            "share_better": float((wn.loc[idx] > wb.loc[idx]).mean()), "average_gain": float((wn - wb).mean())}


# ---------------------------------------------------------------------------------------------------------------
# section 34 completion: one measure at a time, and where predictive value is lost before the portfolio
# ---------------------------------------------------------------------------------------------------------------
def measure(name: str, panel: pd.DataFrame, spec: ValueSpec, now, **kw) -> ComponentValue:
    """One of the eight kinds of value on its own: predictive_effect, movement_prediction, ranking_value, selection_value,
    direction_value, timing_value, risk_value or portfolio_value. The value is measured exactly as in decompose_value (same code, same
    seed), so a separately measured component always equals the one in the full decomposition."""
    if name not in COMPONENTS:
        raise ValueError(f"unknown component {name!r}; choose from {COMPONENTS}")
    return decompose_value(panel, spec, now, **kw)[name]


@dataclass(frozen=True)
class LossStage:
    stage: str
    delta: float
    lo: float
    hi: float
    survives: bool                 # the gain is still significantly positive at this stage
    note: str = ""


@dataclass(frozen=True)
class ValueWaterfall:
    stages: tuple
    first_loss: str | None         # the first stage at which a gain that existed earlier is no longer significant
    lost_between: tuple | None     # (last surviving stage, first failing stage)
    verdict: str

    def render(self) -> str:
        lines = [f"{s.stage:<26} {'kept' if s.survives else 'LOST':<5} delta {s.delta:+.4f} [{s.lo:+.4f},{s.hi:+.4f}] {s.note}" for s in self.stages]
        return "\n".join(lines + [self.verdict])


def value_waterfall(panel: pd.DataFrame, spec: ValueSpec, now, *, k: int = 5, cost_bps: float = 5.0, min_names: int = 8, min_dates: int = 20,
                    n_boot: int = 400, seed: int = 0, window_weeks: int = 26) -> ValueWaterfall:
    """Where does predictive value leak before it reaches the portfolio? The same new-minus-base comparison is made stage by stage:
      1 prediction      rank IC of the score / movement signal
      2 ranking         NDCG@k against in-band moves
      3 selection       picks' in-band rate minus the universe rate
      4 gross weeks     the weekly in-band share of the simulated portfolio, before costs
      5 net weeks       the same after costs
      6 objective       the tiered-objective decision (lexicographic; direction cannot outrank tier 1 or 2)
    A stage `survives` when its paired interval is above zero (stage 6: when the firewall accepts). first_loss names the earliest stage
    that fails after an earlier one held, which is where prediction stopped being portfolio value."""
    p = check_panel(panel, spec, now)
    _check_spacing(p, spec)
    stages: list[LossStage] = []
    dec = decompose_value(p, spec, now, k=k, cost_bps=cost_bps, min_names=min_names, min_dates=min_dates, n_boot=n_boot, seed=seed, window_weeks=window_weeks)
    first = dec["movement_prediction"] if dec["movement_prediction"].status == STATUS_MEASURED else dec["predictive_effect"]
    for label, c in (("1 prediction", first), ("2 ranking", dec["ranking_value"]), ("3 selection", dec["selection_value"])):
        ok = c.status == STATUS_MEASURED
        stages.append(LossStage(label, c.delta if ok else float("nan"), c.lo if ok else float("nan"), c.hi if ok else float("nan"), bool(ok and c.lo > 0), c.note if ok else c.status))
    for label, cost in (("4 gross weeks", 0.0), ("5 net weeks", cost_bps)):
        arms = run_arms(p, spec, now, k=k, cost_bps=cost, min_names=min_names, seed=seed)
        wb, wn = arms["base"][0], arms["new"][0]
        if len(wb) < min_dates:
            stages.append(LossStage(label, float("nan"), float("nan"), float("nan"), False, "too few weeks"))
            continue
        band = lambda w: ((w.abs() >= O.BAND[0]) & (w.abs() <= O.BAND[1])).astype(float)
        diff = (band(wn) - band(wb)).to_numpy()
        bm = TS.cluster_bootstrap_mean(diff, pd.DatetimeIndex(wb.index).strftime("%Y-%m").to_numpy(), n_boot=n_boot, seed=seed + 5)
        stages.append(LossStage(label, bm.mean, bm.lo, bm.hi, bool(bm.lo > 0), "in-band week share, new minus base"))
    acc = dec.portfolio_accept
    stages.append(LossStage("6 objective", float(dec["portfolio_value"].delta) if dec["portfolio_value"].status == STATUS_MEASURED else float("nan"),
                            dec["portfolio_value"].lo, dec["portfolio_value"].hi, bool(acc), dec.portfolio_reason))
    held = False
    first_loss, between = None, None
    for i, s in enumerate(stages):
        if s.survives:
            held = True
        elif held and first_loss is None:
            first_loss, between = s.stage, (next(x.stage for x in reversed(stages[:i]) if x.survives), s.stage)
    if not any(s.survives for s in stages):
        verdict = "no stage shows a reliable gain: nothing was lost because nothing was gained"
    elif not any(s.survives for s in stages[:3]):
        verdict = "a portfolio-level gain appears with NO prediction, ranking or selection gain behind it: treat it as luck or sizing until explained"
    elif first_loss is None:
        verdict = "the gain survives every stage to the tiered objective"
    else:
        verdict = f"predictive value is lost between {between[0]} and {between[1]}"
    return ValueWaterfall(tuple(stages), first_loss, between, verdict)


def conversion_rate(dec: ValueDecomposition) -> float:
    """Share of significant prediction-side gains (predictive, movement, ranking, selection) that coincide with an accepted portfolio
    change: 1.0 or 0.0 for one decomposition, NaN when no prediction gain exists. Averaged over decompositions it is the fraction of
    predictive improvements that ever became portfolio improvements."""
    n = dec.translation()["n_predictive_gains"]
    return float("nan") if n == 0 else float(bool(dec.portfolio_accept))


def risk_breakdown(panel: pd.DataFrame, spec: ValueSpec, now, *, k: int = 5, cost_bps: float = 5.0, min_names: int = 8, window_weeks: int = 26,
                   n_boot: int = 400, seed: int = 0) -> pd.DataFrame:
    """The tier-2 risk value taken apart: per-window worst-5% week, maximum drawdown, catastrophic-week rate and band overshoot, each as
    new minus base (positive = less risk) with a paired bootstrap interval over windows. A risk gain that comes from overshoot alone
    is not the same as one that comes from a shallower drawdown."""
    arms = run_arms(panel, spec, now, k=k, cost_bps=cost_bps, min_names=min_names, seed=seed)
    rb, rn = weeks_to_rows(arms["base"][0].to_numpy(), window_weeks), weeks_to_rows(arms["new"][0].to_numpy(), window_weeks)
    cols = ["measure", "base", "new", "delta", "lo", "hi", "n_windows"]
    if len(rb) < 2 or len(rb) != len(rn):
        return pd.DataFrame(columns=cols)
    rows = []
    for name, key, sign in (("worst_5pct_week", "worst5", 1.0), ("max_drawdown", "max_dd", 1.0), ("catastrophic_weeks", "cat_rate", -1.0), ("overshoot_weeks", "over_band", -1.0)):
        b, n = np.array([r[key] for r in rb]), np.array([r[key] for r in rn])
        d = sign * (n - b)
        bm = TS.cluster_bootstrap_mean(d, None, n_boot=n_boot, seed=seed + len(rows))
        rows.append({"measure": name, "base": float(b.mean()), "new": float(n.mean()), "delta": bm.mean, "lo": bm.lo, "hi": bm.hi, "n_windows": len(d)})
    return pd.DataFrame(rows, columns=cols)


def timing_split(panel: pd.DataFrame, spec: ValueSpec, now, *, k: int = 5, min_names: int = 8, seed: int = 0) -> dict:
    """Timing value separated from exposure level. For the new arm's exposure e_t and the picks' pre-cost return r_t:
    total = mean(e r) = mean(e) mean(r) + cov(e, r). `level` is the first term (being invested more or less on average),
    `timing` the second (being invested more when the picks pay). Only the second is timing skill. Needs an exposure pair."""
    if not spec.exposure or any(c not in panel for c in spec.exposure):
        return {"available": False, "level": float("nan"), "timing": float("nan"), "total": float("nan"), "n": 0}
    p = check_panel(panel, spec, now)
    sig = pick_signal(p, spec, 1)
    picks = top_k_picks(sig, k, np.random.default_rng(seed + 11), min_names)
    r = p.loc[picks, spec.fwd].groupby(level=0).mean()
    e = p[spec.exposure[1]].groupby(level=0).first().reindex(r.index).clip(0, 1)
    j = pd.concat([e.rename("e"), r.rename("r")], axis=1).dropna()
    if len(j) < 3:
        return {"available": False, "level": float("nan"), "timing": float("nan"), "total": float("nan"), "n": int(len(j))}
    cov = float(((j["e"] - j["e"].mean()) * (j["r"] - j["r"].mean())).mean())
    level = float(j["e"].mean() * j["r"].mean())
    return {"available": True, "level": level, "timing": cov, "total": float((j["e"] * j["r"]).mean()), "n": int(len(j))}


def conversion_summary(decs: Sequence[ValueDecomposition]) -> dict:
    """Over many decompositions (versions, eras, seeds): how often a significant prediction gain existed, and how often the portfolio
    objective accepted the change when it did. The gap is the rate at which predictive value never became portfolio value."""
    rates = [conversion_rate(d) for d in decs]
    have = [r for r in rates if np.isfinite(r)]
    return {"n": len(decs), "with_prediction_gain": len(have), "converted": int(sum(have)), "rate": float(np.mean(have)) if have else float("nan"),
            "lost": len(have) - int(sum(have))}


def picked_direction_value(panel: pd.DataFrame, spec: ValueSpec, now, *, k: int = 5, mover_thr: float = 0.05, n_boot: int = 400, min_names: int = 8,
                           seed: int = 0) -> dict:
    """Direction value measured on the names the portfolio actually picks (not the whole universe): accuracy of sign(P(up) - 0.5) on the
    picked movers, and the return difference between picked names the model calls up and those it calls down (a positive spread is
    tradeable direction; accuracy alone is not). Each new-minus-base with a month-cluster bootstrap. Tier 3, reported for the record."""
    if spec.direction is None or any(c not in panel for c in spec.direction):
        return {"available": False}
    p = check_panel(panel, spec, now)
    out = {"available": True}
    for arm, tag in ((0, "base"), (1, "new")):
        picks = top_k_picks(pick_signal(p, spec, arm), k, np.random.default_rng(seed + 11), min_names)
        q = p[picks]
        mov = q[q[spec.fwd].abs() >= mover_thr]
        pu = q[spec.direction[arm]]
        up = q[spec.fwd][pu >= 0.5]
        dn = q[spec.fwd][pu < 0.5]
        out[tag] = {"n_movers": int(len(mov)),
                    "accuracy": float(((mov[spec.direction[arm]] >= 0.5) == (mov[spec.fwd] > 0)).mean()) if len(mov) else float("nan"),
                    "spread": float(up.mean() - dn.mean()) if len(up) and len(dn) else float("nan")}
    months = lambda idx: pd.DatetimeIndex(idx.get_level_values(0)).strftime("%Y-%m").to_numpy()
    q1 = p[top_k_picks(pick_signal(p, spec, 1), k, np.random.default_rng(seed + 11), min_names)]
    mov = q1[q1[spec.fwd].abs() >= mover_thr]
    if len(mov) >= 10:
        hit = ((mov[spec.direction[1]] >= 0.5) == (mov[spec.fwd] > 0)).astype(float).to_numpy()
        out["accuracy_vs_coin"] = TS.cluster_bootstrap_mean(hit - 0.5, months(mov.index), n_boot=n_boot, seed=seed)
    else:
        out["accuracy_vs_coin"] = None
    return out


def value_by_group(panel: pd.DataFrame, spec: ValueSpec, labels: pd.Series, now, *, min_dates: int = 20, **kw) -> pd.DataFrame:
    """Component deltas per group (era, year, regime) as one table: rows = groups, columns = components, cells = delta or NaN when the
    component could not be measured in that group. Thin groups appear with NaN, never disappear."""
    by = decompose_by(panel, spec, labels, now, min_dates=min_dates, **kw)
    rows = []
    for g, d in by.items():
        rows.append({"group": g, **{n: (d[n].delta if d is not None and d[n].status == STATUS_MEASURED else float("nan")) for n in COMPONENTS}})
    return pd.DataFrame(rows, columns=["group"] + list(COMPONENTS))


def selection_over_luck(panel: pd.DataFrame, spec: ValueSpec, now, *, k: int = 5, n_sims: int = 200, min_names: int = 8, seed: int = 0) -> dict:
    """Selection value of BOTH arms against the same random-picker distribution: lift, percentile and z for base and new, and whether the
    new arm's advantage over the base arm exceeds the random pickers' own spread (so a lift that a coin-flipper reaches one time in five
    is not credited). Returns {'base': LuckFloor, 'new': LuckFloor, 'advantage': float, 'exceeds_luck_spread': bool}."""
    b = random_floor(panel, spec, now, arm=0, k=k, n_sims=n_sims, min_names=min_names, seed=seed)
    n = random_floor(panel, spec, now, arm=1, k=k, n_sims=n_sims, min_names=min_names, seed=seed)
    adv = n.arm_lift - b.arm_lift
    spread = max(n.random_sd, b.random_sd)
    return {"base": b, "new": n, "advantage": adv, "exceeds_luck_spread": bool(np.isfinite(spread) and adv > 2 * spread)}


def signal_quality_needed(panel: pd.DataFrame, spec: ValueSpec, now, *, skills: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8), k: int = 5,
                          n_boot: int = 120, seed: int = 0) -> pd.DataFrame:
    """Calibrate the decomposition on the panel's own structure: how much movement skill (correlation of the movement score with the true
    size) does the portfolio objective need before it accepts the change? Blends the panel's true |fwd| rank into the base movement
    signal at each skill level, re-runs the objective comparison, and reports the acceptance and in-band share. Answers: is a
    prediction gain of this size even capable of reaching the portfolio under this objective?"""
    if spec.move is None or spec.move[0] not in panel:
        raise ValueError("needs the base movement column")
    p = check_panel(panel, spec, now)
    truth = _rank(p[spec.fwd].abs()) / p[spec.fwd].groupby(level=0).transform("size")
    base = _rank(p[spec.move[0]]) / p[spec.move[0]].groupby(level=0).transform("size")
    rows = []
    for s in skills:
        q = p.copy()
        q[spec.move[1]] = s * truth + (1 - s) * base
        d = decompose_value(q, dataclasses.replace(spec, pick="move"), now, k=k, n_boot=n_boot, seed=seed)
        rows.append({"skill": float(s), "movement_delta": d["movement_prediction"].delta, "accepted": d.portfolio_accept, "reason": d.portfolio_reason})
    return pd.DataFrame(rows)


def significant_components(dec: ValueDecomposition) -> dict:
    """Components split by what their interval says: gained, lost, unchanged, unmeasured. Never summed: eight different currencies."""
    out = {"gained": [], "lost": [], "unchanged": [], "unmeasured": []}
    for n in COMPONENTS:
        c = dec[n]
        key = "unmeasured" if c.status != STATUS_MEASURED else "gained" if c.significant_gain else "lost" if c.significant_loss else "unchanged"
        out[key].append(n)
    return out
