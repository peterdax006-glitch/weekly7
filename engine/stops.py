"""Bible Phase 16 (Stop / loss engine), under the tiered objective of Phase 20.

CRITICAL RULE (Phase 16): nothing here claims a stop guarantees a -20% maximum loss. Positions are held overnight,
fills happen at the next open, and a gap can jump straight through any stop. So the engine measures instead:
P(loss > 20%) with a Wilson interval, expected loss, worst observed loss, gap frequency and gap severity, and how
much damage gaps did BEYOND the stop level. Catastrophic exposure is reduced by position sizing and candidate
filtering, both learned from training data only.

Candidate stop systems (all `Rule`s of exits.py, so they share its simulator, tiered selection and walk-forward):
ATR-based, volatility-percentile (learned adverse-excursion quantile per vol bucket), stock-type specific (fixed
distance chosen per type by the selector), gap-aware (distance shrunk by the name's learned gap tail), position-size-
aware (distance capped by loss budget / weight), pattern invalidation (needs Paths.fail) and a hybrid of these.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .exits import (CATASTROPHE, R_GAP_STOP, R_GAP_TRAIL, R_STOP, R_TRAIL, R_TIME, CostModel, ExitResult, ExitSpec, Paths, Rule,
                    SpecRule, run_exit, trade_metrics, walk_forward, WalkForward, band_metrics, week_table, _week_codes)

MIN_DIST, MAX_DIST = 0.03, 0.30
NO_GUARANTEE = ("No stop guarantees a maximum loss: positions gap overnight and stops fill at the next open. "
                "Read p_loss_gt20 (with its upper bound) and worst_observed, not a promised floor.")


# ------------------------------------------------------------------ gap and loss measurement
def adverse_gaps(p: Paths) -> np.ndarray:
    """(N, D) overnight adverse gap per session as a positive fraction of the previous close (0 when the open gapped up)."""
    prev = np.concatenate([p.prev_close[:, None], p.c[:, :-1]], axis=1)
    return np.maximum(0.0, 1 - p.o / prev)


def gap_stats(p: Paths, thr: float = 0.03) -> dict:
    """Gap frequency (share of position-nights with an adverse gap >= thr) and severity of those gaps."""
    g = adverse_gaps(p)
    if g.size == 0:
        return {"nights": 0}
    big = g[g >= thr]
    return {"nights": int(g.size), "freq": float((g >= thr).mean()), "freq_gt10": float((g >= 0.10).mean()),
            "freq_gt20": float((g >= 0.20).mean()), "severity_mean": float(big.mean()) if len(big) else 0.0,
            "severity_q95": float(np.quantile(g, 0.95)), "severity_max": float(g.max())}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, mid - half)), float(min(1.0, mid + half))


def loss_risk(res: ExitResult, target: float = -0.20) -> dict:
    """Phase-16 evaluation of a stop system. `guarantee` is always False by design."""
    x = res.net
    n = len(x)
    if n == 0:
        return {"n": 0, "guarantee": False, "note": NO_GUARANTEE}
    k = int((x <= target).sum())
    lo, hi = wilson(k, n)
    stopped = np.isin(res.reason, (R_STOP, R_GAP_STOP, R_TRAIL, R_GAP_TRAIL))
    gapped = np.isin(res.reason, (R_GAP_STOP, R_GAP_TRAIL))
    losers = x[x < 0]
    tail = np.sort(x)[:max(1, int(np.ceil(0.05 * n)))]
    return {"n": n, "p_loss_gt20": k / n, "p_loss_gt20_lo": lo, "p_loss_gt20_hi": hi, "expected_loss": float(np.minimum(x, 0).mean()),
            "mean_loser": float(losers.mean()) if len(losers) else 0.0, "es5": float(tail.mean()), "worst_observed": float(x.min()),
            "stopped_share": float(stopped.mean()), "gap_through_share": float(gapped.sum() / max(1, stopped.sum())),
            "overshoot_mean": float(res.stop_overshoot[stopped].mean()) if stopped.any() else 0.0,
            "overshoot_max": float(res.stop_overshoot.max()), "guarantee": False, "note": NO_GUARANTEE}


def make_feasible(max_p_cat_hi: float = 0.05):
    """Selector constraint: the UPPER Wilson bound of P(loss>20%) must be within budget, else the rule is ineligible."""
    def f(band: dict, tm: dict) -> bool:
        n = tm.get("n", 0)
        return n > 0 and wilson(int(round(tm["cat_rate"] * n)), n)[1] <= max_p_cat_hi
    return f


# ------------------------------------------------------------------ gap model, sizing, filtering
@dataclass
class GapModel:
    """Learned tail of adverse overnight gaps per ticker, shrunk toward its stock type, then toward everything."""
    q: float
    ticker: dict
    kind: dict
    overall: float
    shrink: float = 40.0

    def tail(self, p: Paths) -> np.ndarray:
        out = np.empty(len(p))
        for i, (t, k) in enumerate(zip(p.ticker, p.kind)):
            base = self.kind.get(k, self.overall)
            if t in self.ticker:
                v, n = self.ticker[t]
                out[i] = (n * v + self.shrink * base) / (n + self.shrink)
            else:
                out[i] = base
        return out


def fit_gap_model(train: Paths, q: float = 0.95, shrink: float = 40.0) -> GapModel:
    g = adverse_gaps(train)
    if g.size == 0:
        return GapModel(q, {}, {}, 0.0, shrink)
    overall = float(np.quantile(g, q))
    kinds = {k: float(np.quantile(g[train.kind == k], q)) for k in set(train.kind)}
    tick = {}
    for t in set(train.ticker):
        m = train.ticker == t
        tick[t] = (float(np.quantile(g[m], q)), int(g[m].size))
    return GapModel(q, tick, kinds, overall, shrink)


def size_positions(p: Paths, dist: np.ndarray, gap_tail: np.ndarray, loss_budget: float = 0.02, max_weight: float = 0.25,
                   gap_mult: float = 1.0) -> np.ndarray:
    """Weight per position so that a stop that fills AFTER a typical bad gap (stop distance + gap tail) costs at most
    `loss_budget` of the portfolio. This is exposure control, not a guarantee."""
    worst = np.maximum(dist + gap_mult * gap_tail, 1e-6)
    return np.minimum(max_weight, loss_budget / worst)


def filter_candidates(p: Paths, model: GapModel, max_tail: float = 0.12) -> np.ndarray:
    """Keep-mask: drop names whose learned gap tail alone is too large for any stop to matter."""
    return model.tail(p) <= max_tail


def filter_effect(res: ExitResult, keep: np.ndarray) -> dict:
    """What candidate filtering did to catastrophic exposure on a given result set."""
    def part(x):
        return {"n": int(len(x)), "cat_rate": float((x <= CATASTROPHE).mean()) if len(x) else 0.0,
                "expected_loss": float(np.minimum(x, 0).mean()) if len(x) else 0.0, "worst": float(x.min()) if len(x) else 0.0}
    return {"all": part(res.net), "kept": part(res.net[keep]), "dropped": part(res.net[~keep]), "kept_share": float(keep.mean()) if len(keep) else 0.0}


def sizing_effect(p: Paths, res: ExitResult, dist: np.ndarray, gap_tail: np.ndarray, loss_budget: float = 0.02,
                  max_weight: float = 0.25) -> pd.DataFrame:
    """Weekly-portfolio effect of gap-aware sizing versus equal weights, holding gross exposure fixed (weighted average
    return per week). Reports the mean, the worst week and the mean of the worst 5% of weeks (the CVaR of config.CVAR_LIMIT)."""
    w_eq = np.ones(len(p))
    w_sz = size_positions(p, dist, gap_tail, loss_budget, max_weight)
    codes, W = _week_codes(p) if len(p) else (np.zeros(0, int), 0)
    rows = []
    for name, w in (("equal_weight", w_eq), ("gap_aware_sized", w_sz)):
        num = np.bincount(codes, weights=w * res.net, minlength=W)
        den = np.bincount(codes, weights=w, minlength=W)
        wk = num[den > 0] / den[den > 0]
        if len(wk) == 0:
            continue
        rows.append({"weights": name, "weeks": len(wk), "mean_week": float(wk.mean()), "worst_week": float(wk.min()),
                     "cvar5": float(np.sort(wk)[:max(1, int(np.ceil(0.05 * len(wk))))].mean()),
                     "weeks_below_-12pct": float((wk <= -0.12).mean()), "mean_weight": float(w.mean())})
    return pd.DataFrame(rows)


def gap_calibration(model: GapModel, test: Paths, bins: int = 5) -> pd.DataFrame:
    """Is the learned gap tail honest out of sample? Within bins of predicted tail, the share of nights whose adverse
    gap exceeds the prediction should be about 1 - q. Large excess = the model understates gap risk for those names."""
    if len(test) == 0:
        return pd.DataFrame()
    tail = model.tail(test)
    g = adverse_gaps(test)
    exceed = (g > tail[:, None]).mean(1)
    edges = np.unique(np.quantile(tail, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, tail, side="right") - 1, 0, len(edges) - 2)
    rows = [{"bin": j, "n": int((b == j).sum()), "predicted_tail_mean": float(tail[b == j].mean()),
             "exceed_rate": float(exceed[b == j].mean()), "target_rate": 1 - model.q,
             "excess": float(exceed[b == j].mean() - (1 - model.q))} for j in range(len(edges) - 1) if (b == j).any()]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ stop rules
def _clip(d):
    return np.clip(d, MIN_DIST, MAX_DIST)


@dataclass
class StopRule(Rule):
    """Base: subclasses define `dist(p)` (N,) and optionally `fail_exit`; the position exits on the stop or at week end."""
    name: str = "stop"
    family: str = "stop"
    cost: CostModel = field(default_factory=CostModel)
    fail_exit: bool = False
    skip_tail: float | None = None   # if set, positions whose learned gap tail exceeds this are not traded (cash)
    gap: GapModel | None = None

    def dist(self, p: Paths) -> np.ndarray:
        raise NotImplementedError

    def _fit_gap(self, train: Paths):
        return fit_gap_model(train)

    def fit(self, train: Paths):
        import copy
        r = copy.copy(self)
        r.gap = r._fit_gap(train)
        r._learn(train)
        return r

    def _learn(self, train: Paths):
        pass

    def run(self, p: Paths) -> ExitResult:
        res = run_exit(p, ExitSpec(fail_exit=self.fail_exit), self.cost, stop_dist=self.dist(p))
        if self.skip_tail is not None and self.gap is not None and len(p):
            skip = ~filter_candidates(p, self.gap, self.skip_tail)
            res.net[skip] = 0.0
            res.gross[skip] = 0.0
            res.days[skip] = 0
            res.reason[skip] = R_TIME
            res.stop_overshoot[skip] = 0.0
        return res


@dataclass
class AtrStop(StopRule):
    k: float = 3.0

    def dist(self, p):
        return _clip(self.k * p.atr)


@dataclass
class FixedStop(StopRule):
    """Stock-type specific: a fixed distance; the per-type selector picks which one each type gets."""
    d: float = 0.08

    def dist(self, p):
        return np.full(len(p), self.d)


@dataclass
class VolPercentileStop(StopRule):
    """Stop at the q-quantile of the maximum adverse excursion seen in training among positions of similar volatility."""
    q: float = 0.85
    buckets: int = 5
    edges: np.ndarray | None = None
    dists: np.ndarray | None = None

    def _learn(self, train: Paths):
        if len(train) == 0:
            self.edges, self.dists = np.array([]), np.array([0.10])
            return
        mae = 1 - train.l.min(1) / train.o[:, 0]
        self.edges = np.quantile(train.vol, np.linspace(0, 1, self.buckets + 1)[1:-1])
        b = np.searchsorted(self.edges, train.vol)
        allq = float(np.quantile(mae, self.q))
        self.dists = np.array([np.quantile(mae[b == j], self.q) if (b == j).sum() >= 20 else allq for j in range(self.buckets)])

    def dist(self, p):
        if self.edges is None:
            raise RuntimeError("VolPercentileStop must be fit before use")
        return _clip(self.dists[np.searchsorted(self.edges, p.vol)])


@dataclass
class GapAwareStop(StopRule):
    """ATR stop shrunk so that (stop distance + the name's learned gap tail) stays within `loss_target`."""
    k: float = 3.0
    loss_target: float = 0.15

    def dist(self, p):
        tail = self.gap.tail(p) if self.gap is not None else np.zeros(len(p))
        return _clip(np.minimum(self.k * p.atr, self.loss_target - tail))


@dataclass
class SizeAwareStop(StopRule):
    """ATR stop capped so weight * distance <= risk_budget of the portfolio."""
    k: float = 3.0
    risk_budget: float = 0.012

    def dist(self, p):
        return _clip(np.minimum(self.k * p.atr, self.risk_budget / np.maximum(p.weight, 1e-6)))


@dataclass
class InvalidationStop(StopRule):
    """Pattern invalidation exit plus a wide catastrophe stop (invalidation is not gap-proof either)."""
    k: float = 5.0
    fail_exit: bool = True

    def dist(self, p):
        return _clip(self.k * p.atr)


@dataclass
class HybridStop(StopRule):
    """Tightest of ATR, gap-aware and size-aware distances, optional invalidation exit, optional gap-tail filter."""
    k: float = 3.0
    loss_target: float = 0.15
    risk_budget: float = 0.012

    def dist(self, p):
        tail = self.gap.tail(p) if self.gap is not None else np.zeros(len(p))
        return _clip(np.minimum.reduce([self.k * p.atr, self.loss_target - tail, self.risk_budget / np.maximum(p.weight, 1e-6)]))


def default_stop_rules(has_fail: bool = True, cost: CostModel = CostModel()) -> list[Rule]:
    """Element 0 is the no-stop week-end baseline: a stop must earn its place under the tiered objective."""
    rules: list[Rule] = [SpecRule(ExitSpec(), "no_stop", "baseline", cost)]
    rules += [AtrStop(name=f"atr_{k}", family="atr", cost=cost, k=k) for k in (2.0, 3.0, 4.0)]
    rules += [VolPercentileStop(name=f"volpct_{q}", family="vol_percentile", cost=cost, q=q) for q in (0.75, 0.9)]
    rules += [FixedStop(name=f"fixed_{int(d * 100)}", family="type_specific", cost=cost, d=d) for d in (0.06, 0.10, 0.15)]
    rules += [GapAwareStop(name=f"gap_{k}", family="gap_aware", cost=cost, k=k) for k in (3.0, 4.0)]
    rules += [SizeAwareStop(name="size_aware", family="size_aware", cost=cost)]
    rules += [HybridStop(name="hybrid", family="hybrid", cost=cost),
              HybridStop(name="hybrid_filter", family="hybrid", cost=cost, skip_tail=0.12)]
    if has_fail:
        rules += [InvalidationStop(name="invalidation", family="invalidation", cost=cost),
                  HybridStop(name="hybrid_invalidation", family="hybrid", cost=cost, fail_exit=True)]
    return rules


# ------------------------------------------------------------------ walk-forward and report
def walk_forward_stops(paths: Paths, rules: Sequence[Rule] | None = None, max_p_cat_hi: float = 0.05, **kw) -> WalkForward:
    """exits.walk_forward with the stop rule set and the P(loss>20%) feasibility constraint."""
    rules = list(rules) if rules is not None else default_stop_rules(paths.fail is not None)
    return walk_forward(paths, rules, feasible=make_feasible(max_p_cat_hi), **kw)


def risk_report(wf: WalkForward) -> pd.DataFrame:
    """Per type and ALL: learned stops vs the no-stop baseline, out of sample, with gap frequency/severity of the data."""
    rows, p = [], wf.paths
    for k in sorted(set(p.kind[wf.tested])) + ["ALL"]:
        m = wf.tested & ((p.kind == k) if k != "ALL" else True)
        if not m.any():
            continue
        i = np.flatnonzero(m)
        a, b, g = loss_risk(wf.oos.take(i)), loss_risk(wf.base.take(i)), gap_stats(p.take(i))
        rows.append({"kind": k, "n": a["n"], **{f"learned_{x}": a[x] for x in ("p_loss_gt20", "p_loss_gt20_hi", "expected_loss", "es5",
                     "worst_observed", "stopped_share", "gap_through_share", "overshoot_max")},
                     **{f"base_{x}": b[x] for x in ("p_loss_gt20", "p_loss_gt20_hi", "expected_loss", "es5", "worst_observed")},
                     "gap_freq": g["freq"], "gap_severity": g["severity_mean"], "gap_max": g["severity_max"], "guarantee": False})
    return pd.DataFrame(rows)


def log_stop_walk_forward(wf: WalkForward, cfg=None, seed=None):
    from .improve import log_experiment
    rep = risk_report(wf)
    log_experiment({"event": "stop_walk_forward", "report": rep.to_dict("records"), "note": NO_GUARANTEE}, cfg, seed)
    return rep


# ------------------------------------------------------------------ diagnostics
def stop_distance_curve(p: Paths, dists=(0.04, 0.06, 0.08, 0.10, 0.15, 0.20), cost: CostModel = CostModel()) -> pd.DataFrame:
    """Trade-off table for fixed stop distances: how often the stop fires versus how much catastrophic exposure remains.
    The 'no stop' row is d=None. Shows why a tighter stop is not automatically safer (more whipsaw, same gaps)."""
    rows = []
    for d in (None, *dists):
        res = run_exit(p, ExitSpec(), cost, stop_dist=None if d is None else np.full(len(p), d))
        r = loss_risk(res)
        if r["n"]:
            rows.append({"stop": d, "mean": float(res.net.mean()), **{k: r[k] for k in ("p_loss_gt20", "p_loss_gt20_hi", "expected_loss",
                         "worst_observed", "stopped_share", "gap_through_share", "overshoot_mean")}})
    return pd.DataFrame(rows)


def type_gap_table(p: Paths, thr: float = 0.03) -> pd.DataFrame:
    """Gap frequency and severity per stock type: which types a stop cannot protect."""
    return pd.DataFrame([{"kind": k, **gap_stats(p.take(np.flatnonzero(p.kind == k)), thr)} for k in sorted(set(p.kind))])


def stop_era_breakdown(wf: WalkForward, years_per_era: int = 1) -> pd.DataFrame:
    """Learned stops vs no stop by calendar era: P(loss>20%), expected loss and worst loss out of sample."""
    m, p = wf.tested, wf.paths
    if not m.any():
        return pd.DataFrame()
    yr = p.week.astype("datetime64[Y]").astype(int) + 1970
    era = (yr // years_per_era) * years_per_era
    rows = []
    for e in sorted(set(era[m])):
        i = np.flatnonzero(m & (era == e))
        if len(i) < 20:
            continue
        a, b, g = loss_risk(wf.oos.take(i)), loss_risk(wf.base.take(i)), gap_stats(p.take(i))
        rows.append({"era": int(e), "n": len(i), "base_p_gt20": b["p_loss_gt20"], "learned_p_gt20": a["p_loss_gt20"],
                     "base_expected_loss": b["expected_loss"], "learned_expected_loss": a["expected_loss"],
                     "base_worst": b["worst_observed"], "learned_worst": a["worst_observed"], "gap_freq": g["freq"],
                     "gap_severity": g["severity_mean"]})
    return pd.DataFrame(rows)


def format_risk_report(wf: WalkForward) -> str:
    rep = risk_report(wf)
    if rep.empty:
        return "Stop engine: no out-of-sample blocks (not enough history).\n" + NO_GUARANTEE
    out = ["Stop engine, out of sample vs no stop", NO_GUARANTEE]
    for r in rep.itertuples():
        out.append(f"{r.kind:>12}: n={r.n}; P(loss>20%) {r.base_p_loss_gt20:.2%} -> {r.learned_p_loss_gt20:.2%} "
                   f"(upper {r.learned_p_loss_gt20_hi:.2%}); expected loss {r.base_expected_loss:+.2%} -> {r.learned_expected_loss:+.2%}; "
                   f"worst {r.base_worst_observed:+.1%} -> {r.learned_worst_observed:+.1%}; gap freq {r.gap_freq:.2%}, "
                   f"gap severity {r.gap_severity:.1%}, stopped {r.learned_stopped_share:.0%} of which through a gap {r.learned_gap_through_share:.0%}")
    return "\n".join(out)
