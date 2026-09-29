"""Bible Phase 15 (Exit learner), serving the tiered objective of Phase 20 and the "no look-ahead, fills at next open" rules.

Competing exit families (week-end, fixed +10% target, trailing, volatility-scaled, pattern-failure, hybrids) are all
parameterisations of ONE daily-bar simulator (`run_exit`), so every family sees identical fills, gaps and costs.
Per stock type the candidates are compared OUT OF SAMPLE with a lexicographic (tiered) objective - never by raw
return - and a challenger only displaces the plain week-end exit when it wins a week-clustered bootstrap.
`stops.py` (Phase 16) reuses the simulator, the metrics, the selection machinery and the walk-forward here.

Fill model (per position, daily bars over one holding week, entry at the FIRST bar's open):
* resting orders whose level is fixed by information through the PREVIOUS close (target, stop, trailing level built
  from the prior highs) fill intraday at their level, or at the open if the open already gapped through it;
* if a stop and a target are both touched inside one bar the stop is assumed hit first (conservative);
* decisions taken at a close (pattern failure) fill at the NEXT session's open; the planned week-end / time exit is a
  market-on-close order placed in advance at the final bar's close;
* market-type fills (entry, gap, stop, failure, time) pay slippage; limit target fills do not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

BAND_LO, BAND_HI, WEEKLY_TARGET = 0.05, 0.10, 0.07
CATASTROPHE = -0.20
REASONS = ("time", "target", "gap_target", "stop", "gap_stop", "trail", "gap_trail", "fail")
R_TIME, R_TARGET, R_GAP_TARGET, R_STOP, R_GAP_STOP, R_TRAIL, R_GAP_TRAIL, R_FAIL = range(8)


# ------------------------------------------------------------------ data
@dataclass
class Paths:
    """Daily OHLC paths of N candidate positions over D sessions (entry = first bar's open).
    `week` is the entry date, `end` the date of the last bar (a position is only usable for learning once end <= as_of).
    `vol`/`atr` are DAILY fractions known at entry; `fail[i, d]` says the pattern was invalidated at close d."""
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    prev_close: np.ndarray
    vol: np.ndarray
    atr: np.ndarray
    week: np.ndarray
    end: np.ndarray
    kind: np.ndarray
    ticker: np.ndarray
    fail: np.ndarray | None = None
    weight: np.ndarray | None = None

    def __post_init__(self):
        for k in ("o", "h", "l", "c"):
            setattr(self, k, np.atleast_2d(np.asarray(getattr(self, k), float)))
        n = self.o.shape[0]
        for k in ("prev_close", "vol", "atr"):
            setattr(self, k, np.asarray(getattr(self, k), float).reshape(-1))
        self.week = np.asarray(self.week, "datetime64[D]").reshape(-1)
        self.end = np.asarray(self.end, "datetime64[D]").reshape(-1)
        self.kind = np.asarray(self.kind, object).reshape(-1)
        self.ticker = np.asarray(self.ticker, object).reshape(-1)
        if self.fail is not None:
            self.fail = np.asarray(self.fail, bool).reshape(n, -1)
        self.weight = np.full(n, 0.10) if self.weight is None else np.asarray(self.weight, float).reshape(-1)

    def __len__(self):
        return self.o.shape[0]

    @property
    def D(self):
        return self.o.shape[1]

    def take(self, idx) -> "Paths":
        idx = np.asarray(idx)
        return Paths(self.o[idx], self.h[idx], self.l[idx], self.c[idx], self.prev_close[idx], self.vol[idx], self.atr[idx],
                     self.week[idx], self.end[idx], self.kind[idx], self.ticker[idx],
                     None if self.fail is None else self.fail[idx], self.weight[idx])

    def until(self, as_of) -> "Paths":
        """Positions fully observed by `as_of` (last bar on or before it)."""
        return self.take(np.flatnonzero(self.end <= np.datetime64(as_of, "D")))

    def validate(self) -> list[str]:
        """Return a list of data problems (empty = clean). Bad bars silently make every exit look better or worse."""
        bad = []
        n, D = self.o.shape
        for k in ("h", "l", "c"):
            if getattr(self, k).shape != (n, D):
                bad.append(f"shape mismatch {k}")
        if bad:
            return bad
        for k in ("o", "h", "l", "c"):
            a = getattr(self, k)
            if not np.isfinite(a).all():
                bad.append(f"non-finite {k}: {int((~np.isfinite(a)).sum())}")
            elif (a <= 0).any():
                bad.append(f"non-positive {k}: {int((a <= 0).sum())}")
        if D:
            hi = np.maximum(self.o, self.c)
            lo = np.minimum(self.o, self.c)
            if (self.h < hi - 1e-9).any():
                bad.append(f"high below open/close in {int((self.h < hi - 1e-9).any(1).sum())} rows")
            if (self.l > lo + 1e-9).any():
                bad.append(f"low above open/close in {int((self.l > lo + 1e-9).any(1).sum())} rows")
        for k in ("prev_close", "vol", "atr"):
            a = getattr(self, k)
            if len(a) != n or not np.isfinite(a).all() or (a < 0).any():
                bad.append(f"bad {k}")
        if (self.end < self.week).any():
            bad.append("end before week")
        if len(self.kind) != n or len(self.ticker) != n or len(self.week) != n or len(self.end) != n:
            bad.append("metadata length mismatch")
        return bad

    def check(self):
        bad = self.validate()
        if bad:
            raise ValueError("invalid Paths: " + "; ".join(bad))
        return self


def pattern_fail_flags(scores, entry_score, keep_frac=0.5):
    """Pattern-failure flags from a per-close pattern score path: the thesis is invalid once the score falls below
    `keep_frac` of the score the position was entered with (scores must be computed as-of each close)."""
    s = np.asarray(scores, float)
    e = np.asarray(entry_score, float).reshape(-1, 1)
    return s < keep_frac * e


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 2.0      # per side, paid on every round trip
    slip_bps: float = 5.0     # on market-type fills only


@dataclass(frozen=True)
class ExitSpec:
    """One exit rule. Distances are fractions; `*_vol` are multiples of the position's weekly sigma
    (daily vol * sqrt(D)). If both a fixed and a vol distance are given the tighter one applies."""
    target: float | None = None
    target_vol: float | None = None
    trail: float | None = None
    trail_vol: float | None = None
    arm: float = 0.0             # trailing only arms after the high has gained this much
    stop: float | None = None
    stop_vol: float | None = None
    hold_days: int | None = None  # planned market-on-close exit after this many sessions
    fail_exit: bool = False

    def uses_fail(self):
        return self.fail_exit


def _dist(fixed, mult, sigma, n):
    a = np.full(n, np.nan)
    if fixed is not None:
        a = np.where(np.isnan(a), fixed, a)
    if mult is not None:
        m = mult * sigma
        a = np.where(np.isnan(a), m, np.minimum(a, m))
    return a


@dataclass
class ExitResult:
    net: np.ndarray            # per-position return after slippage and fees
    gross: np.ndarray
    days: np.ndarray           # sessions held (1..D)
    reason: np.ndarray         # index into REASONS
    stop_overshoot: np.ndarray  # fraction of entry lost BEYOND the stop level when a stop/trail filled (gap damage)
    D: int

    def take(self, idx):
        return ExitResult(self.net[idx], self.gross[idx], self.days[idx], self.reason[idx], self.stop_overshoot[idx], self.D)


def run_exit(p: Paths, spec: ExitSpec, cost: CostModel = CostModel(), stop_dist: np.ndarray | None = None) -> ExitResult:
    """Simulate `spec` on every position in `p`. `stop_dist` (N,) overrides spec.stop with a per-position distance
    (used by the stop engine, which learns distances per stock)."""
    n, D = p.o.shape
    if n == 0:
        z = np.zeros(0)
        return ExitResult(z, z, z.astype(int), z.astype(int), z, D)
    if spec.fail_exit and p.fail is None:
        raise ValueError("pattern-failure exit needs Paths.fail")
    slip, fee = cost.slip_bps / 1e4, cost.fee_bps / 1e4
    sigma = p.vol * np.sqrt(D)
    entry = p.o[:, 0] * (1 + slip)
    tgt_d = _dist(spec.target, spec.target_vol, sigma, n)
    tr_d = _dist(spec.trail, spec.trail_vol, sigma, n)
    st_d = _dist(spec.stop, spec.stop_vol, sigma, n) if stop_dist is None else np.asarray(stop_dist, float)
    tgt_lvl = entry * (1 + tgt_d)
    stop_lvl = entry * (1 - st_d)
    hold = D if spec.hold_days is None else int(min(max(spec.hold_days, 1), D))

    alive = np.ones(n, bool)
    px = np.full(n, np.nan)
    day = np.full(n, D, int)
    reason = np.full(n, R_TIME, int)
    over = np.zeros(n)
    pending = np.zeros(n, bool)
    peak = entry.copy()

    def fill(mask, price, d, code):
        px[mask] = price[mask] if np.ndim(price) else price
        day[mask] = d + 1
        reason[mask] = code
        alive[mask] = False

    for d in range(hold):
        o, h, l, c = p.o[:, d], p.h[:, d], p.l[:, d], p.c[:, d]
        m = alive & pending                                  # decided at last close -> this open
        fill(m, o * (1 - slip), d, R_FAIL)
        armed = peak >= entry * (1 + spec.arm)
        trail_lvl = np.where(armed, peak * (1 - tr_d), np.nan)   # built only from PRIOR highs
        lvl = np.fmax(stop_lvl, trail_lvl)
        is_trail = np.nan_to_num(trail_lvl, nan=-np.inf) > np.nan_to_num(stop_lvl, nan=-np.inf)
        gs = alive & (o <= lvl)
        gt = alive & ~gs & (o >= tgt_lvl)
        rest = alive & ~gs & ~gt
        isx = rest & (l <= lvl)
        itg = rest & ~isx & (h >= tgt_lvl)
        for mask, price, code_s, code_t in ((gs, o * (1 - slip), R_GAP_STOP, R_GAP_TRAIL), (isx, lvl * (1 - slip), R_STOP, R_TRAIL)):
            over_m = np.maximum(0.0, (lvl - price) / entry)
            fill(mask & ~is_trail, price, d, code_s)
            fill(mask & is_trail, price, d, code_t)
            over[mask] = over_m[mask]
        fill(gt, o, d, R_GAP_TARGET)
        fill(itg, tgt_lvl, d, R_TARGET)
        peak = np.where(alive, np.maximum(peak, h), peak)
        if d == hold - 1:
            fill(alive, c * (1 - slip), d, R_TIME)
        elif spec.fail_exit:
            pending = alive & p.fail[:, d]

    gross = px / entry - 1
    return ExitResult(gross - 2 * fee, gross, day, reason, over, D)


# ------------------------------------------------------------------ metrics
def trade_metrics(res: ExitResult, cost: CostModel = CostModel()) -> dict:
    """Per-trade Phase-15 measurements: hit rate, average gain/loss, tail loss, time held, early-exit rate, costs."""
    x = res.net
    if len(x) == 0:
        return {"n": 0}
    pos, neg = x[x > 0], x[x <= 0]
    k = max(1, int(np.ceil(0.05 * len(x))))
    tail = np.sort(x)[:k]
    return {"n": int(len(x)), "mean": float(x.mean()), "hit_rate": float((x > 0).mean()),
            "hit10": float((x >= 0.10).mean()),
            "avg_gain": float(pos.mean()) if len(pos) else 0.0, "avg_loss": float(neg.mean()) if len(neg) else 0.0,
            "tail5_mean": float(tail.mean()), "worst": float(x.min()), "cat_rate": float((x <= CATASTROPHE).mean()),
            "days_held": float(res.days.mean()), "early_exit_rate": float((res.days < res.D).mean()),
            "utilisation": float(res.days.mean() / res.D),   # share of the week capital stays deployed
            "cost_drag_bps": float(2 * cost.fee_bps + (cost.slip_bps * (1 + (~np.isin(res.reason, (R_TARGET, R_GAP_TARGET))).mean()))),
            "reasons": {r: float((res.reason == i).mean()) for i, r in enumerate(REASONS) if (res.reason == i).any()}}


@dataclass
class WeekTable:
    """Per-week sufficient statistics of one rule's trades; resampling weeks = indexing these arrays."""
    n: np.ndarray
    total: np.ndarray
    cat: np.ndarray
    hit10: np.ndarray
    pos: np.ndarray


def week_table(net: np.ndarray, week_codes: np.ndarray, n_weeks: int) -> WeekTable:
    b = lambda w: np.bincount(week_codes, weights=w, minlength=n_weeks)
    return WeekTable(b(np.ones(len(net))), b(net), b((net <= CATASTROPHE).astype(float)),
                     b((net >= 0.10).astype(float)), b((net > 0).astype(float)))


def band_metrics(t: WeekTable, order: np.ndarray | None = None) -> dict:
    """Weekly-band behaviour of the equal-weight portfolio of these trades (tier 1/2/3 inputs of Phase 20)."""
    if order is not None:
        t = WeekTable(*(a[order] for a in (t.n, t.total, t.cat, t.hit10, t.pos)))
    keep = t.n > 0
    if not keep.any():
        return {"weeks": 0}
    n = t.n[keep]
    wk = t.total[keep] / n
    in_band = (wk >= BAND_LO) & (wk <= BAND_HI)
    k = max(1, int(np.ceil(0.05 * len(wk))))
    eq = np.cumprod(1 + wk)
    dd = 1 - eq / np.maximum.accumulate(np.maximum(eq, 1.0))
    return {"weeks": int(len(wk)), "mean_week": float(wk.mean()), "mean_dev": float(abs(wk.mean() - WEEKLY_TARGET)),
            "in_band": float(in_band.mean()), "below_band": float((wk < BAND_LO).mean()),
            "cvar5": float(np.sort(wk)[:k].mean()), "maxdd": float(dd.max()),
            "cat_rate": float(t.cat[keep].sum() / n.sum()), "pos_weeks": float((wk > 0).mean()),
            "hit10": float(t.hit10[keep].sum() / n.sum()), "precision": float(t.pos[keep].sum() / n.sum())}


# (tier, metric, sign: +1 higher is better, tolerance below which a difference is a tie)
TIERS = ((1, "mean_dev", -1, 0.005), (1, "in_band", +1, 0.03),
         (2, "cvar5", +1, 0.01), (2, "maxdd", -1, 0.02), (2, "cat_rate", -1, 0.005), (2, "below_band", -1, 0.03),
         (3, "pos_weeks", +1, 0.03), (3, "hit10", +1, 0.01), (3, "precision", +1, 0.01))


def compare(a: dict, b: dict, tiers=TIERS) -> int:
    """Lexicographic comparison (+1 a better, -1 b better, 0 tie). A lower tier is only consulted when every higher
    metric is a tie within tolerance, so a tier-3 gain can never buy a tier-1/2 loss. Raw return is NOT an input."""
    if a.get("weeks", 0) == 0 or b.get("weeks", 0) == 0:
        return 0
    for _, name, sign, tol in tiers:
        d = (a[name] - b[name]) * sign
        if d > tol:
            return 1
        if d < -tol:
            return -1
    return 0


def bootstrap_win(net_a, net_b, week_codes, n_weeks, rng: np.random.Generator, B=200, tiers=TIERS) -> float:
    """Share of week-resamples (ties excluded) in which rule A beats rule B under the tiered objective.
    Weeks, not trades, are resampled: positions in one week share the market."""
    ta, tb = week_table(net_a, week_codes, n_weeks), week_table(net_b, week_codes, n_weeks)
    wins = losses = 0
    for _ in range(B):
        order = np.sort(rng.integers(0, n_weeks, n_weeks))
        r = compare(band_metrics(ta, order), band_metrics(tb, order), tiers)
        wins += r > 0
        losses += r < 0
    return wins / (wins + losses) if wins + losses else 0.5


# ------------------------------------------------------------------ rules and selection
class Rule:
    """A candidate exit/stop rule. `fit` returns a rule fitted on training positions only (default: stateless)."""
    name = "rule"
    family = "rule"

    def fit(self, train: Paths) -> "Rule":
        return self

    def run(self, p: Paths) -> ExitResult:
        raise NotImplementedError


@dataclass
class SpecRule(Rule):
    spec: ExitSpec
    name: str = "week_end"
    family: str = "week_end"
    cost: CostModel = field(default_factory=CostModel)

    def run(self, p):
        return run_exit(p, self.spec, self.cost)


def default_rules(has_fail: bool = True, cost: CostModel = CostModel()) -> list[Rule]:
    """The competing families of Phase 15. Element 0 is always the plain week-end baseline."""
    R = lambda name, fam, **kw: SpecRule(ExitSpec(**kw), name, fam, cost)
    rules = [R("week_end", "week_end")]
    for t in (0.05, 0.07, 0.10, 0.15):
        rules.append(R(f"target_{int(t * 100)}", "fixed_target", target=t))
    for tr in (0.04, 0.06, 0.09):
        rules.append(R(f"trail_{int(tr * 100)}", "trailing", trail=tr, arm=0.03))
    for k in (0.75, 1.0, 1.5):
        rules.append(R(f"volt_{k}", "vol_scaled", target_vol=k))
    for k in (1.0, 1.5, 2.0):
        rules.append(R(f"voltrail_{k}", "vol_scaled", trail_vol=k, arm=0.0))
    for d in (3, 4):
        rules.append(R(f"time_{d}", "time", hold_days=d))
    rules.append(R("t10_trail6", "hybrid", target=0.10, trail=0.06, arm=0.03))
    rules.append(R("t10_trail4", "hybrid", target=0.10, trail=0.04, arm=0.05))
    rules.append(R("volt1_voltrail1.5", "hybrid", target_vol=1.0, trail_vol=1.5))
    rules.append(R("t10_time4", "hybrid", target=0.10, hold_days=4))
    if has_fail:
        rules.append(R("pattern_fail", "pattern_failure", fail_exit=True))
        rules.append(R("t10_fail", "hybrid", target=0.10, fail_exit=True))
        rules.append(R("t10_trail6_fail", "hybrid", target=0.10, trail=0.06, arm=0.03, fail_exit=True))
    return rules


@dataclass
class Selection:
    kind: str
    chosen: str
    family: str
    reason: str
    n_trades: int
    n_weeks: int
    win_rate: float | None = None
    stability: float | None = None
    chosen_metrics: dict | None = None
    baseline_metrics: dict | None = None
    table: pd.DataFrame | None = None     # every candidate's in-sample band metrics (diagnostic)


def _tournament(mets: list, ok: list) -> int:
    champ = 0
    for i in range(1, len(mets)):
        if ok[i] and compare(mets[i], mets[champ]) > 0:
            champ = i
    return champ


def selection_stability(tables: list, ok: list, champ: int, n_weeks: int, rng: np.random.Generator, B: int = 100) -> float:
    """Share of week-resamples in which the whole tournament (all candidates) again crowns `champ`. The bootstrap
    against the baseline alone ignores that the champion was picked out of ~25; a winner that only wins on the
    original weeks is selection noise."""
    hit = 0
    for _ in range(B):
        order = np.sort(rng.integers(0, n_weeks, n_weeks))
        hit += _tournament([band_metrics(t, order) for t in tables], ok) == champ
    return hit / B


def _week_codes(p: Paths):
    u, codes = np.unique(p.week, return_inverse=True)
    return codes, len(u)


def select_rule(train: Paths, rules: Sequence[Rule], seed: int = 0, min_weeks: int = 26, min_trades: int = 60,
                min_win: float = 0.65, boot: int = 200, min_stability: float = 0.25, feasible: Callable[[dict, dict], bool] | None = None,
                kind: str = "all") -> tuple[Rule, Selection]:
    """Pick one rule for one stock type from TRAINING positions only. Falls back to rules[0] (the baseline)
    when data are thin, when nothing beats it under the tiered objective, or when the win is not week-robust."""
    base = rules[0]
    codes, W = _week_codes(train) if len(train) else (np.zeros(0, int), 0)
    if len(train) < min_trades or W < min_weeks:
        return base, Selection(kind, base.name, base.family, "neutral:insufficient_data", len(train), W)
    fitted = [r.fit(train) for r in rules]
    res = [f.run(train) for f in fitted]
    mets = [band_metrics(week_table(r.net, codes, W)) for r in res]
    tmets = [trade_metrics(r) for r in res]
    ok = [True if feasible is None or i == 0 else bool(feasible(mets[i], tmets[i])) for i in range(len(rules))]
    champ = _tournament(mets, ok)
    tab = pd.DataFrame([{"rule": rules[i].name, "family": rules[i].family, "feasible": ok[i], **{k: v for k, v in mets[i].items()
                        if k != "weeks"}, "mean_trade": tmets[i]["mean"], "worst": tmets[i]["worst"]} for i in range(len(rules))])
    sel = Selection(kind, base.name, base.family, "baseline_best", len(train), W, chosen_metrics=mets[0],
                    baseline_metrics=mets[0], table=tab)
    if champ == 0:
        return fitted[0], sel
    win = bootstrap_win(res[champ].net, res[0].net, codes, W, np.random.default_rng(seed), boot)
    sel.win_rate = win
    if win < min_win:
        sel.reason = "baseline_kept:not_week_robust"
        return fitted[0], sel
    tabs = [week_table(r.net, codes, W) for r in res]
    sel.stability = selection_stability(tabs, ok, champ, W, np.random.default_rng(seed + 1), max(20, boot // 2))
    if sel.stability < min_stability:
        sel.reason = "baseline_kept:unstable_choice"
        return fitted[0], sel
    sel.chosen, sel.family, sel.reason, sel.chosen_metrics = rules[champ].name, rules[champ].family, "selected", mets[champ]
    return fitted[champ], sel


@dataclass
class Policy:
    """Per-type chosen rules learned as of `as_of`; unknown types use the baseline."""
    by_kind: dict
    selections: dict
    baseline: Rule
    as_of: np.datetime64 | None = None

    def apply(self, p: Paths) -> ExitResult:
        n = len(p)
        out = ExitResult(np.zeros(n), np.zeros(n), np.full(n, p.D if n else 0), np.zeros(n, int), np.zeros(n), p.D if n else 0)
        done = np.zeros(n, bool)
        for k in set(p.kind):
            m = p.kind == k
            r = self.by_kind.get(k, self.baseline).run(p.take(np.flatnonzero(m)))
            for a in ("net", "gross", "days", "reason", "stop_overshoot"):
                getattr(out, a)[m] = getattr(r, a)
            done |= m
        return out


def learn(paths: Paths, rules: Sequence[Rule], as_of, seed: int = 0, train_window_weeks: int | None = None,
          pool_fallback: bool = True, **kw) -> Policy:
    """Learn one rule per stock type using only positions completed by `as_of`. A type with too little data of its own
    borrows the rule selected on all types pooled - but only if that pooled rule itself passed every gate; otherwise
    it is neutralised to the baseline (never invent reliability from insufficient data)."""
    tr = paths.until(as_of)
    if train_window_weeks and len(tr):
        cut = tr.week.max() - np.timedelta64(7 * train_window_weeks, "D")
        tr = tr.take(np.flatnonzero(tr.week > cut))
    by, sels = {}, {}
    for j, k in enumerate(sorted(set(tr.kind))):
        r, s = select_rule(tr.take(np.flatnonzero(tr.kind == k)), rules, seed + j, kind=k, **kw)
        by[k], sels[k] = r, s
    thin = [k for k, s in sels.items() if s.reason == "neutral:insufficient_data"]
    if pool_fallback and thin and len(tr):
        pr, ps = select_rule(tr, rules, seed + 9999, kind="POOLED", **kw)
        if ps.reason == "selected":
            for k in thin:
                by[k] = pr
                sels[k] = Selection(k, ps.chosen, ps.family, "pooled_fallback", sels[k].n_trades, sels[k].n_weeks, ps.win_rate,
                                    ps.stability, ps.chosen_metrics, ps.baseline_metrics)
    return Policy(by, sels, rules[0], np.datetime64(as_of, "D"))


@dataclass
class WalkForward:
    oos: ExitResult              # learned policy, positions in the test blocks (others zero/NaN-free but flagged)
    base: ExitResult
    tested: np.ndarray           # bool mask of positions that were out of sample
    fold_choices: list           # (block_start, {kind: chosen name})
    paths: Paths

    def report(self) -> pd.DataFrame:
        """Per stock type (and ALL): learned vs baseline out-of-sample, with the tiered verdict."""
        rows, p = [], self.paths
        for k in sorted(set(p.kind[self.tested])) + ["ALL"]:
            m = self.tested & ((p.kind == k) if k != "ALL" else True)
            if not m.any():
                continue
            sub = p.take(np.flatnonzero(m))
            codes, W = _week_codes(sub)
            i = np.flatnonzero(m)
            a, b = self.oos.take(i), self.base.take(i)
            ba, bb = band_metrics(week_table(a.net, codes, W)), band_metrics(week_table(b.net, codes, W))
            ta, tb = trade_metrics(a), trade_metrics(b)
            v = compare(ba, bb)
            rows.append({"kind": k, "n": int(m.sum()), "weeks": W,
                         "verdict": {1: "better", -1: "worse", 0: "same"}[v],
                         **{f"learned_{x}": ba.get(x) for x in ("mean_week", "in_band", "cvar5", "maxdd", "below_band")},
                         **{f"base_{x}": bb.get(x) for x in ("mean_week", "in_band", "cvar5", "maxdd", "below_band")},
                         "learned_hit": ta["hit_rate"], "base_hit": tb["hit_rate"], "learned_avg_gain": ta["avg_gain"],
                         "learned_avg_loss": ta["avg_loss"], "learned_tail5": ta["tail5_mean"], "base_tail5": tb["tail5_mean"],
                         "learned_days": ta["days_held"], "learned_early_exit": ta["early_exit_rate"]})
        return pd.DataFrame(rows)


def walk_forward(paths: Paths, rules: Sequence[Rule], min_train_weeks: int = 52, block_weeks: int = 13, seed: int = 0,
                 embargo_days: int = 0, **kw) -> WalkForward:
    """Expanding-window OOS test: each block of `block_weeks` weeks is traded with a policy learned only from
    positions that finished at least `embargo_days` before the block's first entry."""
    n = len(paths)
    weeks = np.unique(paths.week)
    base_res = rules[0].run(paths)
    net = np.zeros(n)
    oos = ExitResult(net, np.zeros(n), np.full(n, paths.D), np.zeros(n, int), np.zeros(n), paths.D)
    tested = np.zeros(n, bool)
    choices = []
    for j, s in enumerate(range(min_train_weeks, len(weeks), block_weeks)):
        blk = weeks[s:s + block_weeks]
        start = blk[0]
        as_of = start - np.timedelta64(1 + embargo_days, "D")
        pol = learn(paths, rules, as_of, seed + 1000 * j, **kw)
        idx = np.flatnonzero(np.isin(paths.week, blk))
        r = pol.apply(paths.take(idx))
        for a in ("net", "gross", "days", "reason", "stop_overshoot"):
            getattr(oos, a)[idx] = getattr(r, a)
        tested[idx] = True
        choices.append((str(start), {k: v.chosen for k, v in pol.selections.items()}))
    return WalkForward(oos, base_res, tested, choices, paths)


def log_walk_forward(wf: WalkForward, cfg=None, seed=None, event="exit_walk_forward"):
    """Append the OOS report to the experiment registry (provenance-stamped, append-only)."""
    from .improve import log_experiment
    rep = wf.report()
    log_experiment({"event": event, "report": rep.to_dict("records"), "n_folds": len(wf.fold_choices)}, cfg, seed)
    return rep


# ------------------------------------------------------------------ diagnostics
def evaluate_rules(p: Paths, rules: Sequence[Rule], fit_on: Paths | None = None) -> pd.DataFrame:
    """Every Phase-15 measurement for every candidate on `p`, one row per rule. If `fit_on` is given, rules are fitted
    there (so `p` is a true holdout); otherwise on `p` itself (in-sample, label accordingly)."""
    codes, W = _week_codes(p) if len(p) else (np.zeros(0, int), 0)
    rows = []
    for r in rules:
        f = r.fit(fit_on if fit_on is not None else p)
        res = f.run(p)
        tm = trade_metrics(res)
        if tm["n"] == 0:
            continue
        bm = band_metrics(week_table(res.net, codes, W))
        rows.append({"rule": r.name, "family": r.family, **{k: v for k, v in tm.items() if k != "reasons"},
                     **{k: v for k, v in bm.items() if k not in ("weeks",)}, "weeks": W})
    return pd.DataFrame(rows)


def audit_no_lookahead(p: Paths, spec: ExitSpec, cost: CostModel = CostModel(), stop_dist=None) -> int:
    """Runtime look-ahead audit: overwrite every bar AFTER each position's exit with garbage and re-run. Any position
    whose result moves has used future data. Returns the number of violating positions (must be 0)."""
    base = run_exit(p, spec, cost, stop_dist)
    q = p.take(np.arange(len(p)))
    for i in range(len(q)):
        d = base.days[i]
        if 0 < d < q.D:
            q.o[i, d:], q.h[i, d:], q.l[i, d:], q.c[i, d:] = q.o[i, 0] * 7.0, q.o[i, 0] * 9.0, q.o[i, 0] * 0.1, q.o[i, 0] * 5.0
            if q.fail is not None:
                q.fail[i, d:] = ~q.fail[i, d:]
    again = run_exit(q, spec, cost, stop_dist)
    return int((~np.isclose(base.net, again.net) | (base.days != again.days)).sum())


def cost_sensitivity(p: Paths, spec: ExitSpec, bps=(0, 5, 10, 25, 50)) -> pd.DataFrame:
    """Mean/hit-rate of one rule as round-trip friction rises: an exit family that only works at zero cost is not real.
    Early-exit rules trade more, so they degrade faster than week-end."""
    rows = []
    for b in bps:
        res = run_exit(p, spec, CostModel(fee_bps=b / 2, slip_bps=b))
        rows.append({"friction_bps": b, "mean": float(res.net.mean()) if len(res.net) else np.nan,
                     "hit_rate": float((res.net > 0).mean()) if len(res.net) else np.nan})
    return pd.DataFrame(rows)


def oos_gain_interval(wf: WalkForward, kind: str | None = None, B: int = 400, seed: int = 0, level: float = 0.90) -> dict:
    """Week-clustered bootstrap interval for (learned - baseline) mean weekly portfolio return out of sample.
    An interval straddling zero means the learner has not shown a gain, whatever the point estimate."""
    m = wf.tested if kind is None else wf.tested & (wf.paths.kind == kind)
    if not m.any():
        return {"n_weeks": 0}
    i = np.flatnonzero(m)
    codes, W = _week_codes(wf.paths.take(i))
    a = week_table(wf.oos.net[i], codes, W)
    b = week_table(wf.base.net[i], codes, W)
    diff = a.total / a.n - b.total / b.n
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, W, W)].mean() for _ in range(B)])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"n_weeks": int(W), "gain": float(diff.mean()), "lo": float(lo), "hi": float(hi), "level": level,
            "significant": bool(lo > 0 or hi < 0)}


def era_breakdown(wf: WalkForward, years_per_era: int = 1) -> pd.DataFrame:
    """Out-of-sample learned vs baseline by calendar era (and by type inside each era). A gain that lives in one era
    (e.g. a crash year) is a regime bet, not an exit rule."""
    m = wf.tested
    if not m.any():
        return pd.DataFrame()
    p = wf.paths
    yr = p.week.astype("datetime64[Y]").astype(int) + 1970
    era = (yr // years_per_era) * years_per_era
    rows = []
    for e in sorted(set(era[m])):
        for k in ["ALL"] + sorted(set(p.kind[m & (era == e)])):
            sel = m & (era == e) & ((p.kind == k) if k != "ALL" else True)
            i = np.flatnonzero(sel)
            if len(i) < 20:
                continue
            codes, W = _week_codes(p.take(i))
            a, b = (band_metrics(week_table(x.net[i], codes, W)) for x in (wf.oos, wf.base))
            rows.append({"era": int(e), "kind": k, "n": len(i), "weeks": W, "verdict": {1: "better", -1: "worse", 0: "same"}[compare(a, b)],
                         "base_mean_week": b["mean_week"], "learned_mean_week": a["mean_week"], "base_cvar5": b["cvar5"],
                         "learned_cvar5": a["cvar5"], "base_in_band": b["in_band"], "learned_in_band": a["in_band"]})
    return pd.DataFrame(rows)


def rule_usage(wf: WalkForward) -> pd.DataFrame:
    """How often each rule was chosen per type across folds (instability = the choice is noise)."""
    rows = [{"kind": k, "rule": v} for _, ch in wf.fold_choices for k, v in ch.items()]
    if not rows:
        return pd.DataFrame(columns=["kind", "rule", "folds", "share"])
    t = pd.DataFrame(rows).groupby(["kind", "rule"]).size().rename("folds").reset_index()
    t["share"] = t["folds"] / t.groupby("kind")["folds"].transform("sum")
    return t.sort_values(["kind", "folds"], ascending=[True, False]).reset_index(drop=True)


def format_report(wf: WalkForward) -> str:
    """Plain-text summary for the report pages: what was chosen per fold and how it did out of sample."""
    rep = wf.report()
    if rep.empty:
        return "Exit learner: no out-of-sample blocks (not enough history)."
    out = ["Exit learner, out of sample vs plain week-end exit (tiered objective, not raw return)"]
    for r in rep.itertuples():
        ci = oos_gain_interval(wf, None if r.kind == "ALL" else r.kind)
        out.append(f"{r.kind:>12}: n={r.n} weeks={r.weeks} verdict={r.verdict}; mean week {r.base_mean_week:+.2%} -> "
                   f"{r.learned_mean_week:+.2%}; in-band {r.base_in_band:.0%} -> {r.learned_in_band:.0%}; worst-5% week "
                   f"{r.base_cvar5:+.2%} -> {r.learned_cvar5:+.2%}; gain {ci['gain']:+.2%} [{ci['lo']:+.2%}, {ci['hi']:+.2%}]")
    last = wf.fold_choices[-1][1] if wf.fold_choices else {}
    out.append("latest rules: " + ", ".join(f"{k}={v}" for k, v in sorted(last.items())))
    return "\n".join(out)
