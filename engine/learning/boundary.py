"""Boundary learning: learn where a pattern STOPS working (contract C62 section 41; checklist C12 conditions, C13 anti-conditions).
IMPLEMENTED - NOT VALIDATED.

A pattern that is real on average is usually real only inside a region: volatility below a level, liquidity above a level, no
regime transition in progress, patterns agreeing with each other, no market shock in the last few weeks, sector exposure not
concentrated, its own recent reliability not deteriorating.  This module learns those edges as first-class knowledge:

    Boundary        one learned edge (feature, threshold, side that works, edge inside vs outside, evidence, out-of-time check)
    BoundarySet     all accepted boundaries of one pattern -> `contexts` (where it works) and `anti_contexts` (where it stops
                    or reverses) in the format every KnowledgeObject carries
    scope_status()  applies a context/anti-context pair to today's features: IN_SCOPE / OUT_OF_SCOPE / ANTI_HIT / UNKNOWN
                    (a missing feature is UNKNOWN, never silently 'in scope')

Discovery is deliberately conservative: the best split of each feature is found by an exhaustive scan, its significance is
judged against a circular-shift permutation null of the SAME scan (so the search over thresholds and features is paid for),
Holm-adjusted across features, then re-fitted on the older part of history and confirmed on the newer part.  Only boundaries
that survive both are accepted; the rest are recorded as rejected with the reason (that record is knowledge too).

Time rule (C56): the frame's dates are outcome-maturity dates and must all precede `now`; feature builders here only ever use
values known at the decision (trailing statistics are shifted one period).  Builds on engine.pattern_reliability.holm."""
from __future__ import annotations

import dataclasses as dc
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FirewallBreach, Lifecycle, Promotion, Provenance,
                                  Unknown, _StrEnum, as_date, current_code_hash, stable_hash)


class BoundaryError(ValueError):
    pass


class BoundaryKind(_StrEnum):                 # the seven contract examples map onto these
    THRESHOLD = "THRESHOLD"                   # volatility > x, liquidity < y, breadth, ...
    TRANSITION = "TRANSITION"                 # regime transition in progress
    DISAGREEMENT = "DISAGREEMENT"             # patterns disagree with one another
    SHOCK = "SHOCK"                           # market shock in the last k periods
    CONCENTRATION = "CONCENTRATION"           # sector / name concentration of the picks
    CONFIDENCE = "CONFIDENCE"                 # the pattern's own reliability deteriorating


class ScopeStatus(_StrEnum):
    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"             # a required context does not hold
    ANTI_HIT = "ANTI_HIT"                     # an anti-context holds: known-bad territory
    UNKNOWN = "UNKNOWN"                       # a needed feature is missing


class OutsideBehaviour(_StrEnum):
    STOPS = "STOPS"                           # outside effect indistinguishable from zero
    REVERSES = "REVERSES"                     # outside effect significantly opposite
    WEAKENS = "WEAKENS"                       # outside effect same sign but significantly smaller


@dc.dataclass(frozen=True)
class BoundaryConfig:
    min_side: int = 20                        # observations each side of a split
    n_perm: int = 199                         # circular-shift permutations for the scan null
    alpha: float = 0.05
    holdout_frac: float = 0.3                 # newest share of history reserved for out-of-time confirmation
    min_test: int = 30                        # holdout periods needed to confirm (else UNTESTED)
    oot_t: float = 1.28                       # one-sided t needed in the holdout, in the training direction
    min_test_side: int = 5
    zero_z: float = 1.64                      # |z| below this: outside edge is 'zero' (STOPS)
    max_missing: float = 0.4                  # features missing more than this are skipped
    n_boot: int = 40                          # bootstrap refits for threshold spread
    boot_block: int = 4
    min_shift_frac: float = 0.1               # permutation shifts stay this far from 0
    seed: int = 0


DEFAULT_BCFG = BoundaryConfig()


# ------------------------------------------------------------------------------------------------ conditions
def condition_holds(spec: Any, value: Any) -> bool | None:
    """Evaluate one condition spec against a feature value.  None = cannot tell (value missing/NaN).

    spec forms: {"op": "<="|"<"|">="|">"|"==", "value": x} | {"op": "between", "lo": a, "hi": b} | {"op": "in", "values": [...]}."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return None
    if not isinstance(spec, Mapping) or "op" not in spec:
        raise BoundaryError(f"malformed condition spec {spec!r}")
    op = spec["op"]
    if op == "in":
        return value in spec["values"]
    if op == "between":
        return spec["lo"] <= value <= spec["hi"]
    x = spec["value"]
    if op == "<=":
        return value <= x
    if op == "<":
        return value < x
    if op == ">=":
        return value >= x
    if op == ">":
        return value > x
    if op == "==":
        return value == x
    raise BoundaryError(f"unknown op {op!r}")


def _as_list(spec) -> list:
    return list(spec) if isinstance(spec, (list, tuple)) else [spec]


def scope_status(contexts: Mapping[str, Any], anti_contexts: Mapping[str, Any], ctx_now: Mapping[str, Any]) -> ScopeStatus:
    """Where does today's context sit relative to a pattern's learned region?  Known-bad beats everything; a required
    condition that fails is OUT_OF_SCOPE; a needed feature that is missing is UNKNOWN (never assumed fine)."""
    unknown = False
    for dim, spec in (anti_contexts or {}).items():
        for sp in _as_list(spec):
            h = condition_holds(sp, (ctx_now or {}).get(dim))
            if h is True:
                return ScopeStatus.ANTI_HIT
            if h is None:
                unknown = True
    for dim, spec in (contexts or {}).items():
        for sp in _as_list(spec):
            h = condition_holds(sp, (ctx_now or {}).get(dim))
            if h is False:
                return ScopeStatus.OUT_OF_SCOPE
            if h is None:
                unknown = True
    return ScopeStatus.UNKNOWN if unknown else ScopeStatus.IN_SCOPE


# ------------------------------------------------------------------------------------------------ split scan
@dc.dataclass(frozen=True)
class Split:
    t: float                                  # Welch t of (left mean - right mean)
    threshold: float
    n_left: int
    n_right: int
    mean_left: float
    mean_right: float
    se_left: float
    se_right: float


class _Scanner:
    """Exhaustive best-split search of an edge series on one sorted feature, in O(n) per evaluation (cumulative sums)."""

    def __init__(self, f: np.ndarray, min_side: int):
        self.order = np.argsort(f, kind="mergesort")
        self.fs = f[self.order]
        n = len(f)
        ks = np.arange(min_side, n - min_side + 1)
        if len(ks):
            ks = ks[self.fs[ks - 1] < self.fs[ks]]
        self.ks = ks
        self.n = n

    @property
    def empty(self) -> bool:
        return len(self.ks) == 0

    def _stats(self, e: np.ndarray):
        es = e[self.order]
        c1, c2 = np.cumsum(es), np.cumsum(es * es)
        ks, n = self.ks, self.n
        nl, nr = ks, n - ks
        sl = c1[ks - 1]
        sr = c1[-1] - sl
        ml, mr = sl / nl, sr / nr
        vl = np.maximum((c2[ks - 1] - nl * ml ** 2) / np.maximum(nl - 1, 1), 1e-18)
        vr = np.maximum(((c2[-1] - c2[ks - 1]) - nr * mr ** 2) / np.maximum(nr - 1, 1), 1e-18)
        sel, ser = np.sqrt(vl / nl), np.sqrt(vr / nr)
        t = (ml - mr) / np.sqrt(sel ** 2 + ser ** 2)
        return t, ml, mr, sel, ser

    def max_abs_t(self, e: np.ndarray) -> float:
        if self.empty:
            return 0.0
        return float(np.max(np.abs(self._stats(e)[0])))

    def best(self, e: np.ndarray) -> Split | None:
        if self.empty:
            return None
        t, ml, mr, sel, ser = self._stats(e)
        i = int(np.argmax(np.abs(t)))
        k = int(self.ks[i])
        thr = 0.5 * (self.fs[k - 1] + self.fs[k])
        return Split(float(t[i]), float(thr), k, self.n - k, float(ml[i]), float(mr[i]), float(sel[i]), float(ser[i]))


def best_split(e: np.ndarray, f: np.ndarray, min_side: int) -> Split | None:
    return _Scanner(np.asarray(f, float), min_side).best(np.asarray(e, float))


# ------------------------------------------------------------------------------------------------ boundary records
@dc.dataclass(frozen=True)
class Boundary:
    pattern_id: str
    feature: str
    kind: BoundaryKind
    threshold: float
    works_when: str                           # '<=' : pattern works at feature <= threshold ; '>' : at feature > threshold
    n_inside: int
    n_outside: int
    edge_inside: float
    edge_outside: float
    se_inside: float
    se_outside: float
    contrast_t: float
    perm_p: float                             # scan-null p of the best split of this feature
    p_adj: float                              # Holm across features
    threshold_sd: float                       # bootstrap spread of the threshold, in feature-IQR units (instability)
    oot_status: str                           # CONFIRMED / FAILED / UNTESTED
    oot_t: float
    oot_edge_inside: float
    oot_edge_outside: float
    outside: str                              # OutsideBehaviour value
    accepted: bool
    reasons: tuple[str, ...]
    learned_at: str

    @property
    def context(self) -> dict:
        """Where the pattern works, as a condition spec."""
        return {"op": self.works_when, "value": self.threshold}

    @property
    def anti_context(self) -> dict | None:
        """Where it is known to stop/reverse; None when it merely weakens (outside is then just less attractive)."""
        if self.outside == str(OutsideBehaviour.WEAKENS):
            return None
        return {"op": ">" if self.works_when == "<=" else "<=", "value": self.threshold}

    def inside_mask(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return x <= self.threshold if self.works_when == "<=" else x > self.threshold

    def describe(self) -> str:
        side = "at or below" if self.works_when == "<=" else "above"
        return (f"{self.pattern_id}: works {side} {self.feature}={self.threshold:.4g} "
                f"(edge {self.edge_inside:+.4f} in, {self.edge_outside:+.4f} out; outside it {self.outside}; "
                f"p_adj={self.p_adj:.3f}, out-of-time {self.oot_status})")


@dc.dataclass(frozen=True)
class BoundarySet:
    pattern_id: str
    learned_at: str
    n_periods: int
    boundaries: tuple[Boundary, ...]          # accepted only
    rejected: tuple[Boundary, ...]            # tested and refused: recorded, not forgotten
    skipped: tuple[tuple[str, str], ...]      # (feature, reason) never scanned
    overall_edge: float
    config_hash: str

    @property
    def contexts(self) -> dict:
        out: dict[str, Any] = {}
        for b in self.boundaries:
            out.setdefault(b.feature, []).append(b.context)
        return {k: (v[0] if len(v) == 1 else v) for k, v in out.items()}

    @property
    def anti_contexts(self) -> dict:
        out: dict[str, Any] = {}
        for b in self.boundaries:
            a = b.anti_context
            if a is not None:
                out.setdefault(b.feature, []).append(a)
        return {k: (v[0] if len(v) == 1 else v) for k, v in out.items()}

    def scope(self, ctx_now: Mapping[str, Any]) -> ScopeStatus:
        return scope_status(self.contexts, self.anti_contexts, ctx_now)

    @property
    def set_id(self) -> str:
        return stable_hash([self.pattern_id, self.learned_at, [b.feature + str(round(b.threshold, 8)) for b in self.boundaries]])

    def summary(self) -> str:
        if not self.boundaries:
            return f"{self.pattern_id}: no boundary accepted ({len(self.rejected)} rejected, {len(self.skipped)} skipped)"
        return "\n".join(b.describe() for b in self.boundaries)


# ------------------------------------------------------------------------------------------------ feature builders (past-only)
def shift_trailing(s: pd.Series, lag: int = 1) -> pd.Series:
    """The only sanctioned way to use a trailing statistic as a feature: shift so row t sees values up to t-lag."""
    return s.shift(lag)


def disagreement_from_scores(scores: pd.DataFrame) -> pd.Series:
    """Cross-pattern disagreement per decision date.  `scores`: rows = decision dates, columns = patterns, values = the
    signed score/direction each pattern gives.  0 = all patterns agree, 1 = evenly split.  Uses only that date's row."""
    sg = np.sign(scores.astype(float))
    active = sg.abs().sum(axis=1).replace(0, np.nan)
    net = sg.sum(axis=1).abs()
    return (1.0 - net / active).rename("disagreement")


def concentration_hhi(picks: pd.DataFrame, date_col: str = "date", group_col: str = "sector") -> pd.Series:
    """Herfindahl concentration of the picks per date from a long table (one row per pick)."""
    if picks is None or len(picks) == 0:
        return pd.Series(dtype=float, name="concentration")
    g = picks.groupby([date_col, group_col]).size().rename("n").reset_index()
    tot = g.groupby(date_col)["n"].transform("sum")
    g["sh2"] = (g["n"] / tot) ** 2
    return g.groupby(date_col)["sh2"].sum().rename("concentration")


def shock_features(market_ret: pd.Series, lookback: int = 52, z: float = 3.0, min_hist: int = 20) -> pd.DataFrame:
    """Market-shock state per period.  z-score of |return| against the TRAILING window (excludes the current period), then
    `since_shock` = periods since the last shock (large when calm).  Row t is known only after period t's return: use it
    as a feature for period t+1 via shift_trailing."""
    r = market_ret.astype(float)
    hist_sd = r.abs().rolling(lookback, min_periods=min_hist).std().shift(1)
    hist_mu = r.abs().rolling(lookback, min_periods=min_hist).mean().shift(1)
    zs = (r.abs() - hist_mu) / hist_sd.replace(0, np.nan)
    shock = (zs > z).astype(float)
    shock[zs.isna()] = np.nan
    since = np.full(len(r), np.nan)
    last = None
    for i, v in enumerate(shock.values):
        if v == 1.0:
            last = i
        since[i] = (i - last) if last is not None else (np.nan if np.isnan(v) else 999.0)
    return pd.DataFrame({"shock": shock, "since_shock": since}, index=r.index)


def regime_transition_features(regime: pd.Series) -> pd.DataFrame:
    """`since_regime_change`: periods since the regime label last changed (0 on the change period).  The label at row t must
    itself be known at t (caller's responsibility; use a lagged/causal regime classifier)."""
    lab = regime.astype(object)
    since = np.zeros(len(lab))
    for i in range(1, len(lab)):
        since[i] = 0 if (lab.iloc[i] != lab.iloc[i - 1]) else since[i - 1] + 1
    return pd.DataFrame({"since_regime_change": since, "regime_changed": (since == 0).astype(float)}, index=regime.index)


def confidence_deterioration(edge: pd.Series, window: int = 13, lag: int = 1) -> pd.DataFrame:
    """The pattern's own trailing reliability as a feature: rolling mean/t of its edge over `window`, shifted by `lag` so the
    row for period t uses only outcomes matured by t-lag, plus the change from the previous window (deterioration)."""
    e = edge.astype(float)
    mu = e.rolling(window, min_periods=max(4, window // 2)).mean()
    sd = e.rolling(window, min_periods=max(4, window // 2)).std()
    t = mu / (sd / math.sqrt(window)).replace(0, np.nan)
    prev = mu.shift(window)
    return pd.DataFrame({"own_t": t.shift(lag), "own_mean": mu.shift(lag), "own_change": (mu - prev).shift(lag)}, index=e.index)


# ------------------------------------------------------------------------------------------------ learner
def _validate_frame(df: pd.DataFrame, edge_col: str, feature_cols: Sequence[str], now) -> pd.DataFrame:
    if edge_col not in df.columns:
        raise BoundaryError(f"edge column {edge_col!r} missing")
    d = df.copy()
    d.index = pd.to_datetime(d.index)
    if d.index.duplicated().any():
        raise BoundaryError("duplicate dates in boundary frame (one row per period)")
    d = d.sort_index()
    if "matured_at" in d.columns:
        mx = pd.to_datetime(d["matured_at"]).max()
        if pd.notna(mx) and mx >= pd.Timestamp(as_date(now)):
            raise FirewallBreach(f"boundary frame: outcome matured {mx.date()} is not before now={as_date(now)}")
    if len(d) and d.index.max() >= pd.Timestamp(as_date(now)):
        raise FirewallBreach(f"boundary frame: row dated {d.index.max().date()} is not before now={as_date(now)}")
    missing = [c for c in feature_cols if c not in d.columns]
    if missing:
        raise BoundaryError(f"feature columns missing: {missing}")
    return d


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float, float]:
    """(t of a-b, mean a, mean b, se a, se b) with small-sample guards."""
    ma, mb = float(np.mean(a)), float(np.mean(b))
    sa = float(np.std(a, ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("inf")
    sb = float(np.std(b, ddof=1) / math.sqrt(len(b))) if len(b) > 1 else float("inf")
    den = math.sqrt(sa ** 2 + sb ** 2)
    return ((ma - mb) / den if den > 0 and math.isfinite(den) else 0.0), ma, mb, sa, sb


def _stationary_idx(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / max(block, 1)
    out = np.empty(n, int)
    i = 0
    pos = int(rng.integers(0, n))
    while i < n:
        out[i] = pos % n
        i += 1
        pos = int(rng.integers(0, n)) if rng.random() < p else pos + 1
    return out


class BoundaryLearner:
    """Learns the boundaries of one pattern from a period-level frame: one row per period, the pattern's realised `edge`
    that period, and any number of context features known at decision time."""

    def __init__(self, cfg: BoundaryConfig = DEFAULT_BCFG):
        self.cfg = cfg

    def _rng(self, pattern_id: str, salt: str = "") -> np.random.Generator:
        return np.random.default_rng(int(stable_hash({"s": self.cfg.seed, "p": pattern_id, "x": salt}, 8), 16) % (2 ** 32))

    def learn(self, pattern_id: str, df: pd.DataFrame, edge_col: str, feature_cols: Sequence[str], now,
              kinds: Mapping[str, BoundaryKind] | None = None) -> BoundarySet:
        cfg = self.cfg
        d = _validate_frame(df, edge_col, list(feature_cols), now)
        d = d[np.isfinite(d[edge_col].astype(float))]
        kinds = dict(kinds or {})
        n = len(d)
        skipped: list[tuple[str, str]] = []
        e_all = d[edge_col].astype(float).values
        overall = float(np.mean(e_all)) if n else float("nan")
        if n < 2 * cfg.min_side + cfg.min_test:
            return BoundarySet(pattern_id, as_date(now).isoformat(), n, (), (),
                               tuple((c, f"only {n} periods: INSUFFICIENT_DATA") for c in feature_cols), overall, stable_hash(cfg))
        rng = self._rng(pattern_id)
        scanners, ok_cols, obs = {}, [], {}
        for c in feature_cols:
            f = d[c].astype(float).values
            miss = float(np.mean(~np.isfinite(f)))
            if miss > cfg.max_missing:
                skipped.append((c, f"{miss:.0%} missing"))
                continue
            if np.nanstd(f) == 0:
                skipped.append((c, "constant"))
                continue
            keep = np.isfinite(f)
            sc = _Scanner(f[keep], cfg.min_side)
            if sc.empty:
                skipped.append((c, "no valid split with min_side observations each side"))
                continue
            scanners[c] = (sc, keep)
            ok_cols.append(c)
            obs[c] = sc.best(e_all[keep])
        # circular-shift permutation null of the whole scan (shifts of the edge relative to ALL features together)
        nulls = {c: np.empty(cfg.n_perm) for c in ok_cols}
        lo = max(1, int(cfg.min_shift_frac * n))
        for i in range(cfg.n_perm):
            shift = int(rng.integers(lo, n - lo + 1))
            ep = np.roll(e_all, shift)
            for c in ok_cols:
                sc, keep = scanners[c]
                nulls[c][i] = sc.max_abs_t(ep[keep])
        raw_p = np.array([(1 + np.sum(nulls[c] >= abs(obs[c].t) - 1e-12)) / (cfg.n_perm + 1) for c in ok_cols]) if ok_cols else np.array([])
        from engine.pattern_reliability import holm
        adj = holm(raw_p) if len(raw_p) > 1 else raw_p
        accepted, rejected = [], []
        for j, c in enumerate(ok_cols):
            b = self._assemble(pattern_id, c, kinds.get(c, BoundaryKind.THRESHOLD), d, edge_col, obs[c], float(raw_p[j]),
                               float(adj[j]), now, overall)
            (accepted if b.accepted else rejected).append(b)
        accepted.sort(key=lambda b: b.p_adj)
        return BoundarySet(pattern_id, as_date(now).isoformat(), n, tuple(accepted), tuple(rejected), tuple(skipped), overall,
                           stable_hash(cfg))

    # ---- one feature
    def _assemble(self, pid, col, kind, d, edge_col, sp: Split, raw_p, p_adj, now, overall) -> Boundary:
        cfg = self.cfg
        e = d[edge_col].astype(float).values
        f = d[col].astype(float).values
        keep = np.isfinite(f)
        e, f = e[keep], f[keep]
        sgn = 1.0 if overall >= 0 else -1.0
        left_better = sgn * sp.mean_left >= sgn * sp.mean_right
        if left_better:
            works, ei, eo, si, so, ni, no = "<=", sp.mean_left, sp.mean_right, sp.se_left, sp.se_right, sp.n_left, sp.n_right
        else:
            works, ei, eo, si, so, ni, no = ">", sp.mean_right, sp.mean_left, sp.se_right, sp.se_left, sp.n_right, sp.n_left
        z_out = eo / so if so > 0 and math.isfinite(so) else 0.0
        if sgn * eo < 0 and abs(z_out) > cfg.zero_z:
            outside = OutsideBehaviour.REVERSES
        elif abs(z_out) <= cfg.zero_z:
            outside = OutsideBehaviour.STOPS
        else:
            outside = OutsideBehaviour.WEAKENS
        iqr = float(np.subtract(*np.percentile(f, [75, 25]))) or 1.0
        thr_sd = self._threshold_spread(pid, col, e, f, iqr)
        status, ot, oi, oo = self._confirm(e, f)
        reasons = []
        acc = True
        if p_adj >= cfg.alpha:
            acc = False
            reasons.append(f"scan-null p_adj={p_adj:.3f} >= {cfg.alpha}")
        if status == "FAILED":
            acc = False
            reasons.append(f"not confirmed out of time (t={ot:.2f})")
        if status == "UNTESTED":
            acc = False
            reasons.append("too little history to confirm out of time: UNTESTED, not accepted")
        if sgn * ei <= 0:
            acc = False
            reasons.append("inside region has no edge in the pattern's direction")
        if thr_sd > 0.5:
            acc = False
            reasons.append(f"threshold unstable (bootstrap sd {thr_sd:.2f} IQR)")
        if acc:
            reasons.append(f"p_adj={p_adj:.3f}, confirmed out of time (t={ot:.2f})")
        return Boundary(pid, col, kind, float(sp.threshold), works, int(ni), int(no), float(ei), float(eo), float(si), float(so),
                        float(sp.t), float(raw_p), float(p_adj), float(thr_sd), status, float(ot), float(oi), float(oo),
                        str(outside), acc, tuple(reasons), as_date(now).isoformat())

    def _threshold_spread(self, pid, col, e, f, iqr) -> float:
        cfg = self.cfg
        if cfg.n_boot <= 0:
            return 0.0
        rng = self._rng(pid, col)
        ths = []
        for _ in range(cfg.n_boot):
            ix = _stationary_idx(len(e), cfg.boot_block, rng)
            s = best_split(e[ix], f[ix], cfg.min_side)
            if s is not None:
                ths.append(s.threshold)
        if len(ths) < max(5, cfg.n_boot // 4):
            return float("inf")
        return float(np.std(ths) / iqr)

    def _confirm(self, e, f) -> tuple[str, float, float, float]:
        """Refit the best split on the older part, test the same rule on the newer part (chronological order preserved)."""
        cfg = self.cfg
        n = len(e)
        cut = int(round(n * (1 - cfg.holdout_frac)))
        if n - cut < cfg.min_test or cut < 2 * cfg.min_side:
            return "UNTESTED", float("nan"), float("nan"), float("nan")
        tr = best_split(e[:cut], f[:cut], cfg.min_side)
        if tr is None:
            return "FAILED", float("nan"), float("nan"), float("nan")
        eT, fT = e[cut:], f[cut:]
        left = fT <= tr.threshold
        if left.sum() < cfg.min_test_side or (~left).sum() < cfg.min_test_side:
            return "UNTESTED", float("nan"), float("nan"), float("nan")
        t, ml, mr, _, _ = _welch(eT[left], eT[~left])
        direction = 1.0 if tr.t > 0 else -1.0                # training said left better (t>0) or right better
        t_dir = direction * t
        inside, outside = (ml, mr) if direction > 0 else (mr, ml)
        return ("CONFIRMED" if t_dir >= cfg.oot_t else "FAILED"), float(t_dir), float(inside), float(outside)


# ------------------------------------------------------------------------------------------------ using boundaries
def in_scope_mask(bset: BoundarySet, df: pd.DataFrame) -> np.ndarray:
    """Row-wise: is each row inside every accepted boundary?  Rows missing a boundary feature are NOT in scope."""
    m = np.ones(len(df), bool)
    for b in bset.boundaries:
        v = df[b.feature].astype(float).values
        m &= np.isfinite(v) & b.inside_mask(np.where(np.isfinite(v), v, 0.0))
    return m


def gating_benefit(bset: BoundarySet, df: pd.DataFrame, edge_col: str) -> dict:
    """What would gating on the learned boundaries have done on a frame (ideally NEWER than the learning date)?
    Compares the mean edge kept vs skipped and reports coverage; never used to choose the boundaries."""
    e = df[edge_col].astype(float).values
    m = in_scope_mask(bset, df) & np.isfinite(e)
    ok = np.isfinite(e)
    if ok.sum() == 0:
        return {"n": 0, "verdict": str(Unknown.INSUFFICIENT_DATA)}
    kept, skip = e[m], e[ok & ~m]
    out = {"n": int(ok.sum()), "coverage": float(m.sum() / ok.sum()), "edge_all": float(e[ok].mean()),
           "edge_kept": float(kept.mean()) if len(kept) else float("nan"),
           "edge_skipped": float(skip.mean()) if len(skip) else float("nan")}
    out["lift"] = out["edge_kept"] - out["edge_all"] if len(kept) else float("nan")
    if len(kept) >= 8 and len(skip) >= 8:
        t, *_ = _welch(kept, skip)
        out["t_kept_vs_skipped"] = float(t)
    return out


def boundary_health(b: Boundary, df: pd.DataFrame, edge_col: str, now, cfg: BoundaryConfig = DEFAULT_BCFG) -> dict:
    """Does a learned boundary still hold on data newer than it was learned on?  Boundaries are knowledge and can decay."""
    d = _validate_frame(df, edge_col, [b.feature], now)
    d = d[d.index > pd.Timestamp(as_date(b.learned_at))]
    e = d[edge_col].astype(float).values
    f = d[b.feature].astype(float).values
    ok = np.isfinite(e) & np.isfinite(f)
    e, f = e[ok], f[ok]
    if len(e) < cfg.min_test:
        return {"status": "UNTESTED", "n": int(len(e)), "unknown": str(Unknown.INSUFFICIENT_DATA)}
    m = b.inside_mask(f)
    if m.sum() < cfg.min_test_side or (~m).sum() < cfg.min_test_side:
        return {"status": "UNTESTED", "n": int(len(e)), "unknown": str(Unknown.INSUFFICIENT_DATA)}
    t, mi, mo, _, _ = _welch(e[m], e[~m])
    status = "HOLDS" if t >= cfg.oot_t else "DEGRADED" if t > 0 else "REVERSED"
    return {"status": status, "n": int(len(e)), "t": float(t), "edge_inside": mi, "edge_outside": mo,
            "learned_edge_inside": b.edge_inside, "learned_edge_outside": b.edge_outside}


def refine_conjunction(df: pd.DataFrame, edge_col: str, b1: Boundary, b2: Boundary, cfg: BoundaryConfig = DEFAULT_BCFG) -> dict:
    """Do two boundaries together isolate the working region better than either alone?  Returns the contrast (inside-both vs
    everything else) and the improvement over the better single boundary.  A conjunction must still be confirmed out of time by
    the caller (this is a screening statistic)."""
    e = df[edge_col].astype(float).values
    m1 = b1.inside_mask(df[b1.feature].astype(float).values)
    m2 = b2.inside_mask(df[b2.feature].astype(float).values)
    ok = np.isfinite(e)
    res = {}
    for name, m in (("b1", m1), ("b2", m2), ("both", m1 & m2)):
        a, b = e[ok & m], e[ok & ~m]
        if len(a) < cfg.min_side or len(b) < cfg.min_side:
            res[name] = float("nan")
        else:
            res[name] = _welch(a, b)[0]
    best_single = np.nanmax([res["b1"], res["b2"]]) if not (np.isnan(res["b1"]) and np.isnan(res["b2"])) else float("nan")
    res["improvement"] = res["both"] - best_single if np.isfinite(res["both"]) and np.isfinite(best_single) else float("nan")
    res["n_both"] = int(np.sum(ok & m1 & m2))
    res["worthwhile"] = bool(np.isfinite(res["improvement"]) and res["improvement"] > 1.0)
    return res


# ------------------------------------------------------------------------------------------------ boundary as knowledge
@dc.dataclass(frozen=True)
class BoundaryKnowledge:
    """A BoundarySet dressed as a KnowledgeLike record (engine.learning.core.KnowledgeLike): the boundary itself is knowledge
    with its own id, epistemic state, confidence and provenance.  Status is CONDITIONAL because a boundary is by definition a
    statement about conditions, and RESEARCH-promoted until a promotion gate (another module) says otherwise."""
    knowledge_id: str
    version: int
    epistemic: Epistemic
    lifecycle: Lifecycle
    promotion: Promotion
    confidence: Confidence
    provenance: Provenance
    contexts: Mapping[str, Any]
    anti_contexts: Mapping[str, Any]
    decision_effect: tuple[DecisionEffect, ...]
    pattern_id: str = ""
    statement: str = ""


def to_knowledge(bset: BoundarySet, version: int = 1, learned_at_real: str | None = None) -> BoundaryKnowledge:
    """Make the accepted boundaries storable.  `learned_at_real` = the real date the newest evidence matured (trusted side);
    defaults to the set's learned_at (the caller's `now`)."""
    la = learned_at_real or bset.learned_at
    conf = Confidence(context=(max(0.0, 1.0 - min(b.p_adj for b in bset.boundaries)) if bset.boundaries else None))
    prov = Provenance(created_real=bset.learned_at, learned_at=la, code_hash=current_code_hash(),
                      config_hash=bset.config_hash, outcomes_seen_through=la)
    return BoundaryKnowledge(
        knowledge_id=f"boundary:{bset.pattern_id}:{bset.set_id}", version=version,
        epistemic=Epistemic.CONDITIONAL if bset.boundaries else Epistemic.UNKNOWN,
        lifecycle=Lifecycle.ACTIVE if bset.boundaries else Lifecycle.BIRTH, promotion=Promotion.RESEARCH,
        confidence=conf, provenance=prov, contexts=bset.contexts, anti_contexts=bset.anti_contexts,
        decision_effect=(DecisionEffect.ABSTENTION, DecisionEffect.POSITION_SIZE) if bset.boundaries else (DecisionEffect.NONE,),
        pattern_id=bset.pattern_id, statement=bset.summary())


class BoundaryRegistry:
    """Append-only history of boundary sets per pattern.  A re-learned boundary is a new version, never an overwrite; the
    drift between versions (threshold moved, boundary appeared/vanished) is itself reported as knowledge about the pattern."""

    def __init__(self):
        self._sets: dict[str, list[BoundarySet]] = {}

    def add(self, bset: BoundarySet) -> int:
        h = self._sets.setdefault(bset.pattern_id, [])
        if h and as_date(bset.learned_at) < as_date(h[-1].learned_at):
            raise BoundaryError("boundary history is append-only in time")
        h.append(bset)
        return len(h)

    def latest(self, pattern_id: str, as_of=None) -> BoundarySet | None:
        h = self._sets.get(pattern_id, [])
        if as_of is not None:
            h = [s for s in h if as_date(s.learned_at) <= as_date(as_of)]
        return h[-1] if h else None

    def history(self, pattern_id: str) -> tuple[BoundarySet, ...]:
        return tuple(self._sets.get(pattern_id, ()))

    def drift(self, pattern_id: str) -> list[dict]:
        h = self._sets.get(pattern_id, [])
        out = []
        for a, b in zip(h, h[1:]):
            fa = {x.feature: x for x in a.boundaries}
            fb = {x.feature: x for x in b.boundaries}
            out.append({"from": a.learned_at, "to": b.learned_at,
                        "appeared": sorted(set(fb) - set(fa)), "vanished": sorted(set(fa) - set(fb)),
                        "moved": {f: (fa[f].threshold, fb[f].threshold) for f in set(fa) & set(fb)
                                  if abs(fa[f].threshold - fb[f].threshold) > 1e-12}})
        return out

    def patterns(self) -> list[str]:
        return sorted(self._sets)


# ------------------------------------------------------------------------------------------------ event-time profiles
def event_profile(edge: pd.Series, since: pd.Series, max_lag: int = 12, min_n: int = 5) -> pd.DataFrame:
    """Mean edge by periods-since-an-event (0 = the event period, ... max_lag), against the calm baseline (lags beyond).  Shows
    HOW LONG a pattern stays off after a shock or a regime change - the shape of the boundary in time, which a single threshold
    hides.  `since` must already be lagged so row t only knows events up to t-1."""
    j = pd.concat([edge.rename("edge"), since.rename("since")], axis=1, join="inner").dropna()
    base = j[j["since"] > max_lag]["edge"]
    rows = []
    for lag in range(0, max_lag + 1):
        x = j[j["since"] == lag]["edge"].values
        if len(x) < min_n:
            rows.append({"lag": lag, "n": int(len(x)), "mean": float("nan"), "se": float("nan"), "vs_calm_t": float("nan")})
            continue
        t, ma, mb, sa, sb = _welch(x, base.values) if len(base) >= min_n else (float("nan"), float(x.mean()), float("nan"), float("nan"), float("nan"))
        rows.append({"lag": lag, "n": int(len(x)), "mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))),
                     "vs_calm_t": float(t)})
    out = pd.DataFrame(rows)
    out.attrs["calm_mean"] = float(base.mean()) if len(base) else float("nan")
    out.attrs["calm_n"] = int(len(base))
    return out


def off_duration(profile: pd.DataFrame, t_bad: float = -1.64) -> int:
    """Number of consecutive lags from the event onward at which the edge is significantly worse than calm (0 = no effect).
    This is the learned 'wait this many periods after a shock' value."""
    k = 0
    for r in profile.itertuples():
        if math.isfinite(r.vs_calm_t) and r.vs_calm_t <= t_bad:
            k += 1
        else:
            break
    return k


def transition_effect(edge: pd.Series, regime: pd.Series, max_lag: int = 8) -> pd.DataFrame:
    """Edge in the first `max_lag` periods after each regime change versus stable periods (regime label known at decision time)."""
    feats = regime_transition_features(regime)
    return event_profile(edge, shift_trailing(feats["since_regime_change"], 0), max_lag)


def shock_aftermath(edge: pd.Series, market_ret: pd.Series, max_lag: int = 8, z: float = 3.0) -> pd.DataFrame:
    """Edge in the periods after a market shock (row t sees shocks through t-1 only)."""
    sf = shock_features(market_ret, z=z)
    since = shift_trailing(sf["since_shock"], 1) + 1
    return event_profile(edge, since, max_lag)


# ------------------------------------------------------------------------------------------------ assembling the frame
def build_boundary_frame(edge: pd.Series, *, volatility: pd.Series | None = None, liquidity: pd.Series | None = None,
                         breadth: pd.Series | None = None, market_ret: pd.Series | None = None, regime: pd.Series | None = None,
                         pattern_scores: pd.DataFrame | None = None, picks: pd.DataFrame | None = None,
                         own_window: int = 13) -> tuple[pd.DataFrame, dict[str, BoundaryKind]]:
    """One frame with the seven contract boundary families as columns, every one built only from what a decision at that date
    could know (trailing quantities shifted).  Missing inputs simply omit their columns.  Returns (frame, kinds-by-column).

    The `edge` series is the pattern's realised edge indexed by outcome-maturity date; features are aligned to the same index."""
    idx = pd.to_datetime(edge.index)
    fr = pd.DataFrame({"edge": edge.astype(float).values}, index=idx)
    kinds: dict[str, BoundaryKind] = {}

    def put(name, s, kind):
        if s is None:
            return
        x = pd.Series(s).copy()
        x.index = pd.to_datetime(x.index)
        fr[name] = x.reindex(idx).values
        kinds[name] = kind

    put("volatility", volatility, BoundaryKind.THRESHOLD)
    put("liquidity", liquidity, BoundaryKind.THRESHOLD)
    put("breadth", breadth, BoundaryKind.THRESHOLD)
    if market_ret is not None:
        mr = pd.Series(market_ret).astype(float)
        mr.index = pd.to_datetime(mr.index)
        sf = shock_features(mr)
        put("since_shock", shift_trailing(sf["since_shock"], 1) + 1, BoundaryKind.SHOCK)
    if regime is not None:
        rg = pd.Series(regime)
        rg.index = pd.to_datetime(rg.index)
        put("since_regime_change", regime_transition_features(rg.reindex(idx))["since_regime_change"], BoundaryKind.TRANSITION)
    if pattern_scores is not None:
        put("disagreement", disagreement_from_scores(pattern_scores), BoundaryKind.DISAGREEMENT)
    if picks is not None:
        put("concentration", concentration_hhi(picks), BoundaryKind.CONCENTRATION)
    cd = confidence_deterioration(pd.Series(fr["edge"].values, index=idx), window=own_window)
    put("own_t", cd["own_t"], BoundaryKind.CONFIDENCE)
    put("own_change", cd["own_change"], BoundaryKind.CONFIDENCE)
    return fr, kinds


def learn_pattern_boundaries(pattern_id: str, edge: pd.Series, now, learner: BoundaryLearner | None = None, **inputs) -> BoundarySet:
    """Convenience end-to-end: build the seven feature families from raw inputs and learn boundaries."""
    fr, kinds = build_boundary_frame(edge, **inputs)
    return (learner or BoundaryLearner()).learn(pattern_id, fr, "edge", [c for c in fr.columns if c != "edge"], now, kinds)


# ------------------------------------------------------------------------------------------------ reporting
def boundary_table(bset: BoundarySet, include_rejected: bool = True) -> pd.DataFrame:
    """Every tested boundary, accepted or not, as rows - the rejections are part of what was learned."""
    rows = []
    for b in list(bset.boundaries) + (list(bset.rejected) if include_rejected else []):
        rows.append({"pattern": b.pattern_id, "feature": b.feature, "kind": str(b.kind), "works_when": b.works_when,
                     "threshold": b.threshold, "edge_in": b.edge_inside, "edge_out": b.edge_outside, "outside": b.outside,
                     "n_in": b.n_inside, "n_out": b.n_outside, "t": b.contrast_t, "p_adj": b.p_adj,
                     "thr_sd": b.threshold_sd, "oot": b.oot_status, "oot_t": b.oot_t, "accepted": b.accepted,
                     "why": "; ".join(b.reasons)})
    cols = ["pattern", "feature", "kind", "works_when", "threshold", "edge_in", "edge_out", "outside", "n_in", "n_out", "t",
            "p_adj", "thr_sd", "oot", "oot_t", "accepted", "why"]
    return pd.DataFrame(rows, columns=cols)


def anti_condition_summary(bsets: Iterable[BoundarySet]) -> pd.DataFrame:
    """Across many patterns: which features most often mark where patterns stop working?  A feature that ends many patterns is a
    market-level boundary (e.g. volatility), worth promoting to a regime gate rather than a per-pattern rule."""
    rows = []
    for bs in bsets:
        for b in bs.boundaries:
            if b.anti_context is not None:
                rows.append({"feature": b.feature, "kind": str(b.kind), "pattern": b.pattern_id, "outside": b.outside,
                             "threshold": b.threshold, "works_when": b.works_when})
    if not rows:
        return pd.DataFrame(columns=["feature", "kind", "n_patterns", "median_threshold", "n_reverses"])
    df = pd.DataFrame(rows)
    g = df.groupby(["feature", "kind"])
    out = g.agg(n_patterns=("pattern", "nunique"), median_threshold=("threshold", "median"),
                n_reverses=("outside", lambda s: int((s == str(OutsideBehaviour.REVERSES)).sum()))).reset_index()
    return out.sort_values("n_patterns", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ serialisation
def boundary_to_dict(b: Boundary) -> dict:
    d = dc.asdict(b)
    d["kind"] = str(b.kind)
    d["reasons"] = list(b.reasons)
    return d


def boundary_from_dict(d: Mapping) -> Boundary:
    d = dict(d)
    d["kind"] = BoundaryKind.parse(d["kind"])
    d["reasons"] = tuple(d.get("reasons", ()))
    return Boundary(**d)


def set_to_dict(bs: BoundarySet) -> dict:
    return {"pattern_id": bs.pattern_id, "learned_at": bs.learned_at, "n_periods": bs.n_periods,
            "boundaries": [boundary_to_dict(b) for b in bs.boundaries], "rejected": [boundary_to_dict(b) for b in bs.rejected],
            "skipped": [list(s) for s in bs.skipped], "overall_edge": bs.overall_edge, "config_hash": bs.config_hash}


def set_from_dict(d: Mapping) -> BoundarySet:
    return BoundarySet(d["pattern_id"], d["learned_at"], int(d["n_periods"]), tuple(boundary_from_dict(x) for x in d["boundaries"]),
                       tuple(boundary_from_dict(x) for x in d["rejected"]), tuple(tuple(s) for s in d["skipped"]),
                       float(d["overall_edge"]), d["config_hash"])
