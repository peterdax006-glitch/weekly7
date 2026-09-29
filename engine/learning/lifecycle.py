"""Knowledge lifecycle: birth, growth, peak, decay, failure, recovery, retirement (contract C62 section 12; checklists D12, J07).
Bible phase serving: pattern lifecycle (phases 4/9). Generalises engine/pattern_lifecycle.py (which keeps owning pattern-level
mining states) to ANY knowledge item that produces a stream of signed outcomes.

Two jobs. (1) `trace`: a causal per-row stage for an item - the stage at row i uses outcomes of rows < i only - with hysteresis
so it cannot flap, explicit recovery (a failed or decayed item is never assumed dead), DORMANT when the item stopped firing,
and RETIRED only when an external gate says so (retirement.py owns the four-state ledger; `apply_to_ledger` feeds it).
(2) `classify_deterioration`: when an item has deteriorated, decide WHAT KIND: random (indistinguishable from noise), gradual
(a trend), abrupt (a step), or linked to a family of context (regime, market, sector, volatility, liquidity, interaction). Shape
is chosen by BIC among constant / trend / step with a block-permutation p-value for "anything changed at all"; linkage asks
whether context coefficients learned BEFORE the decline already predict the decline, against a block-permuted null. A decline
that no family explains keeps its shape label and is not given an invented cause.

Built on core.Lifecycle vocabulary, engine.learning.retirement (Evidence, RetirementLedger, read-only) and
engine.pattern_reliability.holm. IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine import pattern_reliability as PR

from .core import FailureCause, FirewallBreach, Lifecycle, as_date, stable_hash
from .retirement import Evidence, RetirementLedger, State, Verdict, series_evidence

PARAMS = {
    "window": 26, "peak_se": 2.0, "birth_n": 13, "est_n": 26, "t_est": 2.0,
    "peak_frac": 0.85,          # PEAK while the trailing effect is at least this share of the best it ever was
    "decay_frac": 0.60,         # DECAY when the trailing effect falls under this share of the peak
    "decay_exit": 0.70,         # ... and leaves DECAY (to RECOVERY) only above this share: hysteresis
    "recover_ok_frac": 0.80, "hold": 8,
    "t_fail": 1.5,              # trailing t at or below -t_fail = FAILURE
    "recover_t": 1.0, "recover_n": 13,
    "give_up": 104,             # rows after which an effect that was never established is a failure (phantom)
    "dormant_after": 26, "retire_after": 104,
    # deterioration
    "min_seg": 8, "bic_margin": 6.0, "trend_bias": 2.0, "n_perm": 300, "perm_block": 4, "alpha": 0.05,
    "link_frac": 0.5, "link_alpha": 0.05, "ridge": 1.0, "pre_frac": 0.5, "min_pre": 30,
}
FAMILIES = ("regime", "market", "sector", "volatility", "liquidity", "interaction")


def _cfg(cfg) -> dict:
    return {**PARAMS, **(cfg or {})}


class Deterioration(str, enum.Enum):
    """Section 12's list. RANDOM = the apparent decline is indistinguishable from noise."""
    RANDOM = "RANDOM"
    GRADUAL = "GRADUAL"
    ABRUPT = "ABRUPT"
    REGIME_LINKED = "REGIME_LINKED"
    MARKET_LINKED = "MARKET_LINKED"
    SECTOR_LINKED = "SECTOR_LINKED"
    VOLATILITY_LINKED = "VOLATILITY_LINKED"
    LIQUIDITY_LINKED = "LIQUIDITY_LINKED"
    INTERACTION_LINKED = "INTERACTION_LINKED"
    INSUFFICIENT = "INSUFFICIENT"

    def __str__(self):
        return self.value


CAUSE_BY_KIND = {
    Deterioration.RANDOM: FailureCause.INSUFFICIENT_EVIDENCE, Deterioration.GRADUAL: FailureCause.WEAKENING_EFFECT,
    Deterioration.ABRUPT: FailureCause.UNKNOWN, Deterioration.REGIME_LINKED: FailureCause.REGIME_CHANGE,
    Deterioration.MARKET_LINKED: FailureCause.REGIME_CHANGE, Deterioration.SECTOR_LINKED: FailureCause.WRONG_CONTEXT,
    Deterioration.VOLATILITY_LINKED: FailureCause.REGIME_CHANGE, Deterioration.LIQUIDITY_LINKED: FailureCause.REGIME_CHANGE,
    Deterioration.INTERACTION_LINKED: FailureCause.INTERACTION_FAILURE, Deterioration.INSUFFICIENT: FailureCause.INSUFFICIENT_EVIDENCE,
}


# ------------------------------------------------------------------------------------------------- the trace

@dataclasses.dataclass(frozen=True)
class StageChange:
    at: int
    frm: str
    to: str
    reason: str
    numbers: Mapping[str, float]


@dataclasses.dataclass
class LifecycleTrace:
    """Per-row result of `trace`. `stage[i]` is the stage at the close of row i using outcomes of rows < i."""
    stage: np.ndarray                  # of Lifecycle values (str)
    effect: np.ndarray                 # trailing mean outcome (NaN until half a window exists)
    t_trail: np.ndarray                # trailing t
    peak: np.ndarray                   # best trailing effect seen so far while established (NaN before)
    changes: list
    phantom: bool

    def __len__(self):
        return len(self.stage)

    def at(self, i: int) -> Lifecycle:
        return Lifecycle(str(self.stage[i]))

    @property
    def current(self) -> Lifecycle:
        return self.at(len(self.stage) - 1) if len(self.stage) else Lifecycle.BIRTH

    def phases(self) -> list[dict]:
        out, start = [], 0
        for i in range(1, len(self.stage) + 1):
            if i == len(self.stage) or self.stage[i] != self.stage[start]:
                out.append({"stage": str(self.stage[start]), "start": start, "end": i, "length": i - start})
                start = i
        return out

    def first(self, stage: Lifecycle) -> int | None:
        hit = np.flatnonzero(self.stage == stage.value)
        return int(hit[0]) if len(hit) else None

    def time_in(self) -> dict:
        vals, counts = np.unique(self.stage, return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, counts)}

    def summary(self) -> dict:
        s = self.stage
        return {"current": str(self.current), "born": 0, "established": self.first(Lifecycle.ACTIVE) if self.first(Lifecycle.ACTIVE) is not None else self.first(Lifecycle.PEAK),
                "peak_index": int(np.nanargmax(self.effect)) if np.isfinite(self.effect).any() else None,
                "peak_effect": float(np.nanmax(self.peak)) if np.isfinite(self.peak).any() else float("nan"),
                "failures": int(sum(c.to == "FAILURE" for c in self.changes)),
                "recoveries": int(sum(c.to == "RECOVERY" for c in self.changes)), "phantom": self.phantom,
                "time_in": self.time_in(), "n": len(s)}


def _t_of(v: np.ndarray) -> tuple[float, float, float]:
    """(mean, sd, t) of a finite sample; t is 0 when it cannot be formed."""
    if len(v) < 3:
        return (float(v.mean()) if len(v) else float("nan")), 0.0, 0.0
    sd = float(v.std(ddof=1))
    return float(v.mean()), sd, (float(v.mean() / (sd / math.sqrt(len(v)))) if sd > 1e-12 else 0.0)


class LifecycleMachine:
    """The resumable state machine behind `trace`. `push(x, exposed)` decides the stage of the NEXT row from outcomes already
    pushed (rows before it) and only then records the row's own outcome, so streaming one row at a time is exactly the batch
    trace. `snapshot` / `from_snapshot` checkpoint it, and a resumed machine continues identically."""

    def __init__(self, cfg=None):
        self.P = _cfg(cfg)
        self.state = Lifecycle.BIRTH
        self.inner = Lifecycle.BIRTH               # where a dormant item returns to when it fires again
        self.peak = float("nan")
        self.established = False
        self.phantom = False
        self.fail_start = 0
        self.hold = 0
        self.idle = 0
        self.values: list[float] = []
        self.stage: list[str] = []
        self.effect: list[float] = []
        self.t_trail: list[float] = []
        self.peak_hist: list[float] = []
        self.changes: list[StageChange] = []

    def _move(self, to: Lifecycle, why: str, **nums) -> None:
        if to != self.state:
            self.changes.append(StageChange(len(self.stage), self.state.value, to.value, why, nums))
            self.state = to

    def _peak_of(self, win: np.ndarray) -> float:
        return float(win.mean() - self.P["peak_se"] * _t_of(win)[1] / math.sqrt(len(win)))

    def step(self, exposed: bool = True, retired: bool = False) -> Lifecycle:
        P = self.P
        i, W = len(self.stage), P["window"]
        past = np.array(self.values, float)
        past = past[np.isfinite(past)]
        k = len(past)
        m_e, _, t_e = _t_of(past)
        win = past[-W:]
        m_w = t_w = float("nan")
        if len(win) >= W // 2:
            m_w, _, t_w = _t_of(win)
        if self.established and len(win) >= W and np.isfinite(m_w):
            lcb = self._peak_of(win)
            self.peak = lcb if not np.isfinite(self.peak) else max(self.peak, lcb)
        self.effect.append(m_w)
        self.t_trail.append(t_w)
        self.peak_hist.append(self.peak)
        self.idle = 0 if exposed else self.idle + 1
        if retired:
            self._move(Lifecycle.RETIRED, "retired by an external gate")
        elif self.state == Lifecycle.DORMANT and not exposed:
            pass
        elif self.state != Lifecycle.DORMANT and self.idle >= P["dormant_after"]:
            self.inner = self.state
            self._move(Lifecycle.DORMANT, f"no firing for {self.idle} rows", idle=self.idle)
        else:
            if self.state == Lifecycle.DORMANT:
                self._move(self.inner, "fired again", idle=self.idle)
            self._advance(i, k, m_e, t_e, m_w, t_w, win)
        self.stage.append(self.state.value)
        return self.state

    def _advance(self, i, k, m_e, t_e, m_w, t_w, win) -> None:
        P, st = self.P, self.state
        if st == Lifecycle.BIRTH and k >= P["birth_n"]:
            self._move(Lifecycle.GROWTH, "first evidence gathered", n=k)
        if self.state == Lifecycle.GROWTH:
            if k >= P["est_n"] and t_e >= P["t_est"] and m_e > 0:
                self.established = True
                self.peak = self._peak_of(win) if np.isfinite(m_w) else m_e
                self._move(Lifecycle.ACTIVE, "effect established", t=t_e, n=k)
            elif k >= P["give_up"]:
                self.phantom, self.fail_start = True, i
                self._move(Lifecycle.FAILURE, "never established: phantom", t=t_e, n=k)
            elif np.isfinite(t_w) and t_w <= -P["t_fail"] and k >= P["est_n"]:
                self.phantom, self.fail_start = True, i      # failing before it was ever established is never having existed
                self._move(Lifecycle.FAILURE, "evidence turned against it before it was established", t=t_w)
        if self.state in (Lifecycle.ACTIVE, Lifecycle.PEAK):
            if np.isfinite(t_w) and t_w <= -P["t_fail"]:
                self.fail_start = i
                self._move(Lifecycle.FAILURE, "trailing evidence reversed", t=t_w, effect=m_w)
            elif np.isfinite(m_w) and np.isfinite(self.peak) and self.peak > 0 and m_w < P["decay_frac"] * self.peak:
                self._move(Lifecycle.DECAY, "trailing effect under the decay share of its peak", effect=m_w, peak=self.peak)
            elif np.isfinite(m_w) and np.isfinite(self.peak) and self.peak > 0:
                self._move(Lifecycle.PEAK if m_w >= P["peak_frac"] * self.peak else Lifecycle.ACTIVE, "tracking the peak",
                           effect=m_w, peak=self.peak)
        elif self.state == Lifecycle.DECAY:
            if np.isfinite(t_w) and t_w <= -P["t_fail"]:
                self.fail_start = i
                self._move(Lifecycle.FAILURE, "decay became reversal", t=t_w)
            elif np.isfinite(m_w) and self.peak > 0 and m_w >= P["decay_exit"] * self.peak:
                self.hold = 0
                self._move(Lifecycle.RECOVERY, "trailing effect back above the decay exit", effect=m_w, peak=self.peak)
        elif self.state == Lifecycle.FAILURE:
            since = np.array(self.values[self.fail_start:i], float)
            since = since[np.isfinite(since)]
            m_s, _, t_s = _t_of(since)
            need_n, need_t = (P["est_n"], P["t_est"]) if self.phantom else (P["recover_n"], P["recover_t"])
            if len(since) >= need_n and t_s >= need_t and m_s > 0:
                self.hold = 0
                self._move(Lifecycle.RECOVERY, "new evidence since the failure is positive", t=t_s, n=len(since))
        elif self.state == Lifecycle.RECOVERY:
            if np.isfinite(t_w) and t_w <= -P["t_fail"]:
                self.fail_start = i
                self._move(Lifecycle.FAILURE, "relapse", t=t_w)
            elif np.isfinite(m_w) and np.isfinite(self.peak) and self.peak > 0 and m_w >= P["recover_ok_frac"] * self.peak:
                self.hold += 1
                if self.hold >= P["hold"]:
                    self._move(Lifecycle.ACTIVE, "recovery held", effect=m_w, peak=self.peak)
            elif self.phantom and np.isfinite(t_e) and t_e >= P["t_est"] and m_e > 0:
                self.established, self.phantom = True, False
                self.peak = self._peak_of(win) if np.isfinite(m_w) else m_e
                self._move(Lifecycle.ACTIVE, "effect finally established", t=t_e)
            else:
                self.hold = 0

    def push(self, x: float, exposed: bool = True, retired: bool = False) -> Lifecycle:
        st = self.step(exposed, retired)
        self.values.append(float(x))
        return st

    def to_trace(self) -> LifecycleTrace:
        return LifecycleTrace(np.array(self.stage, dtype=str), np.array(self.effect, float), np.array(self.t_trail, float),
                              np.array(self.peak_hist, float), list(self.changes), self.phantom)

    def snapshot(self) -> dict:
        """JSON-safe checkpoint of everything the machine needs to continue."""
        return {"state": self.state.value, "inner": self.inner.value, "peak": self.peak, "established": self.established,
                "phantom": self.phantom, "fail_start": self.fail_start, "hold": self.hold, "idle": self.idle,
                "values": list(self.values), "stage": list(self.stage), "effect": list(self.effect),
                "t_trail": list(self.t_trail), "peak_hist": list(self.peak_hist),
                "changes": [dataclasses.asdict(c) for c in self.changes]}

    @classmethod
    def from_snapshot(cls, snap: Mapping[str, Any], cfg=None) -> "LifecycleMachine":
        m = cls(cfg)
        m.state, m.inner = Lifecycle(snap["state"]), Lifecycle(snap["inner"])
        for f in ("peak", "established", "phantom", "fail_start", "hold", "idle"):
            setattr(m, f, snap[f])
        for f in ("values", "stage", "effect", "t_trail", "peak_hist"):
            setattr(m, f, list(snap[f]))
        m.changes = [StageChange(**c) for c in snap["changes"]]
        return m


def trace(values: Sequence[float], cfg=None, exposure: Sequence[bool] | None = None, retired_from: int | None = None) -> LifecycleTrace:
    """Causal lifecycle stage of an item for every row. `exposure[i]` False means the item did not fire at row i (its outcome is
    NaN); a run of `dormant_after` such rows makes it DORMANT until it fires again. `retired_from` is the row an external gate
    retired it; nothing after that row is anything but RETIRED (recovery then needs the ledger's revival gate)."""
    m = LifecycleMachine(cfg)
    for i, x in enumerate(np.asarray(values, float)):
        m.push(x, True if exposure is None else bool(exposure[i]), retired_from is not None and i >= retired_from)
    return m.to_trace()


def resume(machine: LifecycleMachine, values: Sequence[float], exposure: Sequence[bool] | None = None) -> LifecycleTrace:
    """Continue a checkpointed machine over new rows and return the full trace so far."""
    for i, x in enumerate(np.asarray(values, float)):
        machine.push(x, True if exposure is None else bool(exposure[i]))
    return machine.to_trace()


def retire_proposal(tr: LifecycleTrace, cfg=None) -> dict:
    """Should a retirement be PROPOSED? Only when the item has been in FAILURE (or DORMANT) continuously for `retire_after`
    rows with no RECOVERY. A proposal, never an action: the ledger's gate decides, and recovery stays possible after it."""
    P = _cfg(cfg)
    n = len(tr)
    if n == 0:
        return {"propose": False, "why": "empty trace"}
    bad = {Lifecycle.FAILURE.value, Lifecycle.DORMANT.value}
    run = 0
    for s in tr.stage[::-1]:
        if s in bad:
            run += 1
        else:
            break
    return {"propose": bool(run >= P["retire_after"]), "run": run, "needed": P["retire_after"],
            "why": f"{run} consecutive rows in FAILURE/DORMANT" if run else "not currently failed or dormant"}


def decay_half_life(effect: Sequence[float], peak_index: int, grid: Sequence[float] = tuple(range(4, 157, 4))) -> dict:
    """Fit effect[t] = c + (peak - c) * 0.5 ** ((t - peak_index) / h) after the peak by grid search over h (rows), c free by least
    squares at each h. Returns h and the fit R-squared; a decay that is really a step (best h tiny or R-squared low) says so."""
    e = np.asarray(effect, float)[peak_index:]
    ok = np.isfinite(e)
    if ok.sum() < 12:
        return {"half_life": float("nan"), "r2": float("nan"), "n": int(ok.sum())}
    t = np.arange(len(e))[ok]
    y = e[ok]
    best = (float("nan"), -np.inf, float("nan"))
    for h in grid:
        f = 0.5 ** (t / h)
        A = np.column_stack([np.ones_like(f), f])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef
        r2 = 1.0 - float(r @ r) / float(((y - y.mean()) ** 2).sum() or 1.0)
        if r2 > best[1]:
            best = (float(h), r2, float(coef[0]))
    return {"half_life": best[0], "r2": best[1], "floor": best[2], "n": int(ok.sum())}


# ------------------------------------------------------------------------------------------------- shape of a deterioration

def _rss_const(y: np.ndarray) -> float:
    return float(((y - y.mean()) ** 2).sum())


def _rss_trend(y: np.ndarray) -> tuple[float, float, float]:
    t = np.arange(len(y), dtype=float)
    A = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    return float(r @ r), float(coef[1]), float(coef[0])


def _rss_step(y: np.ndarray, min_seg: int) -> tuple[float, int, float]:
    """Best single step: (rss, index of the first row of the second segment, size of the step = mean2 - mean1)."""
    n = len(y)
    cs, cs2 = np.cumsum(y), np.cumsum(y * y)
    best = (np.inf, -1, 0.0)
    for k in range(min_seg, n - min_seg + 1):
        m1, m2 = cs[k - 1] / k, (cs[-1] - cs[k - 1]) / (n - k)
        rss = (cs2[k - 1] - k * m1 * m1) + ((cs2[-1] - cs2[k - 1]) - (n - k) * m2 * m2)
        if rss < best[0]:
            best = (float(rss), k, float(m2 - m1))
    return best


def _rss_pulse(y: np.ndarray, min_seg: int) -> tuple[float, int, int]:
    """Best interval [a, b) whose mean differs from the rest (rest pooled as one level): (rss, a, b). Catches 'strong, weak,
    strong again', which no single step can."""
    n = len(y)
    cs, cs2 = np.r_[0.0, np.cumsum(y)], np.r_[0.0, np.cumsum(y * y)]
    best = (np.inf, -1, -1)
    tot, tot2 = cs[-1], cs2[-1]
    for a in range(min_seg, n - 2 * min_seg + 1):
        b = np.arange(a + min_seg, n - min_seg + 1)
        m_in = (cs[b] - cs[a]) / (b - a)
        n_out = n - (b - a)
        m_out = (tot - (cs[b] - cs[a])) / n_out
        rss = (tot2 - (cs2[b] - cs2[a]) - n_out * m_out ** 2) + ((cs2[b] - cs2[a]) - (b - a) * m_in ** 2)
        j = int(np.argmin(rss))
        if rss[j] < best[0]:
            best = (float(rss[j]), a, int(b[j]))
    return best


def _bic(rss: float, n: int, k: int) -> float:
    return n * math.log(max(rss, 1e-300) / n) + k * math.log(n)


def _block_permute(y: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    nb = math.ceil(n / block)
    order = rng.permutation(nb)
    return np.concatenate([y[b * block:(b + 1) * block] for b in order])[:n]


@dataclasses.dataclass(frozen=True)
class ShapeFit:
    shape: str                 # RANDOM | GRADUAL | ABRUPT | INSUFFICIENT
    bic_const: float
    bic_trend: float
    bic_step: float
    slope: float
    step_at: int               # index within the segment (-1 if none)
    step_size: float
    p_change: float
    n: int


def classify_shape(y: Sequence[float], cfg=None, seed: int = 0) -> ShapeFit:
    """Constant vs trend vs step for the outcome segment `y` (oldest first), by BIC with a margin, gated by a block-permutation
    p-value on the best BIC gain (so noise is called RANDOM even when a step 'fits'). Ties between trend and step go to the
    simpler trend unless the step wins by `trend_bias` BIC points."""
    P = _cfg(cfg)
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2 * P["min_seg"]:
        return ShapeFit("INSUFFICIENT", float("nan"), float("nan"), float("nan"), float("nan"), -1, float("nan"), 1.0, n)
    b0 = _bic(_rss_const(y), n, 1)
    r1, slope, _ = _rss_trend(y)
    b1 = _bic(r1, n, 2)
    r2, k2, size = _rss_step(y, P["min_seg"])
    b2 = _bic(r2, n, 3)
    gain = b0 - min(b1, b2)
    rng = np.random.default_rng(seed)
    null = np.empty(P["n_perm"])
    for i in range(P["n_perm"]):
        yp = _block_permute(y, P["perm_block"], rng)
        null[i] = _bic(_rss_const(yp), n, 1) - min(_bic(_rss_trend(yp)[0], n, 2), _bic(_rss_step(yp, P["min_seg"])[0], n, 3))
    p = float((1 + (null >= gain).sum()) / (P["n_perm"] + 1))
    if gain < P["bic_margin"] or p > P["alpha"]:
        return ShapeFit("RANDOM", b0, b1, b2, slope, -1, size, p, n)
    shape = "ABRUPT" if b2 + P["trend_bias"] < b1 else "GRADUAL"
    return ShapeFit(shape, b0, b1, b2, slope, k2 if shape == "ABRUPT" else -1, size, p, n)


# ------------------------------------------------------------------------------------------------- linkage to context families

def _ridge_fit(Z: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    A = np.column_stack([np.ones(len(Z)), Z])
    reg = ridge * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return coef, float(y.mean())


def _explained_fraction(Z: np.ndarray, y: np.ndarray, n_pre: int, ridge: float) -> tuple[float, float]:
    """Fit y ~ Z on the first n_pre rows (the item still working), predict the rest, and report the share of the raw mean drop
    (post minus pre) that the prediction reproduces, and the pre-window R-squared. 1 = the context fully explains the fall."""
    coef, _ = _ridge_fit(Z[:n_pre], y[:n_pre], ridge)
    A = np.column_stack([np.ones(len(Z)), Z])
    fit = A @ coef
    res = y[:n_pre] - fit[:n_pre]
    r2 = 1.0 - float(res @ res) / float(((y[:n_pre] - y[:n_pre].mean()) ** 2).sum() or 1.0)
    drop = float(y[n_pre:].mean() - y[:n_pre].mean())
    if abs(drop) < 1e-12:
        return 0.0, r2
    return float(np.clip((fit[n_pre:].mean() - fit[:n_pre].mean()) / drop, -1.0, 1.5)), r2


@dataclasses.dataclass(frozen=True)
class LinkFit:
    family: str
    explained: float
    pre_r2: float
    p: float
    columns: tuple


def test_links(y: Sequence[float], families: Mapping[str, pd.DataFrame | None], n_pre: int, cfg=None, seed: int = 0) -> list[LinkFit]:
    """Score every context family against the decline in `y` (rows aligned with each family's frame; the first n_pre rows are
    the stretch where the item still worked). p = share of block-permuted context rows that explain the fall as well as the
    real ones do, so a family with many columns cannot win by having more knobs than the others' null."""
    P = _cfg(cfg)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    out = []
    for fam, df in families.items():
        if df is None or df.shape[1] == 0:
            continue
        Z = df.astype(float).values
        ok = np.isfinite(Z).all(axis=1) & np.isfinite(y)
        if ok.sum() < len(y) or n_pre < P["min_pre"]:
            out.append(LinkFit(fam, float("nan"), float("nan"), 1.0, tuple(df.columns)))
            continue
        mu, sd = Z[:n_pre].mean(0), Z[:n_pre].std(0)
        sd[sd <= 1e-12] = 1.0
        Zs = (Z - mu) / sd
        frac, r2 = _explained_fraction(Zs, y, n_pre, P["ridge"])
        null = np.empty(P["n_perm"])
        for i in range(P["n_perm"]):
            null[i] = _explained_fraction(_block_permute(Zs, P["perm_block"], rng), y, n_pre, P["ridge"])[0]
        out.append(LinkFit(fam, frac, r2, float((1 + (null >= frac).sum()) / (P["n_perm"] + 1)), tuple(df.columns)))
    return out


@dataclasses.dataclass(frozen=True)
class DeteriorationVerdict:
    kind: Deterioration
    shape: ShapeFit
    links: tuple
    linked_family: str | None
    cause: FailureCause
    evidence: Mapping[str, float]

    def as_dict(self) -> dict:
        return {"kind": self.kind.value, "cause": self.cause.value, "linked_family": self.linked_family,
                "shape": dataclasses.asdict(self.shape), "links": [dataclasses.asdict(l) for l in self.links],
                "evidence": dict(self.evidence)}


def classify_deterioration(values: Sequence[float], families: Mapping[str, pd.DataFrame | None] | None = None,
                           start: int = 0, end: int | None = None, cfg=None, seed: int = 0) -> DeteriorationVerdict:
    """What kind of deterioration is the stretch values[start:end]? Order of business: is there any change at all (else RANDOM);
    if so is it a trend or a step; then does a context family account for it (else the shape label stands). A verdict of
    INSUFFICIENT is returned for short segments rather than a guess."""
    P = _cfg(cfg)
    x = np.asarray(values, float)
    end = len(x) if end is None else end
    seg = x[start:end]
    shape = classify_shape(seg, P, seed)
    ev = {"n": float(shape.n), "p_change": shape.p_change}
    if shape.shape == "INSUFFICIENT":
        return DeteriorationVerdict(Deterioration.INSUFFICIENT, shape, (), None, FailureCause.INSUFFICIENT_EVIDENCE, ev)
    if shape.shape == "RANDOM":
        return DeteriorationVerdict(Deterioration.RANDOM, shape, (), None, CAUSE_BY_KIND[Deterioration.RANDOM], ev)
    kind = Deterioration.ABRUPT if shape.shape == "ABRUPT" else Deterioration.GRADUAL
    links: list[LinkFit] = []
    linked = None
    if families:
        fam_seg = {f: (None if d is None else d.iloc[start:end]) for f, d in families.items()}
        n_pre = shape.step_at if shape.shape == "ABRUPT" else int(P["pre_frac"] * len(seg))
        links = test_links(seg, fam_seg, n_pre, P, seed + 1)
        adj = PR.holm([l.p for l in links]) if links else []
        good = [(l, a) for l, a in zip(links, adj) if np.isfinite(l.explained) and l.explained >= P["link_frac"] and a <= P["link_alpha"]]
        if good:
            best = max(good, key=lambda z: z[0].explained)[0]
            linked = best.family
            kind = Deterioration[f"{linked.upper()}_LINKED"]
            ev["explained"] = best.explained
    ev["slope"] = shape.slope
    ev["step_size"] = shape.step_size
    return DeteriorationVerdict(kind, shape, tuple(links), linked, CAUSE_BY_KIND[kind], ev)


# ------------------------------------------------------------------------------------------------- recovery

@dataclasses.dataclass(frozen=True)
class RecoveryAssessment:
    recovered: bool
    n: int
    effect: float
    t: float
    needed_t: float
    needed_n: int
    why: str


def assess_recovery(values: Sequence[float], fail_start: int, now_index: int, cfg=None, prior_t: float | None = None) -> RecoveryAssessment:
    """Has a failed item recovered? Recovery is allowed but must be earned: only outcomes AFTER the failure began and before
    `now_index` count, there must be at least `recover_n` of them, their mean must be positive, and their t must clear
    `recover_t` - or, when the t that condemned the item is known (`prior_t`), exceed its magnitude (hysteresis: it must be
    at least as convincing as the evidence that failed it)."""
    P = _cfg(cfg)
    x = np.asarray(values, float)[fail_start:now_index]
    x = x[np.isfinite(x)]
    m, _, t = _t_of(x)
    need_t = max(P["recover_t"], abs(prior_t) if prior_t is not None else 0.0)
    if len(x) < P["recover_n"]:
        return RecoveryAssessment(False, len(x), m, t, need_t, P["recover_n"], f"only {len(x)} outcomes since the failure")
    if not (m > 0):
        return RecoveryAssessment(False, len(x), m, t, need_t, P["recover_n"], "mean outcome since the failure is not positive")
    if t < need_t:
        return RecoveryAssessment(False, len(x), m, t, need_t, P["recover_n"], f"t={t:.2f} below the required {need_t:.2f}")
    return RecoveryAssessment(True, len(x), m, t, need_t, P["recover_n"], "new evidence clears the recovery bar")


# ------------------------------------------------------------------------------------------------- the book and the ledger

@dataclasses.dataclass
class LifecycleBook:
    """Traces for many items with their deterioration verdicts. `as_of` snapshots only look at rows before the cut."""
    traces: dict = dataclasses.field(default_factory=dict)
    verdicts: dict = dataclasses.field(default_factory=dict)
    values: dict = dataclasses.field(default_factory=dict)

    def add(self, kid: str, values: Sequence[float], cfg=None, exposure=None, retired_from=None) -> LifecycleTrace:
        v = np.asarray(values, float)
        self.values[kid] = v
        self.traces[kid] = trace(v, cfg, exposure, retired_from)
        return self.traces[kid]

    def stage_at(self, kid: str, i: int) -> Lifecycle:
        return self.traces[kid].at(i)

    def counts(self, i: int | None = None) -> dict:
        out: dict[str, int] = {}
        for tr in self.traces.values():
            if len(tr) == 0:
                continue
            s = str(tr.stage[(len(tr) - 1) if i is None else min(i, len(tr) - 1)])
            out[s] = out.get(s, 0) + 1
        return out

    def diagnose(self, kid: str, families=None, cfg=None, seed: int = 0) -> DeteriorationVerdict | None:
        """Classify the deterioration of an item that is in DECAY/FAILURE now, over the stretch from its peak to the present."""
        tr = self.traces[kid]
        if tr.current not in (Lifecycle.DECAY, Lifecycle.FAILURE):
            return None
        eff = tr.effect
        pk = int(np.nanargmax(eff)) if np.isfinite(eff).any() else 0
        self.verdicts[kid] = classify_deterioration(self.values[kid], families, pk, None, cfg, seed)
        return self.verdicts[kid]

    def report(self) -> list[dict]:
        rows = []
        for kid, tr in sorted(self.traces.items()):
            s = tr.summary()
            v = self.verdicts.get(kid)
            rows.append({"knowledge_id": kid, "stage": s["current"], "failures": s["failures"], "recoveries": s["recoveries"],
                         "phantom": s["phantom"], "deterioration": None if v is None else v.kind.value,
                         "cause": None if v is None else v.cause.value})
        return rows


def apply_to_ledger(ledger: RetirementLedger, kid: str, tr: LifecycleTrace, values: Sequence[float], dates: Sequence[Any], now,
                    cause: FailureCause = FailureCause.UNKNOWN, apply: bool = False) -> Verdict | None:
    """Turn the current lifecycle stage into a ledger action: FAILURE/DECAY go through the retirement gate, RECOVERY through the
    recovery gate. The evidence window starts at the last stage change (recovery: after the item went inactive) and ends
    strictly before `now`. Returns the ledger's verdict, or None when the stage calls for no action."""
    if len(tr) == 0 or not ledger.known(kid):
        return None
    cur = tr.current
    last = tr.changes[-1].at if tr.changes else 0
    if cur in (Lifecycle.DECAY, Lifecycle.FAILURE):
        start = dates[max(last - 1, 0)] if cur == Lifecycle.FAILURE else dates[max(len(dates) - 2 * PARAMS["window"], 0)]
        return ledger.evaluate(kid, series_evidence(dates, values, start, now, "lifecycle"), now, cause, apply=apply)
    if cur == Lifecycle.RECOVERY:
        st = ledger.state(kid, now)
        if st in (State.DORMANT, State.RETIRED, State.DEGRADED):
            since = ledger.since(kid, now) or dates[0]
            return ledger.attempt_recovery(kid, series_evidence(dates, values, since, now, "lifecycle"), now, apply=apply)
    return None


def stage_table(traces: Mapping[str, LifecycleTrace]) -> pd.DataFrame:
    """Every item's phases as rows (item, stage, start, end, length) - the lifecycle chart as data."""
    rows = []
    for kid, tr in sorted(traces.items()):
        for p in tr.phases():
            rows.append({"knowledge_id": kid, **p})
    return pd.DataFrame(rows, columns=["knowledge_id", "stage", "start", "end", "length"])


def transition_matrix(traces: Mapping[str, LifecycleTrace]) -> pd.DataFrame:
    """Counts of stage-to-stage moves across the book. A recovery shows up as FAILURE -> RECOVERY; an item with no such
    entry anywhere would mean recovery is being assumed impossible."""
    counts: dict[tuple, int] = {}
    for tr in traces.values():
        for c in tr.changes:
            counts[(c.frm, c.to)] = counts.get((c.frm, c.to), 0) + 1
    if not counts:
        return pd.DataFrame()
    names = sorted({a for a, _ in counts} | {b for _, b in counts})
    m = pd.DataFrame(0, index=names, columns=names)
    for (a, b), c in counts.items():
        m.loc[a, b] = c
    return m


def check_trace(tr: LifecycleTrace, cfg=None) -> list[str]:
    """Invariants of a trace: only legal moves, PEAK/DECAY never before ESTABLISHED, RETIRED terminal, no BIRTH after the
    first stage. Returns human-readable violations (empty when clean)."""
    legal = {("BIRTH", "GROWTH"), ("GROWTH", "ACTIVE"), ("GROWTH", "FAILURE"), ("ACTIVE", "PEAK"), ("PEAK", "ACTIVE"),
             ("ACTIVE", "DECAY"), ("PEAK", "DECAY"), ("ACTIVE", "FAILURE"), ("PEAK", "FAILURE"), ("DECAY", "FAILURE"),
             ("DECAY", "RECOVERY"), ("FAILURE", "RECOVERY"), ("RECOVERY", "ACTIVE"), ("RECOVERY", "FAILURE")}
    errs = []
    for c in tr.changes:
        if c.to == "DORMANT" or c.frm == "DORMANT" or c.to == "RETIRED":
            continue
        if (c.frm, c.to) not in legal:
            errs.append(f"illegal move {c.frm}->{c.to} at {c.at}")
    stages = list(tr.stage)
    if "RETIRED" in stages:
        first = stages.index("RETIRED")
        if any(s != "RETIRED" for s in stages[first:]):
            errs.append("RETIRED is not terminal")
    if stages and stages[0] != "BIRTH":
        errs.append("trace does not start at BIRTH")
    return errs


# ------------------------------------------------------------------------------------------------- growth, forecast, history

def fit_growth(effect: Sequence[float], peak_index: int | None = None) -> dict:
    """Logistic growth of the trailing effect up to its peak: e(t) = L / (1 + exp(-k (t - t0))) with L the peak, by grid search
    over the midpoint t0 and steepness k. Returns t0 (the row at which the item reached half its peak), k, and R-squared, so
    'how fast did it establish' is a number. NaN when there is too little rise to fit."""
    e = np.asarray(effect, float)
    pk = int(np.nanargmax(e)) if peak_index is None and np.isfinite(e).any() else peak_index
    if pk is None or pk < 12:
        return {"t0": float("nan"), "k": float("nan"), "r2": float("nan"), "n": 0}
    t = np.arange(pk + 1, dtype=float)
    y = e[: pk + 1]
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) < 12 or y.max() <= 0:
        return {"t0": float("nan"), "k": float("nan"), "r2": float("nan"), "n": len(y)}
    L = float(y.max())
    best = (float("nan"), float("nan"), -np.inf)
    for t0 in np.linspace(t.min(), t.max(), 25):
        for k in (0.02, 0.05, 0.1, 0.2, 0.5):
            f = L / (1.0 + np.exp(-k * (t - t0)))
            r = y - f
            r2 = 1.0 - float(r @ r) / float(((y - y.mean()) ** 2).sum() or 1.0)
            if r2 > best[2]:
                best = (float(t0), float(k), r2)
    return {"t0": best[0], "k": best[1], "r2": best[2], "n": len(y), "peak": L}


def forecast_life(effect: Sequence[float], peak_index: int, bar: float = 0.0, horizon: int = 156, n_boot: int = 200, seed: int = 0) -> dict:
    """How long until the decaying effect falls below `bar`? Extrapolates the fitted exponential decay (decay_half_life) and
    bootstraps its residuals for an interval on the crossing row. Reported as rows AFTER the last observation; None when the
    fitted floor stays above the bar (the decay levels off before it) or the fit is unusable."""
    e = np.asarray(effect, float)
    fit = decay_half_life(e, peak_index)
    if not np.isfinite(fit["half_life"]) or fit["r2"] < 0.3:
        return {"crossing": None, "lo": None, "hi": None, "fit": fit, "why": "no usable decay fit"}
    seg = e[peak_index:]
    ok = np.isfinite(seg)
    t = np.arange(len(seg))[ok]
    y = seg[ok]

    def cross(yy):
        h_best = (np.inf, None)
        for h in range(4, 157, 4):
            f = 0.5 ** (t / h)
            A = np.column_stack([np.ones_like(f), f])
            coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
            rss = float(((yy - A @ coef) ** 2).sum())
            if rss < h_best[0]:
                h_best = (rss, (h, coef))
        h, coef = h_best[1]
        if coef[0] >= bar or coef[1] <= 0:
            return None
        rows = h * math.log2(coef[1] / (bar - coef[0])) if bar > coef[0] else None
        return None if rows is None else float(rows - (len(seg) - 1))

    base = cross(y)
    rng = np.random.default_rng(seed)
    f0 = 0.5 ** (t / fit["half_life"])
    A0 = np.column_stack([np.ones_like(f0), f0])
    c0, *_ = np.linalg.lstsq(A0, y, rcond=None)
    resid = y - A0 @ c0
    draws = []
    for _ in range(n_boot):
        c = cross(A0 @ c0 + rng.choice(resid, size=len(resid), replace=True))
        if c is not None and c <= horizon:
            draws.append(c)
    return {"crossing": base, "lo": float(np.quantile(draws, 0.1)) if len(draws) >= 20 else None,
            "hi": float(np.quantile(draws, 0.9)) if len(draws) >= 20 else None, "fit": fit,
            "why": "" if base is not None else "the fitted decay levels off above the bar"}


def segment_history(values: Sequence[float], cfg=None, max_segments: int = 6) -> list[dict]:
    """Binary segmentation of the outcome history into stretches of different mean (each split must beat a BIC margin and leave
    `min_seg` rows each side). Describes an item's life as 'strong, then weak, then strong again' without any lifecycle
    labels, and gives the lifecycle trace something independent to be compared with."""
    P = _cfg(cfg)
    x = np.asarray(values, float)
    x = np.where(np.isfinite(x), x, np.nanmean(x) if np.isfinite(x).any() else 0.0)
    segs = [(0, len(x))]
    while len(segs) < max_segments:
        best = None
        for a, b in segs:
            y = x[a:b]
            if len(y) < 2 * P["min_seg"]:
                continue
            r0 = _rss_const(y)
            r2, k, size = _rss_step(y, P["min_seg"])
            cands = [(_bic(r0, len(y), 1) - _bic(r2, len(y), 3), [a + k])]
            rp, pa, pb = _rss_pulse(y, P["min_seg"])
            if pa >= 0:
                cands.append((_bic(r0, len(y), 1) - _bic(rp, len(y), 5), [a + pa, a + pb]))
            gain, cuts = max(cands, key=lambda z: z[0])
            if gain > 2 * P["bic_margin"] and (best is None or gain > best[0]):
                best = (gain, (a, b), cuts)
        if best is None:
            break
        _, (a, b), cuts = best
        segs.remove((a, b))
        edges = [a, *cuts, b]
        segs += list(zip(edges[:-1], edges[1:]))
    out = []
    for a, b in sorted(segs):
        y = x[a:b]
        m, sd, t = _t_of(y)
        out.append({"start": a, "end": b, "n": b - a, "mean": m, "t": t})
    return out


# ------------------------------------------------------------------------------------------------- the book over time

def survival_by_age(traces: Mapping[str, LifecycleTrace], step: int = 13) -> pd.DataFrame:
    """Kaplan-Meier survival of items free of FAILURE, by age in rows (age counted from the row an item left BIRTH), with items
    still alive censored at the end of their trace. The book-level answer to 'how long do things last', and the input a
    hazard prior should come from."""
    ages, events = [], []
    for tr in traces.values():
        born = tr.first(Lifecycle.GROWTH)
        if born is None:
            continue
        f = tr.first(Lifecycle.FAILURE)
        ages.append((f if f is not None else len(tr)) - born)
        events.append(f is not None)
    if not ages:
        return pd.DataFrame(columns=["age", "at_risk", "events", "survival"])
    ages, events = np.array(ages), np.array(events)
    rows, surv = [], 1.0
    for a in range(0, int(ages.max()) + step, step):
        lo, hi = a, a + step
        at_risk = int((ages >= lo).sum())
        d = int(((ages >= lo) & (ages < hi) & events).sum())
        if at_risk:
            surv *= 1.0 - d / at_risk
        rows.append({"age": lo, "at_risk": at_risk, "events": d, "survival": surv})
    return pd.DataFrame(rows)


def median_life(surv: pd.DataFrame) -> float:
    """First age at which survival drops to 0.5 or below; NaN if it never does (more than half the items never failed)."""
    hit = surv[surv["survival"] <= 0.5]
    return float(hit["age"].iloc[0]) if len(hit) else float("nan")


def recovery_conditions(ctx: pd.DataFrame, tr: LifecycleTrace, min_events: int = 3) -> pd.DataFrame:
    """Which context columns differ between the rows where recoveries BEGAN and the rows where failures began? A column that
    reliably differs is a candidate recovery condition ('it comes back when...'). Needs at least `min_events` of each; below
    that it returns an empty table rather than a guess. Columns are aligned to the trace's rows."""
    rec = [c.at for c in tr.changes if c.to == "RECOVERY"]
    fail = [c.at for c in tr.changes if c.to == "FAILURE"]
    cols = ["column", "mean_recovery", "mean_failure", "t", "p", "n_recovery", "n_failure"]
    if len(rec) < min_events or len(fail) < min_events:
        return pd.DataFrame(columns=cols)
    rows = []
    for c in ctx.columns:
        x = ctx[c].astype(float).values
        a = np.array([x[i] for i in rec if i < len(x) and np.isfinite(x[i])])
        b = np.array([x[i] for i in fail if i < len(x) and np.isfinite(x[i])])
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = sps.ttest_ind(a, b, equal_var=False)
        rows.append({"column": c, "mean_recovery": float(a.mean()), "mean_failure": float(b.mean()), "t": float(t),
                     "p": float(p), "n_recovery": len(a), "n_failure": len(b)})
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df["p"] = PR.holm(df["p"].values)
    return df.sort_values("p").reset_index(drop=True)


def lifecycle_frame(book: LifecycleBook) -> pd.DataFrame:
    """One row per item with the lifecycle numbers a reviewer wants at a glance."""
    rows = []
    for kid, tr in sorted(book.traces.items()):
        s = tr.summary()
        g = fit_growth(tr.effect)
        rows.append({"knowledge_id": kid, "stage": s["current"], "n": s["n"], "peak_effect": s["peak_effect"],
                     "failures": s["failures"], "recoveries": s["recoveries"], "phantom": s["phantom"],
                     "growth_t0": g["t0"], "time_failed": s["time_in"].get("FAILURE", 0), "time_decay": s["time_in"].get("DECAY", 0)})
    return pd.DataFrame(rows)


def deterioration_report(book: LifecycleBook) -> str:
    """Plain-language page: every diagnosed item, the kind of deterioration, the evidence, and whether a cause is claimed."""
    lines = ["# Deterioration diagnoses", "IMPLEMENTED - NOT VALIDATED", ""]
    if not book.verdicts:
        return "\n".join(lines + ["no item is currently decaying or failed"])
    for kid, v in sorted(book.verdicts.items()):
        s = v.shape
        lines.append(f"## {kid}: {v.kind.value}")
        lines.append(f"cause: {v.cause.value}; segment n={s.n}; p(change)={s.p_change:.3f}; "
                     f"BIC const/trend/step = {s.bic_const:.1f}/{s.bic_trend:.1f}/{s.bic_step:.1f}")
        if v.linked_family:
            best = [l for l in v.links if l.family == v.linked_family][0]
            lines.append(f"linked to {v.linked_family}: explains {best.explained:.0%} of the fall (permutation p={best.p:.3f}) using "
                         f"{', '.join(best.columns)}")
        else:
            lines.append("no context family explains it; the shape label stands and no cause is claimed")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- onset, spells, agreement

def locate_onset(values: Sequence[float], cfg=None, n_boot: int = 200, seed: int = 0) -> dict:
    """Where did the change happen? The best single step in `values`, and a bootstrap interval for its row (resampling the
    residuals around the two segment means). A wide interval says the onset is not localisable - typical of a gradual decline
    that a step happens to fit."""
    P = _cfg(cfg)
    y = np.asarray(values, float)
    y = y[np.isfinite(y)]
    if len(y) < 2 * P["min_seg"]:
        return {"onset": None, "lo": None, "hi": None, "size": float("nan"), "n": len(y)}
    _, k, size = _rss_step(y, P["min_seg"])
    fitted = np.r_[np.full(k, y[:k].mean()), np.full(len(y) - k, y[k:].mean())]
    resid = y - fitted
    rng = np.random.default_rng(seed)
    ks = np.empty(n_boot, int)
    for b in range(n_boot):
        yb = fitted + rng.choice(resid, size=len(y), replace=True)
        ks[b] = _rss_step(yb, P["min_seg"])[1]
    return {"onset": int(k), "lo": int(np.quantile(ks, 0.05)), "hi": int(np.quantile(ks, 0.95)), "size": float(size), "n": len(y),
            "width": int(np.quantile(ks, 0.95) - np.quantile(ks, 0.05))}


def spells(tr: LifecycleTrace, stage: Lifecycle) -> list[dict]:
    """Consecutive runs of one stage as (start, end, length): how long each failure / decay / dormancy lasted."""
    return [p for p in tr.phases() if p["stage"] == stage.value]


def mean_time_to_recover(tr: LifecycleTrace) -> float:
    """Average rows from the start of a FAILURE spell to the start of the RECOVERY that ended it, over failures that did
    recover. NaN when none did - which must not be read as 'recovery takes forever' but as 'no recovery observed'."""
    starts = [c.at for c in tr.changes if c.to == "FAILURE"]
    recs = [c.at for c in tr.changes if c.to == "RECOVERY"]
    gaps = []
    for s in starts:
        later = [r for r in recs if r > s]
        nxt_fail = [f for f in starts if f > s]
        if later and (not nxt_fail or later[0] < nxt_fail[0]):
            gaps.append(later[0] - s)
    return float(np.mean(gaps)) if gaps else float("nan")


def stage_agreement(tr: LifecycleTrace, values: Sequence[float], cfg=None) -> dict:
    """Cross-check the causal trace against the retrospective segmentation: in rows the trace calls FAILURE the segment mean
    should be low, in ACTIVE/PEAK rows high. Reports the mean outcome by stage and the share of stage rows whose segment agrees
    in sign. A trace that disagrees with the segmentation has a threshold problem, not just lag."""
    x = np.asarray(values, float)
    segs = segment_history(x, cfg)
    seg_mean = np.zeros(len(x))
    for s in segs:
        seg_mean[s["start"]:s["end"]] = s["mean"]
    out = {}
    for name in ("ACTIVE", "PEAK", "DECAY", "FAILURE", "RECOVERY"):
        sel = tr.stage[: len(x)] == name
        if sel.any():
            out[name] = {"n": int(sel.sum()), "mean_outcome": float(np.nanmean(x[sel])), "segment_mean": float(seg_mean[sel].mean())}
    good = [out[k] for k in ("ACTIVE", "PEAK") if k in out]
    bad = [out[k] for k in ("FAILURE",) if k in out]
    agree = None
    if good and bad:
        agree = bool(np.mean([g["mean_outcome"] for g in good]) > np.mean([b["mean_outcome"] for b in bad]))
    return {"by_stage": out, "segments": len(segs), "good_beats_failed": agree}


# ------------------------------------------------------------------------------------------------- durations and readiness

def stage_durations(traces: Mapping[str, LifecycleTrace]) -> pd.DataFrame:
    """Distribution of spell lengths per stage across the book (count, median, quartiles, max). How long is a typical DECAY,
    a typical FAILURE, a typical RECOVERY - the numbers a hazard prior and a probation length should be set from."""
    lens: dict[str, list[int]] = {}
    for tr in traces.values():
        for p in tr.phases():
            lens.setdefault(p["stage"], []).append(p["length"])
    rows = [{"stage": s, "spells": len(v), "median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)),
             "q75": float(np.quantile(v, 0.75)), "max": int(max(v))} for s, v in sorted(lens.items())]
    return pd.DataFrame(rows, columns=["stage", "spells", "median", "q25", "q75", "max"])


def rows_to_establish(effect: float, sd: float, cfg=None) -> float:
    """Outcomes needed before an item with true per-period effect `effect` and outcome sd `sd` can clear the establishment bar
    t_est: (t_est * sd / effect) ** 2. inf for a non-positive effect. Tells a researcher how long BIRTH/GROWTH must last for
    an effect of the size they hope for, so 'not yet established' is not read as 'absent'."""
    P = _cfg(cfg)
    if effect <= 0 or sd <= 0:
        return float("inf")
    return float(max((P["t_est"] * sd / effect) ** 2, P["est_n"]))


def exposure_from_counts(counts: Sequence[float], min_signals: int = 1) -> np.ndarray:
    """Exposure mask for `trace` from a per-period count of how many times the item fired: True where it fired at least
    `min_signals` times. NaN counts (no data) are treated as not firing."""
    c = np.asarray(counts, float)
    return np.isfinite(c) & (c >= min_signals)


def validate_trace(tr: LifecycleTrace, n_expected: int | None = None) -> list[str]:
    """Structural checks on a trace object: aligned array lengths, known stage names, sorted change rows, and that the stage
    at every change row equals the change's target."""
    errs = []
    n = len(tr.stage)
    for name in ("effect", "t_trail", "peak"):
        if len(getattr(tr, name)) != n:
            errs.append(f"{name} length {len(getattr(tr, name))} != stage length {n}")
    if n_expected is not None and n != n_expected:
        errs.append(f"trace has {n} rows, expected {n_expected}")
    names = {s.value for s in Lifecycle}
    bad = sorted(set(map(str, tr.stage)) - names)
    if bad:
        errs.append(f"unknown stages {bad}")
    rows = [c.at for c in tr.changes]
    if rows != sorted(rows):
        errs.append("changes are not in time order")
    last_at = {c.at: c.to for c in tr.changes}          # several moves may share a row; the last one is the row's stage
    for at, to in last_at.items():
        if 0 <= at < n and str(tr.stage[at]) != to:
            errs.append(f"stage at {at} is {tr.stage[at]} but the last change says {to}")
    return errs


# ------------------------------------------------------------------------------------------------- vocabulary bridges

HEALTH_BY_STAGE = {Lifecycle.BIRTH: "INSUFFICIENT_EVIDENCE", Lifecycle.GROWTH: "INSUFFICIENT_EVIDENCE", Lifecycle.PEAK: "HEALTHY",
                   Lifecycle.ACTIVE: "HEALTHY", Lifecycle.DECAY: "DEGRADING", Lifecycle.DEGRADED: "DEGRADING",
                   Lifecycle.FAILURE: "BROKEN", Lifecycle.DORMANT: "DORMANT", Lifecycle.RECOVERY: "RECOVERING",
                   Lifecycle.RETIRED: "DORMANT"}


def health_name(stage: Lifecycle) -> str:
    """The section-46 health state a lifecycle stage implies on its own (the health monitor combines this with other sources)."""
    return HEALTH_BY_STAGE[stage]


def first_bad_row(tr: LifecycleTrace) -> int | None:
    """First row at which the trace left the healthy family (DECAY, FAILURE, DORMANT, RETIRED); None if it never did."""
    bad = {Lifecycle.DECAY.value, Lifecycle.FAILURE.value, Lifecycle.DORMANT.value, Lifecycle.RETIRED.value}
    hit = [i for i, s in enumerate(tr.stage) if s in bad]
    return hit[0] if hit else None


LIVE_STAGES = (Lifecycle.ACTIVE, Lifecycle.PEAK, Lifecycle.RECOVERY, Lifecycle.DECAY)


def may_influence_live(stage: Lifecycle) -> bool:
    """Only items that are working, or on probation after a recovery, or merely decaying carry live influence. BIRTH / GROWTH
    (unproven), FAILURE, DORMANT and RETIRED must not; the ledger enforces this with assert_clean, this is the trace-level rule."""
    return stage in LIVE_STAGES


def stage_share(tr: LifecycleTrace) -> dict:
    """Share of the trace's rows spent in each stage (sums to 1); empty for an empty trace."""
    n = len(tr)
    return {s: c / n for s, c in tr.time_in().items()} if n else {}


def influence_mask(tr: LifecycleTrace) -> np.ndarray:
    """Boolean per row: may the item influence decisions at that row?"""
    return np.array([may_influence_live(Lifecycle(s)) for s in tr.stage], dtype=bool)


def n_phases(tr: LifecycleTrace) -> int:
    """Number of distinct consecutive phases in the trace (a life with many phases has flapped or recovered repeatedly)."""
    return len(tr.phases())


def is_flapping(tr: LifecycleTrace, max_changes: int = 8) -> bool:
    """True when the trace changed stage more than `max_changes` times - thresholds that make an item chatter are a design
    fault to be reported, not a life history."""
    return len(tr.changes) > max_changes


def last_change(tr: LifecycleTrace) -> StageChange | None:
    return tr.changes[-1] if tr.changes else None
