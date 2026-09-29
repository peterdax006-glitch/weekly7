"""Direction calibration alternatives and abstention gates (Bible PHASE 13; canon: "if the system cannot produce
sufficient confidence, do not bet"). Complements engine/direction.py, which owns the default Platt/isotonic route.

  PerTypeCalibrator   the global Platt map plus a per-stock-type intercept/slope correction (types from engine.trust),
                      ridge-shrunk toward "no correction", and only for types with enough rows of BOTH classes.
  venn_abers()        inductive Venn-Abers: a probability INTERVAL [p0, p1] per row (isotonic fitted twice, once
                      with the row labelled 0 and once labelled 1) and the merged probability p1 / (1 - p0 + p1).
  VennAbersGate       bets only if the pessimistic end of the interval still clears the gate: the interval width is
                      how the method says "I do not know", so wide intervals abstain by themselves.
  ConformalGate       class-conditional (Mondrian) split-conformal: bet only when the prediction set at error level
                      1 - gate is a single class (the other class is rejected at that level).
  compare_gates()     one table, same rows: none / platt / isotonic / per-type / Venn-Abers gate / conformal gate.
All fitting takes calibration rows only; evaluation rows must be strictly later in time (caller's split)."""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from . import direction as D


def _logit(p):
    return D._logit(p)


# ---- per-type calibration ---------------------------------------------------------------------------------------
def _fit_ab(z, y, lam, iters=50):
    """Ridge logistic fit of  logit q = z*(1+b) + a  (z fixed offset input), penalty lam*(a^2 + b^2). Newton, 2 params."""
    a = b = 0.0
    for _ in range(iters):
        eta = z * (1 + b) + a
        q = expit(eta)
        w = np.clip(q * (1 - q), 1e-6, None)
        g = np.array([np.sum(q - y) + 2 * lam * a, np.sum((q - y) * z) + 2 * lam * b])
        H = np.array([[w.sum() + 2 * lam, (w * z).sum()], [(w * z).sum(), (w * z * z).sum() + 2 * lam]])
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


class PerTypeCalibrator:
    """Global Platt, then a shrunk per-type (a, b) correction. Unknown or thin types get the global map unchanged."""

    def __init__(self, min_n=60, min_class=15, lam=5.0):
        self.min_n, self.min_class, self.lam = min_n, min_class, lam
        self.glob = D._Cal("platt")
        self.ab = {}

    def fit(self, p, y, types):
        p, y, types = np.asarray(p, float), np.asarray(y, float), np.asarray(types)
        self.glob = D._Cal("platt").fit(p, y)
        z = _logit(self.glob(p))
        self.ab = {}
        for t in np.unique(types):
            m = types == t
            k = int(y[m].sum())
            if m.sum() >= self.min_n and min(k, m.sum() - k) >= self.min_class:
                self.ab[t] = _fit_ab(z[m], y[m], self.lam)
        return self

    def __call__(self, p, types):
        p, types = np.asarray(p, float), np.asarray(types)
        z = _logit(self.glob(p))
        a = np.array([self.ab.get(t, (0.0, 0.0))[0] for t in types])
        b = np.array([self.ab.get(t, (0.0, 0.0))[1] for t in types])
        return np.clip(expit(z * (1 + b) + a), 0.02, 0.98)

    def corrections(self) -> pd.DataFrame:
        return pd.DataFrame([(t, a, b) for t, (a, b) in self.ab.items()], columns=["type", "a", "b"])


# ---- Venn-Abers -------------------------------------------------------------------------------------------------
def venn_abers(p_cal, y_cal, p_new, decimals=3):
    """Returns (p0, p1, p_merged) arrays for p_new. Scores are rounded to `decimals` for caching: rows with equal
    rounded score share one pair of isotonic fits, which makes this O(unique scores * m) rather than O(rows * m)."""
    p_cal, y_cal, p_new = np.asarray(p_cal, float), np.asarray(y_cal, float), np.asarray(p_new, float)
    if len(p_cal) == 0:
        raise ValueError("Venn-Abers needs calibration rows")
    key = np.round(p_new, decimals)
    cache = {}
    for s in np.unique(key):
        pair = []
        for lab in (0.0, 1.0):
            xs, ys = np.append(p_cal, s), np.append(y_cal, lab)
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(xs, ys)
            pair.append(float(iso.predict([s])[0]))
        cache[s] = pair
    p0 = np.array([cache[s][0] for s in key])
    p1 = np.array([cache[s][1] for s in key])
    return p0, p1, p1 / (1 - p0 + p1)


class VennAbersGate:
    def __init__(self, gate=0.80, max_width=0.25):
        self.gate, self.max_width = gate, max_width

    def fit(self, p_cal, y_cal):
        self.p_cal, self.y_cal = np.asarray(p_cal, float), np.asarray(y_cal, float)
        return self

    def decide(self, p_new) -> pd.DataFrame:
        p0, p1, pm = venn_abers(self.p_cal, self.y_cal, p_new)
        up = pm >= 0.5
        lower = np.where(up, p0, 1 - p1)                        # pessimistic P(correct direction)
        width = p1 - p0
        bet = (lower >= self.gate) & (width <= self.max_width)
        return pd.DataFrame({"p_up": pm, "lo": p0, "hi": p1, "side": np.where(bet, np.where(up, 1, -1), 0),
                             "bet": bet})


# ---- conformal --------------------------------------------------------------------------------------------------
class ConformalGate:
    """Mondrian split-conformal on calibrated probabilities. nonconformity of a calibration row = 1 - p(true class).
    A class stays in the prediction set if its p-value exceeds eps = 1 - gate. Bet iff the set is exactly one class.
    Conformal guarantees COVERAGE (the true class is in the set >= gate of the time), not the precision of singleton
    sets, so by default a singleton must also carry calibrated confidence >= gate (`require_conf`). Feed it already
    calibrated probabilities."""

    def __init__(self, gate=0.80, require_conf=True):
        self.gate, self.require_conf = gate, require_conf

    def fit(self, p_cal, y_cal):
        p_cal, y_cal = np.asarray(p_cal, float), np.asarray(y_cal, float)
        self.alpha = {1: np.sort(1 - p_cal[y_cal == 1]), 0: np.sort(p_cal[y_cal == 0])}   # 1-(1-p)=p for class 0
        return self

    def _pvalue(self, cls, alpha_new):
        a = self.alpha[cls]
        if len(a) == 0:
            return np.ones_like(alpha_new)
        ge = len(a) - np.searchsorted(a, alpha_new, side="left")
        return (ge + 1) / (len(a) + 1)

    def decide(self, p_new) -> pd.DataFrame:
        p_new = np.asarray(p_new, float)
        eps = 1 - self.gate
        in_up = self._pvalue(1, 1 - p_new) > eps
        in_dn = self._pvalue(0, p_new) > eps
        bet = in_up ^ in_dn
        if self.require_conf:
            bet &= np.maximum(p_new, 1 - p_new) >= self.gate
        return pd.DataFrame({"p_up": p_new, "in_up": in_up, "in_dn": in_dn, "bet": bet,
                             "side": np.where(bet, np.where(in_up, 1, -1), 0)})


# ---- comparison -------------------------------------------------------------------------------------------------
def _row(name, p, y, bet, side, gate):
    y = np.asarray(y, float)
    n_bet = int(bet.sum())
    hit = int(((side[bet] > 0) == (y[bet] > 0.5)).sum())
    lo, hi = D.wilson(hit, n_bet)
    return dict(method=name, n=len(y), brier=D.brier(p, y), log_loss=D.log_loss(p, y), ece=D.ece(p, y),
                n_bet=n_bet, coverage=n_bet / max(len(y), 1), acc_bet=hit / n_bet if n_bet else np.nan,
                acc_lo=lo if n_bet else np.nan)


def compare_gates(raw_cal, y_cal, raw_te, y_te, gate=0.80, types_cal=None, types_te=None) -> pd.DataFrame:
    """Fit every route on the calibration block and evaluate on the later test block, same rows for all methods.
    Probability routes bet when max(p, 1-p) >= gate; the Venn-Abers and conformal routes use their own rule."""
    raw_cal, y_cal = np.asarray(raw_cal, float), np.asarray(y_cal, float)
    raw_te, y_te = np.asarray(raw_te, float), np.asarray(y_te, float)
    rows = []

    def prob_route(name, p):
        p = np.clip(p, 0.02, 0.98)
        conf = np.maximum(p, 1 - p)
        rows.append(_row(name, p, y_te, conf >= gate, np.where(p >= 0.5, 1, -1), gate))

    prob_route("none", raw_te)
    if len(np.unique(y_cal)) < 2 or len(y_cal) < 10:
        return pd.DataFrame(rows)
    plat = D._Cal("platt").fit(raw_cal, y_cal)
    prob_route("platt", plat(raw_te))
    prob_route("isotonic", D._Cal("isotonic").fit(raw_cal, y_cal)(raw_te))
    if types_cal is not None and types_te is not None:
        pt = PerTypeCalibrator().fit(raw_cal, y_cal, types_cal)
        prob_route("per_type_platt", pt(raw_te, types_te))
    va = VennAbersGate(gate).fit(raw_cal, y_cal)
    d = va.decide(raw_te)
    rows.append(_row("venn_abers_gate", d["p_up"].values, y_te, d["bet"].values, d["side"].values, gate))
    cg = ConformalGate(gate).fit(plat(raw_cal), y_cal)
    d = cg.decide(plat(raw_te))
    rows.append(_row("conformal_gate", d["p_up"].values, y_te, d["bet"].values, d["side"].values, gate))
    return pd.DataFrame(rows)
