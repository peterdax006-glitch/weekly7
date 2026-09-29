"""Direction engine (Bible PHASE 13; canons: direction comes AFTER movement identification, calibrate, abstain).

Estimates P(up | stock moves) - trained and scored ONLY on rows the caller flags as movers - by stacking the evidence
sources (pattern score x per-type trust, analog up-fraction, missed-winner probability, model prediction, optional
evidence) in a regularised logistic model, then calibrating on a later time block (Platt / isotonic / none, chosen by
blocked-CV log loss) and evaluating on a still later, untouched block. Time-ordered three-way split by date:
train 60% / calibrate 20% / test 20%; rows whose forward window had not closed by `now` are dropped.

The gate: bet only when P(correct direction) = max(p, 1-p) >= gate (default 0.80) AND calibration is sufficient:
enough out-of-sample rows, ECE under a ceiling, and the gated test rows really are right about as often as claimed
(Wilson lower bound not far below the gate). If calibration is insufficient the engine abstains on every row and says
why. Reaching 0.80 is not assumed; `report` states plainly when it was not reached."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

INPUTS = ["pattern", "analog", "trust", "mw", "model", "evidence"]
EPS = 1e-6


# ---- metrics ----------------------------------------------------------------------------------------------------
def _clip(p):
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def log_loss(p, y):
    p, y = _clip(p), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(p) else float("nan")


def wilson(k, n, z=1.645):
    """Wilson score interval for a binomial proportion; (0, 1) when n == 0."""
    if n <= 0:
        return 0.0, 1.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def reliability_bins(p, y, n_bins=10, strategy="uniform") -> pd.DataFrame:
    """Calibration curve data: per bin mean predicted, observed frequency, n, Wilson interval. Empty bins dropped."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    cols = ["bin", "n", "mean_pred", "obs_freq", "lo", "hi"]
    if len(p) == 0:
        return pd.DataFrame(columns=cols)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for i in np.unique(b):
        m = b == i
        k, n = y[m].sum(), int(m.sum())
        lo, hi = wilson(k, n)
        rows.append((int(i), n, float(p[m].mean()), float(k / n), lo, hi))
    return pd.DataFrame(rows, columns=cols)


def ece(p, y, n_bins=10) -> float:
    """Expected calibration error: n-weighted mean |obs - pred| over uniform bins."""
    rb = reliability_bins(p, y, n_bins)
    return float((rb["n"] * (rb["obs_freq"] - rb["mean_pred"]).abs()).sum() / rb["n"].sum()) if len(rb) else float("nan")


def ece_noise_floor(p, n_bins=10) -> float:
    """ECE a PERFECTLY calibrated model would still show from sampling noise: sum_b (n_b/n) * E|Binom - np|/n_b."""
    p = np.asarray(p, float)
    if len(p) == 0:
        return float("nan")
    rb = reliability_bins(p, np.zeros(len(p)), n_bins)
    return float((rb["n"] / rb["n"].sum() * np.sqrt(2 / np.pi * rb["mean_pred"] * (1 - rb["mean_pred"]) / rb["n"])).sum())


def direction_accuracy(p_up, up, gate=None) -> dict:
    """Out-of-sample directional accuracy; with `gate`, also accuracy and coverage among rows with confidence >= gate."""
    p_up, up = np.asarray(p_up, float), np.asarray(up, float)
    if len(p_up) == 0:
        return dict(n=0, acc=float("nan"), base_rate=float("nan"))
    pred = p_up >= 0.5
    out = dict(n=len(p_up), acc=float((pred == (up > 0.5)).mean()), base_rate=float(up.mean()))
    if gate is not None:
        conf = np.maximum(p_up, 1 - p_up)
        m = conf >= gate
        k = int((pred[m] == (up[m] > 0.5)).sum())
        lo, hi = wilson(k, int(m.sum()))
        out.update(n_gated=int(m.sum()), coverage=float(m.mean()), acc_gated=float(k / m.sum()) if m.any() else float("nan"),
                   acc_gated_lo=lo, acc_gated_hi=hi)
    return out


def reliability_svg(bins: pd.DataFrame, title="reliability", size=320) -> str:
    """Self-contained SVG reliability diagram: diagonal = perfect calibration, dots sized by n, whiskers = Wilson CI."""
    pad, s = 36, size
    f = lambda v: pad + v * (s - 2 * pad)
    g = lambda v: s - pad - v * (s - 2 * pad)
    el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" font-family="sans-serif" font-size="10">',
          f'<rect x="{pad}" y="{pad}" width="{s - 2 * pad}" height="{s - 2 * pad}" fill="none" stroke="#999"/>',
          f'<line x1="{f(0)}" y1="{g(0)}" x2="{f(1)}" y2="{g(1)}" stroke="#bbb" stroke-dasharray="4"/>',
          f'<text x="{s / 2}" y="16" text-anchor="middle">{title}</text>',
          f'<text x="{s / 2}" y="{s - 8}" text-anchor="middle">predicted P(up)</text>']
    nmax = max(bins["n"].max(), 1) if len(bins) else 1
    for r in bins.itertuples():
        x = f(r.mean_pred)
        el.append(f'<line x1="{x:.1f}" y1="{g(r.lo):.1f}" x2="{x:.1f}" y2="{g(r.hi):.1f}" stroke="#36c"/>')
        el.append(f'<circle cx="{x:.1f}" cy="{g(r.obs_freq):.1f}" r="{2 + 5 * np.sqrt(r.n / nmax):.1f}" fill="#36c" opacity="0.7"/>')
    return "\n".join(el + ["</svg>"])


# ---- calibrators ------------------------------------------------------------------------------------------------
def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


class _Cal:
    """Maps raw stacker probability -> calibrated probability. kind in none / platt / isotonic."""

    def __init__(self, kind):
        self.kind = kind
        self.m = None

    def fit(self, p, y):
        if self.kind == "platt":
            self.m = LogisticRegression(C=1e4, max_iter=500).fit(_logit(p)[:, None], y)
        elif self.kind == "isotonic":
            self.m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return self

    def __call__(self, p):
        p = np.asarray(p, float)
        if self.kind == "platt":
            return self.m.predict_proba(_logit(p)[:, None])[:, 1]
        if self.kind == "isotonic":
            return self.m.predict(p)
        return p

    def clipped(self, p):
        """Calibrated probability limited to [0.02, 0.98]: a few hundred calibration rows cannot certify more."""
        return np.clip(self(p), 0.02, 0.98)


def _cv_logloss(kind, p, y, folds=4, min_fit=30):
    """Blocked (contiguous, time-ordered) K-fold log loss of a calibrator; inf when folds cannot be fit."""
    n = len(p)
    edges = np.linspace(0, n, folds + 1).astype(int)
    tot, cnt = 0.0, 0
    for i in range(folds):
        te = np.arange(edges[i], edges[i + 1])
        tr = np.setdiff1d(np.arange(n), te)
        if len(tr) < min_fit or len(np.unique(y[tr])) < 2:
            return float("inf")
        pc = np.clip(_Cal(kind).fit(p[tr], y[tr])(p[te]), 0.01, 0.99)   # isotonic can emit 0/1; cap the penalty
        tot += log_loss(pc, y[te]) * len(te)
        cnt += len(te)
    return tot / cnt


# ---- the engine -------------------------------------------------------------------------------------------------
def build_inputs(pattern=None, analog=None, trust=None, mw=None, model=None, evidence=None, index=None) -> pd.DataFrame:
    """Assemble the input frame. pattern/model/evidence are signed scores (+ = up); analog and mw are probabilities of
    an up move in [0,1]; trust in [0,1] scales the pattern score (0 = neutralized). Missing sources stay NaN."""
    src = dict(pattern=pattern, analog=analog, trust=trust, mw=mw, model=model, evidence=evidence)
    if index is None:
        for v in src.values():
            if v is not None and hasattr(v, "index"):
                index = v.index
                break
    if index is None:
        raise ValueError("need an index or at least one Series input")
    F = pd.DataFrame(index=index, columns=INPUTS, dtype=float)
    for k, v in src.items():
        if v is not None:
            F[k] = pd.Series(v, index=index) if not hasattr(v, "reindex") else v.reindex(index)
    return F


def _design(F: pd.DataFrame) -> np.ndarray:
    """Feature matrix: trust-scaled pattern, centred analog/mw probabilities, model, evidence, + missing flags."""
    F = F.reindex(columns=INPUTS)
    trust = F["trust"].fillna(0.0).clip(0, 1)
    cols = [F["pattern"].fillna(0.0) * trust,
            (F["analog"] - 0.5).fillna(0.0) * 2,
            (F["mw"] - 0.5).fillna(0.0) * 2,
            F["model"].fillna(0.0),
            F["evidence"].fillna(0.0),
            F["analog"].isna().astype(float), F["mw"].isna().astype(float), F["model"].isna().astype(float)]
    return np.column_stack([c.values for c in cols])


class DirectionEngine:
    def __init__(self, gate=0.80, C=0.5, min_rows=300, min_test=100, min_gated=30, max_ece=0.05, gate_slack=0.05,
                 min_inputs=2, horizon_days=7, split=(0.6, 0.2, 0.2), seed=0):
        self.p = dict(gate=gate, C=C, min_rows=min_rows, min_test=min_test, min_gated=min_gated, max_ece=max_ece,
                      gate_slack=gate_slack, min_inputs=min_inputs, horizon_days=horizon_days, split=split, seed=seed)
        self.stack = None
        self.cal = _Cal("none")
        self.mu = self.sd = None
        self.open = False
        self.reason = "not fitted"
        self.diag = {}

    def fit(self, F: pd.DataFrame, up: pd.Series, now, movers: pd.Series = None):
        """F: input frame indexed (date, ticker); up: 1/0 direction of the realised move; movers: bool mask of rows that
        actually moved (rows outside it are ignored - the target is P(up | moves))."""
        now = pd.Timestamp(now)
        self.open, self.reason, self.diag = False, "not fitted", {}
        up = up.reindex(F.index)
        ok = up.notna().to_numpy(copy=True)
        if movers is not None:
            ok = ok & movers.reindex(F.index).fillna(False).astype(bool).to_numpy()
        dates = pd.DatetimeIndex(F.index.get_level_values(0))
        ok = ok & np.asarray(dates + pd.Timedelta(days=self.p["horizon_days"]) <= now)
        F, up, dates = F[ok], up[ok].astype(float), dates[ok]
        n = len(F)
        self.diag["n_rows"] = n
        if n < self.p["min_rows"]:
            self.reason = f"insufficient data: {n} mover rows < {self.p['min_rows']}"
            return self
        order = np.argsort(dates.values, kind="stable")
        F, up, dates = F.iloc[order], up.iloc[order], dates[order]
        ud = np.unique(dates.values)
        c1, c2 = (int(len(ud) * s) for s in (self.p["split"][0], self.p["split"][0] + self.p["split"][1]))
        if c1 < 1 or c2 <= c1 or c2 >= len(ud):
            self.reason = "insufficient distinct dates for a three-way time split"
            return self
        d1, d2 = ud[c1], ud[c2]
        tr, ca, te = (dates.values < d1), (dates.values >= d1) & (dates.values < d2), (dates.values >= d2)
        X, y = _design(F), up.values
        if len(np.unique(y[tr])) < 2:
            self.reason = "training block has one class only"
            return self
        self.mu, self.sd = X[tr].mean(0), X[tr].std(0)
        self.sd[self.sd == 0] = 1.0
        self.stack = LogisticRegression(C=self.p["C"], max_iter=1000).fit((X[tr] - self.mu) / self.sd, y[tr])
        raw = self.stack.predict_proba((X - self.mu) / self.sd)[:, 1]
        # calibrator choice on the calibration block only
        scores = {k: _cv_logloss(k, raw[ca], y[ca]) for k in ("none", "platt", "isotonic")}
        scores["isotonic"] += 0.005          # isotonic must beat the smoother options clearly; it overfits small blocks
        kind = min(scores, key=scores.get)
        self.cal = _Cal(kind).fit(raw[ca], y[ca]) if len(np.unique(y[ca])) > 1 else _Cal("none")
        pc = self.cal.clipped(raw)
        self.diag.update(calibrator=self.cal.kind, cv_logloss=scores, n_train=int(tr.sum()), n_calib=int(ca.sum()),
                         n_test=int(te.sum()), calib_start=str(pd.Timestamp(d1).date()),
                         test_start=str(pd.Timestamp(d2).date()),
                         coef=dict(zip(["pattern*trust", "analog", "mw", "model", "evidence", "analog_na", "mw_na",
                                        "model_na"], self.stack.coef_[0].round(4).tolist())))
        pt, yt = pc[te], y[te]
        base = float(y[tr].mean())
        self.diag["test"] = dict(brier=brier(pt, yt), brier_base=brier(np.full(len(yt), base), yt),
                                 log_loss=log_loss(pt, yt), ece=ece(pt, yt), ece_floor=ece_noise_floor(pt), raw_brier=brier(raw[te], yt),
                                 **direction_accuracy(pt, yt, self.p["gate"]))
        self.diag["bins_test"] = reliability_bins(pt, yt, 10)
        self._decide_gate()
        return self

    def _decide_gate(self):
        t, g = self.diag["test"], self.p["gate"]
        if t["n"] < self.p["min_test"]:
            self.reason = f"calibration unproven: only {t['n']} out-of-sample rows < {self.p['min_test']}"
        elif not t["brier"] < t["brier_base"]:
            self.reason = "no skill: out-of-sample Brier is not better than the base rate"
        elif t["ece"] - t["ece_floor"] > self.p["max_ece"]:
            self.reason = (f"miscalibrated out of sample: ECE {t['ece']:.3f} (sampling floor {t['ece_floor']:.3f}) "
                           f"exceeds it by more than {self.p['max_ece']}")
        elif t.get("n_gated", 0) < self.p["min_gated"]:
            self.reason = f"no evidence at the gate: {t.get('n_gated', 0)} out-of-sample rows reach {g:.2f} (< {self.p['min_gated']})"
        elif t["acc_gated_lo"] < g - self.p["gate_slack"]:
            self.reason = (f"gated rows are not reliably right: accuracy {t['acc_gated']:.3f}, lower bound "
                           f"{t['acc_gated_lo']:.3f} < {g - self.p['gate_slack']:.2f}")
        else:
            self.open, self.reason = True, "open"

    def raw(self, F: pd.DataFrame) -> np.ndarray:
        """Uncalibrated stacker probability (input to the alternative calibrators in direction_calib)."""
        if self.stack is None:
            raise RuntimeError("engine not fitted: " + self.reason)
        return self.stack.predict_proba((_design(F) - self.mu) / self.sd)[:, 1]

    def predict(self, F: pd.DataFrame) -> pd.Series:
        """Calibrated P(up | moves). Raises if the engine could not be fitted (no stacker)."""
        if self.stack is None:
            raise RuntimeError("engine not fitted: " + self.reason)
        raw = self.stack.predict_proba((_design(F) - self.mu) / self.sd)[:, 1]
        return pd.Series(self.cal.clipped(raw), index=F.index, name="p_up")

    def decide(self, F: pd.DataFrame) -> pd.DataFrame:
        """One row per candidate: p_up, conf = P(correct direction), side (+1 up / -1 down), bet, reason.
        Abstains for every row when calibration was insufficient; otherwise when conf < gate or inputs are too sparse."""
        if self.stack is None:
            return pd.DataFrame({"p_up": np.nan, "conf": np.nan, "side": 0, "bet": False,
                                 "reason": "engine_closed: " + self.reason}, index=F.index)
        p = self.predict(F)
        conf = np.maximum(p, 1 - p)
        n_in = F.reindex(columns=["pattern", "analog", "mw", "model", "evidence"]).notna().sum(axis=1)
        trusted = F.reindex(columns=["trust"])["trust"].fillna(0) > 0
        n_in = n_in - ((F["pattern"].notna()) & (~trusted)).astype(int).reindex(F.index).fillna(0)
        reason = np.where(n_in.values < self.p["min_inputs"], "too_few_inputs",
                          np.where(conf.values < self.p["gate"], "below_gate", "bet"))
        if not self.open:
            reason = np.full(len(F), "engine_closed: " + self.reason, dtype=object)
        out = pd.DataFrame({"p_up": p.values, "conf": conf.values, "side": np.where(p.values >= 0.5, 1, -1),
                            "reason": reason}, index=F.index)
        out["bet"] = out["reason"] == "bet"
        out.loc[~out["bet"], "side"] = 0
        return out[["p_up", "conf", "side", "bet", "reason"]]

    def report(self) -> str:
        d = self.diag
        if "test" not in d:
            return f"direction engine: NOT FITTED ({self.reason}); rows={d.get('n_rows', 0)}. It abstains."
        t = d["test"]
        lines = [f"direction engine: gate {self.p['gate']:.2f}, status={'OPEN' if self.open else 'CLOSED - ' + self.reason}",
                 f"  rows train/calib/test = {d['n_train']}/{d['n_calib']}/{d['n_test']}  calibrator={d['calibrator']}",
                 f"  test: acc={t['acc']:.3f} (base rate {t['base_rate']:.3f})  brier={t['brier']:.4f} vs base {t['brier_base']:.4f}"
                 f"  logloss={t['log_loss']:.4f}  ECE={t['ece']:.3f}",
                 f"  at gate: n={t.get('n_gated', 0)} coverage={t.get('coverage', 0):.1%} acc={t.get('acc_gated', float('nan')):.3f} "
                 f"(90% CI {t.get('acc_gated_lo', 0):.3f}-{t.get('acc_gated_hi', 1):.3f})"]
        if not self.open and t.get("acc_gated", 0) < self.p["gate"]:
            lines.append("  0.80 was NOT demonstrated out of sample; the engine does not bet.")
        return "\n".join(lines)
