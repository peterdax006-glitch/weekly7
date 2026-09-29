"""Dynamic reliability (contract C62 section 11; checklists D12, K05; canon C60). Bible phase serving: pattern reliability (9/10).

There is no permanent confidence number. Five dimensions are kept apart and updated as evidence arrives:

  truth_confidence     is the effect real?               P(long-run effect > 0), slow memory, skeptical prior, AR(1)-deflated n
  current_reliability  is it delivering NOW?             P(recent effect >= a fraction of what it used to deliver), fast memory
  transfer_confidence  will it hold somewhere new?       random-effects predictive P(effect > 0 in an unseen domain: year, regime, sector)
  context_confidence   does it work in TODAY's context?  kernel-weighted P(win | contexts like today's), shrunk to the base rate,
                                                         capped by any matching anti-context
  failure_risk         how likely is a break soon?       gamma-prior hazard from the item's own break history x current CUSUM stress

A dimension with too little evidence is None (UNTESTED), never a default number. `decide` turns a state into a decision, and
two contrasting profiles from the contract (high truth / low current / high transfer / high risk versus moderate truth / high
current / low transfer / low risk) get visibly different decisions. `walk_forward_reliability` scores current_reliability
against what happened next, against the base rate, so a reliability number that predicts nothing says so (the real-panel study
in state/research/pattern_reliability found AUC 0.4975; this module must not hide that).

Built on engine.pattern_reliability (`break_states`, `cusum_threshold`, `auc`, `brier`, `ece`, `walk_forward`,
`score_predictions`) and core.Confidence; lifts them from patterns to any evidence stream. Nothing reads a row dated at or after
`now`. IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import expit

from engine import pattern_reliability as PR

from .core import Confidence, FirewallBreach, as_date, clip01, stable_hash

DIMS = ("truth", "current_reliability", "transfer", "context", "failure_risk")

PARAMS = {
    "min_n": 12,                 # effective observations before a dimension is anything but UNTESTED
    "half_life_truth": 156.0,    # periods
    "half_life_current": 13.0,
    "half_life_context": 104.0,
    "tau_mult": 0.25,            # skeptical prior sd of the long-run effect, in units of the outcome sd
    "tau_current": 0.5,          # prior sd of the recent effect around the long-run effect
    "current_frac": 0.5,         # "delivering" = at least this fraction of the long-run effect
    "min_domains": 3, "min_domain_n": 8,
    "knn": 25, "ctx_prior": 10.0,
    "hazard_a0": 0.5, "hazard_b0": 52.0, "stress_beta": 1.0, "horizon": 13, "stress_window": 52,
    "arl0": 500, "k_floor": 0.08,
    "stale_half_life": 26.0,     # periods without evidence after which current_reliability halves its distance from 0.5
    "collapse_rho": 0.95,
    # decision thresholds
    "truth_hi": 0.90, "current_lo": 0.40, "current_hi": 0.70, "risk_hi": 0.50, "risk_lo": 0.25, "transfer_lo": 0.50,
    "block": 4, "boot": 300, "min_pred_n": 40,
}


def _cfg(cfg) -> dict:
    return {**PARAMS, **(cfg or {})}


# ------------------------------------------------------------------------------------------------- state record

@dataclasses.dataclass(frozen=True)
class ReliabilityState:
    """The five dimensions of one item at one moment plus the evidence behind each. None = UNTESTED (never 0)."""
    knowledge_id: str
    as_of: str
    truth: float | None
    current_reliability: float | None
    transfer: float | None
    context: float | None
    failure_risk: float | None
    n_obs: int
    n_eff_truth: float
    n_eff_current: float
    n_domains: int
    context_match: str = "unknown"             # in | out | unknown  (stated contexts / anti-contexts versus today's)
    notes: tuple = ()

    def check(self) -> list[str]:
        errs = []
        for d in DIMS:
            v = getattr(self, d)
            if v is not None and not (isinstance(v, float) and 0.0 <= v <= 1.0):
                errs.append(f"{d}={v!r} outside [0,1]")
        if self.n_obs < 0:
            errs.append("n_obs<0")
        return errs

    def untested(self) -> tuple[str, ...]:
        return tuple(d for d in DIMS if getattr(self, d) is None)

    def to_confidence(self) -> Confidence:
        return Confidence(truth=self.truth, current_reliability=self.current_reliability, context=self.context,
                          transfer=self.transfer, failure_risk=self.failure_risk)

    @property
    def state_id(self) -> str:
        return stable_hash([self.knowledge_id, self.as_of, [getattr(self, d) for d in DIMS], self.n_obs], 16)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["state_id"] = self.state_id
        return d


# ------------------------------------------------------------------------------------------------- estimators

def _finite(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def ar1_rho(x: np.ndarray) -> float:
    """Lag-1 autocorrelation clipped to [-0.5, 0.9]; 0 when it cannot be measured."""
    x = _finite(x)
    if len(x) < 8:
        return 0.0
    xc = x - x.mean()
    den = float(xc @ xc)
    return float(np.clip((xc[1:] @ xc[:-1]) / den, -0.5, 0.9)) if den > 1e-18 else 0.0


def discounted_stats(x: np.ndarray, half_life: float) -> tuple[float, float, float]:
    """(weighted mean, weighted sd, effective n) with weight 0.5 ** (age / half_life); age counted in observations."""
    x = _finite(x)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), 0.0
    w = 0.5 ** ((n - 1 - np.arange(n)) / max(half_life, 1e-9))
    W = w.sum()
    m = float((w * x).sum() / W)
    n_eff = float(W * W / (w * w).sum())
    var = float((w * (x - m) ** 2).sum() / W) * (n_eff / (n_eff - 1.0)) if n_eff > 1.0 else 0.0
    return m, math.sqrt(max(var, 0.0)), n_eff


def posterior_positive(m: float, sd: float, n_eff: float, rho: float, prior_mean: float, tau: float, threshold: float = 0.0) -> tuple[float, float]:
    """P(effect > threshold) and the posterior mean for a normal effect with a normal prior. The effective n is deflated for
    serial dependence, n * (1 - rho) / (1 + rho), when rho > 0 - overlapping or persistent outcomes are not independent."""
    if not (np.isfinite(m) and np.isfinite(sd)) or sd <= 1e-12 or n_eff <= 1.0:
        return 0.5, float(prior_mean)
    n_adj = n_eff * (1.0 - rho) / (1.0 + rho) if rho > 0 else n_eff
    se2 = sd * sd / max(n_adj, 1e-9)
    tau2 = max(tau * tau, 1e-18)
    prec = 1.0 / se2 + 1.0 / tau2
    post = (m / se2 + prior_mean / tau2) / prec
    return float(sps.norm.cdf((post - threshold) * math.sqrt(prec))), float(post)


def truth_confidence(values: Sequence[float] | np.ndarray, cfg=None) -> tuple[float | None, dict]:
    """Is the effect real? P(long-run mean outcome > 0) under a skeptical zero-centred prior. None below `min_n` effective
    observations. Returns (value, detail) - detail carries the shrunk effect the other dimensions compare against."""
    P = _cfg(cfg)
    x = _finite(values)
    m, sd, n_eff = discounted_stats(x, P["half_life_truth"])
    rho = ar1_rho(x)
    n_adj = n_eff * (1.0 - rho) / (1.0 + rho) if rho > 0 else n_eff
    if n_adj < P["min_n"] or not np.isfinite(sd) or sd <= 1e-12:
        return None, {"n_eff": n_adj, "effect": float("nan"), "sd": sd, "rho": rho}
    p, post = posterior_positive(m, sd, n_eff, rho, 0.0, P["tau_mult"] * sd)
    return float(p), {"n_eff": n_adj, "effect": post, "sd": sd, "rho": rho}


def current_reliability(values: Sequence[float] | np.ndarray, truth_detail: Mapping[str, float], cfg=None) -> tuple[float | None, dict]:
    """Is it delivering now? P(recent effect >= current_frac x the long-run effect), the recent effect having fast-decaying
    memory and a prior centred on the long-run effect. If the long-run effect is not positive the bar is zero."""
    P = _cfg(cfg)
    x = _finite(values)
    m, sd, n_eff = discounted_stats(x, P["half_life_current"])
    if not np.isfinite(sd) or sd <= 1e-12 or n_eff < 4.0 or len(x) < P["min_n"]:
        return None, {"n_eff": n_eff}
    lr = truth_detail.get("effect", float("nan"))
    lr = 0.0 if not np.isfinite(lr) else lr
    bar = P["current_frac"] * max(lr, 0.0)
    rho = ar1_rho(x[-max(P["min_n"] * 4, 40):])
    p, post = posterior_positive(m, sd, n_eff, rho, max(lr, 0.0), P["tau_current"] * sd, bar)
    return float(p), {"n_eff": n_eff, "recent_effect": post, "bar": bar}


def transfer_confidence(values: Sequence[float] | np.ndarray, domains: Sequence[Any] | np.ndarray, cfg=None) -> tuple[float | None, dict]:
    """Will it hold in a domain it has not seen? Random-effects view: the domain means (years, regimes, sectors) are draws from
    a population; P(a new draw > 0) = t-CDF of the mean over the between-domain spread. Needs `min_domains` domains of at
    least `min_domain_n` observations each - otherwise UNTESTED, because transfer cannot be inferred from a single domain."""
    P = _cfg(cfg)
    x = np.asarray(values, float)
    d = np.asarray(domains, dtype=object)
    ok = np.isfinite(x)
    x, d = x[ok], d[ok]
    means = []
    for g in sorted(set(d.astype(str))):
        v = x[d.astype(str) == g]
        if len(v) >= P["min_domain_n"]:
            means.append(float(v.mean()))
    k = len(means)
    if k < P["min_domains"]:
        return None, {"n_domains": k}
    mu, s = float(np.mean(means)), float(np.std(means, ddof=1))
    if s <= 1e-12:
        return (1.0 if mu > 0 else 0.0), {"n_domains": k, "between_sd": s, "mean": mu}
    t = mu / (s * math.sqrt(1.0 + 1.0 / k))
    return float(sps.t.cdf(t, k - 1)), {"n_domains": k, "between_sd": s, "mean": mu, "positive_domains": int(sum(m > 0 for m in means))}


def match_context(spec: Mapping[str, Any] | None, ctx_now: Mapping[str, Any] | None) -> str:
    """'in' when every stated condition holds today, 'out' when one is violated, 'unknown' when a needed column is missing or
    the spec is empty. A condition is (lo, hi) numeric bounds (None = open) or a collection of allowed values."""
    if not spec:
        return "unknown"
    if not ctx_now:
        return "unknown"
    verdict = "in"
    for col, cond in spec.items():
        v = ctx_now.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "unknown"
        if isinstance(cond, tuple) and len(cond) == 2 and all(c is None or isinstance(c, (int, float)) for c in cond):
            lo, hi = cond
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                verdict = "out"
        elif v not in cond:
            verdict = "out"
    return verdict


def context_confidence(hist_ctx: pd.DataFrame | None, values: Sequence[float] | np.ndarray, ctx_now: Mapping[str, Any] | None, cfg=None) -> tuple[float | None, dict]:
    """Does it work in a context like today's? Recency- and distance-weighted P(outcome > 0) over the `knn` nearest historical
    contexts (standardised numeric columns), shrunk toward the item's own base rate with `ctx_prior` pseudo-observations. None
    when today's context is missing a column or fewer than `knn` outcomes exist."""
    P = _cfg(cfg)
    if hist_ctx is None or hist_ctx.shape[1] == 0 or not ctx_now:
        return None, {"reason": "no context"}
    cols = [c for c in hist_ctx.columns if c in ctx_now]
    if not cols or any(ctx_now[c] is None or not np.isfinite(float(ctx_now[c])) for c in cols):
        return None, {"reason": "context columns missing today"}
    H = hist_ctx[cols].astype(float).values
    y = np.asarray(values, float)
    ok = np.isfinite(y) & np.isfinite(H).all(axis=1)
    H, y = H[ok], y[ok]
    if len(y) < P["knn"]:
        return None, {"reason": f"only {len(y)} outcomes with context"}
    mu, sd = H.mean(0), H.std(0)
    sd[sd <= 1e-12] = 1.0
    z = (H - mu) / sd
    now = (np.array([float(ctx_now[c]) for c in cols]) - mu) / sd
    d2 = ((z - now) ** 2).sum(1)
    near = np.argsort(d2)[: P["knn"]]
    h2 = max(float(np.median(d2[near])), 1e-9)
    age = (len(y) - 1 - near).astype(float)
    w = np.exp(-d2[near] / (2.0 * h2)) * 0.5 ** (age / P["half_life_context"])
    k_eff = float(w.sum() ** 2 / (w * w).sum())
    p_local = float((w * (y[near] > 0)).sum() / w.sum())
    base = float((y > 0).mean())
    prior = P["ctx_prior"]
    p = (k_eff * p_local + prior * base) / (k_eff + prior)
    return float(p), {"k_eff": k_eff, "p_local": p_local, "base": base, "distance_median": float(np.median(np.sqrt(d2[near])))}


def cusum_stress(values: Sequence[float] | np.ndarray, truth_detail: Mapping[str, float], cfg=None) -> float:
    """Downward CUSUM over the last `stress_window` outcomes against the shrunk long-run effect, as a share of its alarm
    threshold (0 = calm, 1 = at the alarm). 0 when the item has no positive long-run effect to fall short of."""
    P = _cfg(cfg)
    x = _finite(values)
    mu0 = truth_detail.get("effect", float("nan"))
    sd = truth_detail.get("sd", float("nan"))
    if len(x) < P["min_n"] or not np.isfinite(mu0) or mu0 <= 0 or not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    k = max(0.5 * mu0 / sd, P["k_floor"])
    h = PR.cusum_threshold(k, P["arl0"])
    S = 0.0
    for r in x[-P["stress_window"]:]:
        S = max(0.0, S + (mu0 - r) / sd - k)
    return float(S / h)


def break_history(values: Sequence[float] | np.ndarray) -> tuple[int, int]:
    """(number of breaks, periods spent in the working state) of the item's own history by the causal state machine."""
    x = _finite(values)
    if len(x) < 30:
        return 0, len(x)
    states, ev = PR.break_states(pd.DataFrame({"v": x}))
    return int(len(ev)), int((states.iloc[:, 0].values == 1).sum())


def failure_risk(values: Sequence[float] | np.ndarray, truth: float | None, truth_detail: Mapping[str, float], current: float | None, cfg=None) -> tuple[float | None, dict]:
    """P(a break within `horizon` periods) = 1 - exp(-hazard x horizon x exp(beta x stress)). The hazard is (breaks + a0) /
    (working periods + b0), a gamma-prior rate so an item with no history is not given zero risk; stress is the CUSUM share.
    Floored at 1 - truth for an effect not yet established (a phantom is a break waiting to be noticed). None when there is
    too little evidence to say anything."""
    P = _cfg(cfg)
    x = _finite(values)
    if truth is None or len(x) < P["min_n"]:
        return None, {"reason": "insufficient evidence"}
    n_break, exposure = break_history(x)
    hazard = (n_break + P["hazard_a0"]) / (exposure + P["hazard_b0"])
    stress = cusum_stress(x, truth_detail, P)
    rate = hazard * P["horizon"] * math.exp(P["stress_beta"] * min(stress, 1.5))
    risk = 1.0 - math.exp(-rate)
    if current is not None and current < 0.5:
        risk = max(risk, 0.5 * (1.0 - current))
    risk = max(risk, 1.0 - truth if truth < 0.5 else 0.0)
    return float(min(max(risk, 0.0), 1.0)), {"hazard": hazard, "stress": stress, "breaks": n_break, "exposure": exposure}


# ------------------------------------------------------------------------------------------------- state from evidence

def assert_no_future(frame: pd.DataFrame, now) -> None:
    """Fail closed if any evidence row is dated at or after `now`."""
    late = [i for i in frame.index if as_date(i) >= as_date(now)]
    if late:
        raise FirewallBreach(f"reliability evidence dated at/after now={now}: {late[:3]}")


def compute_state(knowledge_id: str, frame: pd.DataFrame, now, ctx_now: Mapping[str, Any] | None = None,
                  contexts: Mapping[str, Any] | None = None, anti_contexts: Mapping[str, Any] | None = None,
                  last_evidence=None, cfg=None) -> ReliabilityState:
    """All five dimensions from the evidence dated strictly before `now`. `frame` is indexed by the date each outcome matured and
    has a `value` column (signed; positive = worked), optionally `domain` (year / regime / sector label per row) and numeric
    context columns describing the situation the outcome was observed in."""
    P = _cfg(cfg)
    fr = frame.loc[[i for i in frame.index if as_date(i) < as_date(now)]]
    vals = fr["value"].astype(float).values if len(fr) else np.zeros(0)
    notes = []
    truth, td = truth_confidence(vals, P)
    cur, cd = current_reliability(vals, td, P)
    dom = fr["domain"].values if "domain" in fr.columns else np.array([str(as_date(i).year) for i in fr.index], dtype=object)
    trn, trd = transfer_confidence(vals, dom, P)
    ctx_cols = [c for c in fr.columns if c not in ("value", "domain")]
    ctxc, cxd = context_confidence(fr[ctx_cols] if ctx_cols else None, vals, ctx_now, P)
    match = match_context(contexts, ctx_now)
    if match == "unknown" and anti_contexts:
        match = "out" if match_context(anti_contexts, ctx_now) == "in" else "unknown"
    if anti_contexts and match_context(anti_contexts, ctx_now) == "in":
        match = "out"
        notes.append("an anti-context matches today: context confidence capped")
        ctxc = min(ctxc, 0.2) if ctxc is not None else 0.2
    elif match == "in" and ctxc is not None:
        notes.append("stated context matches today")
    risk, rd = failure_risk(vals, truth, td, cur, P)
    if last_evidence is not None and cur is not None:
        cur = stale_adjust(cur, periods_since(fr.index, last_evidence, now), P)
        notes.append("current_reliability decayed toward 0.5 for missing recent evidence")
    for name, v in (("truth", truth), ("current_reliability", cur), ("transfer", trn), ("context", ctxc), ("failure_risk", risk)):
        if v is None:
            notes.append(f"{name} UNTESTED")
    return ReliabilityState(knowledge_id, str(as_date(now)), None if truth is None else clip01(truth),
                            None if cur is None else clip01(cur), None if trn is None else clip01(trn),
                            None if ctxc is None else clip01(ctxc), None if risk is None else clip01(risk), int(len(vals)),
                            float(td.get("n_eff", 0.0)), float(cd.get("n_eff", 0.0)), int(trd.get("n_domains", 0)), match,
                            tuple(notes))


def periods_since(index, last_evidence, now) -> float:
    """Days between the newest evidence and `now`, in units of the median spacing of the index (>= 0)."""
    dts = [as_date(i) for i in index]
    if len(dts) < 3:
        return 0.0
    gaps = np.diff([d.toordinal() for d in dts])
    unit = float(np.median(gaps)) or 1.0
    return max((as_date(now).toordinal() - as_date(last_evidence).toordinal()) / unit - 1.0, 0.0)


def stale_adjust(current: float, periods_missing: float, cfg=None) -> float:
    """Evidence goes stale: with no new outcomes the reliability estimate regresses toward 0.5 with half-life
    `stale_half_life`. A reliability that never moves when the world goes quiet is a permanent number in disguise."""
    P = _cfg(cfg)
    keep = 0.5 ** (max(periods_missing, 0.0) / P["stale_half_life"])
    return float(0.5 + (current - 0.5) * keep)


class ReliabilityTracker:
    """Keeps every item's evidence and its state history. `update` appends new evidence (must be strictly older than `now` and
    strictly newer than what is stored), `state` recomputes as of any `now` from stored evidence only."""

    def __init__(self, cfg=None):
        self.cfg = _cfg(cfg)
        self._frames: dict[str, pd.DataFrame] = {}
        self._contexts: dict[str, tuple] = {}
        self._history: dict[str, list[ReliabilityState]] = {}

    def register(self, kid: str, contexts: Mapping[str, Any] | None = None, anti_contexts: Mapping[str, Any] | None = None) -> None:
        self._frames.setdefault(kid, pd.DataFrame({"value": []}, index=pd.DatetimeIndex([])))
        self._contexts[kid] = (dict(contexts or {}), dict(anti_contexts or {}))
        self._history.setdefault(kid, [])

    def known(self, kid: str) -> bool:
        return kid in self._frames

    def n_obs(self, kid: str) -> int:
        return len(self._frames[kid])

    def update(self, kid: str, records: pd.DataFrame, now, ctx_now: Mapping[str, Any] | None = None) -> ReliabilityState:
        """Add new evidence and return the new state. Raises FirewallBreach on any record dated at/after `now`, and ValueError on
        a record not newer than the stored evidence (history is append-only)."""
        if kid not in self._frames:
            raise KeyError(f"unregistered item {kid}")
        if len(records):
            assert_no_future(records, now)
            old = self._frames[kid]
            if len(old) and pd.Timestamp(records.index.min()) <= pd.Timestamp(old.index.max()):
                raise ValueError("new evidence must be newer than stored evidence")
            self._frames[kid] = pd.concat([old, records]) if len(old) else records.copy()
        return self.state(kid, now, ctx_now, record=True)

    def state(self, kid: str, now, ctx_now: Mapping[str, Any] | None = None, record: bool = False) -> ReliabilityState:
        fr = self._frames[kid]
        ctxs, anti = self._contexts[kid]
        last = fr.index.max() if len(fr) else None
        st = compute_state(kid, fr, now, ctx_now, ctxs, anti, last, self.cfg)
        if record:
            self._history[kid].append(st)
        return st

    def history(self, kid: str) -> list[ReliabilityState]:
        return list(self._history[kid])

    def snapshot(self, now, ctx_now: Mapping[str, Any] | None = None) -> list[ReliabilityState]:
        return [self.state(k, now, ctx_now) for k in sorted(self._frames)]


# ------------------------------------------------------------------------------------------------- decisions

@dataclasses.dataclass(frozen=True)
class ReliabilityDecision:
    action: str            # USE_FULL | USE_LOCAL | USE_REDUCED | STANDBY | ABSTAIN_UNTESTED
    weight: float          # multiplier on the item's influence in [0, 1]
    scope: str             # anywhere | current_conditions_only | none
    reasons: tuple


def decide(state: ReliabilityState, cfg=None, in_context: bool = True) -> ReliabilityDecision:
    """The decision the five dimensions imply. Profile A (high truth, low current reliability, high transfer, high failure risk)
    goes to STANDBY with zero weight: proven but not delivering, and about to break; it is watched, not used. Profile B
    (moderate truth, high current reliability, low transfer, low risk) is USED LOCALLY: it works here and now, but is not
    carried into a new context. `in_context=False` means the decision is for a context unlike the ones observed, where
    transfer confidence multiplies the weight."""
    P = _cfg(cfg)
    s = state
    why = []
    if s.truth is None or s.current_reliability is None:
        return ReliabilityDecision("ABSTAIN_UNTESTED", 0.0, "none", (f"untested: {', '.join(s.untested())}",))
    risk = 0.5 if s.failure_risk is None else s.failure_risk
    ctx = 1.0 if s.context is None else s.context
    if s.context_match == "out":
        why.append("today is outside the item's stated contexts (or inside an anti-context)")
        return ReliabilityDecision("STANDBY", 0.0, "none", tuple(why))
    if s.current_reliability < P["current_lo"]:
        why.append(f"current reliability {s.current_reliability:.2f} below {P['current_lo']}")
        if s.truth >= P["truth_hi"]:
            why.append(f"historical truth {s.truth:.2f} is high, so it is watched for recovery rather than retired")
        if risk >= P["risk_hi"]:
            why.append(f"failure risk {risk:.2f} at or above {P['risk_hi']}")
        return ReliabilityDecision("STANDBY", 0.0, "none", tuple(why))
    base = (s.truth ** 0.5) * s.current_reliability * (1.0 - risk) * (0.5 + 0.5 * ctx)
    transfer = 1.0 if s.transfer is None else s.transfer
    if not in_context:
        base *= transfer
        why.append(f"unfamiliar context: weight scaled by transfer confidence {transfer:.2f}")
    if s.current_reliability >= P["current_hi"] and risk <= P["risk_lo"]:
        if s.transfer is not None and s.transfer < P["transfer_lo"]:
            why.append(f"works now (reliability {s.current_reliability:.2f}) but transfer {s.transfer:.2f} is low: local use only")
            return ReliabilityDecision("USE_LOCAL", float(min(base, 1.0)), "current_conditions_only", tuple(why))
        why.append("reliable now, low failure risk")
        return ReliabilityDecision("USE_FULL", float(min(base, 1.0)), "anywhere", tuple(why))
    why.append("mixed evidence: reduced weight")
    return ReliabilityDecision("USE_REDUCED", float(min(base * 0.5, 1.0)), "anywhere" if s.transfer is None or s.transfer >= P["transfer_lo"]
                               else "current_conditions_only", tuple(why))


def profile_a() -> dict:
    """The first contrasting profile of section 11 as concrete numbers (used by tests and by the dashboard's legend)."""
    return {"truth": 0.97, "current_reliability": 0.20, "transfer": 0.90, "context": 0.50, "failure_risk": 0.70}


def profile_b() -> dict:
    return {"truth": 0.70, "current_reliability": 0.85, "transfer": 0.25, "context": 0.75, "failure_risk": 0.10}


def state_from_profile(kid: str, prof: Mapping[str, float], as_of="2000-01-01") -> ReliabilityState:
    return ReliabilityState(kid, str(as_of), prof["truth"], prof["current_reliability"], prof["transfer"], prof["context"],
                            prof["failure_risk"], 200, 100.0, 20.0, 5)


# ------------------------------------------------------------------------------------------------- diagnostics

def confidence_delta(prev: ReliabilityState | None, cur: ReliabilityState, tol: float = 0.02) -> list[dict]:
    """Which dimensions moved between two states and by how much - the 'what evidence changed it' half of the dashboard."""
    out = []
    for d in DIMS:
        b, a = (None if prev is None else getattr(prev, d)), getattr(cur, d)
        if a is None and b is None:
            continue
        if b is None or a is None or abs(a - b) > tol:
            out.append({"dimension": d, "before": b, "after": a, "delta": None if a is None or b is None else a - b})
    return out


def dimension_table(states: Sequence[ReliabilityState]) -> pd.DataFrame:
    """One row per item; None shows as NaN so 'untested' is never read as zero."""
    return pd.DataFrame([{"knowledge_id": s.knowledge_id, **{d: (np.nan if getattr(s, d) is None else getattr(s, d)) for d in DIMS},
                          "n_obs": s.n_obs} for s in states])


def collapse_check(states: Sequence[ReliabilityState], cfg=None) -> list[dict]:
    """Section 11 forbids one permanent confidence number. If two dimensions rank items almost identically across the book
    (|Spearman| >= collapse_rho) they have collapsed into one number and one of them is not measuring anything of its own."""
    P = _cfg(cfg)
    df = dimension_table(states)
    out = []
    for i, a in enumerate(DIMS):
        for b in DIMS[i + 1:]:
            ok = df[[a, b]].dropna()
            if len(ok) < 6 or ok[a].nunique() < 3 or ok[b].nunique() < 3:
                continue
            rho = float(sps.spearmanr(ok[a], ok[b])[0])
            if abs(rho) >= P["collapse_rho"]:
                out.append({"a": a, "b": b, "rho": rho, "n": int(len(ok))})
    return out


@dataclasses.dataclass(frozen=True)
class ReliabilityScore:
    n: int
    auc: float
    auc_lo: float
    auc_hi: float
    brier: float
    brier_base: float
    skill: float
    ece: float
    predictive: bool
    verdict: str
    skill_recalibrated: float = float("nan")      # Brier skill after a logistic recalibration fitted on the first half

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def walk_forward_reliability(values: Sequence[float], start: int = 60, step: int = 2, horizon: int = 4, cfg=None, seed: int = 0) -> tuple[ReliabilityScore, pd.DataFrame]:
    """Score current_reliability as a forecaster: at each origin t the estimate uses values[:t] only; the target is whether the
    mean of values[t:t+horizon] is positive. Compared with the expanding base rate of that target on the same origins'
    past (Brier skill) and with chance (AUC with a block-bootstrap interval). `predictive` is True only when the AUC
    interval excludes 0.5 AND Brier beats the base rate - otherwise the verdict says the number is not informative."""
    P = _cfg(cfg)
    x = np.asarray(values, float)
    rows = []
    for t in range(start, len(x) - horizon + 1, step):
        past = x[:t]
        truth, td = truth_confidence(past, P)
        cur, _ = current_reliability(past, td, P)
        if cur is None:
            continue
        fut = x[t:t + horizon]
        if np.isfinite(fut).sum() < horizon:
            continue
        y = float(np.nanmean(fut) > 0)
        hist_y = [float(np.nanmean(x[s:s + horizon]) > 0) for s in range(0, t - horizon + 1, step) if np.isfinite(x[s:s + horizon]).all()]
        rows.append({"t": t, "p": cur, "truth": truth, "y": y, "base": float(np.mean(hist_y)) if hist_y else 0.5})
    df = pd.DataFrame(rows, columns=["t", "p", "truth", "y", "base"])
    if len(df) < P["min_pred_n"]:
        return ReliabilityScore(len(df), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                                float("nan"), False, f"only {len(df)} scored origins (need {P['min_pred_n']})"), df
    a = PR.auc(df["p"].values, df["y"].values > 0)
    rng = np.random.default_rng(seed)
    n = len(df)
    blk = max(P["block"], 1)
    boots = []
    for _ in range(P["boot"]):
        idx = np.concatenate([(int(s) + np.arange(blk)) % n for s in rng.integers(0, n, size=math.ceil(n / blk))])[:n]
        boots.append(PR.auc(df["p"].values[idx], df["y"].values[idx] > 0))
    lo, hi = float(np.nanquantile(boots, 0.05)), float(np.nanquantile(boots, 0.95))
    b, bb = PR.brier(df["p"], df["y"]), PR.brier(df["base"], df["y"])
    skill = 1.0 - b / bb if bb > 1e-12 else float("nan")
    half = n // 2
    a0, a1 = platt_fit(df["p"].values[:half], df["y"].values[:half])
    pr2 = expit(a0 + a1 * _logit(df["p"].values[half:]))
    b2, bb2 = PR.brier(pr2, df["y"].values[half:]), PR.brier(df["base"].values[half:], df["y"].values[half:])
    skill2 = 1.0 - b2 / bb2 if bb2 > 1e-12 else float("nan")
    predictive = bool(lo > 0.5)
    if not predictive:
        verdict = "AUC interval includes chance: current_reliability does not rank what happens next"
    elif skill > 0:
        verdict = "informative"
    else:
        verdict = "ranks what happens next but is over-confident as a probability (raw Brier worse than the base rate)"
    return ReliabilityScore(n, float(a), lo, hi, b, bb, float(skill), PR.ece(df["p"], df["y"]), predictive, verdict,
                            float(skill2)), df


def _logit(p, eps: float = 1e-4):
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def platt_fit(p: np.ndarray, y: np.ndarray, iters: int = 40, ridge: float = 1e-3) -> tuple[float, float]:
    """(intercept, slope) of a logistic recalibration y ~ sigmoid(a + b * logit(p)) by ridge-damped Newton steps."""
    z = _logit(p)
    y = np.asarray(y, float)
    a, b = 0.0, 1.0
    for _ in range(iters):
        q = expit(a + b * z)
        g = np.array([(q - y).sum(), ((q - y) * z).sum()]) + ridge * np.array([a, b - 1.0])
        w = q * (1.0 - q)
        H = np.array([[w.sum(), (w * z).sum()], [(w * z).sum(), (w * z * z).sum()]]) + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if float(np.abs(step).max()) < 1e-8:
            break
    return float(a), float(b)


def calibration_table(scored: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Predicted vs realised frequency by equal-mass bin of current_reliability."""
    if len(scored) < bins * 4:
        return pd.DataFrame(columns=["bin", "n", "predicted", "realised"])
    order = np.argsort(scored["p"].values, kind="mergesort")
    rows = []
    for k, chunk in enumerate(np.array_split(order, bins)):
        rows.append({"bin": k, "n": len(chunk), "predicted": float(scored["p"].values[chunk].mean()),
                     "realised": float(scored["y"].values[chunk].mean())})
    return pd.DataFrame(rows)


def pooled_model_scores(tl, kind: str = "logit", cfg=None, seed: int = 0) -> dict:
    """The pooled meta-model of engine.pattern_reliability (one calibrated model across patterns and time, walk-forward) scored
    against base rates, returned as plain numbers. A thin, honest wrapper: it neither tunes nor re-implements anything."""
    pl = PR.walk_forward(tl, cfg, kind=kind, seed=seed)
    sc = PR.score_predictions(pl, seed=seed)
    return dict(sc) if isinstance(sc, Mapping) else {"scores": sc}


def apply_decision_to_weights(weights: Mapping[str, float], states: Mapping[str, ReliabilityState], cfg=None) -> dict[str, float]:
    """Multiply each item's proposed weight by its decision weight; items without a state are zeroed (unknown is not free)."""
    out = {}
    for k, w in weights.items():
        s = states.get(k)
        out[k] = 0.0 if s is None else float(w) * decide(s, cfg).weight
    return out


# ------------------------------------------------------------------------------------------------- streaming (O(1) per outcome)

class StreamingReliability:
    """Recursive version of truth_confidence / current_reliability: exponentially discounted sums updated in O(1) per outcome, so a
    live book of thousands of items does not recompute its whole history every period. Reproduces the batch estimators (the
    tests assert equality), including the lag-1 autocorrelation, from running sums."""

    def __init__(self, cfg=None):
        self.cfg = _cfg(cfg)
        self.lam_t = 0.5 ** (1.0 / self.cfg["half_life_truth"])
        self.lam_c = 0.5 ** (1.0 / self.cfg["half_life_current"])
        self.n = 0
        self.t = {"W": 0.0, "S1": 0.0, "S2": 0.0, "Q": 0.0}
        self.c = {"W": 0.0, "S1": 0.0, "S2": 0.0, "Q": 0.0}
        self.sum = self.sumsq = self.cross = 0.0
        self.first = self.last = None
        self.tail: list[float] = []

    def push(self, x: float) -> None:
        """Add one realised outcome. Non-finite outcomes are ignored, exactly as the batch estimators drop them."""
        x = float(x)
        if not math.isfinite(x):
            return
        for acc, lam in ((self.t, self.lam_t), (self.c, self.lam_c)):
            acc["W"] = lam * acc["W"] + 1.0
            acc["S1"] = lam * acc["S1"] + x
            acc["S2"] = lam * acc["S2"] + x * x
            acc["Q"] = lam * lam * acc["Q"] + 1.0
        if self.n:
            self.cross += x * self.last
        else:
            self.first = x
        self.sum += x
        self.sumsq += x * x
        self.last = x
        self.n += 1
        self.tail.append(x)
        if len(self.tail) > max(self.cfg["min_n"] * 4, 40):
            self.tail.pop(0)

    def _stats(self, acc) -> tuple[float, float, float]:
        if self.n == 0:
            return float("nan"), float("nan"), 0.0
        m = acc["S1"] / acc["W"]
        n_eff = acc["W"] ** 2 / acc["Q"]
        var = max(acc["S2"] / acc["W"] - m * m, 0.0) * (n_eff / (n_eff - 1.0)) if n_eff > 1.0 else 0.0
        return float(m), math.sqrt(var), float(n_eff)

    def rho(self) -> float:
        """Exact lag-1 autocorrelation about the plain mean from running sums (matches ar1_rho)."""
        if self.n < 8:
            return 0.0
        m = self.sum / self.n
        den = self.sumsq - self.n * m * m
        if den <= 1e-18:
            return 0.0
        num = self.cross - m * ((self.sum - self.first) + (self.sum - self.last)) + (self.n - 1) * m * m
        return float(np.clip(num / den, -0.5, 0.9))

    def truth(self) -> tuple[float | None, dict]:
        m, sd, n_eff = self._stats(self.t)
        rho = self.rho()
        n_adj = n_eff * (1.0 - rho) / (1.0 + rho) if rho > 0 else n_eff
        if n_adj < self.cfg["min_n"] or not np.isfinite(sd) or sd <= 1e-12:
            return None, {"n_eff": n_adj, "effect": float("nan"), "sd": sd, "rho": rho}
        p, post = posterior_positive(m, sd, n_eff, rho, 0.0, self.cfg["tau_mult"] * sd)
        return float(p), {"n_eff": n_adj, "effect": post, "sd": sd, "rho": rho}

    def current(self, truth_detail: Mapping[str, float]) -> tuple[float | None, dict]:
        m, sd, n_eff = self._stats(self.c)
        if not np.isfinite(sd) or sd <= 1e-12 or n_eff < 4.0 or self.n < self.cfg["min_n"]:
            return None, {"n_eff": n_eff}
        lr = truth_detail.get("effect", float("nan"))
        lr = 0.0 if not np.isfinite(lr) else lr
        bar = self.cfg["current_frac"] * max(lr, 0.0)
        rho = ar1_rho(np.array(self.tail))
        p, post = posterior_positive(m, sd, n_eff, rho, max(lr, 0.0), self.cfg["tau_current"] * sd, bar)
        return float(p), {"n_eff": n_eff, "recent_effect": post, "bar": bar}


# ------------------------------------------------------------------------------------------------- domains and transfer

def domain_breakdown(values: Sequence[float], domains: Sequence[Any], cfg=None) -> pd.DataFrame:
    """Per-domain mean, sd, t, and P(effect > 0) under the same skeptical prior as truth_confidence. Domains below
    `min_domain_n` are listed but flagged, not silently dropped."""
    P = _cfg(cfg)
    x = np.asarray(values, float)
    d = np.asarray(domains, dtype=object).astype(str)
    rows = []
    for g in sorted(set(d)):
        v = _finite(x[d == g])
        if len(v) < 2:
            rows.append({"domain": g, "n": len(v), "mean": float("nan"), "sd": float("nan"), "t": float("nan"),
                         "p_positive": float("nan"), "enough": False})
            continue
        sd = float(v.std(ddof=1))
        p, _ = posterior_positive(float(v.mean()), sd, float(len(v)), ar1_rho(v), 0.0, P["tau_mult"] * sd) if sd > 1e-12 else (0.5, 0.0)
        rows.append({"domain": g, "n": len(v), "mean": float(v.mean()), "sd": sd,
                     "t": float(v.mean() / (sd / math.sqrt(len(v)))) if sd > 1e-12 else 0.0,
                     "p_positive": p, "enough": len(v) >= P["min_domain_n"]})
    return pd.DataFrame(rows, columns=["domain", "n", "mean", "sd", "t", "p_positive", "enough"])


def heterogeneity(values: Sequence[float], domains: Sequence[Any], min_n: int = 8) -> dict:
    """Cochran's Q, I-squared and the DerSimonian-Laird between-domain variance of the domain means. High I-squared says the
    effect differs across domains, which is exactly when transfer_confidence should be low."""
    tab = domain_breakdown(values, domains, {"min_domain_n": min_n})
    tab = tab[tab["enough"]]
    k = len(tab)
    if k < 2:
        return {"k": k, "Q": float("nan"), "p": 1.0, "I2": float("nan"), "tau2": float("nan")}
    se2 = (tab["sd"] ** 2 / tab["n"]).replace(0.0, np.nan).fillna(1e-12).values
    w = 1.0 / se2
    mbar = float((w * tab["mean"].values).sum() / w.sum())
    Q = float((w * (tab["mean"].values - mbar) ** 2).sum())
    df = k - 1
    c = float(w.sum() - (w * w).sum() / w.sum())
    return {"k": k, "Q": Q, "p": float(sps.chi2.sf(Q, df)), "I2": float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0,
            "tau2": float(max(0.0, (Q - df) / c)) if c > 0 else 0.0}


def leave_one_domain_out(values: Sequence[float], domains: Sequence[Any], cfg=None) -> dict:
    """Validate transfer_confidence itself: for every domain, compute the transfer confidence from the OTHER domains and check
    it against whether the held-out domain's mean was positive. Reports the hit rate of sign(p > 0.5), the Brier score against
    always predicting the pooled rate, and the domains tested. A transfer number that fails this is decoration."""
    P = _cfg(cfg)
    x = np.asarray(values, float)
    d = np.asarray(domains, dtype=object).astype(str)
    rows = []
    for g in sorted(set(d)):
        held = _finite(x[d == g])
        if len(held) < P["min_domain_n"]:
            continue
        p, det = transfer_confidence(x[d != g], d[d != g], P)
        if p is None:
            continue
        rows.append({"domain": g, "p": p, "y": float(held.mean() > 0), "held_mean": float(held.mean())})
    if not rows:
        return {"n": 0, "hit_rate": float("nan"), "brier": float("nan"), "brier_base": float("nan"), "rows": []}
    df = pd.DataFrame(rows)
    base = float(df["y"].mean())
    return {"n": len(df), "hit_rate": float(((df["p"] > 0.5) == (df["y"] > 0.5)).mean()), "brier": PR.brier(df["p"], df["y"]),
            "brier_base": PR.brier(np.full(len(df), base), df["y"]), "rows": df.to_dict("records")}


# ------------------------------------------------------------------------------------------------- learning where it works

def context_reliability_map(hist_ctx: pd.DataFrame, values: Sequence[float] | np.ndarray, n_bins: int = 4) -> pd.DataFrame:
    """For every context column: quantile bins, and in each the number of outcomes, mean, hit rate and t. The raw material for
    'it works when...'. Bins with fewer than 5 outcomes are kept and reported with NaN t."""
    y = np.asarray(values, float)
    rows = []
    for c in hist_ctx.columns:
        x = hist_ctx[c].astype(float).values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 4 * n_bins:
            continue
        edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, n_bins + 1)))
        for k in range(len(edges) - 1):
            hi = edges[k + 1]
            sel = ok & (x >= edges[k]) & ((x <= hi) if k == len(edges) - 2 else (x < hi))
            v = y[sel]
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            rows.append({"column": c, "bin": k, "lo": float(edges[k]), "hi": float(hi), "n": int(sel.sum()),
                         "mean": float(v.mean()) if len(v) else float("nan"), "hit": float((v > 0).mean()) if len(v) else float("nan"),
                         "t": float(v.mean() / (sd / math.sqrt(len(v)))) if len(v) >= 5 and sd > 1e-12 else float("nan")})
    return pd.DataFrame(rows, columns=["column", "bin", "lo", "hi", "n", "mean", "hit", "t"])


def context_heterogeneity(hist_ctx: pd.DataFrame, values: Sequence[float] | np.ndarray, n_bins: int = 4) -> pd.DataFrame:
    """Does the outcome differ across the bins of each context column? One-way F test per column, Holm-adjusted across the
    columns searched. Columns with a significant adjusted p are the ones worth stating as contexts."""
    y = np.asarray(values, float)
    rows = []
    for c in hist_ctx.columns:
        x = hist_ctx[c].astype(float).values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 6 * n_bins:
            continue
        edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, n_bins + 1)))
        lab = np.clip(np.searchsorted(edges[1:-1], x[ok], side="right"), 0, len(edges) - 2)
        groups = [y[ok][lab == k] for k in range(len(edges) - 1) if (lab == k).sum() >= 3]
        if len(groups) < 2:
            continue
        f, p = sps.f_oneway(*groups)
        rows.append({"column": c, "F": float(f), "p": float(p), "spread": float(max(g.mean() for g in groups) - min(g.mean() for g in groups))})
    df = pd.DataFrame(rows, columns=["column", "F", "p", "spread"])
    if len(df):
        df["p_holm"] = PR.holm(df["p"].values)
    else:
        df["p_holm"] = []
    return df.sort_values("p_holm").reset_index(drop=True)


def learn_context_spec(hist_ctx: pd.DataFrame, values: Sequence[float], cfg=None, fit_frac: float = 0.6, alpha: float = 0.05) -> dict:
    """Learn a stated context ('works when column in [lo, hi]') from the first `fit_frac` of the evidence and keep it only if it
    also holds on the rest: same sign, a better mean inside than outside, and one-sided p under `alpha`. Returns
    {'spec': {col: (lo, hi)}, 'report': [...]}; the spec is empty when nothing validates - an item without a validated
    context simply has none."""
    n = len(values)
    cut = int(fit_frac * n)
    y = np.asarray(values, float)
    fit_h, fit_y = hist_ctx.iloc[:cut], y[:cut]
    het = context_heterogeneity(fit_h, fit_y)
    spec, report = {}, []
    for _, r in het[het["p_holm"] <= 0.10].iterrows():
        col = r["column"]
        m = context_reliability_map(fit_h[[col]], fit_y)
        good = m[(m["mean"] > 0) & (m["n"] >= 5)]
        if not len(good):
            continue
        lo, hi = float(good["lo"].min()), float(good["hi"].max())
        x_out = hist_ctx[col].astype(float).values[cut:]
        y_out = y[cut:]
        inside = (x_out >= lo) & (x_out <= hi) & np.isfinite(y_out)
        a, b = y_out[inside], y_out[~inside & np.isfinite(y_out)]
        if len(a) < 8 or len(b) < 8:
            report.append({"column": col, "validated": False, "why": "too few held-out rows on a side"})
            continue
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = (a.mean() - b.mean()) / se if se > 1e-12 else 0.0
        p = float(sps.norm.sf(t))
        ok = a.mean() > 0 and a.mean() > b.mean() and p <= alpha
        report.append({"column": col, "validated": bool(ok), "lo": lo, "hi": hi, "mean_in": float(a.mean()),
                       "mean_out": float(b.mean()), "p_oos": p})
        if ok:
            spec[col] = (lo, hi)
    return {"spec": spec, "report": report}


# ------------------------------------------------------------------------------------------------- scoring failure risk

def hazard_backtest(values: Sequence[float], start: int = 80, step: int = 4, horizon: int = 13, cfg=None) -> dict:
    """Score failure_risk as a forecaster of breaks: at each origin (item currently working) the estimate uses values[:t]; the
    label is whether the causal state machine detects a break in the next `horizon` periods. Reports AUC, Brier versus the
    base rate, and the number of origins/events, so a risk number with no skill is visible as such."""
    P = _cfg(cfg)
    x = np.asarray(values, float)
    rows = []
    for t in range(start, len(x) - horizon, step):
        past = x[:t]
        truth, td = truth_confidence(past, P)
        cur, _ = current_reliability(past, td, P)
        risk, _ = failure_risk(past, truth, td, cur, P)
        if risk is None:
            continue
        states, ev = PR.break_states(pd.DataFrame({"v": x[:t + horizon]}))
        if states.iloc[t - 1, 0] == 2:
            continue                                   # already broken: not a forecast of a new break
        hit = float(any(t <= int(e) < t + horizon for e in ev["detect"]))
        rows.append({"t": t, "risk": risk, "y": hit})
    df = pd.DataFrame(rows, columns=["t", "risk", "y"])
    if len(df) < 20 or df["y"].nunique() < 2:
        return {"n": len(df), "events": int(df["y"].sum()) if len(df) else 0, "auc": float("nan"), "brier": float("nan"),
                "brier_base": float("nan"), "informative": False}
    base = float(df["y"].mean())
    b, bb = PR.brier(df["risk"], df["y"]), PR.brier(np.full(len(df), base), df["y"])
    a = PR.auc(df["risk"].values, df["y"].values > 0)
    return {"n": len(df), "events": int(df["y"].sum()), "auc": float(a), "brier": b, "brier_base": bb,
            "informative": bool(a > 0.55 and b < bb)}


def state_series(frame: pd.DataFrame, start: int = 40, step: int = 4, cfg=None) -> pd.DataFrame:
    """The five dimensions over time for one evidence stream (walk-forward: each row uses only earlier rows) - the trajectory
    that lets a reviewer see reliability move as evidence arrives."""
    rows = []
    idx = list(frame.index)
    for t in range(start, len(idx), step):
        st = compute_state("series", frame.iloc[:t], idx[t], None, None, None, idx[t - 1], cfg)
        rows.append({"as_of": idx[t], **{d: getattr(st, d) for d in DIMS}, "n_obs": st.n_obs})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- decision-rule properties

def decision_grid(cfg=None, levels: Sequence[float] = (0.1, 0.5, 0.9), transfer: float = 0.8) -> pd.DataFrame:
    """The decision for every combination of truth / current / failure-risk levels. Used to inspect the rule as a table and to
    check its monotonicity."""
    rows = []
    for tr in levels:
        for cu in levels:
            for rk in levels:
                s = ReliabilityState("grid", "2000-01-01", tr, cu, transfer, 0.7, rk, 200, 100.0, 20.0, 5)
                d = decide(s, cfg)
                rows.append({"truth": tr, "current": cu, "risk": rk, "action": d.action, "weight": d.weight})
    return pd.DataFrame(rows)


def monotonicity_violations(grid: pd.DataFrame, tol: float = 1e-9) -> list[dict]:
    """Weight must not rise when failure risk rises, and must not fall when current reliability or truth rises, all else equal.
    A violation means the rule rewards the wrong thing."""
    out = []
    key = ["truth", "current", "risk"]
    for axis, sign in (("risk", -1), ("current", +1), ("truth", +1)):
        others = [k for k in key if k != axis]
        for _, sub in grid.groupby(others):
            s = sub.sort_values(axis)
            w = s["weight"].values * sign
            for i in range(len(w) - 1):
                if w[i + 1] < w[i] - tol:
                    out.append({"axis": axis, "at": {k: float(s.iloc[i][k]) for k in others}, "from": float(s.iloc[i][axis]),
                                "to": float(s.iloc[i + 1][axis]), "weights": (float(s.iloc[i]["weight"]), float(s.iloc[i + 1]["weight"]))})
    return out


def contrast_profiles(cfg=None) -> dict:
    """Decisions for the two contrasting profiles of section 11 side by side (they must differ)."""
    a = decide(state_from_profile("profile_a", profile_a()), cfg)
    b = decide(state_from_profile("profile_b", profile_b()), cfg)
    return {"A": dataclasses.asdict(a), "B": dataclasses.asdict(b), "differ": a.action != b.action or abs(a.weight - b.weight) > 1e-9}


# ------------------------------------------------------------------------------------------------- attribution and persistence

def explain_update(prev: ReliabilityState | None, cur: ReliabilityState, new_records: pd.DataFrame | None) -> dict:
    """Why did the state move? The new evidence (count, mean, t), the dimensions that moved and the dominant one. The 'what
    evidence caused the change' answer for the health dashboard."""
    deltas = confidence_delta(prev, cur)
    ev = {"n_new": 0, "mean_new": float("nan"), "t_new": float("nan")}
    if new_records is not None and len(new_records):
        v = _finite(new_records["value"].values)
        ev["n_new"] = int(len(v))
        if len(v):
            ev["mean_new"] = float(v.mean())
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            ev["t_new"] = float(v.mean() / (sd / math.sqrt(len(v)))) if sd > 1e-12 else 0.0
    movers = [d for d in deltas if d["delta"] is not None]
    top = max(movers, key=lambda d: abs(d["delta"]))["dimension"] if movers else None
    if top is None:
        sentence = "no dimension moved by more than the tolerance"
    else:
        d = [x for x in movers if x["dimension"] == top][0]
        sentence = (f"{top} moved {d['before']:.2f} -> {d['after']:.2f} after {ev['n_new']} new outcomes "
                    f"(mean {ev['mean_new']:+.3%})")
    return {"evidence": ev, "deltas": deltas, "dominant": top, "summary": sentence}


def state_from_dict(d: Mapping[str, Any]) -> ReliabilityState:
    """Inverse of ReliabilityState.as_dict (state_id is recomputed and must match if present)."""
    fields = {f.name for f in dataclasses.fields(ReliabilityState)}
    kw: dict[str, Any] = {k: (tuple(v) if k == "notes" else v) for k, v in d.items() if k in fields}
    st = ReliabilityState(**kw)
    if "state_id" in d and d["state_id"] != st.state_id:
        raise ValueError("state_id mismatch: record was altered")
    return st


def export_tracker(tr: ReliabilityTracker) -> dict:
    """Plain-data export of a tracker (evidence rows, contexts, state history) for the archive."""
    out = {}
    for k, fr in tr._frames.items():
        out[k] = {"evidence": [{"date": str(as_date(i)), **{c: (None if pd.isna(r[c]) else r[c]) for c in fr.columns}}
                               for i, r in fr.iterrows()],
                  "contexts": tr._contexts[k][0], "anti_contexts": tr._contexts[k][1],
                  "history": [s.as_dict() for s in tr._history[k]]}
    return out


def import_tracker(data: Mapping[str, Any], cfg=None) -> ReliabilityTracker:
    tr = ReliabilityTracker(cfg)
    for k, blob in data.items():
        tr.register(k, blob.get("contexts"), blob.get("anti_contexts"))
        rows = blob.get("evidence", [])
        if rows:
            fr = pd.DataFrame(rows)
            fr.index = pd.DatetimeIndex(fr.pop("date"))
            tr._frames[k] = fr
        tr._history[k] = [state_from_dict(h) for h in blob.get("history", [])]
    return tr


# ------------------------------------------------------------------------------------------------- choosing memory, uncertainty, the book

def select_half_life(values: Sequence[float], grid: Sequence[float] = (6.0, 13.0, 26.0, 52.0, 104.0), start: int = 60) -> dict:
    """Which memory length predicts this item best? For each candidate half-life, forecast the next outcome by the discounted
    mean of the outcomes before it and score the one-step squared error out of sample from `start` on. Returns the per-half-life
    error and the winner; a flat error profile (best within 1% of the worst) says the item's memory is not identifiable and the
    default should stay. It is a diagnostic of the fast-memory parameter, not a tuner that changes it."""
    x = np.asarray(values, float)
    n = len(x)
    if n < start + 20:
        return {"errors": {}, "best": None, "identifiable": False, "n": max(n - start, 0)}
    errs = {}
    for h in grid:
        lam = 0.5 ** (1.0 / h)
        num = den = 0.0
        se = []
        for t in range(n):
            if t >= start and np.isfinite(x[t]) and den > 0:
                se.append((x[t] - num / den) ** 2)
            if np.isfinite(x[t]):
                num, den = lam * num + x[t], lam * den + 1.0
            else:
                num, den = lam * num, lam * den
        errs[float(h)] = float(np.mean(se)) if se else float("nan")
    best = min(errs, key=lambda k: errs[k] if np.isfinite(errs[k]) else np.inf)
    spread = (max(errs.values()) - min(errs.values())) / max(max(errs.values()), 1e-18)
    return {"errors": errs, "best": best, "identifiable": bool(spread > 0.01), "spread": float(spread), "n": n - start}


def bootstrap_state(frame: pd.DataFrame, now, n_boot: int = 100, block: int = 8, seed: int = 0, cfg=None) -> dict:
    """Uncertainty of the five dimensions: recompute the state on block-bootstrapped evidence and report the 5th-95th percentile of
    each. A dimension whose interval spans most of [0, 1] is barely measured, whatever its point value says. Rows dated at
    or after `now` are excluded before resampling."""
    P = _cfg(cfg)
    fr = frame.loc[[i for i in frame.index if as_date(i) < as_date(now)]]
    n = len(fr)
    base = compute_state("boot", fr, now, None, None, None, None, P)
    if n < 2 * block:
        return {"point": base, "intervals": {}, "n": n}
    rng = np.random.default_rng(seed)
    draws: dict[str, list] = {d: [] for d in DIMS}
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boot = fr.iloc[np.sort(idx)].copy()
        boot.index = fr.index[:n]
        st = compute_state("boot", boot, now, None, None, None, None, P)
        for d in DIMS:
            v = getattr(st, d)
            if v is not None:
                draws[d].append(v)
    iv = {d: (float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))) for d, v in draws.items() if len(v) >= max(10, n_boot // 5)}
    return {"point": base, "intervals": iv, "n": n,
            "barely_measured": sorted(d for d, (lo, hi) in iv.items() if hi - lo > 0.6)}


def rank_items(states: Sequence[ReliabilityState], cfg=None, in_context: bool = True) -> pd.DataFrame:
    """Items ordered by the decision weight their state implies, with the action and reasons. Ties break by id so the order is
    deterministic. The table a portfolio layer would read."""
    rows = []
    for s in states:
        d = decide(s, cfg, in_context)
        rows.append({"knowledge_id": s.knowledge_id, "action": d.action, "weight": d.weight, "scope": d.scope,
                     "truth": s.truth, "current": s.current_reliability, "risk": s.failure_risk, "reasons": "; ".join(d.reasons)})
    df = pd.DataFrame(rows, columns=["knowledge_id", "action", "weight", "scope", "truth", "current", "risk", "reasons"])
    return df.sort_values(["weight", "knowledge_id"], ascending=[False, True]).reset_index(drop=True)


def book_summary(states: Sequence[ReliabilityState], cfg=None) -> dict:
    """Whole-book view: counts by action, how many items have each dimension untested, mean of each measured dimension, and
    whether any pair of dimensions has collapsed into one number."""
    acts: dict[str, int] = {}
    for s in states:
        a = decide(s, cfg).action
        acts[a] = acts.get(a, 0) + 1
    df = dimension_table(states)
    return {"n": len(states), "actions": acts,
            "untested": {d: int(df[d].isna().sum()) for d in DIMS} if len(df) else {},
            "means": {d: (float(df[d].mean()) if df[d].notna().any() else None) for d in DIMS} if len(df) else {},
            "collapsed": collapse_check(states, cfg)}


def decision_sensitivity(state: ReliabilityState, cfg=None, delta: float = 0.1) -> list[dict]:
    """How fragile is this item's decision? Nudge each measured dimension by +/- delta (clipped) and report which nudges change
    the action. An item that flips on a 0.1 nudge sits on a boundary and its weight should be read as uncertain."""
    base = decide(state, cfg)
    out = []
    for d in DIMS:
        v = getattr(state, d)
        if v is None:
            continue
        for sign in (-1, 1):
            changed: dict[str, Any] = {d: float(min(max(v + sign * delta, 0.0), 1.0))}
            moved = dataclasses.replace(state, **changed)
            dec = decide(moved, cfg)
            if dec.action != base.action:
                out.append({"dimension": d, "shift": sign * delta, "from": base.action, "to": dec.action})
    return out


# ------------------------------------------------------------------------------------------------- linking to the break engine

def contexts_from_condition(condition: Any) -> tuple[dict, dict]:
    """Turn a validated break-engine condition (terms of column / op / cut, the item works when ALL hold) into the two mappings the
    reliability state understands: `contexts` (works when column is in (lo, hi)) and `anti_contexts` (a single-term condition
    yields its complement; a conjunction has no single complement, so none is claimed). A condition without terms (a fitted
    state) yields nothing rather than a guess."""
    terms = getattr(condition, "terms", ())
    if not terms:
        return {}, {}
    ctx: dict[str, tuple] = {}
    for t in terms:
        lo, hi = ctx.get(t.column, (None, None))
        ctx[t.column] = (max(lo, t.cut) if (t.op == ">=" and lo is not None) else (t.cut if t.op == ">=" else lo),
                         min(hi, t.cut) if (t.op == "<=" and hi is not None) else (t.cut if t.op == "<=" else hi))
    anti: dict[str, tuple] = {}
    if len(terms) == 1:
        t = terms[0]
        anti[t.column] = (None, t.cut) if t.op == ">=" else (t.cut, None)
    return ctx, anti


def evidence_summary(frame: pd.DataFrame, now) -> dict:
    """What evidence a state rests on: outcome count, span in days, per-domain counts, share missing, newest outcome date."""
    fr = frame.loc[[i for i in frame.index if as_date(i) < as_date(now)]]
    if not len(fr):
        return {"n": 0, "span_days": 0, "domains": {}, "missing_share": float("nan"), "newest": None}
    v = fr["value"].astype(float)
    dom = fr["domain"].astype(str) if "domain" in fr.columns else pd.Series([str(as_date(i).year) for i in fr.index], index=fr.index)
    return {"n": int(v.notna().sum()), "span_days": (as_date(fr.index[-1]) - as_date(fr.index[0])).days,
            "domains": {k: int(c) for k, c in dom[v.notna()].value_counts().sort_index().items()},
            "missing_share": float(v.isna().mean()), "newest": str(as_date(fr.index[-1]))}


def dimension_history(states: Sequence[ReliabilityState]) -> pd.DataFrame:
    """A tracker's history for one item as a table with the change in each dimension from the previous state - how reliability
    moved as evidence arrived."""
    rows = []
    prev = None
    for s in states:
        row = {"as_of": s.as_of, "n_obs": s.n_obs}
        for d in DIMS:
            v = getattr(s, d)
            row[d] = np.nan if v is None else v
            p: float | None = None if prev is None else getattr(prev, d)
            row[f"d_{d}"] = np.nan if (v is None or p is None) else v - p
        rows.append(row)
        prev = s
    return pd.DataFrame(rows)


def regime_conditional_truth(values: Sequence[float], labels: Sequence[Any], cfg=None) -> pd.DataFrame:
    """truth_confidence computed separately inside each regime / sector / year label. Where the item's truth is high in one
    label and near 0.5 or lower in another, one global truth number was hiding a conditional item. Labels with too little
    evidence show NaN (untested), never a default."""
    P = _cfg(cfg)
    x = np.asarray(values, float)
    lab = np.asarray(labels, dtype=object).astype(str)
    rows = []
    for g in sorted(set(lab)):
        v = x[lab == g]
        p, det = truth_confidence(v, {**P, "min_n": min(P["min_n"], 8)})
        rows.append({"label": g, "n": int(np.isfinite(v).sum()), "truth": np.nan if p is None else p,
                     "effect": det.get("effect", np.nan)})
    return pd.DataFrame(rows, columns=["label", "n", "truth", "effect"])


def worst_label(table: pd.DataFrame) -> dict | None:
    """The label with the lowest measured truth, or None when no label was measurable."""
    t = table.dropna(subset=["truth"])
    if not len(t):
        return None
    r = t.sort_values("truth").iloc[0]
    return {"label": r["label"], "truth": float(r["truth"]), "n": int(r["n"])}


def normalised_weights(base: Mapping[str, float], states: Mapping[str, ReliabilityState], cfg=None, keep_total: bool = True) -> dict[str, float]:
    """Decision-scaled weights, optionally rescaled so the total equals the original total (capital freed from STANDBY items is
    redistributed pro rata). If every item is zero-weighted nothing is redistributed: an empty result is the honest answer."""
    scaled = apply_decision_to_weights(base, states, cfg)
    tot_new, tot_old = sum(scaled.values()), sum(base.values())
    if keep_total and tot_new > 1e-12:
        return {k: v * tot_old / tot_new for k, v in scaled.items()}
    return scaled


def untested_report(states: Sequence[ReliabilityState]) -> list[dict]:
    """Per item, which dimensions are UNTESTED and how many more outcomes a rough rule says are needed (min_n effective
    observations for truth / current, `min_domains` domains for transfer). A to-do list, not a verdict."""
    out = []
    for s in states:
        need = {}
        if s.truth is None:
            need["truth"] = f"needs about {max(PARAMS['min_n'] - int(s.n_eff_truth), 1)} more effective outcomes"
        if s.transfer is None:
            need["transfer"] = f"needs {max(PARAMS['min_domains'] - s.n_domains, 1)} more domain(s) with {PARAMS['min_domain_n']}+ outcomes"
        if s.context is None:
            need["context"] = "needs context columns for today and at least knn historical outcomes"
        if need:
            out.append({"knowledge_id": s.knowledge_id, "untested": s.untested(), "need": need})
    return out


def is_untested(state: ReliabilityState) -> bool:
    """True when any of truth / current reliability is missing - the two without which no decision can be made."""
    return state.truth is None or state.current_reliability is None


def tested_share(states: Sequence[ReliabilityState]) -> float:
    """Share of items whose truth AND current reliability are both measured; NaN for an empty book."""
    return sum(not is_untested(s) for s in states) / len(states) if states else float("nan")
