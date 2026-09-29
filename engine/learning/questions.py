"""Three separate questions (contract C62 section 33; feeds checklist C10/C11).  IMPLEMENTED - NOT VALIDATED.

Every learned relationship must answer, independently:
    1. IS IT REAL?        statistical reality of the relation (multiplicity-adjusted, permutation-checked, stable across eras)
    2. IS IT USEFUL?      incremental decision value out of sample (net of cost, tail-safe, not redundant)
    3. IS IT USEFUL NOW?  current applicability (inside its conditions, not in an anti-condition, not decayed, not broken)

The answers are three separate records with three separate evidence sources.  They never collapse into one score: the
ThreeAnswers object refuses to be converted to a number or ordered, and `disposition()` maps the *combination* to a decision
so that "real but not useful" and "useful but not now" lead to different actions.

Time rule: every series is dated by the day its outcome matured; anything at/after `now` raises FirewallBreach.
Builds on engine.pattern_reliability.nw_t (Newey-West) and engine.learning.boundary.scope_status (conditions)."""
from __future__ import annotations

import dataclasses as dc
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine.learning.boundary import ScopeStatus, scope_status
from engine.learning.core import (Confidence, DecisionEffect, Epistemic, FirewallBreach, Health, Unknown, _StrEnum,
                                  as_date, clip01, stable_hash)


class QuestionError(ValueError):
    """Malformed input to a question (duplicate dates, mismatched series).  Distinct from FirewallBreach."""


class Answer(_StrEnum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


@dc.dataclass(frozen=True)
class QuestionConfig:
    alpha: float = 0.05
    min_n: int = 20                       # periods needed before REAL can be answered YES/NO
    min_n_eff: float = 12.0
    min_stability: float = 0.6            # share of era blocks with the full-sample sign
    n_blocks: int = 4
    equiv_margin: float = 0.001           # |effect| below this counts as 'zero' for evidence of absence
    n_perm: int = 999
    perm_block: int = 4
    nw_lags: int = 4
    gain_margin: float = 0.0              # incremental gain must exceed this to be USEFUL
    tail_frac: float = 0.10
    tail_tolerance: float = 0.5           # tail may worsen by at most this many sd(without) before gain is vetoed
    max_redundancy: float = 0.9           # |corr| with existing knowledge above which a relation adds nothing
    min_oos_n: int = 12
    recent_window: int = 13
    min_recent: int = 6
    shortfall_z: float = 2.0
    cusum_k: float = 0.5
    cusum_h: float = 4.0
    decay_half_life: float = 8.0          # periods, for the decayed effect estimate
    seed: int = 0


DEFAULT_QCFG = QuestionConfig()


# ------------------------------------------------------------------------------------------------ helpers
def clean_series(s: Any, now, what: str) -> pd.Series:
    """Float series on a DatetimeIndex, sorted, NaN dropped.  Duplicate dates raise; any date >= now is a firewall breach."""
    if s is None:
        return pd.Series(dtype=float)
    x = pd.Series(s).astype(float)
    if len(x) == 0:
        return x
    x.index = pd.to_datetime(x.index)
    if x.index.duplicated().any():
        raise QuestionError(f"{what}: duplicate dates")
    x = x.sort_index()
    if x.index.max() >= pd.Timestamp(as_date(now)):
        raise FirewallBreach(f"{what}: observation dated {x.index.max().date()} is not before now={as_date(now)}")
    return x[np.isfinite(x.values)]


def effective_n(x: np.ndarray, max_lag: int = 8) -> float:
    """Autocorrelation-adjusted sample size (Bartlett weights).  n_eff <= n; positively autocorrelated weekly returns
    carry less independent information than their count suggests."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return float(n)
    xc = x - x.mean()
    g0 = float(xc @ xc)
    if g0 <= 0:
        return float(n)
    K = min(max_lag, n // 4)
    s = 0.0
    for k in range(1, K + 1):
        rho = float(xc[k:] @ xc[:-k]) / g0
        s += (1 - k / (K + 1.0)) * rho
    return float(np.clip(n / max(1.0 + 2.0 * s, 1.0 / n), 1.0, n))


def block_sign_flip_p(x: np.ndarray, block: int, n_perm: int, rng: np.random.Generator) -> float:
    """Randomisation test of 'mean effect is zero and symmetric': flip signs of whole blocks (keeping short-range
    dependence inside a block).  One-sided in |mean|.  Valid p-value with the +1 correction."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4:
        return 1.0
    block = max(1, min(block, n))
    ids = np.arange(n) // block
    nb = ids.max() + 1
    signs = rng.choice([-1.0, 1.0], size=(n_perm, nb))
    perm_means = (signs[:, ids] * x[None, :]).mean(axis=1)
    return float((1 + np.sum(np.abs(perm_means) >= abs(x.mean()) - 1e-15)) / (n_perm + 1))


def era_blocks(x: np.ndarray, k: int) -> list[float]:
    """Mean of each of k contiguous eras (fewer if there is too little data)."""
    k = max(1, min(k, len(x) // 4)) if len(x) else 1
    return [float(np.mean(b)) for b in np.array_split(np.asarray(x, float), k) if len(b)]


def _tail_mean(x: np.ndarray, frac: float) -> float:
    if len(x) == 0:
        return float("nan")
    k = max(1, int(math.ceil(frac * len(x))))
    return float(np.sort(x)[:k].mean())


def _nw_mean_se(x: np.ndarray, lags: int) -> tuple[float, float, float]:
    """(mean, se, t) with Newey-West se; falls back to iid when the NW estimate is undefined."""
    from engine.pattern_reliability import nw_t
    x = np.asarray(x, float)
    m = float(x.mean()) if len(x) else float("nan")
    if len(x) < 5:
        return m, float("nan"), float("nan")
    t = nw_t(x, lags=min(lags, len(x) - 1))
    iid = float(x.std(ddof=1) / math.sqrt(len(x)))
    se = abs(m / t) if math.isfinite(t) and abs(t) > 1e-12 else iid
    se = max(se, 0.5 * iid, 1e-12)                        # NW may not be wildly under iid
    return m, se, m / se


def _seed(cfg: QuestionConfig, rid: str) -> int:
    return int(stable_hash({"s": cfg.seed, "r": rid}, 8), 16) % (2 ** 32)


# ------------------------------------------------------------------------------------------------ input record
@dc.dataclass(eq=False)
class RelationEvidence:
    """Everything the three questions need about one relation.  All series are dated by outcome-maturity day.

    effect           per-period realised effect of the relation itself (e.g. long-short return of the pattern).
    n_trials         how many candidate relations were searched to find this one (multiple-testing burden).
    with_decision / without_decision   per-period decision outcome with and without the relation (paired by date).
    oos_start        first date that counts as out-of-sample for usefulness.
    expected_effect  what the current belief predicts per period (defaults to the pre-window mean of `effect`).
    contexts / anti_contexts   condition specs (see boundary.condition_holds); context_now = today's context features."""
    relation_id: str
    effect: pd.Series
    n_trials: int = 1
    with_decision: pd.Series | None = None
    without_decision: pd.Series | None = None
    oos_start: Any = None
    cost_per_period: float = 0.0
    redundancy: float | None = None
    expected_effect: float | None = None
    contexts: Mapping[str, Any] = dc.field(default_factory=dict)
    anti_contexts: Mapping[str, Any] = dc.field(default_factory=dict)
    context_now: Mapping[str, float] = dc.field(default_factory=dict)
    health: Health | None = None
    last_evidence_at: Any = None


# ------------------------------------------------------------------------------------------------ 1. IS IT REAL?
@dc.dataclass(frozen=True)
class RealAnswer:
    verdict: Answer
    unknown: Unknown | None
    effect: float
    se: float
    t: float
    p_nw: float
    p_perm: float
    p_final: float                        # multiplicity-adjusted, the larger of the two tests
    n: int
    n_eff: float
    ci_lo: float
    ci_hi: float
    stability: float                      # share of eras with the full-sample sign
    era_means: tuple[float, ...]
    n_trials: int
    reasons: tuple[str, ...]


def answer_real(rel: RelationEvidence, now, cfg: QuestionConfig = DEFAULT_QCFG, p_family: float | None = None) -> RealAnswer:
    """Statistical reality.  YES needs an adjusted p below alpha from BOTH the Newey-West and the permutation test AND a
    sign that holds in most eras.  NO needs enough data and a confidence interval that sits inside the equivalence margin
    (evidence of absence).  Everything else is UNKNOWN with the reason, never a silent NO."""
    x = clean_series(rel.effect, now, f"real[{rel.relation_id}]")
    n = len(x)
    if n < cfg.min_n:
        return RealAnswer(Answer.UNKNOWN, Unknown.INSUFFICIENT_DATA, float("nan"), float("nan"), float("nan"), 1.0, 1.0, 1.0,
                          n, float(n), float("nan"), float("nan"), 0.0, (), rel.n_trials, (f"only {n} periods (< {cfg.min_n})",))
    v = x.values
    m, se, t = _nw_mean_se(v, cfg.nw_lags)
    neff = effective_n(v)
    p_nw = float(2 * sps.norm.sf(abs(t))) if math.isfinite(t) else 1.0
    rng = np.random.default_rng(_seed(cfg, rel.relation_id))
    p_perm = block_sign_flip_p(v, cfg.perm_block, cfg.n_perm, rng)
    trials = max(int(rel.n_trials), 1)
    p_final = min(1.0, max(p_nw * trials, p_perm * trials))
    if p_family is not None:
        p_final = max(p_final, p_family)
    z = float(sps.norm.ppf(1 - cfg.alpha / 2))
    lo, hi = m - z * se, m + z * se
    eras = era_blocks(v, cfg.n_blocks)
    sign = np.sign(m) if m != 0 else 1.0
    stab = float(np.mean([np.sign(e) == sign for e in eras])) if eras else 0.0
    reasons = []
    if p_final < cfg.alpha and stab >= cfg.min_stability and neff >= cfg.min_n_eff:
        return RealAnswer(Answer.YES, None, m, se, t, p_nw, p_perm, p_final, n, neff, lo, hi, stab, tuple(eras), trials,
                          (f"adj p={p_final:.4f}", f"stable in {stab:.0%} of eras"))
    if p_final < cfg.alpha and stab < cfg.min_stability:
        reasons.append(f"significant but sign holds in only {stab:.0%} of eras")
        return RealAnswer(Answer.UNKNOWN, Unknown.CONFLICTED, m, se, t, p_nw, p_perm, p_final, n, neff, lo, hi, stab,
                          tuple(eras), trials, tuple(reasons))
    if p_final < cfg.alpha and neff < cfg.min_n_eff:
        reasons.append(f"n_eff={neff:.1f} below {cfg.min_n_eff}")
        return RealAnswer(Answer.UNKNOWN, Unknown.INSUFFICIENT_DATA, m, se, t, p_nw, p_perm, p_final, n, neff, lo, hi, stab,
                          tuple(eras), trials, tuple(reasons))
    if neff >= cfg.min_n_eff and lo > -cfg.equiv_margin and hi < cfg.equiv_margin:
        return RealAnswer(Answer.NO, None, m, se, t, p_nw, p_perm, p_final, n, neff, lo, hi, stab, tuple(eras), trials,
                          (f"CI [{lo:.4f},{hi:.4f}] inside +-{cfg.equiv_margin}",))
    reasons.append(f"adj p={p_final:.3f} not below {cfg.alpha} but CI [{lo:.4f},{hi:.4f}] too wide to call it absent")
    return RealAnswer(Answer.UNKNOWN, Unknown.INSUFFICIENT_DATA, m, se, t, p_nw, p_perm, p_final, n, neff, lo, hi, stab,
                      tuple(eras), trials, tuple(reasons))


# ------------------------------------------------------------------------------------------------ 2. IS IT USEFUL?
@dc.dataclass(frozen=True)
class UsefulAnswer:
    verdict: Answer
    unknown: Unknown | None
    gain: float                           # mean per-period decision outcome with minus without, before cost
    net_gain: float                       # after cost_per_period
    gain_se: float
    gain_lo: float
    gain_hi: float
    win_share: float                      # share of periods where using the relation helped
    risk_ratio: float                     # sd(with) / sd(without)
    tail_change: float                    # tail-mean(with) - tail-mean(without); positive = safer
    redundancy: float | None
    n: int
    reasons: tuple[str, ...]


def answer_useful(rel: RelationEvidence, now, cfg: QuestionConfig = DEFAULT_QCFG) -> UsefulAnswer:
    """Incremental decision value out of sample: does the portfolio decision do better WITH the relation than WITHOUT it,
    on the same dates, net of cost, without a worse tail, and not merely duplicating knowledge already in use."""
    def unk(why, u=Unknown.UNTESTED, n=0):
        return UsefulAnswer(Answer.UNKNOWN, u, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                            float("nan"), float("nan"), rel.redundancy, n, (why,))

    if rel.with_decision is None or rel.without_decision is None:
        return unk("no paired with/without decision series: usefulness untested")
    if rel.oos_start is None:
        return unk("no oos_start: usefulness cannot be judged in-sample")
    w = clean_series(rel.with_decision, now, f"useful.with[{rel.relation_id}]")
    wo = clean_series(rel.without_decision, now, f"useful.without[{rel.relation_id}]")
    j = pd.concat([w.rename("w"), wo.rename("wo")], axis=1, join="inner").dropna()
    if len(j) < 0.7 * min(len(w), len(wo)):
        return unk("with/without series are misaligned (under 70% of dates paired)", Unknown.CONFLICTED, len(j))
    j = j[j.index >= pd.Timestamp(as_date(rel.oos_start))]
    if len(j) < cfg.min_oos_n:
        return unk(f"only {len(j)} paired out-of-sample periods (< {cfg.min_oos_n})", Unknown.INSUFFICIENT_DATA, len(j))
    d = (j["w"] - j["wo"]).values
    gain, se, _ = _nw_mean_se(d, cfg.nw_lags)
    net = gain - rel.cost_per_period
    z = float(sps.norm.ppf(1 - cfg.alpha))                   # one-sided: does it beat the margin?
    lo, hi = net - z * se, net + z * se
    tail = _tail_mean(j["w"].values, cfg.tail_frac) - _tail_mean(j["wo"].values, cfg.tail_frac)
    sd_wo = float(j["wo"].std(ddof=1)) or 1e-12
    risk_ratio = float(j["w"].std(ddof=1)) / sd_wo
    win = float(np.mean(d > 0))
    reasons = [f"net gain {net:.5f}/period, one-sided CI [{lo:.5f},{hi:.5f}]"]
    if rel.redundancy is not None and abs(rel.redundancy) >= cfg.max_redundancy:
        reasons.append(f"redundant with existing knowledge (|corr|={abs(rel.redundancy):.2f})")
        return UsefulAnswer(Answer.NO, None, gain, net, se, lo, hi, win, risk_ratio, tail, rel.redundancy, len(j), tuple(reasons))
    tail_bad = tail < -cfg.tail_tolerance * sd_wo
    if lo > cfg.gain_margin and not tail_bad:
        return UsefulAnswer(Answer.YES, None, gain, net, se, lo, hi, win, risk_ratio, tail, rel.redundancy, len(j), tuple(reasons))
    if lo > cfg.gain_margin and tail_bad:
        reasons.append(f"gain vetoed: tail worsens by {-tail / sd_wo:.2f} sd")
        return UsefulAnswer(Answer.NO, None, gain, net, se, lo, hi, win, risk_ratio, tail, rel.redundancy, len(j), tuple(reasons))
    if hi <= cfg.gain_margin:
        reasons.append("upper bound of gain does not reach the margin: adds nothing material")
        return UsefulAnswer(Answer.NO, None, gain, net, se, lo, hi, win, risk_ratio, tail, rel.redundancy, len(j), tuple(reasons))
    reasons.append("gain CI straddles the margin")
    return UsefulAnswer(Answer.UNKNOWN, Unknown.INSUFFICIENT_DATA, gain, net, se, lo, hi, win, risk_ratio, tail, rel.redundancy,
                        len(j), tuple(reasons))


# ------------------------------------------------------------------------------------------------ 3. IS IT USEFUL NOW?
@dc.dataclass(frozen=True)
class UsefulNowAnswer:
    verdict: Answer
    unknown: Unknown | None
    scope: str                            # ScopeStatus value
    recent_mean: float
    expected: float
    shortfall_z: float                    # (recent - expected) / se_recent; very negative = underperforming its belief
    cusum: float                          # most negative standardised cumulative shortfall in the window
    decayed_effect: float                 # recency-weighted effect estimate
    recent_n: int
    days_since_evidence: int | None
    health: str
    reasons: tuple[str, ...]


def answer_useful_now(rel: RelationEvidence, now, cfg: QuestionConfig = DEFAULT_QCFG) -> UsefulNowAnswer:
    """Current applicability.  Independent of the other two: a real, useful relation can be NO here (outside its
    conditions, decayed, or broken) and a relation that is unproven can look fine here."""
    x = clean_series(rel.effect, now, f"now[{rel.relation_id}]")
    sc = scope_status(rel.contexts, rel.anti_contexts, rel.context_now)
    hstr = str(rel.health) if rel.health is not None else ""
    days = (as_date(now) - as_date(rel.last_evidence_at)).days if rel.last_evidence_at is not None else \
        ((as_date(now) - x.index.max().date()).days if len(x) else None)

    def out(v, u, reasons, rm=float("nan"), ex=float("nan"), sz=float("nan"), cu=float("nan"), de=float("nan"), rn=0):
        return UsefulNowAnswer(v, u, str(sc), rm, ex, sz, cu, de, rn, days, hstr, tuple(reasons))

    if rel.health in (Health.BROKEN, Health.CONTRADICTED):
        return out(Answer.NO, None, [f"health monitor says {rel.health}"])
    if sc in (ScopeStatus.ANTI_HIT, ScopeStatus.OUT_OF_SCOPE):
        return out(Answer.NO, None, [f"today's context is {sc}"])
    w = cfg.recent_window
    if len(x) < w + cfg.min_n // 2:
        rec = x.values[-w:] if len(x) else np.array([])
        if len(rec) < cfg.min_recent:
            return out(Answer.UNKNOWN, Unknown.INSUFFICIENT_DATA, [f"only {len(rec)} recent periods"], rn=len(rec))
    rec = x.values[-w:]
    hist = x.values[:-w] if len(x) > w else x.values
    expected = float(rel.expected_effect) if rel.expected_effect is not None else float(np.mean(hist))
    sd_h = float(np.std(hist, ddof=1)) if len(hist) > 2 else float(np.std(x.values, ddof=1)) if len(x) > 2 else 0.0
    sd_h = sd_h or 1e-9
    rmean = float(np.mean(rec))
    sz = (rmean - expected) / (sd_h / math.sqrt(len(rec)))
    s, cmin = 0.0, 0.0
    for v in (rec - expected) / sd_h:
        s = min(0.0, s + v + cfg.cusum_k)
        cmin = min(cmin, s)
    wts = 0.5 ** (np.arange(len(rec))[::-1] / cfg.decay_half_life)
    dec = float(np.sum(wts * rec) / np.sum(wts))
    args = dict(rm=rmean, ex=expected, sz=float(sz), cu=float(cmin), de=dec, rn=len(rec))
    reasons = []
    if len(rec) < cfg.min_recent:
        return out(Answer.UNKNOWN, Unknown.INSUFFICIENT_DATA, [f"only {len(rec)} recent periods"], **args)
    if sc == ScopeStatus.UNKNOWN:
        return out(Answer.UNKNOWN, Unknown.UNKNOWN, ["context features needed to check conditions are missing"], **args)
    reversed_ = expected != 0 and np.sign(dec) != np.sign(expected)
    if sz < -cfg.shortfall_z or abs(cmin) > cfg.cusum_h or reversed_:
        if sz < -cfg.shortfall_z:
            reasons.append(f"recent mean {rmean:.4f} is {-sz:.1f} se below expected {expected:.4f}")
        if abs(cmin) > cfg.cusum_h:
            reasons.append(f"CUSUM alarm ({cmin:.1f})")
        if reversed_:
            reasons.append("recency-weighted effect has the opposite sign")
        return out(Answer.NO, None, reasons, **args)
    return out(Answer.YES, None, [f"recent {rmean:.4f} vs expected {expected:.4f} (z={sz:.1f}); scope {sc}"], **args)


# ------------------------------------------------------------------------------------------------ the triple
class ThreeAnswers:
    """Real / useful / useful-now for one relation.  Deliberately not a number: no __float__, no ordering, no `score`."""
    __slots__ = ("relation_id", "as_of", "real", "useful", "useful_now")

    def __init__(self, relation_id: str, as_of: str, real: RealAnswer, useful: UsefulAnswer, useful_now: UsefulNowAnswer):
        object.__setattr__(self, "relation_id", relation_id)
        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(self, "real", real)
        object.__setattr__(self, "useful", useful)
        object.__setattr__(self, "useful_now", useful_now)

    def __setattr__(self, k, v):
        raise AttributeError("ThreeAnswers is immutable")

    def __float__(self):
        raise TypeError("the three answers must never be collapsed into one number (contract section 33)")

    __int__ = __float__

    def _no_order(self, other):
        raise TypeError("ThreeAnswers are not ordered: compare the three verdicts explicitly")

    __lt__ = __le__ = __gt__ = __ge__ = _no_order

    @property
    def verdicts(self) -> tuple[Answer, Answer, Answer]:
        return self.real.verdict, self.useful.verdict, self.useful_now.verdict

    @property
    def code(self) -> str:
        return "".join(v.value[0] for v in self.verdicts)             # e.g. 'YYN'

    def inconsistencies(self) -> list[str]:
        out = []
        if self.real.verdict == Answer.NO and self.useful.verdict == Answer.YES:
            out.append("USEFUL_WITHOUT_REAL: paid off out of sample but shows no statistical reality: luck, leak or a proxy")
        if self.real.verdict == Answer.NO and self.useful_now.verdict == Answer.YES:
            out.append("NOW_WITHOUT_REAL: recent performance of an effect that is not real: probably noise")
        if self.useful.verdict == Answer.NO and self.useful_now.verdict == Answer.YES and self.useful.unknown is None \
                and self.real.verdict == Answer.YES:
            out.append("REAL_NOW_BUT_NOT_USEFUL: fine in isolation, adds nothing to the decision")
        return out

    def confidence(self) -> Confidence:
        """Fill the SEPARATE confidence dimensions of core.Confidence from the separate answers (None = untested)."""
        r, u, n = self.real, self.useful, self.useful_now
        truth = None if r.verdict == Answer.UNKNOWN and r.unknown == Unknown.INSUFFICIENT_DATA and not math.isfinite(r.effect) \
            else clip01(1.0 - r.p_final)
        if u.verdict == Answer.UNKNOWN and not math.isfinite(u.gain):
            usefulness = None
        else:
            zz = (u.net_gain / u.gain_se) if u.gain_se and math.isfinite(u.gain_se) and u.gain_se > 0 else 0.0
            usefulness = clip01(float(sps.norm.cdf(zz)))
        if n.verdict == Answer.UNKNOWN and not math.isfinite(n.recent_mean):
            cur = None
        else:
            cur = clip01(float(sps.norm.cdf(n.shortfall_z + 2.0))) if math.isfinite(n.shortfall_z) else None
            if n.verdict == Answer.NO:
                cur = min(cur if cur is not None else 0.2, 0.2)
        return Confidence(truth=truth, usefulness=usefulness, current_reliability=cur)

    def to_dict(self) -> dict:
        return {"relation_id": self.relation_id, "as_of": self.as_of, "code": self.code,
                "real": _plain(self.real), "useful": _plain(self.useful), "useful_now": _plain(self.useful_now),
                "inconsistencies": self.inconsistencies()}

    def __repr__(self):
        return f"ThreeAnswers({self.relation_id!r}, real={self.real.verdict}, useful={self.useful.verdict}, now={self.useful_now.verdict})"


def _plain(rec) -> dict:
    d = {}
    for f in dc.fields(rec):
        v = getattr(rec, f.name)
        d[f.name] = str(v) if isinstance(v, _StrEnum) else (list(v) if isinstance(v, tuple) else v)
    return d


# ------------------------------------------------------------------------------------------------ decision from the triple
@dc.dataclass(frozen=True)
class Disposition:
    action: str
    epistemic: Epistemic
    allowed_effects: tuple[DecisionEffect, ...]
    size_multiplier: float                # 0..1 cap on influence
    reason: str


_FULL = (DecisionEffect.RANKING, DecisionEffect.SELECTION, DecisionEffect.PATTERN_WEIGHTING, DecisionEffect.CONFIDENCE)


def disposition(t: ThreeAnswers) -> Disposition:
    """Different combinations, different decisions.  This is the reason the answers are kept apart."""
    R, U, N = t.verdicts
    Y, No, Q = Answer.YES, Answer.NO, Answer.UNKNOWN
    if t.inconsistencies() and R == No:
        return Disposition("QUARANTINE", Epistemic.GATED, (DecisionEffect.RESEARCH_PRIORITY,), 0.0,
                           "looks useful without being real: investigate for leakage or luck before any use")
    if R == No:
        return Disposition("REJECT", Epistemic.RETIRED, (DecisionEffect.NONE,), 0.0, "not statistically real")
    if R == Q:
        return Disposition("COLLECT_EVIDENCE", Epistemic.HYPOTHESIS, (DecisionEffect.RESEARCH_PRIORITY,), 0.0,
                           "reality not established: gather data, do not act")
    # real
    if U == No:
        return Disposition("KEEP_AS_KNOWLEDGE", Epistemic.SUPPORTED, (DecisionEffect.RESEARCH_PRIORITY, DecisionEffect.CONFIDENCE),
                           0.0, "real but adds no decision value: keep as explanation, do not trade on it")
    if U == Q:
        return Disposition("RUN_EXPERIMENT", Epistemic.SUPPORTED, (DecisionEffect.RESEARCH_PRIORITY,), 0.0,
                           "real, decision value untested: shadow-test before use")
    # real and useful
    if N == Y:
        return Disposition("USE", Epistemic.SUPPORTED, _FULL, 1.0, "real, useful and currently applicable")
    if N == Q:
        return Disposition("USE_REDUCED", Epistemic.CONDITIONAL, _FULL, 0.4, "real and useful; current applicability unclear")
    return Disposition("DORMANT", Epistemic.DEGRADED, (DecisionEffect.RESEARCH_PRIORITY,), 0.0,
                       "real and useful historically but not now: monitor for recovery of its conditions")


# ------------------------------------------------------------------------------------------------ engine
class QuestionEngine:
    """Asks the three questions for one relation or a whole family (Holm across the family for the REAL question)."""

    def __init__(self, cfg: QuestionConfig = DEFAULT_QCFG):
        self.cfg = cfg

    def ask(self, rel: RelationEvidence, now, p_family: float | None = None) -> ThreeAnswers:
        r = answer_real(rel, now, self.cfg, p_family)
        u = answer_useful(rel, now, self.cfg)
        n = answer_useful_now(rel, now, self.cfg)
        return ThreeAnswers(rel.relation_id, as_date(now).isoformat(), r, u, n)

    def ask_many(self, rels: Sequence[RelationEvidence], now) -> list[ThreeAnswers]:
        """Family-wise control: the REAL verdicts use Holm-adjusted p-values over all relations asked together."""
        from engine.pattern_reliability import holm
        if not rels:
            return []
        first = [answer_real(r, now, self.cfg) for r in rels]
        ps = np.array([a.p_final for a in first])
        adj = holm(ps) if len(ps) > 1 else ps
        out = []
        for r, a in zip(rels, adj):
            out.append(self.ask(r, now, p_family=float(a)))
        return out


def dissociation_table(answers: Iterable[ThreeAnswers]) -> pd.DataFrame:
    """Counts of every real/useful/now combination.  A table where only YYY and NNN occur means the three questions
    are secretly one question (a red flag the audit must raise)."""
    rows = [a.code for a in answers]
    if not rows:
        return pd.DataFrame({"code": [], "count": []})
    s = pd.Series(rows).value_counts().rename_axis("code").reset_index(name="count")
    return s.sort_values(["count", "code"], ascending=[False, True]).reset_index(drop=True)


def collapse_risk(answers: Sequence[ThreeAnswers]) -> dict:
    """How independent are the three verdicts across relations?  Mean absolute pairwise agreement of YES/NO codes; 1.0 means
    the questions always agree (collapsed)."""
    codes = [a.code for a in answers if "U" not in a.code]
    if len(codes) < 5:
        return {"n": len(codes), "agreement": float("nan"), "collapsed": False, "note": "too few decided relations"}
    arr = np.array([[c[i] == "Y" for i in range(3)] for c in codes], float)
    pair = [np.mean(arr[:, i] == arr[:, j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    ag = float(np.mean(pair))
    return {"n": len(codes), "agreement": ag, "pairwise": [float(p) for p in pair], "collapsed": ag > 0.95}


def format_answers(answers: Sequence[ThreeAnswers]) -> str:
    lines = [f"{'relation':<26}{'REAL':<9}{'USEFUL':<9}{'NOW':<9}{'action':<20}reason"]
    for a in answers:
        d = disposition(a)
        lines.append(f"{a.relation_id[:25]:<26}{str(a.real.verdict):<9}{str(a.useful.verdict):<9}{str(a.useful_now.verdict):<9}"
                     f"{d.action:<20}{d.reason}")
    return "\n".join(lines)
