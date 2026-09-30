"""The FULL evidence bundle the research quality gate needs (contract C66 section 42 'research quality gate', 32 replication, 29-31
anti-cheating, 47 'the full cycle must actually carry data'; canon C62, C63, C66). IMPLEMENTED - NOT VALIDATED (C63: code and
planted-world tests only).

W01 found the loop's quality gate could never promote: it supplied only point-in-time, leak, replication and provenance evidence, so
ten gates stayed MISSING and every candidate ended NEEDS_MORE_EVIDENCE. This module computes EVERY field of
engine.research.quality_gate.QualityEvidence for one research finding (a derived feature, its orientation sign and the problem it
ranks) from the matured research panel, using the modules that own each kind of evidence:

  statistical validity / OOS    per-date rank AUC after a purged train window (engine.learning.promotion Statistical/OOSEvidence)
  incremental value             the same series against a zero-feature baseline (engine.learning.complexity Candidates)
  cross-context transfer        per-sector OOS effect (promotion.TransferEvidence)
  risk                          the per-period excess return of the rule's top bucket (promotion.RiskEvidence)
  anti-memorisation / identity  engine.learning.identity_firewall.IdentityHarness on the rule (ticker / date permutation, stock
                                substitution) + per-name share of the effect (promotion.MemorizationEvidence)
  future-information audit      engine.learning.firewalls.LearningFirewallGate on the candidate's own GateContext, the firewall's
                                planted corpus AND a candidate-specific planted label leak that the leak screen must catch
  reproducibility               re-computations on repeated and fresh seeds with the current code / data hash (promotion.ReproEvidence)
  replication                   engine.research.replication: a Discovery on half the names in the train window, runs on the OTHER
                                half in every new quarter-block of the test period (fresh period + fresh stocks, shuffled-label
                                control), accumulated across looks and assessed in the ONE ledger
  calibration                   engine.learning.calibration.PlattCalibrator fitted on train, scored out of sample
  failure behaviour             negative-effect episodes, the contexts that explain them, perturbation retention, retirement trigger,
                                out-of-scope abstention (probed, not asserted)
  provenance / PIT              engine.learning.core.Provenance fully filled; quality_gate.pit_evidence with the purge gap

No gate is passed by default: anything that cannot be computed is left None (the gate reads None as MISSING) and listed in
`Bundle.missing` with the reason. Public entry: `assemble(frame, spec, now, ...)`; `gate(bundles, now, ...)` runs the gate.
Re-gating one finding as its evidence grows (F12) is a sequential design, `SequentialPlan`: look k is judged at an alpha that sums to
alpha over all looks, and a finding is retired on measured futility, never on a count of looks."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.research.core import FirewallBreach, Problem, as_date, require_past, stable_hash

LABEL = "IMPLEMENTED - NOT VALIDATED"
JUSTIFICATION = ("oos_transfer", "replication", "calibration")


# ================================================================================================================ configuration
@dataclasses.dataclass(frozen=True)
class EvidenceConfig:
    orient_frac: float = 0.30            # earliest share of matured dates: the train window (orientation + in-sample effect)
    min_names: int = 8                   # per date, both classes and at least this many names
    mover_frac: float = 0.20             # predicted-mover universe for direction / loss findings
    horizon_days: int = 5
    purge_days: int = 7                  # calendar gap between the train window's last outcome and the first test date (>= horizon)
    top_q: float = 0.80                  # the rule's 'selection' for risk and calibration: its top quintile
    perturb_sd: float = 0.25             # planted data degradation: noise added to the feature, in its own sd units
    identity_boot: int = 100
    repl_block: int = 13                 # decision dates per replication run: consecutive, non-overlapping quarters of the test period
    repl_min_periods: int = 12
    cal_rows: int = 4000
    seeds: tuple = (1, 1, 2, 3)          # reproducibility reruns: one seed repeated (determinism) and fresh seeds
    catastrophic: float = -0.20
    fail_frac: float = 0.5               # a failure week delivers less than this share of the rule's typical (mean) effect
    noise_alpha: float = 0.05            # family-wise level: a failure week no further below the mean than the most extreme of the
                                         # n test weeks would fall by chance (Bonferroni over n) is sampling noise
    align_years: bool = True             # F14: move the train cut to a year end when the boundary would cost unseen years
    align_min_frac: float = 0.15         # a moved train window's share of dates must stay in [align_min_frac, align_max_frac]
    align_max_frac: float = 0.55
    max_failure_rate: float = 0.10       # F14: exposure route of the failure floor - the failure rate's upper bound must be <= this

    def validate(self) -> list[str]:
        errs = []
        if not 0.1 <= self.orient_frac <= 0.6 or self.min_names < 4 or not 0 < self.mover_frac < 0.5:
            errs.append("orient_frac in [0.1,0.6], min_names >= 4, mover_frac in (0,0.5) required")
        if self.purge_days < self.horizon_days or not 0.5 <= self.top_q < 1 or self.perturb_sd <= 0:
            errs.append("purge_days >= horizon_days, top_q in [0.5,1), perturb_sd > 0 required")
        if len(self.seeds) < 3 or len(set(self.seeds)) < 2 or len(set(self.seeds)) == len(self.seeds):
            errs.append("seeds need >= 3 reruns, >= 2 distinct seeds and one repeated seed")
        if self.repl_block < 4:
            errs.append("repl_block >= 4 dates required (a replication run needs periods to speak)")
        if not 0.05 <= self.align_min_frac <= self.orient_frac <= self.align_max_frac <= 0.6:
            errs.append("0.05 <= align_min_frac <= orient_frac <= align_max_frac <= 0.6 required")
        if not 0.0 < self.max_failure_rate <= 0.25:
            errs.append("max_failure_rate in (0, 0.25] required: 'fails rarely' must mean rarely")
        return errs


@dataclasses.dataclass(frozen=True)
class FindingSpec:
    """What is being judged: a derived feature with its orientation sign, for a problem, identified by the loop's subject id."""
    subject_id: str
    feature: str
    sign: float = 1.0
    problem: str = "VOLATILITY"
    n_tests_searched: int = 1            # how many features the search screened before this one (multiplicity)
    has_falsifier: bool = False          # science memory holds a declared falsifier (a retirement trigger) for this finding
    experiment_id: str = ""
    run_id: str = "research"
    seed: int = 0

    def validate(self) -> list[str]:
        from engine.research import vol_hypotheses as VH
        errs = []
        if not self.subject_id:
            errs.append("subject_id required")
        if self.feature not in VH.DERIVED:
            errs.append(f"unknown derived feature {self.feature!r}")
        if self.sign not in (1.0, -1.0):
            errs.append("sign must be +1 or -1")
        return errs


@dataclasses.dataclass(frozen=True)
class SequentialPlan:
    """F12 (C69 W-06 / W-12): the honest sequential design for re-gating ONE finding as its out-of-sample evidence accumulates.

    W-12 showed the genuine planted pattern (lv20) gated three times and then RETIRED by a hard cap of three looks, although the evidence
    the gate needs (a second unseen year, enough fresh replication quarters, enough failure episodes) could not exist yet. A look cap
    kills true effects for lack of time; unlimited looks at a fixed alpha manufacture a pass for a null (optional stopping). Instead:

      alpha spending   look k (k = 1, 2, ...) is judged at alpha_k = alpha * 6 / (pi^2 k^2). The series sums to alpha over infinitely
                       many looks, so by the union bound the probability that a finding with no effect passes the statistical gate at ANY
                       look is <= alpha - however long it is watched. The statistical gate's alpha and the OOS t floor become
                       z(1 - alpha_k) (never below their fixed-sample values: thresholds are only ever raised), and the replication
                       policy's alpha / CI level tighten the same way (its cumulative assessment is re-read at every look).
      futility         a finding stops being looked at only on MEASURED grounds: its out-of-sample effect's one-sided upper bound is at
                       or below the minimum effect with at least the gate's minimum OOS periods, or replication FAILED (a powered
                       fresh-data refutation and no support). Integrity failures (QUARANTINED) are never retried (the loop's rule).
      horizon          a compute backstop counted in fresh EVIDENCE, never in looks: `horizon_dates` matured decision dates after the
                       first look (five years of weekly dates by default), long after the gate's own needs can be met.
    A true effect's p-value falls exponentially in the sample while alpha_k falls only as 1/k^2, so it is never starved of alpha."""
    alpha: float = 0.05
    futility_z: float = 1.645
    horizon_dates: int = 260

    def validate(self) -> list[str]:
        errs = []
        if not 0.0 < self.alpha < 0.5 or self.futility_z <= 0:
            errs.append("alpha in (0, 0.5) and futility_z > 0 required")
        if self.horizon_dates < 52:
            errs.append("horizon_dates >= 52 required: a finding must be allowed at least a year of fresh evidence")
        return errs

    def alpha_at(self, look: int) -> float:
        if int(look) < 1:
            raise ValueError(f"looks are numbered from 1 (got {look})")
        return float(self.alpha * 6.0 / (math.pi ** 2 * int(look) ** 2))

    def spent(self, looks: int) -> float:
        """Alpha consumed by the first `looks` looks (always < alpha)."""
        return float(sum(self.alpha_at(k) for k in range(1, int(looks) + 1)))

    @staticmethod
    def _z(a: float) -> float:
        from statistics import NormalDist
        return float(NormalDist().inv_cdf(1.0 - a))

    def promotion_policy(self, look: int, base=None):
        from engine.learning import promotion as PR
        base = base if base is not None else PR.PromotionPolicy()
        a = min(float(base.alpha), self.alpha_at(look))
        return dataclasses.replace(base, alpha=a, min_oos_t=max(float(base.min_oos_t), self._z(a)))

    def replication_policy(self, look: int, base=None):
        from engine.research import replication as RP
        base = base if base is not None else RP.DEFAULT_POLICY
        a = min(float(base.alpha), self.alpha_at(look))
        return dataclasses.replace(base, alpha=a, ci_level=max(float(base.ci_level), 1.0 - 2.0 * a))

    def quality_policy(self, look: int, code_hash: str):
        from engine.research import quality_gate as QG
        return QG.QualityPolicy(promotion=self.promotion_policy(look), code_hash=code_hash)

    def futility(self, bundle: "Bundle", min_effect: float = 0.0) -> str | None:
        """Why this finding should stop being looked at, from measured evidence only; None = keep looking."""
        from engine.learning import promotion as PR
        from engine.research import replication as RP
        ev = bundle.evidence
        a = getattr(ev, "replication", None)
        if a is not None and a.status == RP.Status.FAILED:
            return f"replication FAILED: {a.n_refuting} powered fresh-data refutation(s) and no support"
        o = getattr(ev, "oos", None)
        if o is not None and o.oos is not None:
            e = np.asarray(o.oos.oos_effects, dtype=float)
            e = e[np.isfinite(e)]
            if e.size >= PR.PromotionPolicy().min_oos_periods:
                ub = float(e.mean() + self.futility_z * e.std(ddof=1) / math.sqrt(e.size))
                if ub <= min_effect:
                    return f"out-of-sample effect measured absent: upper bound {ub:+.4f} <= {min_effect} over {e.size} periods"
        return None


@dataclasses.dataclass
class Bundle:
    subject_id: str
    evidence: Any                        # quality_gate.QualityEvidence
    parts: dict                          # diagnostics per evidence kind (numbers the report and tests read)
    missing: dict                        # evidence kind -> why it could not be computed (left None: the gate blocks)

    def summary(self) -> dict:
        return {"subject": self.subject_id, "missing": dict(self.missing),
                "parts": {k: v for k, v in self.parts.items() if isinstance(v, (int, float, str, bool, type(None)))}}


# ================================================================================================================ the design
def _design(F: pd.DataFrame, spec: FindingSpec, ec: EvidenceConfig) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(rows, oriented score, label) for the finding's problem, the same designs the loop's experiments use: volatility = touch on
    every name; direction / coverage = up among predicted movers; loss avoidance = a losing close among predicted movers."""
    from engine.research import loop as LP
    from engine.research import vol_hypotheses as VH
    prob = Problem.parse(spec.problem)
    G = F[F["touch"].notna()] if "touch" in F else F.iloc[0:0]
    if prob in (Problem.DIRECTION, Problem.COVERAGE, Problem.LOSS_AVOIDANCE):
        G = G[LP._mover_mask(G, ec.mover_frac)]
        y = G["up"].astype(float) if prob != Problem.LOSS_AVOIDANCE else (G["close"] < 0).astype(float)
    else:
        y = G["touch"].astype(float)
    s = spec.sign * VH.derive(G, (spec.feature,))[spec.feature].astype(float)
    return G, s, y


def per_date_effect(score: pd.Series, y: pd.Series, min_names: int, jitter_seed: int | None = None) -> pd.Series:
    """Per-date rank AUC - 0.5 (dates with both classes and >= min_names finite rows). `jitter_seed` breaks score ties with a seeded
    infinitesimal jitter (the only randomness in the measurement; reruns on seeds show it does not matter)."""
    from engine.pattern_movers import auc as rank_auc
    s = score.to_numpy(float)
    if jitter_seed is not None:
        s = s + np.random.default_rng(jitter_seed).normal(0, 1e-9, len(s))
    codes, uniq = pd.factorize(score.index.get_level_values(0), sort=True)
    yy = y.to_numpy(float)
    out = {}
    for c in range(len(uniq)):
        m = (codes == c) & np.isfinite(s) & np.isfinite(yy)
        if m.sum() < min_names:
            continue
        lab = yy[m] >= 0.5
        if lab.all() or not lab.any():
            continue
        a = rank_auc(s[m], lab)
        if np.isfinite(a):
            out[pd.Timestamp(uniq[c])] = a - 0.5
    return pd.Series(out, dtype=float).sort_index()


def unseen_dates(years: np.ndarray, c: int) -> int:
    """Test dates in calendar years with no train date when the first `c` sorted dates train (train years are contiguous, so a test
    date is unseen exactly when its year is after the last train date's year)."""
    return int((years[c:] > years[c - 1]).sum()) if 0 < c <= len(years) else 0


def train_cut(dates: Sequence, ec: EvidenceConfig) -> tuple[int, bool]:
    """(number of train dates, moved to a year end?) for the decision dates.

    F14 defect (F12 structural issue a): the train window was the earliest `orient_frac` of dates wherever that fell. The quality gate
    counts UNSEEN CALENDAR YEARS (years with no train date), so a train window that ran two weeks past 1 January made that whole year
    'seen': on the planted world (data from April 2016, a sliding three-year frame) every December look cut train in mid-January of
    the previous year and was left with ONE unseen year, while the June look of the same year had two - the verdict depended on where
    the calendar boundary fell, not on the evidence. Now the cut is the one, among the fractional cut and every calendar-year end whose
    train share lies in [align_min_frac, align_max_frac], that leaves the most UNSEEN test dates (dates in years with no train date);
    ties keep the cut nearest the orient_frac target (the fractional cut itself when it loses nothing). So the cut moves only when the
    boundary would cost unseen evidence, and then only to the nearest year end: December and June looks see the same unseen years.
    When no year end qualifies (a frame shorter than about a year and a half) the fractional cut stands and the gate judges the split
    year as seen - it is never counted as unseen."""
    uniq = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates))).unique().sort_values()
    n = len(uniq)
    k = max(2, int(n * ec.orient_frac))
    if not ec.align_years or n <= k:
        return k, False
    yr = uniq.year.to_numpy()
    ends = [i + 1 for i in range(n - 1) if yr[i] != yr[i + 1]]                  # train size when the cut is that year's last date
    cands = [k] + [c for c in ends if ec.align_min_frac * n <= c <= ec.align_max_frac * n and c >= 2 and c != k]
    best = max(cands, key=lambda c: (unseen_dates(yr, c), -abs(c - k), -c))
    return best, best != k


def split_dates(G: pd.DataFrame, ec: EvidenceConfig) -> tuple[pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray]:
    """(train_end, first_test, train mask, test mask): the train window is the earliest ~`orient_frac` of dates, moved to a calendar-year
    end when the boundary would cost unseen evidence (train_cut), purged so that every train outcome ended before the first test date and the gap is at least
    `purge_days` (the section-42 PIT rule)."""
    d = pd.to_datetime(G.index.get_level_values(0))
    uniq = np.array(sorted(pd.unique(d)))
    if len(uniq) < 8:
        raise ValueError(f"{len(uniq)} dates: too few for a train / test split")
    k, _ = train_cut(uniq, ec)
    train_end = pd.Timestamp(uniq[k - 1])
    first_test = next((pd.Timestamp(x) for x in uniq[k:] if pd.Timestamp(x) >= train_end + pd.Timedelta(days=ec.purge_days)), None)
    if first_test is None:
        raise ValueError("no test date after the purge gap")
    ends = pd.to_datetime(G["end"])
    tr = np.asarray(d <= train_end) & np.asarray(ends < first_test)
    te = np.asarray(d >= first_test)
    return train_end, first_test, tr, te


# ================================================================================================================ the parts
def statistical_and_oos(eff_tr: pd.Series, eff_te: pd.Series, spec: FindingSpec, train_end, now):
    from engine.learning import promotion as PR
    x = eff_te.to_numpy(float)
    t = PR.t_stat(x) if len(x) >= 2 else float("nan")
    stat = PR.StatisticalEvidence(float(x.mean()) if len(x) else 0.0, int(len(x)), t, PR.one_sided_p(t) if np.isfinite(t) else 1.0,
                                  int(max(1, spec.n_tests_searched)), float(len(x)))
    dates = tuple(str(d.date()) for d in eff_te.index)
    for d in dates:
        require_past(d, now, "out-of-sample date")
    oos = PR.OOSEvidence(str(as_date(train_end)), dates, tuple(float(v) for v in x), float(eff_tr.mean()) if len(eff_tr) else 0.0,
                         (f"{dates[0]}..{dates[-1]}",) if dates else (), (f"..{as_date(train_end)}",))
    return stat, oos


def identity(G: pd.DataFrame, score: pd.Series, y: pd.Series, tr: np.ndarray, te: np.ndarray, spec: FindingSpec, ec: EvidenceConfig):
    """IdentityHarness on the rule itself (the 'learner' ranks by the oriented feature: it can only memorise identities if the
    feature encodes them), plus the per-name share of the out-of-sample effect."""
    from engine.learning import identity_firewall as IDF
    from engine.learning import promotion as PR
    X = pd.DataFrame({"f": score.to_numpy(float)}, index=G.index)

    def learner(X_train, y_train, X_eval, seed: int = 0) -> pd.Series:
        return X_eval["f"].astype(float)
    rep = IDF.IdentityHarness(learner, attacks=("ticker_permutation", "date_permutation", "stock_substitution"), seed=spec.seed + 11,
                              boot=ec.identity_boot).run(X[tr], y[tr], X[te], y[te])
    ret = rep.retention_by_kind("eval")
    vals = [float(v) for v in ret.values if np.isfinite(v)]
    shuf = ret.get("ticker_permutation", np.nan)
    r = score[te].groupby(level=0).rank(pct=True) - 0.5
    contrib = (r * (y[te] - y[te].groupby(level=0).transform("mean"))).groupby(level=1).sum()
    pos = float(contrib[contrib > 0].sum())
    top = float(contrib.max() / pos) if pos > 0 else 1.0
    memo = PR.MemorizationEvidence(float(shuf) if np.isfinite(shuf) else None, float(min(vals)) if vals else None, (spec.feature,),
                                   False, top, int(contrib.size))
    return rep, memo, {"identity_retention_min": min(vals) if vals else None, "top_identity_share": top}


def transfer(G: pd.DataFrame, score: pd.Series, y: pd.Series, te: np.ndarray, ec: EvidenceConfig):
    """Per-sector out-of-sample effect; home = the sector with the most test rows, away = the rest."""
    from engine.learning import promotion as PR
    if "sector" not in G:
        return None, "no sector column: transfer across contexts cannot be measured"
    eff, n = {}, {}
    sec = G["sector"].astype(str)
    for s in sorted(sec[te].unique()):
        m = te & (sec == s).to_numpy()
        e = per_date_effect(score[m], y[m], max(4, ec.min_names // 2))
        if len(e):
            eff[s], n[s] = float(e.mean()), int(len(e))
    if len(eff) < 2:
        return None, f"only {len(eff)} sector(s) with an out-of-sample effect"
    home = max(n, key=lambda k: (n[k], k))
    return PR.TransferEvidence(eff, n, (home,), eff[home]), ""


def risk(G: pd.DataFrame, score: pd.Series, te: np.ndarray, ec: EvidenceConfig):
    """Per period: the equal-weight horizon return of the rule's top bucket in excess of the universe (the book it would add)."""
    from engine.learning import promotion as PR
    H = G[te]
    r = score[te].groupby(level=0).rank(pct=True)
    top = H["close"].where(r >= ec.top_q).groupby(level=0).mean()
    ex = (top - H["close"].groupby(level=0).mean()).dropna()
    if len(ex) < 2:
        return None, None, "fewer than two test periods with a top bucket"
    cum = ex.cumsum()
    k = max(1, int(0.05 * len(ex)))
    return (PR.RiskEvidence(int(len(ex)), float(ex.min()), float((cum - cum.cummax()).min()), float(ex.nsmallest(k).mean()),
                            int((ex <= ec.catastrophic).sum())), ex, "")


def calibration(G: pd.DataFrame, score: pd.Series, y: pd.Series, tr: np.ndarray, te: np.ndarray, spec: FindingSpec, ec: EvidenceConfig):
    """P(outcome) from the rule: Platt on the train window's within-date rank, scored on the test window (never refitted there)."""
    from engine.learning import calibration as CAL
    from engine.research import quality_gate as QG
    rk = score.groupby(level=0).rank(pct=True).to_numpy(float).clip(0.01, 0.99)
    yy = y.to_numpy(float)
    okt, oke = tr & np.isfinite(rk) & np.isfinite(yy), te & np.isfinite(rk) & np.isfinite(yy)
    if okt.sum() < 50 or oke.sum() < 50 or len(np.unique(yy[okt])) < 2:
        return None, "too few rows to calibrate"
    cal = CAL.PlattCalibrator().fit(rk[okt], yy[okt].astype(int))
    p = np.clip(np.asarray(cal.predict(rk[oke]), float), 0.001, 0.999)
    idx = np.random.default_rng(spec.seed + 5).permutation(len(p))[:ec.cal_rows]
    return QG.CalibrationEvidence(tuple(float(v) for v in p[idx]), tuple(int(v) for v in yy[oke][idx]), spec.seed), ""


def no_feature_effects(score: pd.Series, y: pd.Series, mask: np.ndarray, ec: EvidenceConfig, seed: int) -> pd.Series:
    """The simplest alternative MEASURED: a ranking that carries no information (a seeded random score on the same rows and dates),
    scored exactly like the finding. Its per-date effect is zero in expectation with the true sampling spread of a weekly AUC."""
    noise = pd.Series(np.random.default_rng(seed).normal(0.0, 1.0, len(score)), index=score.index)
    return per_date_effect(noise[mask], y[mask], ec.min_names)


def complexity(eff_te: pd.Series, eff_tr: pd.Series, spec: FindingSpec, base_te: pd.Series | None = None):
    """The finding (one feature, one orientation) against the simplest alternative: no feature at all, measured on the same dates
    (`base_te`, a no-information ranking; see no_feature_effects). The candidate must earn its one unit of complexity.

    F12 defect fixed (reported by F11, verified by tests/test_regate_sequential.py): the baseline used to be an exactly-zero series.
    engine.learning.complexity then (a) judged the tail against a tolerance of 0.5 x sd(baseline) = 0, so ANY finding with a negative
    week in its bottom decile failed 'tail worse' (a genuine t = 5 effect failed); (b) divided the optimism gap by the baseline's gap of 0
    and got inf, which its isfinite guard silently skipped - a check that could not fail. Now the baseline is measured noise (a real
    spread for the tail), the baseline has no in-sample fit (in_sample None: optimism is not applicable to a zero-parameter rule and is
    reported as such; the finding's own in-sample -> out-of-sample decay is judged by the OOS gate's retention), and the folds are
    calendar quarters: calendar years gave two folds on every sliding three-year frame, so 'transfer across folds' was never testable
    (cross-YEAR transfer is required separately by the out-of-sample gate's unseen-years rule)."""
    from engine.learning import complexity as CX
    from engine.research import quality_gate as QG
    q = lambda ix: pd.Series([f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in ix], index=ix)        # noqa: E731
    cand = CX.Candidate(CX.RuleSpec(f"rule_{spec.feature}", n_features=1, n_free_params=1), eff_te, q(eff_te.index),
                        float(eff_tr.mean()) if len(eff_tr) else None)
    if base_te is None:
        base_te = eff_te * 0.0                          # unmeasured baseline: kept only for callers without the rows (never the loop)
    base_te = base_te.reindex(eff_te.index).fillna(0.0)
    base = CX.Candidate(CX.RuleSpec("no_feature", n_features=0, n_free_params=0), base_te, q(eff_te.index), None)
    return QG.ComplexityEvidence(cand, base, float(len(eff_te)))


def failure(G: pd.DataFrame, score: pd.Series, y: pd.Series, te: np.ndarray, eff_te: pd.Series, worst: float | None, spec: FindingSpec,
            ec: EvidenceConfig):
    """Failure behaviour measured, not asserted: episodes = test weeks delivering under `fail_frac` of the typical effect (below
    zero when there is no positive effect); conditions = market-volatility and
    sector contexts whose mean effect is negative; unknown-cause share = failure dates outside every such context; perturbation
    retention = effect with the feature degraded by seeded noise; abstention = deriving the feature without its inputs is refused
    (a missing input is never silently zero-filled)."""
    from engine.research import quality_gate as QG
    from engine.research import vol_hypotheses as VH
    typical = float(eff_te.mean()) if len(eff_te) else 0.0
    fails = eff_te[eff_te < (ec.fail_frac * typical if typical > 0 else 0.0)]
    conds, explained = [], set()
    if "m_vol" in G:
        mv = G["m_vol"].groupby(level=0).median()
        hi = mv > mv.median()
        for lab, mask in (("high market volatility", hi), ("low market volatility", ~hi)):
            ds = [d for d in eff_te.index if bool(mask.get(d, False))]
            if ds and eff_te.loc[ds].mean() < 0:
                conds.append(f"weak in {lab}")
                explained |= set(ds)
    if "sector" in G:
        sec = G["sector"].astype(str)
        for s in sorted(sec[te].unique()):
            m = te & (sec == s).to_numpy()
            e = per_date_effect(score[m], y[m], max(4, ec.min_names // 2))
            if len(e) >= 4 and e.mean() < 0:
                conds.append(f"weak in sector group {stable_hash(s, 4)}")
    if len(fails):                                       # a failure week inside the sampling spread of its own AUC is explained
        mean = float(eff_te.mean())
        pos = (y[te] >= 0.5).groupby(level=0).sum()
        neg = (y[te] < 0.5).groupby(level=0).sum()
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1.0 - ec.noise_alpha / (2.0 * max(1, len(eff_te))))
        noise = []
        for d, e in fails.items():
            if mean > 0 and (mean - e) <= z * hanley_mcneil_se(0.5 + mean, int(pos.get(d, 0)), int(neg.get(d, 0))):
                noise.append(d)
        if noise:
            conds.append(f"sampling noise: weekly effect within {z:.2f} standard errors of its mean (family-wise {ec.noise_alpha:g})")
            explained |= set(noise)
    unknown = float(np.mean([d not in explained for d in fails.index])) if len(fails) else 0.0
    rng = np.random.default_rng(spec.seed + 17)
    sd = float(np.nanstd(score.to_numpy(float))) or 1.0
    noisy = score + rng.normal(0, ec.perturb_sd * sd, len(score))
    e_noisy = per_date_effect(noisy[te], y[te], ec.min_names)
    base = float(eff_te.mean()) if len(eff_te) else 0.0
    retention = float(e_noisy.mean() / base) if base > 0 and len(e_noisy) else 0.0
    abstains = False
    try:
        VH.derive(G.drop(columns=list(VH.required_columns((spec.feature,)))).iloc[:5], (spec.feature,))
    except KeyError:
        abstains = True
    return QG.FailureEvidence(tuple(conds) or ("negative-effect weeks with no dominant context",), int(len(fails)), bool(spec.has_falsifier),
                              abstains, retention, unknown, worst), {"perturbation_retention": retention, "unknown_cause_share": unknown}


def hanley_mcneil_se(a: float, n1: int, n0: int) -> float:
    """Standard error of an AUC `a` from n1 positives and n0 negatives (Hanley & McNeil 1982): how far one week's AUC may fall from
    the true one by sampling alone."""
    if n1 < 1 or n0 < 1:
        return float("inf")
    a = min(max(a, 1e-6), 1 - 1e-6)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    v = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    return float(math.sqrt(max(v, 0.0)))


def reproducibility(score: pd.Series, y: pd.Series, te: np.ndarray, spec: FindingSpec, ec: EvidenceConfig, code_hash: str, data_hash: str):
    from engine.learning import promotion as PR
    rr = tuple(PR.RerunRecord(float(per_date_effect(score[te], y[te], ec.min_names, jitter_seed=s).mean()), int(s), code_hash, data_hash)
               for s in ec.seeds)
    return PR.ReproEvidence(rr, data_hash)


def stock_halves(tickers: Sequence[str], salt: str) -> tuple[frozenset, frozenset]:
    """A deterministic split of the names: the discovery sees half A, replication runs use half B (fresh stocks)."""
    a, b = set(), set()
    for t in sorted(set(map(str, tickers))):
        (a if int(stable_hash([salt, t], 8), 16) % 2 == 0 else b).add(t)
    return frozenset(a), frozenset(b)


def replication_blocks(test_dates: Sequence, after, block: int) -> list[np.ndarray]:
    """Complete, consecutive, non-overlapping blocks of `block` test dates strictly after `after` (the end of the newest window
    already in the ledger). A partial block waits for its remaining dates; nothing is ever cut twice."""
    ted = pd.DatetimeIndex(pd.to_datetime(pd.Index(test_dates))).unique().sort_values()
    ted = ted[ted > pd.Timestamp(after)].to_numpy()
    return [ted[k * block:(k + 1) * block] for k in range(len(ted) // block)]


def replicate(G: pd.DataFrame, score: pd.Series, y: pd.Series, tr: np.ndarray, te: np.ndarray, spec: FindingSpec, ec: EvidenceConfig,
              now, code_hash: str, data_hash: str, ledger=None, policy=None):
    """Register the finding ONCE as a replication.Discovery on half the names in its first train window, then run it on the OTHER half
    in every complete `repl_block`-date block of the test period that no recorded run covers yet (fresh period + fresh stocks, a regime
    label per block, a shuffled-label control on the same periods), all in the ONE replication ledger when one is given.

    F12 defect fixed: runs used to be ids '-w0', '-w1' of a two-way split of the CURRENT test window. With a persistent ledger the ids
    already existed at every later look, so no new run was ever added and the assessment of the first look (W-12: PARTIALLY_REPLICATED,
    heterogeneity I2 of two runs) was frozen for ever: re-gating could not change the replication verdict whatever time brought.
    Blocks are now keyed by their first date and only NEW blocks are added, so every look sees the fresh quarters that matured since."""
    from engine.research import replication as RP
    names = G.index.get_level_values(1).astype(str)
    A, B = stock_halves(names, spec.subject_id)
    inA, inB = np.isin(names, list(A)), np.isin(names, list(B))
    mn = max(4, ec.min_names // 2)
    led = ledger if ledger is not None else RP.ReplicationLedger()
    did = "E" + spec.subject_id
    d = pd.to_datetime(G.index.get_level_values(0))
    ends = pd.to_datetime(G["end"])
    if did not in led.discoveries():
        e0 = per_date_effect(score[tr & inA], y[tr & inA], mn)
        if len(e0) < 3 or float(e0.std()) <= 0:
            return None, {"why": f"{len(e0)} train periods on the discovery half"}
        trd = d[tr]
        led.add_discovery(RP.Discovery(did, float(e0.mean()), float(e0.std()), int(len(e0)), (str(trd.min().date()), str(trd.max().date())),
                                       ec.horizon_days, A, (spec.seed,), frozenset({"train"}), code_hash, data_hash, str(ends[tr].max().date()),
                                       max(1, spec.n_tests_searched)))
    disc = led.discoveries()[did]
    prior = led.runs_for(did, now)
    after = max([pd.Timestamp(r.window[1]) for r in prior]
                + [pd.Timestamp(disc.window[1]) + pd.Timedelta(days=max(ec.purge_days, disc.horizon_days))])
    mvol = G["m_vol"].groupby(level=0).median() if "m_vol" in G else None
    added = 0
    for c in replication_blocks(d[te], after, ec.repl_block):
        m = te & inB & np.isin(d, c)
        e = per_date_effect(score[m], y[m], mn)
        if len(e) < 2:
            continue
        first = str(pd.Timestamp(c[0]).date())
        seed = int(stable_hash([spec.subject_id, first], 6), 16) % 1_000_000 + 1000     # a fresh seed per block, never the discovery's
        rng = np.random.default_rng(seed)
        ys = y[m].groupby(level=0).transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
        ctl = per_date_effect(score[m], ys, mn).reindex(e.index).fillna(0.0)
        regime = "calm"
        if mvol is not None:
            regime = "high_vol" if float(mvol.reindex(c).median()) > float(mvol.median()) else "calm"
        mat = str(ends[m].max().date())
        require_past(mat, now, "replication run")
        led.add_run(RP.ReplicationRun(f"{did}-{first}", did, (first, str(pd.Timestamp(c[-1]).date())), B, seed, regime,
                                      tuple(float(v) for v in e.to_numpy()), tuple(float(v) for v in ctl.to_numpy()), code_hash, data_hash,
                                      mat), now)
        added += 1
    runs = led.runs_for(did, now)
    a = RP.assess(disc, runs, now, policy) if policy is not None else RP.assess(disc, runs, now)
    return a, {"repl_status": str(a.status), "repl_runs": len(runs), "repl_runs_added": added, "repl_discovery_periods": int(disc.n_periods),
               "repl_i2": float(a.pooled.get("i2", 0.0)), "repl_supporting": int(a.n_supporting)}


def leakage(G: pd.DataFrame, score: pd.Series, y: pd.Series, spec: FindingSpec, now, train_end, first_test, identity_report,
            code_hash: str, prov, n_decisions: int, used: np.ndarray | None = None):
    """The future-information audit: (1) the learning firewall on the candidate's OWN context (its panel, horizon, experiment and
    evaluation records, identity report, code state), (2) the firewall's planted corpus (the audit can still see planted leaks),
    (3) a candidate-specific planted leak: the label copied into a feature column must be flagged by the leak screen, (4) outcomes
    dated at/after now are counted (never assumed zero)."""
    from engine.learning import firewalls as FW
    from engine.research import quality_gate as QG
    if used is not None:                                 # only the rows the evidence used (train + test; the purged gap is out)
        G, score = G[used], score[used]
    X = pd.DataFrame({spec.feature: score.to_numpy(float)}, index=G.index)
    yy = G["close"].astype(float).rename("ret")
    d = pd.to_datetime(G.index.get_level_values(0))
    label_close = pd.Series(pd.to_datetime(G["end"]).to_numpy(), index=G.index)
    item = {"knowledge_id": spec.subject_id, "version": 1, "provenance": prov, "contexts": {}, "anti_contexts": {}}
    exp = FW.ExperimentRecord(spec.experiment_id or spec.subject_id, prov.created_real, prov.created_real, prov.config_hash, spec.seed,
                              f"{spec.feature} ranks {spec.problem.lower()} outcomes", n_variants_tried=max(1, spec.n_tests_searched),
                              selected_from=max(1, spec.n_tests_searched), multiplicity_correction="holdout",
                              training_windows=((str(d.min().date()), str(as_date(train_end))),), baseline_declared=True)
    ev = FW.EvaluationRecord(evaluation_windows=((str(as_date(first_test)), str(d.max().date())),),
                             training_windows=((str(d.min().date()), str(as_date(train_end))),), sealed_real=prov.created_real,
                             training_started_real=prov.created_real, state_hash_before=code_hash, state_hash_after=code_hash,
                             n_decisions=int(n_decisions), paired_baseline=True)
    ctx = FW.GateContext(now=str(as_date(now)), subject=spec.subject_id, items=[item], X=X, y=yy, horizon=5, identity_report=identity_report,
                         label_close=label_close, experiment=exp, evaluation=ev, code=FW.CodeState(recorded={"code_hash": code_hash, "code_files": [], "code_mixed": []},
                                                                          current_hash=code_hash), test_start=str(as_date(first_test)))
    fwv = FW.LearningFirewallGate().evaluate(ctx)
    corp = FW.run_corpus()
    leaky = X.assign(planted_label_copy=yy.to_numpy())
    caught_own = any("planted_label_copy" in f for f in QG.screen_label_leak(leaky, yy))
    probe = bool(corp.get("clean_passed")) and not corp.get("missed") and caught_own
    after = int((pd.to_datetime(G["end"]) >= pd.Timestamp(as_date(now))).sum())
    return QG.leak_evidence_from_panel(X, yy, fwv, probe, after), {"firewall_passed": bool(getattr(fwv, "passed", False)),
                                                                   "planted_probe_caught": probe, "own_probe_caught": caught_own,
                                                                   "outcomes_after_now": after}


# ================================================================================================================ the bundle
def assemble(frame: pd.DataFrame, spec: FindingSpec, now, *, code_hash: str, data_hash: str, created_real: str,
             cfg: EvidenceConfig = EvidenceConfig(), ledger=None, look: int | None = None, plan: SequentialPlan | None = None) -> Bundle:
    """PUBLIC ENTRY. Every QualityEvidence field for `spec` from the matured research `frame` (rows whose outcome ended strictly
    before now; a row at/after now is a FirewallBreach, not a filter). Anything not computable stays None and is named in `missing`.
    `look` (1, 2, ...) = this is the finding's look-th sequential look: replication is assessed under the plan's spent alpha."""
    from engine.learning.core import Provenance
    from engine.research import quality_gate as QG
    errs = cfg.validate() + spec.validate()
    if look is not None:
        plan = plan or SequentialPlan()
        errs += plan.validate() + ([] if int(look) >= 1 else [f"look must be >= 1 (got {look})"])
    if errs:
        raise ValueError("invalid evidence request: " + "; ".join(errs))
    if len(frame) and (pd.to_datetime(frame["end"]) >= pd.Timestamp(as_date(now))).any():
        raise FirewallBreach(f"evidence frame carries outcomes that end on/after now={as_date(now)}")
    missing: dict[str, str] = {}
    parts: dict[str, Any] = {}
    empty = QG.QualityEvidence()
    if len(frame) == 0:
        return Bundle(spec.subject_id, empty, parts, {"all": "no matured rows"})
    G, score, y = _design(frame, spec, cfg)
    try:
        train_end, first_test, tr, te = split_dates(G, cfg)
    except ValueError as e:
        return Bundle(spec.subject_id, empty, parts, {"all": str(e)})
    eff_tr = per_date_effect(score[tr], y[tr], cfg.min_names)
    eff_te = per_date_effect(score[te], y[te], cfg.min_names)
    parts.update(train_end=str(train_end.date()), first_test=str(first_test.date()), n_train=len(eff_tr), n_test=len(eff_te),
                 train_moved_to_year_end=bool(train_cut(G.index.get_level_values(0), cfg)[1]),
                 effect_train=float(eff_tr.mean()) if len(eff_tr) else None, effect_test=float(eff_te.mean()) if len(eff_te) else None)
    through = str(pd.to_datetime(G["end"]).max().date())
    prov = Provenance(created_real, through, code_hash, data_hash, stable_hash(dataclasses.asdict(cfg), 12),
                      spec.experiment_id or spec.subject_id, spec.run_id, spec.seed, through)
    stat, oos = statistical_and_oos(eff_tr, eff_te, spec, train_end, now) if len(eff_te) else (None, None)
    if stat is None:
        missing["oos"] = "no evaluable test date"
    train_years = tuple(sorted({d.year for d in pd.to_datetime(G.index.get_level_values(0)[tr])}))
    ident = None
    try:
        rep, memo, p = identity(G, score, y, tr, te, spec, cfg)
        ident = QG.IdentityEvidence(rep, memo)
        parts.update(p)
    except Exception as e:                              # noqa: BLE001 - the harness refusing is recorded, the gate blocks
        missing["identity"] = f"{type(e).__name__}: {str(e)[:120]}"
        rep = None
    tev, why = transfer(G, score, y, te, cfg)
    if tev is None:
        missing["transfer"] = why
    rk, ex, why = risk(G, score, te, cfg)
    if rk is None:
        missing["risk"] = why
    cal, why = calibration(G, score, y, tr, te, spec, cfg)
    if cal is None:
        missing["calibration"] = why
    cx = complexity(eff_te, eff_tr, spec, no_feature_effects(score, y, te, cfg, spec.seed + 23)) if len(eff_te) else None
    fe, p = failure(G, score, y, te, eff_te, rk.worst_period if rk is not None else None, spec, cfg)
    parts.update(p)
    rp = reproducibility(score, y, te, spec, cfg, code_hash, data_hash)
    repl, p = replicate(G, score, y, tr, te, spec, cfg, now, code_hash, data_hash, ledger,
                        plan.replication_policy(look) if look is not None else None)
    parts.update(p)
    if look is not None:
        parts.update(look=int(look), alpha_look=plan.alpha_at(look), alpha_spent=plan.spent(look))
    if repl is None:
        missing["replication"] = p.get("why", "not computable")
    feats = {c: None for c in _base_columns(spec.feature)}
    pit = QG.pit_evidence(feats, str(first_test.date()), str(train_end.date()), str(first_test.date()), cfg.horizon_days, through,
                          fills_next_open=True, known_before=tuple(feats))
    leak, p = leakage(G, score, y, spec, now, train_end, first_test, rep, code_hash, prov, int(te.sum()), tr | te)
    parts.update(p)
    ev = QG.QualityEvidence(pit=pit, leak=leak, identity=ident, oos=QG.OOSBundle(stat, oos, train_years) if stat is not None else None,
                            replication=repl, outputs_probabilities=True, calibration=cal, changes_risk_decisions=True, risk=rk,
                            complexity=cx, transfer=tev, failure=fe, repro=rp, justification=JUSTIFICATION, provenance=prov)
    return Bundle(spec.subject_id, ev, parts, missing)


def _base_columns(feature: str) -> tuple[str, ...]:
    """The panel columns a derived feature is computed from; each is point-in-time by construction (price / volume data through the
    decision close, events public strictly before it), which is what `known_before` declares to the PIT gate."""
    from engine.research import vol_hypotheses as VH
    return tuple(VH.required_columns((feature,)))


def failure_rate_upper(k: int, n: int, alpha: float) -> float:
    """One-sided (1 - alpha) Clopper-Pearson upper bound on a failure rate after k failures in n periods (1.0 when n = 0)."""
    if n <= 0:
        return 1.0
    k = int(min(max(k, 0), n))
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


@dataclasses.dataclass(frozen=True)
class FailureFloor:
    """How much failure evidence a finding must show (F14, F12 structural issue b)."""
    required: int                        # the episode count the failure gate is run with for this finding
    episodes: int | None
    periods: int
    rate_upper: float                    # (1 - alpha) upper bound on the failure rate
    alpha: float
    route: str                           # 'episodes' (floor met) | 'exposure' (failures bounded as rare) | 'waiting' | 'missing'


def failure_floor(bundle: Bundle, pol, alpha: float, cfg: EvidenceConfig = EvidenceConfig()) -> FailureFloor:
    """The failure gate asks for `pol.min_failure_episodes` studied failure episodes. F12 found that a very strong genuine effect cannot
    collect them in time (planted seeds 0 and 4: 3 failure weeks in ~100 test weeks) and waits at NEEDS_MORE_EVIDENCE for ever - it is
    punished for failing rarely. The requirement is ENOUGH EVIDENCE ABOUT FAILURES, which is met either by the episodes, or by enough
    exposure that the failure rate is bounded as rare: the one-sided (1 - alpha) Clopper-Pearson upper bound on (episodes / test
    periods) is at most `cfg.max_failure_rate`. `alpha` is the look's spent alpha (SequentialPlan.alpha_at), so the bound tightens with
    every look exactly like the statistical gate: repeated looks cannot manufacture 'rare'. Only the episode COUNT is relaxed - the
    documented conditions, retirement trigger, out-of-scope abstention, perturbation retention and unknown-cause share are still judged,
    and no other gate (min_effect, the alpha plan, OOS, replication) is touched. A null or weak rule fails in a large share of weeks
    (below half its typical effect, below zero when it has none), so its bound never reaches the exposure route."""
    floor = int(pol.min_failure_episodes)
    fe = getattr(bundle.evidence, "failure", None)
    o = getattr(bundle.evidence, "oos", None)
    k = getattr(fe, "n_failure_episodes", None)
    n = 0
    if o is not None and o.oos is not None:
        e = np.asarray(o.oos.oos_effects, dtype=float)
        n = int(np.isfinite(e).sum())
    ub = failure_rate_upper(k, n, alpha) if k is not None else 1.0
    if k is None:
        return FailureFloor(floor, None, n, ub, float(alpha), "missing")
    if k >= floor:
        return FailureFloor(floor, k, n, ub, float(alpha), "episodes")
    if n >= int(pol.promotion.min_oos_periods) and ub <= cfg.max_failure_rate:
        return FailureFloor(int(k), k, n, ub, float(alpha), "exposure")
    return FailureFloor(floor, k, n, ub, float(alpha), "waiting")


def gate(bundles: Sequence[Bundle], now, code_hash: str, store=None, looks: Mapping[str, int] | None = None,
         plan: SequentialPlan | None = None, cfg: EvidenceConfig = EvidenceConfig()):
    """Run engine.research.quality_gate.step on the bundles with the policy pinned to the loop's code hash (reproducibility is judged
    against the code that produced the reruns). Without `looks` it is one fixed-sample gate at the policy's own alpha. With `looks`
    (subject -> look number) every bundle is judged under its look's spent alpha (SequentialPlan); a bundle missing from `looks` is
    refused (a sequential caller must number every look, or optional stopping creeps back in). Each bundle's failure-episode floor is
    `failure_floor` at that alpha (episodes, or exposure bounding the failure rate as rare)."""
    from engine.research import quality_gate as QG
    if looks is None:
        base = QG.QualityPolicy(code_hash=code_hash)
        groups: dict[tuple, list] = {}
        for b in bundles:
            ff = failure_floor(b, base, float(base.promotion.alpha), cfg)
            groups.setdefault((0, ff.required), []).append(QG.Candidate(b.subject_id, b.evidence))
        reps = {g: QG.step(c, now, policy=dataclasses.replace(base, min_failure_episodes=g[1]), store=store)
                for g, c in sorted(groups.items())}
        dec = {d.subject_id: d for r in reps.values() for d in r.decisions}
        decisions = tuple(dec[b.subject_id] for b in bundles)
        return QG.GateReport(str(as_date(now)), decisions, tuple(s for r in reps.values() for s in r.promoted),
                             tuple(s for r in reps.values() for s in r.newly_quarantined), QG.funnel(list(decisions)))
    plan = plan or SequentialPlan()
    unnumbered = [b.subject_id for b in bundles if b.subject_id not in looks]
    if unnumbered:
        raise ValueError(f"sequential gate: no look number for {unnumbered}")
    by_look: dict[tuple, list] = {}
    for b in bundles:
        k = int(looks[b.subject_id])
        ff = failure_floor(b, plan.quality_policy(k, code_hash), plan.alpha_at(k), cfg)
        by_look.setdefault((k, ff.required), []).append(QG.Candidate(b.subject_id, b.evidence))
    reps = {g: QG.step(c, now, policy=dataclasses.replace(plan.quality_policy(g[0], code_hash), min_failure_episodes=g[1]), store=store)
            for g, c in sorted(by_look.items())}
    dec = {d.subject_id: d for r in reps.values() for d in r.decisions}
    decisions = tuple(dec[b.subject_id] for b in bundles)
    return QG.GateReport(str(as_date(now)), decisions, tuple(s for r in reps.values() for s in r.promoted),
                         tuple(s for r in reps.values() for s in r.newly_quarantined), QG.funnel(list(decisions)))


def verdicts(report) -> dict[str, str]:
    return {getattr(d, "subject_id", "?"): str(d.verdict) for d in report.decisions}


def blocking(report, subject_id: str) -> dict[str, str]:
    """Gate -> state for every gate that blocked `subject_id` (for reports and tests)."""
    for d in report.decisions:
        if getattr(d, "subject_id", "") == subject_id:
            return {g.gate: f"{g.state}: {g.detail[:90]}" for g in d.gates if not g.ok}
    return {}
