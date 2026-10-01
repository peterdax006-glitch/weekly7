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
alpha over all looks, and a finding is retired on measured futility, never on a count of looks.

F26 (C75 Phase 3/10, driven by the F19 real-vs-noise benchmark; each fix names the defect it closes in its own docstring):
  leak-shaped features     construction_audit (truncation + future scramble of the candidate's own derivation) and the
                           future-dependence screen with a planted probe (future_dependence -> quality_gate.screen_future_dependence)
  calibration              the claim-shaped conditional forecasts (calibration, fixed_margin_forecast) with a reference forecast
  replication              power-designed run length (run_length), averaged matched control (matched_control), measured regime
  risk                     claim_risk: a magnitude claim signs no holding
  retirement               SequentialPlan.retire: repeated FAILED looks retire
  multiplicity             FindingSpec.n_scanned / n_search: the screen's full search size
  cost                     vectorised per_date_effect and identity-harness scoring (adopted by identity_firewall in F28): ~0.75 s/bundle

F28 (the F26 remainder, same benchmark; each names its defect in its docstring):
  proxies                  rival_ranks / incremental: the finding against its strongest correlated rival among every scored feature
                           (quality_gate.rival_check - a proxy that adds nothing beyond it FAILS out of sample)
  identity units           name_units: an ordering that is mostly a per-name level is tested with NAMES as the units
  weak leaks               future_dependence: a per-candidate suspicion tier under the family-wise integrity bar (UNKNOWN, never
                           QUARANTINE), and documented_availability: a dated pre-decision publication record explains a magnitude
                           signature (a genuine scheduled event) instead of refusing it
EvidenceConfig.f26_* / f28_* switches exist only so the benchmark can attribute each fix; every default is the fixed behaviour."""
from __future__ import annotations

import contextlib
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
    repl_block: int = 13                 # the SHORTEST replication run: a quarter of consecutive, non-overlapping test dates
    repl_max_block: int = 52             # F26: the longest run a power design may ask for (a year of weekly dates)
    repl_min_periods: int = 12
    leak_alpha: float = 0.05             # F26: family-wise level of the future-dependence screen over the whole search (both signs)
    leak_z_floor: float = 3.0            # ... and its bar is never below this z, however small the search
    leak_min_cell: int = 6               # rows per (date, outcome class) cell the future-dependence screen needs
    repl_control_perms: int = 20         # F26: shuffled-label draws averaged into each replication run's matched control
    # F26 fix switches. Every default is the fixed behaviour; False restores the pre-F26 evidence ONLY so the benchmark can attribute
    # each fix (pattern_benchmark BenchConfig.evidence_ablation). Nothing else may turn them off.
    f26_leak_screen: bool = True         # construction audit + future-dependence screen in the leak audit
    f26_claim_calibration: bool = True   # the claim-shaped (within-date, fixed-margin) calibration evidence
    f26_repl_design: bool = True         # power-designed run length, averaged matched control, measured discovery regime
    f26_claim_risk: bool = True          # a magnitude (VOLATILITY) claim bears no directional position risk of its own
    f26_fold_se: bool = True             # the complexity gate's worst-fold tolerance in fold standard errors (QualityPolicy)
    # F28 (the F26 remainder): the proxy test, identity units and the weak-leak tier. Same rule as the f26_ switches: False only for
    # the benchmark's attribution of each fix.
    rival_min_corr: float = 0.3          # |within-date rank correlation| at which a scored feature is a rival the finding must beat
    name_share: float = 0.5              # between-name share of the ordering at which names, not name-weeks, are the units
    name_min_rows: int = 4               # test rows a name needs to enter the name-level test
    leak_suspect_alpha: float = 0.01     # per-candidate two-sided level of the future-dependence SUSPICION tier (withholds, never quarantines)
    f28_rival: bool = True
    f28_name_units: bool = True
    f28_leak_suspect: bool = True
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
        if self.repl_block < 4 or self.repl_max_block < self.repl_block:
            errs.append("repl_block >= 4 dates and repl_max_block >= repl_block required (a replication run needs periods to speak)")
        if not 0.0 < self.leak_alpha <= 0.1 or self.leak_z_floor < 3.0 or self.leak_min_cell < 4:
            errs.append("leak_alpha in (0, 0.1], leak_z_floor >= 3 and leak_min_cell >= 4 required: the future-dependence screen must "
                        "not fire on noise")
        if not 0.05 <= self.align_min_frac <= self.orient_frac <= self.align_max_frac <= 0.6:
            errs.append("0.05 <= align_min_frac <= orient_frac <= align_max_frac <= 0.6 required")
        if self.repl_control_perms < 1:
            errs.append("repl_control_perms >= 1 required")
        if not 0.0 < self.max_failure_rate <= 0.25:
            errs.append("max_failure_rate in (0, 0.25] required: 'fails rarely' must mean rarely")
        if not 0.1 <= self.rival_min_corr < 1.0 or not 0.2 <= self.name_share <= 1.0 or self.name_min_rows < 2:
            errs.append("rival_min_corr in [0.1, 1), name_share in [0.2, 1] and name_min_rows >= 2 required")
        if not 0.0 < self.leak_suspect_alpha <= 0.05:
            errs.append("leak_suspect_alpha in (0, 0.05] required: the suspicion tier must not withhold genuine findings freely")
        return errs


@dataclasses.dataclass(frozen=True)
class FindingSpec:
    """What is being judged: a derived feature with its orientation sign, for a problem, identified by the loop's subject id."""
    subject_id: str
    feature: str
    sign: float = 1.0
    problem: str = "VOLATILITY"
    n_tests_searched: int = 1            # how many candidates the search RAISED before this one
    has_falsifier: bool = False          # science memory holds a declared falsifier (a retirement trigger) for this finding
    experiment_id: str = ""
    run_id: str = "research"
    seed: int = 0
    n_scanned: int = 0                   # F26: how many features the screen SCORED to raise it (0 = not reported by the caller)

    def validate(self) -> list[str]:
        from engine.research import vol_hypotheses as VH
        errs = []
        if not self.subject_id:
            errs.append("subject_id required")
        if self.feature not in VH.DERIVED:
            errs.append(f"unknown derived feature {self.feature!r}")
        if self.sign not in (1.0, -1.0):
            errs.append("sign must be +1 or -1")
        if self.n_tests_searched < 1 or self.n_scanned < 0:
            errs.append("n_tests_searched >= 1 and n_scanned >= 0 required")
        return errs

    @property
    def n_search(self) -> int:
        """F26 (F19 failure 6): the multiplicity the gate corrects for is the size of the SEARCH - every feature the screen scored - not
        the number of candidates it raised. The screen chose this candidate as the best of `n_scanned` scored features (on data that
        overlaps the gate's test window), so correcting for the ~60-100 raised understated the search about tenfold."""
        return int(max(1, self.n_tests_searched, self.n_scanned))


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
    retire_failed_looks: int = 2         # F26: consecutive FAILED looks (each on a new quarter of evidence) that retire a finding

    def validate(self) -> list[str]:
        errs = []
        if not 0.0 < self.alpha < 0.5 or self.futility_z <= 0:
            errs.append("alpha in (0, 0.5) and futility_z > 0 required")
        if self.horizon_dates < 52:
            errs.append("horizon_dates >= 52 required: a finding must be allowed at least a year of fresh evidence")
        if self.retire_failed_looks < 2:
            errs.append("retire_failed_looks >= 2 required: one FAILED look may be a bad quarter, a repeated one is measured")
        return errs

    def retire(self, bundle: "Bundle", verdicts: Sequence[str], min_effect: float = 0.0) -> str | None:
        """F26 (F19 failure 5): why a finding leaves the re-gate queue, or None to keep it. `verdicts` are its gate verdicts so far,
        oldest first, INCLUDING this look. FAILED means the gate MEASURED negative evidence (it does not survive out of sample, an unsafe
        tail, a miscalibrated claim ...), unlike NEEDS_MORE_EVIDENCE. F19 found FAILED candidates re-gated every quarter for ever (the
        queue only grows). A finding is now retired when (a) the measured futility rules hold (`futility`), or (b) its last
        `retire_failed_looks` looks were all FAILED: the negative evidence was re-measured on a fresh quarter and held. One FAILED look
        is never enough (a single bad quarter must not kill a true effect, the F12 lesson), and NEEDS_MORE_EVIDENCE never counts."""
        why = self.futility(bundle, min_effect)
        if why is not None:
            return why
        v = [str(x) for x in verdicts]
        k = int(self.retire_failed_looks)
        if len(v) >= k and all(x == "FAILED" for x in v[-k:]):
            return f"FAILED at {k} consecutive looks on fresh evidence: the negative measurement held"
        return None

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
    """Per-date rank AUC - 0.5 (dates with both classes and >= min_names finite rows). `jitter_seed` adds a seeded infinitesimal
    jitter (the only randomness in the measurement; reruns on seeds show it does not matter).

    F28: the jitter used to be drawn per ROW, so it broke exact ties at random - and a tied score's AUC under random tie-breaking is
    a different number on every seed (the average-rank AUC only in expectation). A binary feature (a scheduled-event flag, 90% zeros)
    then failed the reproducibility gate's across-seed spread although nothing about it was irreproducible. The jitter is now drawn per
    DISTINCT VALUE: equal values stay equal (ties keep their average rank), distinct values are perturbed by ~1e-9."""
    s = score.to_numpy(float)
    if jitter_seed is not None:
        fin = np.isfinite(s)
        if fin.any():
            u, inv = np.unique(s[fin], return_inverse=True)
            s = s.copy()
            s[fin] = s[fin] + np.random.default_rng(jitter_seed).normal(0, 1e-9, len(u))[inv]
    codes, uniq = pd.factorize(score.index.get_level_values(0), sort=True)
    yy = y.to_numpy(float)
    ok = np.isfinite(s) & np.isfinite(yy)
    if not ok.any():
        return pd.Series(dtype=float)
    # F26: one vectorised pass (average ranks within each date = scipy rankdata's tie rule, so every value equals the old per-date
    # engine.pattern_movers.auc loop exactly: the rank sums are sums of half-integers, exact in float64)
    c, lab = codes[ok], yy[ok] >= 0.5
    r = pd.Series(s[ok]).groupby(c).rank(method="average").to_numpy(float)
    k = len(uniq)
    n = np.bincount(c, minlength=k).astype(float)
    n1 = np.bincount(c, weights=lab.astype(float), minlength=k)
    rs = np.bincount(c, weights=np.where(lab, r, 0.0), minlength=k)
    n0 = n - n1
    keep = (n >= min_names) & (n1 > 0) & (n0 > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (rs - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    keep &= np.isfinite(a)
    return pd.Series(a[keep] - 0.5, index=pd.DatetimeIndex(pd.to_datetime(uniq[keep])), dtype=float).sort_index()


def unseen_dates(years: np.ndarray, c: int) -> int:
    """Test dates in calendar years with no train date when the first `c` sorted dates train (train years are contiguous, so a test
    date is unseen exactly when its year is after the last train date's year)."""
    return int((years[c:] > years[c - 1]).sum()) if 0 < c <= len(years) else 0


def train_cut(dates: Sequence | np.ndarray, ec: EvidenceConfig) -> tuple[int, bool]:
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
                                  spec.n_search, float(len(x)))
    dates = tuple(str(d.date()) for d in eff_te.index)
    for d in dates:
        require_past(d, now, "out-of-sample date")
    oos = PR.OOSEvidence(str(as_date(train_end)), dates, tuple(float(v) for v in x), float(eff_tr.mean()) if len(eff_tr) else 0.0,
                         (f"{dates[0]}..{dates[-1]}",) if dates else (), (f"..{as_date(train_end)}",))
    return stat, oos


def fast_per_date_ic(scores: pd.Series, y: pd.Series, min_names: int = 5) -> pd.Series:
    """F26's vectorised per-date IC, ADOPTED by engine.learning.identity_firewall.per_date_ic in F28 (defect c): kept as a name only."""
    from engine.learning import identity_firewall as IDF
    return IDF.per_date_ic(scores, y, min_names)


def fast_top_k_spread(scores: pd.Series, y: pd.Series, k: int = 5) -> float:
    from engine.learning import identity_firewall as IDF
    return IDF.top_k_spread(scores, y, k)


def fast_multiset_key(X: pd.DataFrame, y: pd.Series | None) -> str:
    from engine.learning import identity_firewall as IDF
    return IDF._multiset_key(X, y)


@contextlib.contextmanager
def fast_identity_scoring():
    """F26 swapped vectorised scoring into IdentityHarness for the duration of a call; F28 moved it into identity_firewall itself (the
    owner-side fix F26 asked for), so this is now a no-op kept for callers that still enter it."""
    yield


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


def date_margins(y: pd.Series, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(date code per row, outcomes per date, usable rows per date) over the rows in `ok`."""
    codes, _ = pd.factorize(y.index.get_level_values(0), sort=True)
    yy = np.where(ok, y.to_numpy(float), 0.0)
    return codes, np.bincount(codes, weights=yy), np.bincount(codes, weights=ok.astype(float))


def fixed_margin_forecast(x: np.ndarray, b: float, codes: np.ndarray, k: np.ndarray, n: np.ndarray, iters: int = 60) -> np.ndarray:
    """P(y_i = 1) = expit(alpha_d + b x_i) with each date's alpha_d solved so that the date's forecasts sum to its realised count k_d
    (conditioning on the margin, the sufficient statistic of a date effect). With b = 0 every name on a date gets k_d / n_d, so the
    date's rate adds no information about WHICH name moved: the forecasts only carry the claimed within-date ordering. Newton on each
    alpha_d (the sum is monotone in alpha_d); dates with k_d in {0, n_d} are NaN (nothing to order)."""
    rate = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    alpha = np.log(np.clip(rate, 1e-6, 1 - 1e-6) / (1 - np.clip(rate, 1e-6, 1 - 1e-6)))
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(alpha[codes] + b * x)))
        f = np.bincount(codes, weights=q, minlength=len(k)) - k
        g = np.bincount(codes, weights=q * (1 - q), minlength=len(k))
        step = np.where(g > 1e-12, f / np.maximum(g, 1e-12), 0.0)
        alpha = alpha - np.clip(step, -5, 5)
        if np.nanmax(np.abs(step)) < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(-(alpha[codes] + b * x)))
    trivial = (k <= 0) | (k >= n)
    return np.where(trivial[codes], np.nan, p)


def calibration(G: pd.DataFrame, score: pd.Series, y: pd.Series, tr: np.ndarray, te: np.ndarray, spec: FindingSpec, ec: EvidenceConfig):
    """The probability model the finding's CLAIM implies, fitted on train and scored out of sample (never refitted there).

    F26 defect (F19 failure 2: calibration blocked 96% of the gated real patterns): the claim a derived-feature finding makes is a
    RANKING within each date (its evidence is a per-date AUC); it never claims the level of the outcome rate, which moves with the market
    era (touch rates of 8% in calm, 30% in crisis). The old evidence fitted engine.learning.calibration.PlattCalibrator on the pooled
    within-date rank and scored it against pooled outcomes in a later era, so the gate judged a base-rate forecast the finding never made
    (Brier skill -0.03 for an AUC 0.60 rule); worse, platt_slope's undamped Newton diverges on a rank input (slope -2e7: tests/
    test_gate_vs_benchmark.py pins the defect), so every forecast was 0 or 1. The claim-shaped model is the conditional logit
        P(y = 1 | rank r, date d) = expit(alpha_d + b (r - 0.5)),
    with b fitted on the train window (damped Newton, date offsets) and, out of sample, each alpha_d fixed by the date's realised count
    (fixed_margin_forecast): the forecasts state exactly the finding's relative claim and nothing about the era. The REFERENCE forecast
    is the same model with b = 0 (every name at the date's rate), so the gate's Brier skill and recalibration slope measure the claimed
    lift and nothing else. (A leave-one-out date rate was tried first: it anti-correlates each forecast with its own outcome and made
    ECE 0.07 an artefact - rejected.) The gate's thresholds are unchanged; an overconfident lift (a train slope too steep out of
    sample, e.g. a decayed rule) still FAILS, and dates on which every name or no name moved carry nothing to order and are left out."""
    from engine.research import quality_gate as QG
    if not ec.f26_claim_calibration:
        return legacy_calibration(G, score, y, tr, te, spec, ec)
    rk = score.groupby(level=0).rank(pct=True).to_numpy(float).clip(0.01, 0.99)
    yy = y.to_numpy(float)
    ok = np.isfinite(rk) & np.isfinite(yy)
    codes, k_tr, n_tr = date_margins(y, ok & tr)
    rate = np.where(n_tr > 0, k_tr / np.maximum(n_tr, 1), np.nan)[codes]
    okt = ok & tr & np.isfinite(rate)
    if okt.sum() < 50 or len(np.unique(yy[okt])) < 2:
        return None, "too few rows to calibrate"
    fit = QG.logistic_offset_fit(rk[okt] - 0.5, yy[okt], QG.logit(np.clip(rate[okt], 1e-3, 1 - 1e-3)))
    if not fit["converged"]:
        return None, "the claim's lift model did not converge on the train window"
    _, k_te, n_te = date_margins(y, ok & te)
    x = np.where(ok & te, rk - 0.5, 0.0)
    p = fixed_margin_forecast(x, float(fit["b"]), codes, k_te, n_te)
    ref = fixed_margin_forecast(x, 0.0, codes, k_te, n_te)
    oke = ok & te & np.isfinite(p) & np.isfinite(ref)
    if oke.sum() < 50:
        return None, "too few test rows on dates with both outcomes to calibrate"
    p, ref, yt = np.clip(p[oke], 0.001, 0.999), np.clip(ref[oke], 0.001, 0.999), yy[oke]
    idx = np.random.default_rng(spec.seed + 5).permutation(len(p))[:ec.cal_rows]
    return QG.CalibrationEvidence(tuple(float(v) for v in p[idx]), tuple(int(v) for v in yt[idx]), spec.seed,
                                  tuple(float(v) for v in ref[idx]), "within-date rank lift at the date's realised rate"), ""


def legacy_calibration(G: pd.DataFrame, score: pd.Series, y: pd.Series, tr: np.ndarray, te: np.ndarray, spec: FindingSpec, ec: EvidenceConfig):
    """The pre-F26 calibration evidence (pooled Platt on the within-date rank), kept ONLY so the benchmark can attribute each F26 fix
    (EvidenceConfig.f26_claim_calibration = False); never the default."""
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
    a: set[str] = set()
    b: set[str] = set()
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
    Blocks are now keyed by their first date and only NEW blocks are added, so every look sees the fresh quarters that matured since.

    F26 defects fixed (F19 failure 2: replication blocked 91% of the gated real patterns): (a) every run was a fixed 13-date quarter on
    half the names, whatever the effect size, so a run of a real moderate effect had little power (replication.judge_run itself calls
    most of them 'under-powered'), yet each counted as an ATTEMPT in the success share - the design demanded the effect in samples too
    small to show it. The run length is now a POWER DESIGN fixed when the discovery is registered (`run_length`: the periods
    replication.required_n needs to see the winner's-curse-adjusted discovery effect at the policy's alpha and power, between repl_block
    and repl_max_block), so a run is sized to be able to answer. Nothing in the replication policy is relaxed. (b) The discovery's regime
    was the literal label 'train', so every run earned the REGIME freshness axis for free; it is now the train window's own volatility
    label, measured like each run's."""
    from engine.research import replication as RP
    names = G.index.get_level_values(1).astype(str)
    A, B = stock_halves(names, spec.subject_id)
    inA, inB = np.isin(names, list(A)), np.isin(names, list(B))
    mn = max(4, ec.min_names // 2)
    led = ledger if ledger is not None else RP.ReplicationLedger()
    did = "E" + spec.subject_id
    d = pd.to_datetime(G.index.get_level_values(0))
    ends = pd.to_datetime(G["end"])
    mvol = G["m_vol"].groupby(level=0).median() if "m_vol" in G else None

    def regime_of(dates) -> str:
        if mvol is None:
            return "calm"
        return "high_vol" if float(mvol.reindex(pd.DatetimeIndex(dates).unique()).median()) > float(mvol.median()) else "calm"
    if did not in led.discoveries():
        e0 = per_date_effect(score[tr & inA], y[tr & inA], mn)
        if len(e0) < 3 or float(e0.std()) <= 0:
            return None, {"why": f"{len(e0)} train periods on the discovery half"}
        trd = d[tr]
        led.add_discovery(RP.Discovery(did, float(e0.mean()), float(e0.std()), int(len(e0)), (str(trd.min().date()), str(trd.max().date())),
                                       ec.horizon_days, A, (spec.seed,), frozenset({regime_of(trd) if ec.f26_repl_design else "train"}),
                                       code_hash, data_hash, str(ends[tr].max().date()),
                                       spec.n_search))
    disc = led.discoveries()[did]
    prior = led.runs_for(did, now)
    after = max([pd.Timestamp(r.window[1]) for r in prior]
                + [pd.Timestamp(disc.window[1]) + pd.Timedelta(days=max(ec.purge_days, disc.horizon_days))])
    added = 0
    block = run_length(disc, ec) if ec.f26_repl_design else ec.repl_block
    for c in replication_blocks(d[te], after, block):
        m = te & inB & np.isin(d, c)
        e = per_date_effect(score[m], y[m], mn)
        if len(e) < 2:
            continue
        first = str(pd.Timestamp(c[0]).date())
        seed = int(stable_hash([spec.subject_id, first], 6), 16) % 1_000_000 + 1000     # a fresh seed per block, never the discovery's
        ctl = matched_control(score[m], y[m], mn, seed, ec.repl_control_perms if ec.f26_repl_design else 1).reindex(e.index).fillna(0.0)
        regime = regime_of(c)
        mat = str(ends[m].max().date())
        require_past(mat, now, "replication run")
        led.add_run(RP.ReplicationRun(f"{did}-{first}", did, (first, str(pd.Timestamp(c[-1]).date())), B, seed, regime,
                                      tuple(float(v) for v in e.to_numpy()), tuple(float(v) for v in ctl.to_numpy()), code_hash, data_hash,
                                      mat), now)
        added += 1
    runs = led.runs_for(did, now)
    a = RP.assess(disc, runs, now, policy) if policy is not None else RP.assess(disc, runs, now)
    return a, {"repl_status": str(a.status), "repl_runs": len(runs), "repl_runs_added": added, "repl_discovery_periods": int(disc.n_periods),
               "repl_i2": float(a.pooled.get("i2", 0.0)), "repl_supporting": int(a.n_supporting), "repl_run_length": int(block)}


def matched_control(score: pd.Series, y: pd.Series, min_names: int, seed: int, perms: int) -> pd.Series:
    """Per date: the effect the same score shows against the date's labels SHUFFLED across names, averaged over `perms` seeded shuffles.
    F26: a single shuffle (the old control) is one noisy draw from the null with the effect's own spread, so 'beats the control' needed
    the effect to beat noise twice over - it halved the power of every run (a run of a real effect at t = 4 did not beat its control).
    The average of many shuffles estimates the same null expectation with a fraction of the noise; the comparison it enters is
    unchanged."""
    rng = np.random.default_rng(seed)
    codes = pd.factorize(y.index.get_level_values(0), sort=True)[0]
    yy = y.to_numpy(float)
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(order)]
    acc: pd.Series | None = None
    for _ in range(max(1, int(perms))):
        ys = yy.copy()
        for a, b in zip(bounds[:-1], bounds[1:]):
            idx = order[a:b]
            ys[idx] = yy[idx][rng.permutation(b - a)]
        e = per_date_effect(score, pd.Series(ys, index=y.index), min_names)
        acc = e if acc is None else acc.add(e, fill_value=0.0)
    return (acc / max(1, int(perms))) if acc is not None else pd.Series(dtype=float)


def run_length(disc, ec: EvidenceConfig) -> int:
    """F26: the replication run length for a registered discovery, a pure function of the discovery record (so it never changes
    between looks and blocks are never cut twice): the periods engine.research.replication.required_n needs to detect the discovery's
    winner's-curse-adjusted effect with the DEFAULT policy's alpha and power, clamped to [repl_block, repl_max_block]. An effect the
    adjustment leaves at zero (indistinguishable from search luck) gets the longest run - the most power the design can give it."""
    from engine.research import replication as RP
    need = RP.required_n(RP.adjusted_effect(disc), float(disc.sd), RP.DEFAULT_POLICY.alpha, RP.DEFAULT_POLICY.power)
    if not math.isfinite(need):
        return int(ec.repl_max_block)
    return int(min(ec.repl_max_block, max(ec.repl_block, math.ceil(need))))


# ================================================================================================================ F28: proxies and identity units
def centered_ranks(X: pd.DataFrame) -> pd.DataFrame:
    """Per date, every column's percentile rank minus that date's mean rank (NaN where the value is missing). A column with no
    within-date variation (a market-level series) is all zero and correlates with nothing."""
    if X.empty:
        return X.astype("float32")
    R = X.groupby(level=0).rank(pct=True)
    return (R - R.groupby(level=0).transform("mean")).astype("float32")


def normal_scores(X: pd.DataFrame) -> pd.DataFrame:
    """Per date, every column's van der Waerden score Phi^-1((rank - 0.5) / n) minus the date's mean (NaN where missing; a column with
    no within-date variation is all zero). F28 (found on the 100-world run's development worlds): the proxy test regressed one feature's
    within-date PERCENTILE rank on another's, but the percentile ranks of two correlated normal features are not linearly related, so a
    proxy of a strong pattern kept a nonlinear piece of it (increment t 2-3.6 for proxies of regime / lifecycle patterns, 10 of 11
    remaining development false positives). Normal scores make a jointly normal pair linear again; incremental() also removes a cubic
    in the rival's score, so what is left is information the rival does not carry in any monotone smooth form."""
    if X.empty:
        return X.astype("float32")
    from scipy.stats import norm
    R = X.groupby(level=0).rank(method="average")
    n = X.notna().groupby(level=0).transform("sum")
    Z = pd.DataFrame(norm.ppf(((R - 0.5) / n).to_numpy(float)), index=X.index, columns=X.columns)
    return (Z - Z.groupby(level=0).transform("mean")).astype("float32")


def rival_pool(frame: pd.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    """Every derived feature computable on `frame` that a screen scores (the scan filter: no interaction forms, no history-dependent
    features): the candidates a finding must be told apart from when its caller does not supply its own search universe."""
    from engine.research import two_stage as TS
    from engine.research import vol_hypotheses as VH
    cols, ex = set(frame.columns), set(exclude)
    return [f for f in VH.DERIVED if not f.startswith(("ix__", "ixnull")) and f not in TS.HISTORY_DEPENDENT and f not in ex
            and not VH.missing_columns((f,), cols)]


_RIVAL_CACHE: dict[str, Any] = {}


def rival_ranks(frame: pd.DataFrame, features: Sequence[str] | None = None) -> pd.DataFrame:
    """Within-date normal scores (normal_scores) of `features` (default: rival_pool) on `frame`, float32. Market-level columns (no within-date
    variation) are dropped. The last result is kept for the same frame object and feature list (the loop gates many findings on one
    matured frame; rebuilding the ranks per finding would dominate the cost)."""
    from engine.research import vol_hypotheses as VH
    feats = list(dict.fromkeys(features if features is not None else rival_pool(frame)))
    key = stable_hash([len(frame), feats], 12)
    if _RIVAL_CACHE.get("frame") is frame and _RIVAL_CACHE.get("key") == key:
        return _RIVAL_CACHE["ranks"]
    if not feats or frame.empty:
        out = pd.DataFrame(index=frame.index, dtype="float32")
    else:
        D = VH.derive(frame, tuple(feats))[feats].astype(float)
        R = normal_scores(D)
        out = R.loc[:, (R.abs() > 0).any(axis=0).to_numpy()]
    _RIVAL_CACHE.update(frame=frame, key=key, ranks=out)
    return out


def _within_date_resid(a: np.ndarray, b: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """a with its within-date least-squares dependence on b removed (both already centred per date; missing b counts as the date's
    mean, 0). Rows where a is missing stay missing."""
    k = int(codes.max()) + 1 if len(codes) else 0
    a0, b0 = np.nan_to_num(a), np.nan_to_num(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.bincount(codes, weights=a0 * b0, minlength=k) / np.bincount(codes, weights=b0 * b0, minlength=k)
    beta = np.where(np.isfinite(beta), beta, 0.0)
    return np.where(np.isfinite(a), a0 - beta[codes] * b0, np.nan)


def _within_date_resid_poly(a: np.ndarray, b: np.ndarray, codes: np.ndarray, deg: int = 3) -> np.ndarray:
    """a with a within-date least-squares POLYNOMIAL (degree `deg`) in b removed: per date the (deg+1) x (deg+1) normal equations are
    accumulated with bincount and solved in one batched call (a singular date falls back to the linear residual). Missing b counts as
    the date's mean (0); rows where a is missing stay missing."""
    k = int(codes.max()) + 1 if len(codes) else 0
    if k == 0:
        return np.asarray(a, float).copy()
    a0, b0 = np.nan_to_num(a), np.nan_to_num(b)
    B = np.column_stack([b0 ** j for j in range(deg + 1)])
    m = deg + 1
    XtX = np.zeros((k, m, m))
    Xty = np.zeros((k, m))
    for i in range(m):
        Xty[:, i] = np.bincount(codes, weights=B[:, i] * a0, minlength=k)
        for j in range(i, m):
            XtX[:, i, j] = XtX[:, j, i] = np.bincount(codes, weights=B[:, i] * B[:, j], minlength=k)
    XtX += 1e-9 * np.eye(m)[None]
    try:
        coef = np.linalg.solve(XtX, Xty[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return _within_date_resid(a, b, codes)
    fit = np.einsum("ij,ij->i", B, coef[codes])
    return np.where(np.isfinite(a), a0 - fit, np.nan)


def incremental(G: pd.DataFrame, score: pd.Series, y: pd.Series, tr: np.ndarray, te: np.ndarray, eff_te: pd.Series, spec: FindingSpec,
                ec: EvidenceConfig, rivals: pd.DataFrame | None):
    """F28 (F26: 16 of the 21 remaining false positives were PROXIES - candidates correlated 0.6-0.9 with a real pattern and adding
    nothing beyond it; the gate judged each candidate alone, so a good proxy of a real pattern passes every test the pattern passes).
    The finding is set against its STRONGEST CORRELATED RIVAL among every feature the search scored (`rivals`: within-date normal
    scores from rival_ranks; filed knowledge belongs in it too): the rival is the scored feature with the largest |within-date
    correlation| with the finding on the TRAIN rows (a feature-feature relation: no outcome is used to choose it). Then, per test date:
      incremental  the effect (rank AUC - 0.5) of the finding's score with a cubic in the rival's score regressed out (within date)
      reverse      the rival's (oriented by its own train effect) with a cubic in the finding's regressed out
    quality_gate.rival_check reads them: a finding that keeps nothing once its rival is removed while the rival keeps something is a
    proxy; a real pattern keeps sqrt(1 - r^2) of its effect beyond any proxy of it. Returns (RivalEvidence, parts)."""
    from engine.learning import promotion as PR
    from engine.research import quality_gate as QG
    if rivals is None or rivals.shape[1] == 0:
        return QG.RivalEvidence(None, 0.0, 0, ec.rival_min_corr), {"rival": None, "rival_corr": 0.0, "rival_pool": 0}
    R = rivals.reindex(G.index)
    R = R[[c for c in R.columns if c != spec.feature]]
    ra = normal_scores(score.to_frame("f"))["f"].to_numpy(float)
    trm = np.asarray(tr, bool) & np.isfinite(ra)
    A = np.nan_to_num(ra[trm])
    B = np.nan_to_num(R.to_numpy(np.float32)[trm]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (A @ B) / np.sqrt((A @ A) * (B * B).sum(0))
    corr = np.where(np.isfinite(corr), corr, 0.0)
    n_pool = int(R.shape[1])
    if n_pool == 0 or float(np.abs(corr).max()) < ec.rival_min_corr:
        top = float(np.abs(corr).max()) if n_pool else 0.0
        return QG.RivalEvidence(None, top, n_pool, ec.rival_min_corr), {"rival": None, "rival_corr": top, "rival_pool": n_pool}
    j = int(np.argmax(np.abs(corr)))
    name = str(R.columns[j])
    rb = R.iloc[:, j].to_numpy(float)
    codes = pd.factorize(G.index.get_level_values(0), sort=True)[0]
    ea = pd.Series(_within_date_resid_poly(ra, rb, codes), index=G.index)
    eb = _within_date_resid_poly(np.where(np.isfinite(rb), rb, np.nan), np.nan_to_num(ra), codes)
    b_tr = per_date_effect(pd.Series(rb, index=G.index)[tr], y[tr], ec.min_names)
    sb = -1.0 if len(b_tr) and float(b_tr.mean()) < 0 else 1.0
    ebs = pd.Series(sb * eb, index=G.index)
    d_a = per_date_effect(ea[te], y[te], ec.min_names)
    d_b = per_date_effect(ebs[te], y[te], ec.min_names)
    riv = per_date_effect(pd.Series(sb * rb, index=G.index)[te], y[te], ec.min_names)
    own = float(eff_te.mean()) if len(eff_te) else None
    rival_eff = float(riv.mean()) if len(riv) else None
    ev = QG.RivalEvidence(name, float(corr[j]), n_pool, ec.rival_min_corr, PR.IncrementalEvidence(tuple(float(v) for v in d_a), spec.seed + 31),
                          PR.IncrementalEvidence(tuple(float(v) for v in d_b), spec.seed + 37), own, rival_eff)
    return ev, {"rival": name, "rival_corr": float(corr[j]), "rival_pool": n_pool, "increment": float(d_a.mean()) if len(d_a) else None,
                "increment_t": PR.t_stat(d_a.to_numpy()) if len(d_a) >= 2 else None,
                "rival_increment_t": PR.t_stat(d_b.to_numpy()) if len(d_b) >= 2 else None, "rival_effect": rival_eff}


def name_units(G: pd.DataFrame, score: pd.Series, y: pd.Series, tr: np.ndarray, te: np.ndarray, ec: EvidenceConfig):
    """F28 (F26: identity_null false positives - a per-name artefact counted as name-weeks). A column that is mostly a fixed per-name
    level ranks the names the same way every week; if those names' outcome base rates happen to line up with it (48 names: a chance
    correlation of sd 0.15, and it persists, because the base rates persist), every weekly AUC repeats ONE cross-name association and the
    per-date t-test counts it ~150 times. Measured here:
      between_share  share of the train rows' within-date rank variance that the names' train means explain (0 for a feature that
                     varies within names, 1 for a pure per-name constant)
      name test      names as the units: Spearman correlation across names of the train mean rank (known at the first test date)
                     with the name's out-of-sample outcome excess over its dates' rates; one-sided t with n_names - 2 df
      within test    dates as the units: the per-date effect of the rank minus the name's train mean (the name levels removed)
    quality_gate.name_units_check applies the gate's own alpha and search-size correction to whichever units the evidence has.
    Returns (NameUnitsEvidence, parts)."""
    from scipy import stats as sps
    from engine.learning import promotion as PR
    from engine.research import quality_gate as QG
    r = centered_ranks(score.to_frame("f"))["f"].astype(float)
    names = G.index.get_level_values(1)
    trs = pd.Series(np.asarray(tr, bool), index=G.index)
    rt = r[trs.to_numpy() & np.isfinite(r.to_numpy())]
    if len(rt) < 20:
        return QG.NameUnitsEvidence(0.0, False, 0, threshold=ec.name_share), {"between_share": None}
    m = rt.groupby(level=1).mean()
    tot = float((rt ** 2).sum())
    share = float((m.reindex(rt.index.get_level_values(1)).to_numpy() ** 2).sum() / tot) if tot > 0 else 0.0
    applies = share >= ec.name_share
    tem = np.asarray(te, bool)
    yt = y[tem]
    exc = (yt - yt.groupby(level=0).transform("mean")).groupby(level=1).agg(["mean", "size"])
    exc = exc[exc["size"] >= ec.name_min_rows]
    common = m.index.intersection(exc.index)
    rho = p_name = None
    if len(common) >= 5:
        rho = float(sps.spearmanr(m.loc[common], exc.loc[common, "mean"]).statistic)
        if np.isfinite(rho):
            n = len(common)
            t = rho * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1e-12, 1 - rho * rho))
            p_name = float(sps.t.sf(t, n - 2))
        else:
            rho = None
    w = (r - pd.Series(m.reindex(names).to_numpy(), index=G.index))[tem]
    ew = per_date_effect(w, yt, ec.min_names)
    tw = PR.t_stat(ew.to_numpy()) if len(ew) >= 2 else float("nan")
    p_within = PR.one_sided_p(tw) if np.isfinite(tw) else None
    ev = QG.NameUnitsEvidence(share, bool(applies), int(len(common)), rho, p_name, float(ew.mean()) if len(ew) else None, p_within, ec.name_share)
    return ev, {"between_share": share, "name_units_apply": bool(applies), "name_corr": rho, "name_p": p_name, "within_p": p_within}


# ================================================================================================================ F28: dated availability
PUBLISHED_SUFFIX = "__published_at"


def documented_availability(G: pd.DataFrame, feature: str) -> tuple[bool | None, str]:
    """(documented?, detail) for the feature's inputs: every base column c needs a per-row publication time `c__published_at` (a
    DATA record, e.g. the time an event calendar entry was published) strictly before the row's decision date on every row where the
    feature has a value. None = no publication record at all (nothing is documented); False = a record exists but some value was
    published at/after its decision (the record itself shows the leak). A documented magnitude signal is new information that was
    public before the decision - a scheduled event - which is observationally identical to a leak on the outcome window alone."""
    from engine.research import vol_hypotheses as VH
    base = list(VH.required_columns((feature,))) if feature in VH.DERIVED else [feature]
    cols = [c + PUBLISHED_SUFFIX for c in base]
    if not base or not all(c in G.columns for c in cols):
        return None, "no publication record for " + ", ".join(c for c in cols if c not in G.columns)
    d = pd.to_datetime(G.index.get_level_values(0))
    has = np.ones(len(G), bool)
    for c in base:
        has &= np.isfinite(pd.to_numeric(G[c], errors="coerce").to_numpy(float))
    for c in cols:
        pub = pd.to_datetime(G[c], errors="coerce")
        bad = has & ~(pub.to_numpy() < d.to_numpy())
        if bad.any():
            return False, f"{int(bad.sum())} value(s) of {c[:-len(PUBLISHED_SUFFIX)]} were published at/after their decision (or undated)"
    return True, f"every value of {', '.join(base)} was published strictly before its decision ({int(has.sum())} rows)"


# columns that MATURE after the decision: the realised outcome of the row's own window (never an input)
OUTCOME_COLUMNS = ("touch", "up", "close", "absmove", "tday", "end")


def construction_audit(frame: pd.DataFrame, feature: str, seed: int = 0) -> list[str]:
    """F26 (F19 failure 1), the F09 method on the candidate's OWN construction: the derived feature is recomputed (a) with every
    outcome column permuted across names within each date (a future scramble of the row's own window) and (b) on the frame truncated
    at its middle date (a truncation: rows after the cut removed); every value dated at or before the cut must be unchanged by both.
    Also refused outright: a derivation that names an outcome column as an input. A stored column (a feature whose derivation is the
    column itself) passes by construction - its values are audited statistically by the future-dependence screen."""
    from engine.research import vol_hypotheses as VH
    if feature not in VH.DERIVED:
        return []
    need = tuple(VH.required_columns((feature,)))
    out = [f"{feature} is derived from outcome column(s) {sorted(set(need) & set(OUTCOME_COLUMNS))}: they mature after the decision"] \
        if set(need) & set(OUTCOME_COLUMNS) else []
    if VH.missing_columns((feature,), frame.columns) or len(frame) == 0:
        return out
    F = frame[[c for c in dict.fromkeys(list(need) + [c for c in OUTCOME_COLUMNS if c in frame.columns])]].sort_index()
    base = VH.derive(F, (feature,))[feature].astype(float)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 26]))
    S = F.copy()
    codes = pd.factorize(S.index.get_level_values(0), sort=True)[0]
    perm = np.argsort(codes * 1.0 + rng.random(len(S)) * 0.5, kind="stable")          # a random order of the rows inside each date
    for c in OUTCOME_COLUMNS:
        if c in S.columns:
            S[c] = S[c].to_numpy()[perm]
    alt = VH.derive(S, (feature,))[feature].astype(float).reindex(base.index)
    if not np.allclose(base.to_numpy(), alt.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True):
        out.append(f"{feature} changes when the rows' future outcomes are scrambled within each date: it reads the outcome window")
    dates = pd.DatetimeIndex(pd.to_datetime(F.index.get_level_values(0))).unique().sort_values()
    if len(dates) >= 4:
        cut = dates[len(dates) // 2]
        T = F[pd.to_datetime(F.index.get_level_values(0)) <= cut]
        tv = VH.derive(T, (feature,))[feature].astype(float)
        bv = base.reindex(tv.index)
        if not np.allclose(bv.to_numpy(), tv.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True):
            out.append(f"{feature} dated on or before {cut.date()} changes when the data after it are removed: it reads later rows")
    return out


def past_magnitude(G: pd.DataFrame, mag: pd.Series, lag: int = 2) -> pd.Series:
    """Each name's outcome magnitude `lag` decision dates earlier, kept only where that outcome had MATURED before the row's date
    (end strictly before the decision): a past quantity the decision could know."""
    if "end" not in G:
        return pd.Series(np.nan, index=G.index)
    H = pd.DataFrame({"m": mag.to_numpy(float), "e": pd.to_datetime(G["end"]).to_numpy()}, index=G.index).sort_index()
    by = H.groupby(level=1)
    pm, pe = by["m"].shift(lag), by["e"].shift(lag)
    known = pe < pd.to_datetime(H.index.get_level_values(0))
    return pm.where(known).reindex(G.index)


def future_dependence(G: pd.DataFrame, score: pd.Series, y: pd.Series, spec: FindingSpec, ec: EvidenceConfig) -> tuple[list[str], dict]:
    """quality_gate.screen_future_dependence on the candidate (two-sided: an oriented score may carry the leak with either sign), with a
    PLANTED probe on the same rows - noise plus the future magnitude - that the screen must catch, or the audit is blind here.
    Future magnitude = |realised outcome| (`absmove` when the frame has it, else |close|); control = the audited volatility state.
    Returns (integrity findings, measures); measures['suspicions'] and measures['documented'] carry the F28 tiers.

    F28 (F26: the weakest leaks, strength 0.3, reached z 2.6-3.4 under a 4.06 bar and were promoted in low-power worlds). Why the bar
    misses them: the screen conditions on the outcome class, which removes exactly the part of a magnitude leak that predicts the label
    (a touch week is a big-magnitude week), so only the WITHIN-class magnitude spread is left to see it - small, and thinner still in a
    calm era where the touch class holds a handful of names a week (cells under leak_min_cell are dropped). Its bar is family-wise over
    the whole search (615 features, both signs) because a finding over it is an integrity accusation that QUARANTINES; lowering it would
    quarantine genuine findings at the search's scale. The fix is a second tier that makes no accusation: a signature at the
    PER-CANDIDATE two-sided level `leak_suspect_alpha` (z 2.58 at 0.01; a genuine predictor with no magnitude dependence crosses it 1% of
    the time) is a SUSPICION - the leak gate answers UNKNOWN, promotion waits, nothing is quarantined. A genuine new-information magnitude
    signal (a scheduled event inside the window) has the same signature on the outcome window - no statistic of the outcome window
    separates it from a leak - so what separates them is WHEN the value was public: documented_availability. A signature whose inputs
    carry a per-row publication record strictly before every decision is recorded under 'documented' and not held against the finding
    (a record that shows a publication at/after the decision is itself a finding)."""
    from statistics import NormalDist
    from engine.research import quality_gate as QG
    if "absmove" in G:
        mag = G["absmove"].astype(float)
    elif "close" in G:
        mag = G["close"].astype(float).abs()
    else:
        return [], {"z": None, "probe": None, "suspicions": [], "documented": []}
    past = past_magnitude(G, mag)
    ctl = G["vol20"].astype(float) if "vol20" in G else None
    zb = leak_bar(spec.n_search, ec)
    f1, m = QG.screen_future_dependence(score, y, mag, past, ctl, zb, ec.leak_min_cell, name=spec.feature)
    f2, m2 = QG.screen_future_dependence(-score, y, mag, past, ctl, zb, ec.leak_min_cell, name=spec.feature)
    rng = np.random.default_rng(np.random.SeedSequence([int(spec.seed), 27]))
    z = (mag - mag.mean()) / (mag.std() or 1.0)
    probe = pd.Series(rng.normal(0.0, 1.0, len(G)), index=G.index) + z
    fp, mp = QG.screen_future_dependence(probe, y, mag, past, ctl, zb, ec.leak_min_cell, name="planted magnitude probe")
    zs = [v for v in (m.get("z"), m2.get("z")) if v is not None and math.isfinite(v)]
    zmax = max(zs) if zs else None
    findings = f1 + f2
    suspect_bar = float(NormalDist().inv_cdf(1.0 - ec.leak_suspect_alpha / 2.0))
    sus = []
    if ec.f28_leak_suspect and not findings and zmax is not None and zmax >= suspect_bar:
        sus = [f"{spec.feature} shows a future-dependence signature (z = {zmax:.2f} >= {suspect_bar:.2f}, the per-candidate "
               f"{ec.leak_suspect_alpha:g} level; integrity bar {zb:.2f}): it may know how the coming outcome realises - promotion waits "
               f"for a documented pre-decision availability of its inputs"]
    documented: list[str] = []
    if findings or sus:
        ok, why = documented_availability(G, spec.feature)
        if ok:
            documented = [f"explained by dated availability: {why}"] + findings + sus
            findings, sus = [], []
        elif ok is False:
            findings = findings + [f"{spec.feature}: {why}"]
    return findings, {**m, "z": zmax, "z_pos": m.get("z"), "z_neg": m2.get("z"), "probe_caught": bool(fp), "probe_z": mp.get("z"),
                      "cells": m.get("cells"), "bar": zb, "suspect_bar": suspect_bar, "suspicions": sus, "documented": documented}


def leak_bar(n_search: int, ec: EvidenceConfig) -> float:
    """The future-dependence screen's z bar: Bonferroni at `leak_alpha` over the whole search, both signs (a null feature anywhere in
    the scan crosses it with probability <= leak_alpha), never below `leak_z_floor`. Set by the search size, not by any result."""
    from statistics import NormalDist
    return float(max(ec.leak_z_floor, NormalDist().inv_cdf(1.0 - ec.leak_alpha / (2.0 * max(1, int(n_search))))))


def leakage(G: pd.DataFrame, score: pd.Series, y: pd.Series, spec: FindingSpec, now, train_end, first_test, identity_report,
            code_hash: str, prov, n_decisions: int, used: np.ndarray | None = None, construction: Sequence[str] = (),
            cfg: EvidenceConfig | None = None):
    """The future-information audit: (1) the learning firewall on the candidate's OWN context (its panel, horizon, experiment and
    evaluation records, identity report, code state), (2) the firewall's planted corpus (the audit can still see planted leaks),
    (3) a candidate-specific planted leak: the label copied into a feature column must be flagged by the leak screen, (4) outcomes
    dated at/after now are counted (never assumed zero), and F26: (5) `construction` - the findings of construction_audit (the F09
    truncation / future-scramble test on the candidate's own derivation) - and (6) the future-dependence screen on the candidate's
    values (quality_gate.screen_future_dependence: does it know how the coming outcome realises?)."""
    from engine.learning import firewalls as FW
    from engine.research import quality_gate as QG
    ec = cfg or EvidenceConfig()
    extra, fd = future_dependence(G, score, y, spec, ec) if ec.f26_leak_screen else ([], {})
    extra = list(construction) + extra
    if used is not None:                                 # only the rows the evidence used (train + test; the purged gap is out)
        G, score = G[used], score[used]
    X = pd.DataFrame({spec.feature: score.to_numpy(float)}, index=G.index)
    yy = G["close"].astype(float).rename("ret")
    d = pd.to_datetime(G.index.get_level_values(0))
    label_close = pd.Series(pd.to_datetime(G["end"]).to_numpy(), index=G.index)
    item = {"knowledge_id": spec.subject_id, "version": 1, "provenance": prov, "contexts": {}, "anti_contexts": {}}
    exp = FW.ExperimentRecord(spec.experiment_id or spec.subject_id, prov.created_real, prov.created_real, prov.config_hash, spec.seed,
                              f"{spec.feature} ranks {spec.problem.lower()} outcomes", n_variants_tried=spec.n_search,
                              selected_from=spec.n_search, multiplicity_correction="holdout",
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
    fd_blind = fd.get("probe_caught") is False           # the future-dependence screen missed its planted probe on these rows
    probe = bool(corp.get("clean_passed")) and not corp.get("missed") and caught_own and not fd_blind
    after = int((pd.to_datetime(G["end"]) >= pd.Timestamp(as_date(now))).sum())
    return QG.leak_evidence_from_panel(X, yy, fwv, probe, after, extra_findings=extra, suspicions=fd.get("suspicions", ()),
                                       documented=fd.get("documented", ())), {
        "firewall_passed": bool(getattr(fwv, "passed", False)), "planted_probe_caught": probe, "own_probe_caught": caught_own,
        "outcomes_after_now": after, "construction_findings": len(construction), "future_dependence_z": fd.get("z"),
        "future_dependence_rho": fd.get("rho_future"), "future_probe_caught": fd.get("probe_caught"),
        "leak_findings": len(extra), "leak_suspicions": len(fd.get("suspicions", ())), "leak_documented": len(fd.get("documented", ()))}


# ================================================================================================================ the bundle
def assemble(frame: pd.DataFrame, spec: FindingSpec, now, *, code_hash: str, data_hash: str, created_real: str,
             cfg: EvidenceConfig = EvidenceConfig(), ledger=None, look: int | None = None, plan: SequentialPlan | None = None,
             rivals: pd.DataFrame | None = None) -> Bundle:
    """PUBLIC ENTRY. Every QualityEvidence field for `spec` from the matured research `frame` (rows whose outcome ended strictly
    before now; a row at/after now is a FirewallBreach, not a filter). Anything not computable stays None and is named in `missing`.
    `look` (1, 2, ...) = this is the finding's look-th sequential look: replication is assessed under the plan's spent alpha.
    `rivals` (F28) = within-date centred ranks (rival_ranks) of every feature the search scored plus filed knowledge, indexed like the
    frame; None = the rival pool is built from the frame itself (rival_pool: every scannable derived feature it can compute)."""
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
    rival_ev = units_ev = None
    if stat is not None and cfg.f28_rival:
        rv = rivals if rivals is not None else rival_ranks(frame, rival_pool(frame, exclude=(spec.feature,)))
        rival_ev, p = incremental(G, score, y, tr, te, eff_te, spec, cfg, rv)
        parts.update(p)
    if stat is not None and cfg.f28_name_units:
        units_ev, p = name_units(G, score, y, tr, te, cfg)
        parts.update(p)
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
                        plan.replication_policy(look) if look is not None and plan is not None else None)
    parts.update(p)
    if look is not None and plan is not None:
        parts.update(look=int(look), alpha_look=plan.alpha_at(look), alpha_spent=plan.spent(look))
    if repl is None:
        missing["replication"] = p.get("why", "not computable")
    feats = {c: None for c in _base_columns(spec.feature)}
    pit = QG.pit_evidence(feats, str(first_test.date()), str(train_end.date()), str(first_test.date()), cfg.horizon_days, through,
                          fills_next_open=True, known_before=tuple(feats))
    leak, p = leakage(G, score, y, spec, now, train_end, first_test, rep, code_hash, prov, int(te.sum()), tr | te,
                      construction=construction_audit(frame, spec.feature, spec.seed) if cfg.f26_leak_screen else (), cfg=cfg)
    parts.update(p)
    risky, risk_note = claim_risk(spec, cfg)
    ev = QG.QualityEvidence(pit=pit, leak=leak, identity=ident,
                            oos=QG.OOSBundle(stat, oos, train_years, rival_ev, units_ev) if stat is not None else None,
                            replication=repl, outputs_probabilities=True, calibration=cal, changes_risk_decisions=risky,
                            risk_note=risk_note, risk=rk,
                            complexity=cx, transfer=tev, failure=fe, repro=rp, justification=JUSTIFICATION, provenance=prov)
    return Bundle(spec.subject_id, ev, parts, missing)


def claim_risk(spec: FindingSpec, ec: EvidenceConfig) -> tuple[bool, str]:
    """(does the claim itself select and sign a holding?, why not). F26 (F19: risk blocked 25% of the gated real patterns): the risk
    evidence is the drawdown of holding the finding's top bucket LONG. A VOLATILITY finding claims only that its top names will MOVE
    (a +-10% touch), not which way; on symmetric outcomes that long book is a coin-flip whose swings grow with exactly the volatility
    the finding correctly predicts, so the better the finding, the deeper the drawdown (planted AUC 0.6 rules failed at -38% to -70%).
    The gate was judging a position the claim never takes. Direction, loss-avoidance and coverage findings sign a holding and keep the
    risk gate; a magnitude finding's position risk is gated where the position is decided (direction, exits, stops). The risk evidence
    is still computed and reported."""
    if ec.f26_claim_risk and Problem.parse(spec.problem) == Problem.VOLATILITY:
        return False, ("a magnitude claim (which names will move, not which way) selects candidates but signs no holding: position risk "
                       "is gated where direction, exits and stops decide it")
    return True, ""


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
        reps = {g: QG.step(c, now, policy=dataclasses.replace(base, min_failure_episodes=g[1], worst_fold_in_fold_se=cfg.f26_fold_se),
                           store=store) for g, c in sorted(groups.items())}
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
    reps = {g: QG.step(c, now, policy=dataclasses.replace(plan.quality_policy(g[0], code_hash), min_failure_episodes=g[1],
                                                         worst_fold_in_fold_se=cfg.f26_fold_se), store=store)
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
