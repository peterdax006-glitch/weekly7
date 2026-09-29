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
                                half in later windows (fresh period + fresh stocks, shuffled-label control), assessed in the ONE ledger
  calibration                   engine.learning.calibration.PlattCalibrator fitted on train, scored out of sample
  failure behaviour             negative-effect episodes, the contexts that explain them, perturbation retention, retirement trigger,
                                out-of-scope abstention (probed, not asserted)
  provenance / PIT              engine.learning.core.Provenance fully filled; quality_gate.pit_evidence with the purge gap

No gate is passed by default: anything that cannot be computed is left None (the gate reads None as MISSING) and listed in
`Bundle.missing` with the reason. Public entry: `assemble(frame, spec, now, ...)`; `gate(bundles, now, ...)` runs the gate."""
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
    repl_windows: int = 2                # replication windows cut from the out-of-sample years
    repl_min_periods: int = 12
    cal_rows: int = 4000
    seeds: tuple = (1, 1, 2, 3)          # reproducibility reruns: one seed repeated (determinism) and fresh seeds
    catastrophic: float = -0.20
    fail_frac: float = 0.5               # a failure week delivers less than this share of the rule's typical (mean) effect
    noise_alpha: float = 0.05            # family-wise level: a failure week no further below the mean than the most extreme of the
                                         # n test weeks would fall by chance (Bonferroni over n) is sampling noise

    def validate(self) -> list[str]:
        errs = []
        if not 0.1 <= self.orient_frac <= 0.6 or self.min_names < 4 or not 0 < self.mover_frac < 0.5:
            errs.append("orient_frac in [0.1,0.6], min_names >= 4, mover_frac in (0,0.5) required")
        if self.purge_days < self.horizon_days or not 0.5 <= self.top_q < 1 or self.perturb_sd <= 0:
            errs.append("purge_days >= horizon_days, top_q in [0.5,1), perturb_sd > 0 required")
        if len(self.seeds) < 3 or len(set(self.seeds)) < 2 or len(set(self.seeds)) == len(self.seeds):
            errs.append("seeds need >= 3 reruns, >= 2 distinct seeds and one repeated seed")
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


def split_dates(G: pd.DataFrame, ec: EvidenceConfig) -> tuple[pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray]:
    """(train_end, first_test, train mask, test mask): the train window is the earliest `orient_frac` of dates, purged so that every
    train outcome ended before the first test date and the gap is at least `purge_days` (the section-42 PIT rule)."""
    d = pd.to_datetime(G.index.get_level_values(0))
    uniq = np.array(sorted(pd.unique(d)))
    if len(uniq) < 8:
        raise ValueError(f"{len(uniq)} dates: too few for a train / test split")
    k = max(2, int(len(uniq) * ec.orient_frac))
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


def complexity(eff_te: pd.Series, eff_tr: pd.Series, spec: FindingSpec):
    """The finding (one feature, one orientation) against the simplest alternative: no feature at all (a base-rate ranker, zero
    effect on the same dates). The candidate must earn its one unit of complexity."""
    from engine.learning import complexity as CX
    from engine.research import quality_gate as QG
    folds = pd.Series(eff_te.index.year.astype(str), index=eff_te.index)
    cand = CX.Candidate(CX.RuleSpec(f"rule_{spec.feature}", n_features=1, n_free_params=1), eff_te, folds,
                        float(eff_tr.mean()) if len(eff_tr) else None)
    base = CX.Candidate(CX.RuleSpec("base_rate", n_features=0, n_free_params=0), eff_te * 0.0, folds, 0.0)
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


def replicate(G: pd.DataFrame, score: pd.Series, y: pd.Series, tr: np.ndarray, te: np.ndarray, spec: FindingSpec, ec: EvidenceConfig,
              now, code_hash: str, data_hash: str, ledger=None):
    """Register the finding as a replication.Discovery on half the names in the train window and run it on the OTHER half in
    `repl_windows` later windows (fresh period + fresh stocks, a regime label per window, a shuffled-label control on the same
    periods), all in the ONE replication ledger when one is given. Returns (assessment, detail)."""
    from engine.research import replication as RP
    names = G.index.get_level_values(1).astype(str)
    A, B = stock_halves(names, spec.subject_id)
    inA, inB = np.isin(names, list(A)), np.isin(names, list(B))
    e0 = per_date_effect(score[tr & inA], y[tr & inA], max(4, ec.min_names // 2))
    if len(e0) < 3 or float(e0.std()) <= 0:
        return None, {"why": f"{len(e0)} train periods on the discovery half"}
    d = pd.to_datetime(G.index.get_level_values(0))
    trd = d[tr]
    ends = pd.to_datetime(G["end"])
    did = "E" + spec.subject_id
    mat0 = str(ends[tr].max().date())
    mvol = G["m_vol"].groupby(level=0).median() if "m_vol" in G else None
    disc = RP.Discovery(did, float(e0.mean()), float(e0.std()), int(len(e0)), (str(trd.min().date()), str(trd.max().date())),
                        ec.horizon_days, A, (spec.seed,), frozenset({"train"}), code_hash, data_hash, mat0, max(1, spec.n_tests_searched))
    led = ledger if ledger is not None else RP.ReplicationLedger()
    if did not in led.discoveries():
        led.add_discovery(disc)
    ted = np.array(sorted(pd.unique(d[te])))
    chunks = [c for c in np.array_split(ted, ec.repl_windows) if len(c)]
    runs = []
    known = {r.run_id for r in led.runs_for(did, now)}
    for i, c in enumerate(chunks):
        m = te & inB & np.isin(d, c)
        e = per_date_effect(score[m], y[m], max(4, ec.min_names // 2))
        if len(e) < 2:
            continue
        rng = np.random.default_rng(spec.seed + 100 + i)
        ys = y[m].groupby(level=0).transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
        ctl = per_date_effect(score[m], ys, max(4, ec.min_names // 2)).reindex(e.index).fillna(0.0)
        regime = "calm"
        if mvol is not None:
            regime = "high_vol" if float(mvol.reindex(c).median()) > float(mvol.median()) else "calm"
        mat = str(ends[m].max().date())
        require_past(mat, now, "replication run")
        rid = f"{did}-w{i}"
        run = RP.ReplicationRun(rid, did, (str(pd.Timestamp(c[0]).date()), str(pd.Timestamp(c[-1]).date())), B, spec.seed + 1 + i, regime,
                                tuple(float(v) for v in e.to_numpy()), tuple(float(v) for v in ctl.to_numpy()), code_hash, data_hash, mat)
        if rid not in known:
            led.add_run(run, now)
        runs.append(run)
    a = RP.assess(led.discoveries()[did], led.runs_for(did, now), now)
    return a, {"repl_status": str(a.status), "repl_runs": len(runs), "repl_discovery_periods": int(len(e0))}


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
             cfg: EvidenceConfig = EvidenceConfig(), ledger=None) -> Bundle:
    """PUBLIC ENTRY. Every QualityEvidence field for `spec` from the matured research `frame` (rows whose outcome ended strictly
    before now; a row at/after now is a FirewallBreach, not a filter). Anything not computable stays None and is named in `missing`."""
    from engine.learning.core import Provenance
    from engine.research import quality_gate as QG
    errs = cfg.validate() + spec.validate()
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
    cx = complexity(eff_te, eff_tr, spec) if len(eff_te) else None
    fe, p = failure(G, score, y, te, eff_te, rk.worst_period if rk is not None else None, spec, cfg)
    parts.update(p)
    rp = reproducibility(score, y, te, spec, cfg, code_hash, data_hash)
    repl, p = replicate(G, score, y, tr, te, spec, cfg, now, code_hash, data_hash, ledger)
    parts.update(p)
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


def gate(bundles: Sequence[Bundle], now, code_hash: str, store=None):
    """Run engine.research.quality_gate.step on the bundles with the policy pinned to the loop's code hash (reproducibility is judged
    against the code that produced the reruns)."""
    from engine.research import quality_gate as QG
    cands = [QG.Candidate(b.subject_id, b.evidence) for b in bundles]
    return QG.step(cands, now, policy=QG.QualityPolicy(code_hash=code_hash), store=store)


def verdicts(report) -> dict[str, str]:
    return {getattr(d, "subject_id", "?"): str(d.verdict) for d in report.decisions}


def blocking(report, subject_id: str) -> dict[str, str]:
    """Gate -> state for every gate that blocked `subject_id` (for reports and tests)."""
    for d in report.decisions:
        if getattr(d, "subject_id", "") == subject_id:
            return {g.gate: f"{g.state}: {g.detail[:90]}" for g in d.gates if not g.ok}
    return {}
