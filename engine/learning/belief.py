"""Evidence-weighted belief updating (contract C62 section 32; checklist C10).  IMPLEMENTED - NOT VALIDATED.

A belief is never overwritten with the newest result.  Each subject (a pattern, a relation, a rule) keeps a *prior*, a
growing set of immutable *evidence* records, and an append-only chain of *posterior* states.  New evidence moves the belief
in proportion to its quality (kind of test, independence, reliability, multiple-testing burden, age) and its precision;
contradictory evidence is counted and inflates uncertainty instead of being averaged away.

The maths is a normal-normal pool with quality-tempered precisions and a DerSimonian-Laird between-evidence variance
(heterogeneity = contradiction), plus optional exponential forgetting so stale evidence loses weight.  It is
order-independent: the posterior is a function of the evidence SET, so replaying the ledger reproduces every state.

Builds on: engine.pattern_reliability (nw_t Newey-West t, used to size the standard error of a return series).
Time rule (C56): evidence must have matured strictly before `now`; anything else raises FirewallBreach (fail closed)."""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine.learning.core import (Epistemic, FirewallBreach, Unknown, _StrEnum, as_date, canonical_json, clip01,
                                  require_past, stable_hash)

Z90 = float(sps.norm.ppf(0.95))


class BeliefError(ValueError):
    """Malformed belief operation (immutable-history violation, unknown version, bad prior).  Not a firewall breach."""


class EvidenceKind(_StrEnum):
    OOS_TEST = "OOS_TEST"                 # out-of-sample test of a frozen claim
    LIVE_OUTCOME = "LIVE_OUTCOME"         # realised outcome of a decision that used the claim
    REPLICATION = "REPLICATION"           # independent re-test on other data
    TRANSFER = "TRANSFER"                 # tested on another year / sector / stock group
    PLACEBO = "PLACEBO"                   # planted-null run; evidence about the *test*, still evidence about the claim
    SIMULATION = "SIMULATION"             # synthetic world
    IN_SAMPLE_FIT = "IN_SAMPLE_FIT"       # where the claim was found: inflated by selection


@dc.dataclass(frozen=True)
class BeliefConfig:
    """Numerical policy.  Frozen: the config that produced a state is part of that state's audit trail (config_hash)."""
    prior_sd: float = 0.05                # default prior standard deviation of the effect
    half_life_days: float | None = 1095.0  # evidence age at which weight halves; None = never forget
    z_contra: float = 2.0                 # predictive z for an opposite-signed record to count as a contradiction
    use_random_effects: bool = True       # DerSimonian-Laird heterogeneity inflation
    tau2_cap_mult: float = 25.0           # cap on between-evidence variance in units of the largest se^2
    min_se: float = 1e-9
    trial_penalty: float = 0.5            # quality /= 1 + trial_penalty * ln(n_trials)
    credible_level: float = 0.90
    supported_prob: float = 0.95          # P(sign is right) needed to call a belief SUPPORTED
    contradicted_mass: float = 0.35       # contradiction precision share that blocks SUPPORTED
    kind_quality: tuple[tuple[str, float], ...] = (
        ("OOS_TEST", 1.0), ("LIVE_OUTCOME", 1.0), ("REPLICATION", 0.9), ("TRANSFER", 0.7),
        ("PLACEBO", 1.0), ("SIMULATION", 0.5), ("IN_SAMPLE_FIT", 0.25))

    def quality_of_kind(self, kind) -> float:
        return dict(self.kind_quality).get(str(kind), 0.0)

    def config_hash(self) -> str:
        return stable_hash(self)

    def validate(self) -> list[str]:
        errs = []
        if not self.prior_sd > 0:
            errs.append("prior_sd must be > 0")
        if self.half_life_days is not None and not self.half_life_days > 0:
            errs.append("half_life_days must be > 0 or None")
        if not 0.5 < self.credible_level < 1:
            errs.append("credible_level in (0.5,1)")
        for k, v in self.kind_quality:
            if not 0 <= v <= 1:
                errs.append(f"kind_quality[{k}] outside [0,1]")
        return errs


DEFAULT_CFG = BeliefConfig()


# ------------------------------------------------------------------------------------------------ evidence
@dc.dataclass(frozen=True)
class Evidence:
    """One immutable observation about a subject.  `estimate`/`se` are on the effect scale of the belief (e.g. mean
    weekly excess return); `n` is the count of independent observations behind it; `n_trials` is how many candidate
    claims were searched to find this one (multiple-testing burden)."""
    subject: str
    observed_at: str                      # ISO date the evidence MATURED (real date, trusted side)
    estimate: float
    se: float
    n: int
    kind: EvidenceKind = EvidenceKind.OOS_TEST
    independence: float = 1.0             # 1 = no overlap with evidence already counted; 0 = a copy
    reliability: float = 1.0              # measurement reliability of the outcome/label
    n_trials: int = 1
    source: str = ""
    context: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "kind", EvidenceKind.parse(self.kind))
        object.__setattr__(self, "observed_at", as_date(self.observed_at).isoformat())
        object.__setattr__(self, "estimate", float(self.estimate))
        object.__setattr__(self, "se", float(self.se))
        ctx = self.context
        if isinstance(ctx, Mapping):
            ctx = tuple(sorted((str(k), str(v)) for k, v in ctx.items()))
        object.__setattr__(self, "context", tuple(ctx))

    @property
    def evidence_id(self) -> str:
        return stable_hash({"s": self.subject, "t": self.observed_at, "e": round(self.estimate, 12),
                            "se": round(self.se, 12), "n": self.n, "k": self.kind, "src": self.source,
                            "c": self.context, "i": self.independence, "r": self.reliability, "nt": self.n_trials})

    def validate(self) -> list[str]:
        errs = []
        if not self.subject:
            errs.append("empty subject")
        if not math.isfinite(self.estimate):
            errs.append("estimate not finite")
        if not (math.isfinite(self.se) and self.se > 0):
            errs.append("se must be finite and > 0")
        if not (isinstance(self.n, (int, np.integer)) and self.n >= 1):
            errs.append("n must be an integer >= 1")
        for f in ("independence", "reliability"):
            v = getattr(self, f)
            if not (0.0 <= v <= 1.0):
                errs.append(f"{f} outside [0,1]")
        if self.n_trials < 1:
            errs.append("n_trials < 1")
        return errs

    def quality(self, cfg: BeliefConfig = DEFAULT_CFG) -> float:
        """Evidence quality in [0,1]: the fraction of its nominal precision this record is allowed to carry."""
        base = cfg.quality_of_kind(self.kind)
        burden = 1.0 + cfg.trial_penalty * math.log(max(self.n_trials, 1))
        return clip01(base * self.independence * self.reliability / burden)

    def to_dict(self) -> dict:
        d = dc.asdict(self)
        d["kind"] = str(self.kind)
        d["context"] = [list(c) for c in self.context]
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "Evidence":
        d = dict(d)
        d["context"] = tuple(tuple(c) for c in d.get("context", ()))
        return cls(**d)


def evidence_from_returns(subject: str, returns: pd.Series, now, *, kind=EvidenceKind.OOS_TEST, n_trials: int = 1,
                          independence: float = 1.0, reliability: float = 1.0, lags: int = 4, source: str = "",
                          context: Mapping | tuple = ()) -> Evidence | None:
    """Turn a dated series of realised effects (one value per independent period) into Evidence.

    Returns None when fewer than 5 finite observations exist: that is INSUFFICIENT_DATA, not a zero-effect finding.
    Fails closed if any observation is dated at/after `now`."""
    if returns is None or len(returns) == 0:
        return None
    s = pd.Series(returns).astype(float)
    idx = pd.to_datetime(s.index)
    for d in (idx.max(),):
        require_past(d, now, f"evidence[{subject}]")
    s = s[np.isfinite(s.values)]
    if len(s) < 5:
        return None
    from engine.pattern_reliability import nw_t
    t = nw_t(s.values, lags=min(lags, len(s) - 1))
    m = float(s.mean())
    if math.isfinite(t) and abs(t) > 1e-9:
        se = abs(m / t)
    else:
        se = float(s.std(ddof=1) / math.sqrt(len(s)))
    se = max(se, float(s.std(ddof=1) / math.sqrt(len(s))) * 0.5, 1e-9)   # a Newey-West se may never fall far below iid
    return Evidence(subject, as_date(idx.max()).isoformat(), m, se, int(len(s)), kind, independence, reliability,
                    n_trials, source, tuple(sorted(dict(context).items())) if isinstance(context, Mapping) else context)


# ------------------------------------------------------------------------------------------------ the pool
@dc.dataclass(frozen=True)
class PoolResult:
    mean: float
    sd: float
    tau2: float                           # between-evidence variance (heterogeneity)
    evidence_precision: float
    prior_precision: float
    n_support: int
    n_contra: int
    n_weak_opposed: int
    contradiction_mass: float
    weights: tuple[float, ...]            # per-evidence share of the evidence precision (input order)


def pool(prior_mean: float, prior_sd: float, evidence: Sequence[Evidence], as_of, cfg: BeliefConfig = DEFAULT_CFG) -> PoolResult:
    """Posterior of the effect given a prior and a set of evidence, using only records that matured before `as_of`."""
    if not prior_sd > 0:
        raise BeliefError("prior_sd must be positive")
    tau0 = 1.0 / prior_sd ** 2
    k = len(evidence)
    if k == 0:
        return PoolResult(prior_mean, prior_sd, 0.0, 0.0, tau0, 0, 0, 0, 0.0, ())
    a = as_date(as_of)
    x = np.array([e.estimate for e in evidence], float)
    se2 = np.array([max(e.se, cfg.min_se) ** 2 for e in evidence], float)
    age = np.array([(a - as_date(e.observed_at)).days for e in evidence], float)
    if (age <= 0).any():
        bad = [e.evidence_id for e, g in zip(evidence, age) if g <= 0]
        raise FirewallBreach(f"pool: evidence {bad[:3]} not strictly before as_of={a}")
    decay = np.ones(k) if cfg.half_life_days is None else 0.5 ** (age / cfg.half_life_days)
    w = np.array([e.quality(cfg) for e in evidence], float) * decay
    live = w > 1e-12
    var_i = np.where(live, se2 / np.maximum(w, 1e-12), np.inf)
    fixed = np.where(live, 1.0 / var_i, 0.0)
    tau2 = 0.0
    if cfg.use_random_effects and live.sum() >= 2:
        W = fixed[live]
        xbar = float((W * x[live]).sum() / W.sum())
        Q = float((W * (x[live] - xbar) ** 2).sum())
        C = float(W.sum() - (W ** 2).sum() / W.sum())
        if C > 0:
            tau2 = max(0.0, (Q - (live.sum() - 1)) / C)
        tau2 = min(tau2, cfg.tau2_cap_mult * float(se2.max()))
    prec = np.where(live, 1.0 / (var_i + tau2), 0.0)
    ptot = tau0 + float(prec.sum())
    mean = (tau0 * prior_mean + float((prec * x).sum())) / ptot
    # contradiction accounting against leave-one-out fixed-effect posteriors (prior included)
    P = tau0 + float(fixed.sum())
    N = tau0 * prior_mean + float((fixed * x).sum())
    p_lo = P - fixed
    m_lo = (N - fixed * x) / p_lo
    z = (x - m_lo) / np.sqrt(np.where(live, var_i, 1.0) + 1.0 / p_lo)
    opp = (np.sign(x) * np.sign(m_lo) < 0) & live
    contra = opp & (np.abs(z) > cfg.z_contra)
    weak = opp & ~contra
    supp = (np.sign(x) * np.sign(m_lo) > 0) & live
    mass = float(prec[contra].sum() / prec.sum()) if prec.sum() > 0 else 0.0
    ws = tuple(float(v) for v in (prec / prec.sum())) if prec.sum() > 0 else tuple(0.0 for _ in range(k))
    return PoolResult(float(mean), float(1.0 / math.sqrt(ptot)), float(tau2), float(prec.sum()), tau0,
                      int(supp.sum()), int(contra.sum()), int(weak.sum()), mass, ws)


# ------------------------------------------------------------------------------------------------ states
@dc.dataclass(frozen=True)
class BeliefState:
    """One immutable version of a belief.  A new version is appended whenever evidence arrives; old versions stay."""
    subject: str
    version: int
    as_of: str
    prior_mean: float
    prior_sd: float
    mean: float
    sd: float
    n_obs: int                            # total independent observations behind the evidence
    n_evidence: int
    n_support: int
    n_contra: int
    n_weak_opposed: int
    contradiction_mass: float
    tau2: float
    evidence_ids: tuple[str, ...]
    last_evidence_at: str
    config_hash: str
    parent: str = ""

    @property
    def state_hash(self) -> str:
        return stable_hash(self)

    def interval(self, level: float = 0.90) -> tuple[float, float]:
        z = float(sps.norm.ppf(0.5 + level / 2))
        return self.mean - z * self.sd, self.mean + z * self.sd

    def prob_positive(self) -> float:
        return float(sps.norm.cdf(self.mean / self.sd)) if self.sd > 0 else float(self.mean > 0)

    def prob_above(self, threshold: float) -> float:
        return float(sps.norm.sf((threshold - self.mean) / self.sd)) if self.sd > 0 else float(self.mean > threshold)

    def prob_sign_right(self) -> float:
        p = self.prob_positive()
        return max(p, 1 - p)

    def shrinkage(self) -> float:
        """Share of the posterior precision still owed to the prior (1 = no learning yet)."""
        return float((self.sd ** 2) / (self.prior_sd ** 2))

    def to_dict(self) -> dict:
        d = dc.asdict(self)
        d["evidence_ids"] = list(self.evidence_ids)
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "BeliefState":
        d = dict(d)
        d["evidence_ids"] = tuple(d["evidence_ids"])
        return cls(**d)


def _make_state(subject, version, as_of, prior_mean, prior_sd, evidence: Sequence[Evidence], cfg, parent) -> BeliefState:
    r = pool(prior_mean, prior_sd, evidence, as_of, cfg)
    last = max((e.observed_at for e in evidence), default="")
    return BeliefState(subject, version, as_date(as_of).isoformat(), float(prior_mean), float(prior_sd), r.mean, r.sd,
                       int(sum(e.n for e in evidence)), len(evidence), r.n_support, r.n_contra, r.n_weak_opposed,
                       r.contradiction_mass, r.tau2, tuple(sorted(e.evidence_id for e in evidence)), last,
                       cfg.config_hash(), parent)


def belief_status(state: BeliefState, cfg: BeliefConfig = DEFAULT_CFG) -> tuple[Epistemic, Unknown | None]:
    """Map a posterior to the contract's epistemic vocabulary WITHOUT collapsing it to a score.  The Unknown returned
    is the honest reason when the answer is not a confident yes/no."""
    if state.n_evidence == 0:
        return Epistemic.HYPOTHESIS, Unknown.UNTESTED
    if state.n_obs < 10:
        return Epistemic.HYPOTHESIS, Unknown.INSUFFICIENT_DATA
    strength = state.prob_sign_right()
    if state.contradiction_mass >= cfg.contradicted_mass:
        if strength >= cfg.supported_prob:
            return Epistemic.CONDITIONAL, Unknown.CONFLICTED       # strong net sign but the records disagree: a context is hiding
        return Epistemic.CONTRADICTED, Unknown.CONFLICTED
    if strength >= cfg.supported_prob:
        return Epistemic.SUPPORTED, None
    return Epistemic.HYPOTHESIS, Unknown.INSUFFICIENT_DATA


def explain_update(before: BeliefState | None, after: BeliefState, evidence: Iterable[Evidence] = (),
                   cfg: BeliefConfig = DEFAULT_CFG) -> dict:
    """Why did the belief move?  Shift in prior-sd units, precision share of new evidence vs prior, and whether the
    move was contradictory.  Diagnostic only - never used to make the update."""
    evs = list(evidence)
    m0 = before.mean if before else after.prior_mean
    s0 = before.sd if before else after.prior_sd
    new_prec = sum(e.quality(cfg) / max(e.se, cfg.min_se) ** 2 for e in evs)
    old_prec = 1.0 / s0 ** 2
    return {"subject": after.subject, "version": after.version, "mean_before": m0, "mean_after": after.mean,
            "shift": after.mean - m0, "shift_in_sd": (after.mean - m0) / s0, "sd_before": s0, "sd_after": after.sd,
            "sd_ratio": after.sd / s0, "new_evidence_precision_share": new_prec / (new_prec + old_prec) if new_prec + old_prec else 0.0,
            "qualities": [round(e.quality(cfg), 4) for e in evs],
            "contradictions_now": after.n_contra, "contradiction_mass": after.contradiction_mass}


# ------------------------------------------------------------------------------------------------ ledger
class BeliefLedger:
    """Append-only store of priors, evidence and belief states.

    * Evidence ids are content hashes; re-adding the same record is a no-op, adding a *different* record under a used id
      cannot happen (the id is the content).
    * update() moves time forward only: `now` may not precede the newest state (a back-dated update would rewrite what was
      believed at a past decision).
    * view(subject, as_of) recomputes what would have been believed at any past date from evidence matured before it -
      the audit tool for 'could this belief have existed then?'."""

    def __init__(self, cfg: BeliefConfig = DEFAULT_CFG):
        errs = cfg.validate()
        if errs:
            raise BeliefError("; ".join(errs))
        self.cfg = cfg
        self._priors: dict[str, tuple[float, float]] = {}
        self._evidence: dict[str, dict[str, Evidence]] = {}
        self._states: dict[str, list[BeliefState]] = {}
        self._log: dict[str, list[UpdateRecord]] = {}        # one record per admitted evidence: the audit of every update
        self._retracted: dict[str, dict[str, tuple[str, str]]] = {}   # subject -> {evidence_id: (reason, when)}
        self.rejected: list[tuple[str, str]] = []            # (evidence_id or '?', reason): kept, never silently dropped
        self.duplicates: int = 0

    # ---- registration
    def register(self, subject: str, prior_mean: float = 0.0, prior_sd: float | None = None) -> None:
        if not subject:
            raise BeliefError("empty subject")
        sd = self.cfg.prior_sd if prior_sd is None else float(prior_sd)
        if not (sd > 0 and math.isfinite(sd) and math.isfinite(prior_mean)):
            raise BeliefError(f"bad prior for {subject}: mean={prior_mean} sd={sd}")
        if subject in self._priors:
            if self._priors[subject] != (float(prior_mean), sd):
                raise BeliefError(f"prior for {subject} is immutable once registered")
            return
        self._priors[subject] = (float(prior_mean), sd)
        self._evidence[subject] = {}
        self._states[subject] = []
        self._log[subject] = []
        self._retracted[subject] = {}

    def subjects(self) -> list[str]:
        return sorted(self._priors)

    def prior(self, subject: str) -> tuple[float, float]:
        return self._priors[subject]

    # ---- updating
    def update(self, subject: str, evidence: Evidence | Iterable[Evidence], now) -> BeliefState:
        evs = [evidence] if isinstance(evidence, Evidence) else list(evidence)
        if subject not in self._priors:
            self.register(subject)
        # 1. fail closed BEFORE any mutation: a batch with one future record changes nothing
        for e in evs:
            if e.subject != subject:
                raise BeliefError(f"evidence for {e.subject!r} sent to {subject!r}")
            require_past(e.observed_at, now, f"evidence[{subject}]")
        hist = self._states[subject]
        if hist and as_date(now) < as_date(hist[-1].as_of):
            raise BeliefError(f"update at {as_date(now)} precedes newest state {hist[-1].as_of}: history is append-only")
        # 2. admit
        fresh = 0
        admitted: list[Evidence] = []
        prior_ev = list(self._evidence[subject].values())
        for e in evs:
            bad = e.validate()
            if bad:
                self.rejected.append((e.evidence_id, "; ".join(bad)))
                continue
            if e.evidence_id in self._evidence[subject]:
                self.duplicates += 1
                continue
            self._evidence[subject][e.evidence_id] = e
            admitted.append(e)
            fresh += 1
        if fresh == 0 and hist:
            return hist[-1]
        pm, ps = self._priors[subject]
        version = len(hist) + 1
        seen = prior_ev
        for e in sorted(admitted, key=lambda e: (e.observed_at, e.evidence_id)):
            self._log[subject].append(update_record(subject, version, pm, ps, seen, e, now, self.cfg))
            seen = seen + [e]
        allev = self._evidence[subject].values()
        gone = self._retracted[subject]
        used = [e for e in allev if as_date(e.observed_at) < as_date(now) and e.evidence_id not in gone]
        st = _make_state(subject, len(hist) + 1, now, pm, ps, used, self.cfg, hist[-1].state_hash if hist else "")
        hist.append(st)
        return st

    # ---- reading
    def history(self, subject: str) -> tuple[BeliefState, ...]:
        return tuple(self._states.get(subject, ()))

    def current(self, subject: str, as_of=None) -> BeliefState | None:
        h = self._states.get(subject, [])
        if as_of is None:
            return h[-1] if h else None
        a = as_date(as_of)
        ok = [s for s in h if as_date(s.as_of) <= a]
        return ok[-1] if ok else None

    def retract(self, subject: str, evidence_id: str, reason: str, now) -> BeliefState:
        """Withdraw one evidence record from the belief (e.g. it was later found contaminated or mis-dated).  Nothing is deleted:
        the record stays in the store, the retraction is logged with its reason and date, every earlier state keeps the evidence it
        was built on (that is what was believed then), and a NEW state is appended without it."""
        if subject not in self._priors or evidence_id not in self._evidence[subject]:
            raise BeliefError(f"unknown evidence {evidence_id!r} for {subject!r}")
        if evidence_id in self._retracted[subject]:
            raise BeliefError(f"evidence {evidence_id!r} already retracted")
        if not reason:
            raise BeliefError("a retraction needs a reason")
        hist = self._states[subject]
        if hist and as_date(now) < as_date(hist[-1].as_of):
            raise BeliefError("retraction cannot be back-dated")
        self._retracted[subject][evidence_id] = (reason, as_date(now).isoformat())
        pm, ps = self._priors[subject]
        used = [e for e in self._evidence[subject].values()
                if as_date(e.observed_at) < as_date(now) and e.evidence_id not in self._retracted[subject]]
        st = _make_state(subject, len(hist) + 1, now, pm, ps, used, self.cfg, hist[-1].state_hash if hist else "")
        hist.append(st)
        return st

    def retractions(self, subject: str) -> dict[str, tuple[str, str]]:
        return dict(self._retracted.get(subject, {}))

    def evidence_for(self, subject: str, as_of=None, include_retracted: bool = False) -> list[Evidence]:
        """Evidence records (oldest first).  Retracted records are excluded unless asked for: a view built today never trusts
        what has since been withdrawn, even when asked about a past date."""
        gone = set() if include_retracted else set(self._retracted.get(subject, {}))
        ev = [e for e in self._evidence.get(subject, {}).values() if e.evidence_id not in gone]
        if as_of is not None:
            ev = [e for e in ev if as_date(e.observed_at) < as_date(as_of)]
        return sorted(ev, key=lambda e: (e.observed_at, e.evidence_id))

    def view(self, subject: str, as_of) -> BeliefState:
        """What would be believed at `as_of` (recomputed; adds no version).  Staleness shows up as a wider sd."""
        if subject not in self._priors:
            raise BeliefError(f"unknown subject {subject!r}")
        pm, ps = self._priors[subject]
        cur = self.current(subject)
        return _make_state(subject, (cur.version if cur else 0), as_of, pm, ps, self.evidence_for(subject, as_of), self.cfg,
                           cur.state_hash if cur else "")

    def drift(self, subject: str) -> list[dict]:
        """Version-to-version movement of the belief (mean, sd, contradiction) - the belief's own timeline."""
        out, prev = [], None
        for s in self._states.get(subject, []):
            out.append({"version": s.version, "as_of": s.as_of, "mean": s.mean, "sd": s.sd, "n_obs": s.n_obs,
                        "d_mean": (s.mean - prev.mean) if prev else s.mean - s.prior_mean,
                        "contra": s.n_contra, "mass": s.contradiction_mass})
            prev = s
        return out

    def updates(self, subject: str) -> tuple["UpdateRecord", ...]:
        return tuple(self._log.get(subject, ()))

    def verify_integrity(self) -> list[str]:
        """Recompute every state from raw evidence and check hash chain and versions.  [] = intact."""
        errs = []
        for subj, hist in self._states.items():
            pm, ps = self._priors[subj]
            parent = ""
            for i, s in enumerate(hist, 1):
                if s.version != i:
                    errs.append(f"{subj}: version {s.version} at position {i}")
                if s.parent != parent:
                    errs.append(f"{subj} v{s.version}: parent hash mismatch")
                ev = [self._evidence[subj][h] for h in s.evidence_ids if h in self._evidence[subj]]
                if len(ev) != len(s.evidence_ids):
                    errs.append(f"{subj} v{s.version}: evidence missing from store")
                    parent = s.state_hash
                    continue
                re = _make_state(subj, s.version, s.as_of, pm, ps, ev, self.cfg, parent)
                if abs(re.mean - s.mean) > 1e-9 or abs(re.sd - s.sd) > 1e-9:
                    errs.append(f"{subj} v{s.version}: state does not replay from evidence")
                if any(as_date(e.observed_at) >= as_date(s.as_of) for e in ev):
                    errs.append(f"{subj} v{s.version}: contains evidence not before its as_of")
                parent = s.state_hash
        return errs

    # ---- reporting
    def report(self, as_of=None) -> list[dict]:
        rows = []
        for subj in self.subjects():
            st = self.view(subj, as_of) if as_of is not None else (self.current(subj) or self.view(subj, "2100-01-01"))
            ep, why = belief_status(st, self.cfg)
            lo, hi = st.interval(self.cfg.credible_level)
            rows.append({"subject": subj, "mean": st.mean, "sd": st.sd, "lo": lo, "hi": hi, "p_pos": st.prob_positive(),
                         "n_obs": st.n_obs, "n_evidence": st.n_evidence, "contra": st.n_contra,
                         "contra_mass": st.contradiction_mass, "epistemic": str(ep), "unknown": str(why) if why else "",
                         "versions": len(self._states[subj])})
        return rows

    def format_report(self, as_of=None) -> str:
        rows = self.report(as_of)
        if not rows:
            return "(no beliefs)"
        head = f"{'subject':<28}{'mean':>9}{'sd':>8}{'P(+)':>7}{'n':>7}{'ev':>4}{'contra':>7}  status"
        lines = [head]
        for r in rows:
            lines.append(f"{r['subject'][:27]:<28}{r['mean']:>9.4f}{r['sd']:>8.4f}{r['p_pos']:>7.2f}{r['n_obs']:>7d}"
                         f"{r['n_evidence']:>4d}{r['contra']:>7d}  {r['epistemic']}" + (f" ({r['unknown']})" if r['unknown'] else ""))
        return "\n".join(lines)

    # ---- persistence (append-only event records)
    def to_records(self) -> list[dict]:
        out = [{"type": "config", "config": dc.asdict(self.cfg)}]
        for subj in self.subjects():
            pm, ps = self._priors[subj]
            out.append({"type": "prior", "subject": subj, "mean": pm, "sd": ps})
            out += [{"type": "evidence", **e.to_dict()} for e in self.evidence_for(subj, include_retracted=True)]
            out += [{"type": "retraction", "subject": subj, "evidence_id": k, "reason": r, "when": w}
                    for k, (r, w) in sorted(self._retracted[subj].items())]
            out += [{"type": "state", **s.to_dict()} for s in self._states[subj]]
            out += [{"type": "update", **dc.asdict(u)} for u in self._log[subj]]
        return out

    @classmethod
    def from_records(cls, recs: Iterable[Mapping]) -> "BeliefLedger":
        recs = list(recs)
        cfgd = next((r["config"] for r in recs if r.get("type") == "config"), None)
        if cfgd is not None:
            cfgd = dict(cfgd)
            cfgd["kind_quality"] = tuple(tuple(kv) for kv in cfgd["kind_quality"])
        led = cls(BeliefConfig(**cfgd) if cfgd else DEFAULT_CFG)
        for r in recs:
            t = r.get("type")
            if t == "prior":
                led.register(r["subject"], r["mean"], r["sd"])
            elif t == "evidence":
                e = Evidence.from_dict({k: v for k, v in r.items() if k != "type"})
                led._evidence[e.subject][e.evidence_id] = e
            elif t == "state":
                s = BeliefState.from_dict({k: v for k, v in r.items() if k != "type"})
                led._states[s.subject].append(s)
            elif t == "retraction":
                led._retracted[r["subject"]][r["evidence_id"]] = (r["reason"], r["when"])
            elif t == "update":
                u = UpdateRecord(**{k: v for k, v in r.items() if k != "type"})
                led._log[u.subject].append(u)
        return led

    def save_jsonl(self, path) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in self.to_records():
                f.write(canonical_json(r) + "\n")

    @classmethod
    def load_jsonl(cls, path) -> "BeliefLedger":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_records(json.loads(line) for line in f if line.strip())


# ------------------------------------------------------------------------------------------------ binary beliefs
@dc.dataclass(frozen=True)
class BetaBelief:
    """Belief about a hit rate: Beta(alpha, beta) with quality-tempered fractional counts.  Immutable; updated() returns a
    new object so a hit-rate history can be replayed."""
    alpha: float = 1.0
    beta: float = 1.0
    n_updates: int = 0

    def updated(self, successes: float, trials: float, quality: float = 1.0, forget: float = 1.0) -> "BetaBelief":
        if trials < 0 or successes < 0 or successes > trials + 1e-9:
            raise BeliefError(f"impossible counts successes={successes} trials={trials}")
        q = clip01(quality)
        f = float(forget)
        if not 0 < f <= 1:
            raise BeliefError("forget factor must be in (0,1]")
        # forgetting shrinks the *evidence* part back toward the (1,1) prior; it never goes below the prior
        a = 1.0 + (self.alpha - 1.0) * f + q * successes
        b = 1.0 + (self.beta - 1.0) * f + q * (trials - successes)
        return BetaBelief(a, b, self.n_updates + 1)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        return self.alpha + self.beta - 2.0                   # effective sample size

    def interval(self, level: float = 0.90) -> tuple[float, float]:
        lo, hi = sps.beta.interval(level, self.alpha, self.beta)
        return float(lo), float(hi)

    def prob_above(self, p0: float) -> float:
        return float(sps.beta.sf(p0, self.alpha, self.beta))


# ------------------------------------------------------------------------------------------------ pooling across contexts
def partial_pool(estimates: Sequence[float], ses: Sequence[float], min_tau2: float = 0.0) -> dict:
    """Empirical-Bayes shrinkage of many group estimates toward their common mean (contract section 8: no context
    explosion).  Groups with large se are pulled hard toward the pooled mean; precise groups keep their own value.
    Delegates to engine.pattern_stats.eb_shrink (DerSimonian-Laird tau2); this wrapper only adds a tau2 floor and names.

    Returns pooled mean `mu`, between-group variance `tau2`, `shrunk` means and `weight_own` (share each group keeps)."""
    from engine.pattern_stats import eb_shrink
    x = np.asarray(estimates, float)
    s = np.asarray(ses, float)
    ok = np.isfinite(x) & np.isfinite(s) & (s > 0)
    if ok.sum() == 0:
        return {"mu": float("nan"), "tau2": float("nan"), "shrunk": np.full(len(x), np.nan), "weight_own": np.full(len(x), np.nan)}
    if ok.sum() == 1:
        return {"mu": float(x[ok][0]), "tau2": float(min_tau2), "shrunk": x.copy(), "weight_own": np.where(ok, 1.0, np.nan)}
    r = eb_shrink(x, s, center=None)
    tau2 = max(float(r["tau2"]), min_tau2)
    own = np.full(len(x), np.nan)
    own[ok] = tau2 / (tau2 + s[ok] ** 2)
    shrunk = np.full(len(x), np.nan)
    shrunk[ok] = r["mu"] + own[ok] * (x[ok] - r["mu"])
    return {"mu": float(r["mu"]), "tau2": tau2, "shrunk": shrunk, "weight_own": own}


# ------------------------------------------------------------------------------------------------ self-diagnostics
def posterior_coverage(n_sims: int = 2000, prior_sd: float = 0.05, se: float = 0.02, n_evidence: int = 3,
                       seed: int = 0, level: float = 0.90, cfg: BeliefConfig | None = None) -> dict:
    """Calibration check of the update itself: draw truth from the prior, generate evidence, and count how often the
    credible interval covers truth.  With well-behaved (homogeneous) evidence coverage must be close to `level`; a
    result far from it means the update over- or under-states its certainty."""
    cfg = cfg or BeliefConfig(prior_sd=prior_sd, half_life_days=None, use_random_effects=False)
    rng = np.random.default_rng(seed)
    hit = 0
    widths = []
    for i in range(n_sims):
        theta = rng.normal(0.0, prior_sd)
        ev = [Evidence("s", f"2020-01-{j + 1:02d}", rng.normal(theta, se), se, 20) for j in range(n_evidence)]
        r = pool(0.0, prior_sd, ev, "2021-01-01", cfg)
        z = float(sps.norm.ppf(0.5 + level / 2))
        lo, hi = r.mean - z * r.sd, r.mean + z * r.sd
        hit += int(lo <= theta <= hi)
        widths.append(hi - lo)
    return {"coverage": hit / n_sims, "target": level, "mean_width": float(np.mean(widths)), "n_sims": n_sims}


def proportionality_check(cfg: BeliefConfig | None = None) -> dict:
    """The contract's key requirement: evidence changes belief in proportion to its quality.  Feeds the same estimate at
    quality 1.0 and 0.5 and at zero (a copy) and returns the three shifts; they must be ordered and the zero must be 0."""
    cfg = cfg or BeliefConfig(half_life_days=None)
    out = {}
    for name, ind in (("full", 1.0), ("half", 0.5), ("copy", 0.0)):
        led = BeliefLedger(cfg)
        led.register("s", 0.0, cfg.prior_sd)
        st = led.update("s", Evidence("s", "2020-01-01", 0.04, 0.02, 30, independence=ind), "2020-02-01")
        out[name] = st.mean
    return out


# ------------------------------------------------------------------------------------------------ update audit records
@dc.dataclass(frozen=True)
class UpdateRecord:
    """Everything the contract asks a belief update to keep: prior, evidence, evidence strength, posterior, uncertainty,
    sample size and whether the record contradicted what was believed.  One per admitted evidence, never edited."""
    subject: str
    version: int
    evidence_id: str
    observed_at: str
    prior_mean: float                     # belief BEFORE this evidence (all earlier evidence pooled)
    prior_sd: float
    estimate: float
    se: float
    quality: float
    n: int
    predictive_z: float                   # how surprising the estimate was under the prior belief
    log_pred_density: float
    log_bayes_factor: float               # ln BF(effect ~ prior belief  vs  effect exactly 0): strength of evidence for a real effect
    post_mean: float
    post_sd: float
    shift_in_sd: float
    contradiction: bool


def update_record(subject, version, pm, ps, earlier: Sequence[Evidence], e: Evidence, now, cfg: BeliefConfig) -> UpdateRecord:
    before = pool(pm, ps, list(earlier), now, cfg)
    after = pool(pm, ps, list(earlier) + [e], now, cfg)
    q = max(e.quality(cfg), 1e-12)
    var_e = max(e.se, cfg.min_se) ** 2 / q
    v1 = before.sd ** 2 + var_e
    z = (e.estimate - before.mean) / math.sqrt(v1)
    lp1 = float(sps.norm.logpdf(e.estimate, before.mean, math.sqrt(v1)))
    lp0 = float(sps.norm.logpdf(e.estimate, 0.0, math.sqrt(var_e)))
    contra = bool(earlier) and (e.estimate * before.mean < 0) and abs(z) > cfg.z_contra
    return UpdateRecord(subject, version, e.evidence_id, e.observed_at, before.mean, before.sd, e.estimate, e.se, q, e.n, z,
                        lp1, lp1 - lp0, after.mean, after.sd, (after.mean - before.mean) / before.sd, contra)


def evidence_strength(state: BeliefState, e: Evidence, cfg: BeliefConfig = DEFAULT_CFG) -> float:
    """ln Bayes factor of the belief in `state` against 'no effect' for one new record (positive = the record supports a
    real effect).  Does not modify anything: a what-if for research prioritisation."""
    q = max(e.quality(cfg), 1e-12)
    var_e = max(e.se, cfg.min_se) ** 2 / q
    return float(sps.norm.logpdf(e.estimate, state.mean, math.sqrt(state.sd ** 2 + var_e))
                 - sps.norm.logpdf(e.estimate, 0.0, math.sqrt(var_e)))


def evidence_influence(ledger: BeliefLedger, subject: str, as_of) -> list[dict]:
    """Leave-one-out influence of each evidence record on the current belief (how much the mean moves if it is removed).
    A belief resting on one dominant record is fragile; this shows it."""
    pm, ps = ledger.prior(subject)
    ev = ledger.evidence_for(subject, as_of)
    full = pool(pm, ps, ev, as_of, ledger.cfg)
    rows = []
    for i, e in enumerate(ev):
        rest = pool(pm, ps, ev[:i] + ev[i + 1:], as_of, ledger.cfg)
        rows.append({"evidence_id": e.evidence_id, "observed_at": e.observed_at, "kind": str(e.kind),
                     "d_mean": full.mean - rest.mean, "d_sd": full.sd - rest.sd, "weight": full.weights[i],
                     "quality": e.quality(ledger.cfg)})
    return sorted(rows, key=lambda r: -abs(r["d_mean"]))


# ------------------------------------------------------------------------------------------------ staleness
def staleness(ledger: BeliefLedger, subject: str, as_of) -> dict:
    """How much of the belief has decayed unrefreshed.  `sd_growth` > 1 means the belief is less certain now than when
    the newest evidence arrived; `days_since` is the age of that newest evidence."""
    cur = ledger.current(subject)
    if cur is None or cur.n_evidence == 0:
        return {"days_since": None, "sd_growth": 1.0, "stale": False, "reason": "no evidence"}
    now_view = ledger.view(subject, as_of)
    days = (as_date(as_of) - as_date(cur.last_evidence_at)).days
    growth = now_view.sd / cur.sd if cur.sd > 0 else float("inf")
    return {"days_since": days, "sd_growth": growth, "mean_shift": now_view.mean - cur.mean,
            "stale": growth > 1.25, "reason": "evidence decayed" if growth > 1.25 else "fresh enough"}


# ------------------------------------------------------------------------------------------------ sequential evidence
class EValueMonitor:
    """Anytime-valid evidence that the effect is non-zero: a mixture likelihood-ratio process.  E-values can be watched
    after every record without inflating false positives (unlike repeatedly re-testing a p-value).
    H1 mixture: effect ~ N(0, tau^2).  Crossing 1/alpha rejects 'no effect'.  Losing evidence drives it back down."""

    def __init__(self, tau: float, alpha: float = 0.05):
        if not tau > 0 or not 0 < alpha < 1:
            raise BeliefError("EValueMonitor needs tau > 0 and 0 < alpha < 1")
        self.tau2 = tau ** 2
        self.alpha = alpha
        self.log_e = 0.0
        self.n = 0
        self._sum_prec = 0.0
        self._sum_prec_x = 0.0
        self.trace: list[float] = []

    def add(self, estimate: float, se: float, quality: float = 1.0) -> float:
        """Add one record; returns the current e-value (mixture LR on the running pooled estimate)."""
        if not (se > 0 and math.isfinite(estimate)):
            raise BeliefError("bad evidence for e-value")
        prec = max(quality, 1e-12) / se ** 2
        self._sum_prec += prec
        self._sum_prec_x += prec * estimate
        v = 1.0 / self._sum_prec                       # variance of pooled estimate
        xbar = self._sum_prec_x * v
        self.log_e = float(sps.norm.logpdf(xbar, 0, math.sqrt(v + self.tau2)) - sps.norm.logpdf(xbar, 0, math.sqrt(v)))
        self.n += 1
        self.trace.append(self.log_e)
        return math.exp(min(self.log_e, 700))

    @property
    def rejects(self) -> bool:
        return self.log_e >= math.log(1.0 / self.alpha)


# ------------------------------------------------------------------------------------------------ hierarchical family
class HierarchicalBeliefs:
    """Beliefs for one relation in many contexts (regimes, sectors, size buckets) that pool toward a shared mean.
    Prevents the context explosion of section 8: a context with 6 observations borrows strength from its siblings and only
    departs from the family as its own evidence earns it.  Every context keeps its own BeliefLedger subject."""

    def __init__(self, family: str, cfg: BeliefConfig = DEFAULT_CFG):
        self.family = family
        self.ledger = BeliefLedger(cfg)
        self.cfg = cfg
        self._contexts: list[str] = []

    def _subject(self, context: str) -> str:
        return f"{self.family}|{context}"

    def add(self, context: str, evidence: Evidence, now) -> BeliefState:
        subj = self._subject(context)
        if context not in self._contexts:
            self._contexts.append(context)
        if evidence.subject != subj:
            evidence = dc.replace(evidence, subject=subj)
        return self.ledger.update(subj, evidence, now)

    def contexts(self) -> list[str]:
        return sorted(self._contexts)

    def family_estimate(self, as_of) -> dict:
        views = [self.ledger.view(self._subject(c), as_of) for c in self.contexts()]
        seen = [v for v in views if v.n_evidence > 0]
        if not seen:
            return {"mu": 0.0, "tau2": 0.0, "n_contexts": 0}
        r = partial_pool([v.mean for v in seen], [v.sd for v in seen])
        return {"mu": r["mu"], "tau2": r["tau2"], "n_contexts": len(seen)}

    def pooled(self, context: str, as_of) -> dict:
        """The context's belief after shrinkage toward the family: own mean vs pooled mean and how much weight it kept."""
        own = self.ledger.view(self._subject(context), as_of)
        fam = self.family_estimate(as_of)
        if own.n_evidence == 0 or fam["n_contexts"] < 2:
            return {"context": context, "mean": own.mean, "sd": own.sd, "weight_own": 1.0 if own.n_evidence else 0.0,
                    "own_mean": own.mean, "family_mean": fam["mu"]}
        w = fam["tau2"] / (fam["tau2"] + own.sd ** 2) if fam["tau2"] + own.sd ** 2 > 0 else 1.0
        mean = fam["mu"] + w * (own.mean - fam["mu"])
        sd = math.sqrt(max(w * own.sd ** 2, 1e-18))
        return {"context": context, "mean": mean, "sd": sd, "weight_own": w, "own_mean": own.mean, "family_mean": fam["mu"]}

    def departures(self, as_of, z: float = 2.0) -> list[dict]:
        """Contexts whose own belief differs from the family by more than z standard errors: candidate real conditions."""
        fam = self.family_estimate(as_of)
        out = []
        for c in self.contexts():
            o = self.ledger.view(self._subject(c), as_of)
            if o.n_evidence == 0:
                continue
            d = o.mean - fam["mu"]
            zz = d / math.sqrt(o.sd ** 2 + fam["tau2"] / max(fam["n_contexts"], 1))
            if abs(zz) >= z:
                out.append({"context": c, "diff": d, "z": zz, "mean": o.mean})
        return sorted(out, key=lambda r: -abs(r["z"]))


# ------------------------------------------------------------------------------------------------ merging workers
def merge_ledgers(a: BeliefLedger, b: BeliefLedger, now) -> BeliefLedger:
    """Union of two ledgers' evidence (e.g. two workers), re-pooled at `now`.  Identical evidence collapses by content id;
    priors must agree.  Histories of the inputs are not copied (a merged ledger starts a new chain) but nothing is lost:
    the evidence is."""
    if a.cfg.config_hash() != b.cfg.config_hash():
        raise BeliefError("cannot merge ledgers with different BeliefConfig")
    out = BeliefLedger(a.cfg)
    for led in (a, b):
        for subj in led.subjects():
            pm, ps = led.prior(subj)
            out.register(subj, pm, ps)
    for led in (a, b):
        for subj in led.subjects():
            ev = led.evidence_for(subj)                      # retracted records are not carried into a merge
            if ev:
                out.update(subj, ev, now)
    return out


# ------------------------------------------------------------------------------------------------ predictive calibration
class PredictiveCalibration:
    """Are the beliefs honest about their own uncertainty?  Before each new evidence record arrives the belief predicts it,
    N(mean, sd^2 + se^2/quality).  The probability-integral transforms of the realised records must be ~Uniform(0,1);
    if beliefs are overconfident the PITs pile up near 0 and 1.  Built from a ledger's UpdateRecords."""

    def __init__(self, records: Iterable[UpdateRecord]):
        self.records = [r for r in records]

    def pits(self) -> np.ndarray:
        out = []
        for r in self.records:
            var_e = r.se ** 2 / max(r.quality, 1e-12)
            sd = math.sqrt(r.prior_sd ** 2 + var_e)
            out.append(float(sps.norm.cdf((r.estimate - r.prior_mean) / sd)))
        return np.asarray(out)

    def summary(self) -> dict:
        p = self.pits()
        if len(p) < 8:
            return {"n": int(len(p)), "verdict": str(Unknown.INSUFFICIENT_DATA)}
        ks = sps.kstest(p, "uniform")
        tail = float(np.mean((p < 0.05) | (p > 0.95)))
        verdict = "OVERCONFIDENT" if (tail > 0.25 and ks.pvalue < 0.05) else "CALIBRATED" if ks.pvalue >= 0.05 else "MISCALIBRATED"
        return {"n": int(len(p)), "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "tail_share": tail,
                "mean_z": float(np.mean(sps.norm.ppf(np.clip(p, 1e-6, 1 - 1e-6)))), "verdict": verdict}


# ------------------------------------------------------------------------------------------------ adapters and decision use
def evidence_from_pattern_row(row: Mapping, now, *, n_candidates: int = 1, n_disc: int | None = None,
                              n_conf: int | None = None, source: str = "PatternMiner") -> list[Evidence]:
    """Two Evidence records from one engine.patterns.PatternMiner row: the DISCOVERY evidence (in-sample, quality-cut and
    charged for the `n_candidates` searched) and the CONFIRMATION evidence (held-out, full quality).  Rows without a usable
    t-statistic yield nothing for that stage - a missing stage is untested, not zero.

    Row keys read: key_named (subject), effect, t_disc, t_conf, and optionally end_disc / end_conf (ISO dates the stage's data
    ended; otherwise `now` minus one day is used, which is conservative only in the sense of never being in the future)."""
    subj = str(row.get("key_named", row.get("subject", "")))
    eff = float(row.get("effect", float("nan")))
    out: list[Evidence] = []
    default_end = (as_date(now) - dt.timedelta(days=1)).isoformat()
    for stage, tkey, ekey, kind, nn, trials in (("disc", "t_disc", "end_disc", EvidenceKind.IN_SAMPLE_FIT, n_disc, n_candidates),
                                                  ("conf", "t_conf", "end_conf", EvidenceKind.OOS_TEST, n_conf, 1)):
        t = row.get(tkey)
        if t is None or not math.isfinite(float(t)) or abs(float(t)) < 1e-9 or not math.isfinite(eff):
            continue
        end = str(row.get(ekey, default_end))
        require_past(end, now, f"pattern row {subj} {stage}")
        se = abs(eff / float(t))
        out.append(Evidence(subj, end, eff, se, int(nn) if nn else max(int(round((eff / se) ** 2)), 1), kind, n_trials=trials,
                            source=f"{source}:{stage}"))
    return out


def conservative_effect(state: BeliefState, direction: int = 1, level: float = 0.80) -> float:
    """The effect a decision should *plan on*: the credible bound on the side of zero (lower bound for a positive claim).
    Zero when the interval reaches across zero.  Using the posterior mean instead is what turns noise into position size."""
    lo, hi = state.interval(level)
    if direction >= 0:
        return max(lo, 0.0)
    return min(hi, 0.0)


def influence_weight(state: BeliefState, cfg: BeliefConfig = DEFAULT_CFG) -> float:
    """0..1 cap on how much a decision may lean on this belief: P(sign right) above one half, scaled down by contradictory
    precision share and by the share of the posterior still owed to the prior.  A single number for one *decision use*; it is
    not a substitute for the separate real/useful/now answers (engine.learning.questions)."""
    if state.n_evidence == 0:
        return 0.0
    edge = max(0.0, 2.0 * state.prob_sign_right() - 1.0)
    return float(clip01(edge * (1.0 - state.contradiction_mass) * (1.0 - min(state.shrinkage(), 1.0))))


def contradiction_report(ledger: "BeliefLedger", subject: str) -> list[dict]:
    """The records that contradicted what was believed when they arrived, newest first, with the context they carried."""
    ev = {e.evidence_id: e for e in ledger.evidence_for(subject)}
    rows = []
    for u in ledger.updates(subject):
        if u.contradiction:
            e = ev.get(u.evidence_id)
            rows.append({"observed_at": u.observed_at, "estimate": u.estimate, "believed": u.prior_mean, "z": u.predictive_z,
                         "shift_in_sd": u.shift_in_sd, "kind": str(e.kind) if e else "", "context": dict(e.context) if e else {},
                         "source": e.source if e else ""})
    return sorted(rows, key=lambda r: r["observed_at"], reverse=True)


def explain_heterogeneity(ledger: "BeliefLedger", subject: str, as_of=None, min_group: int = 2) -> list[dict]:
    """Does a context tag explain why the evidence disagrees?  For every context key carried by the evidence, split the records
    by that key's value, pool each group, and test between-group differences (Cochran Q vs chi-square).  A key that explains the
    disagreement is a candidate condition for the relation (hand it to boundary learning); heterogeneity with no explaining key
    stays honestly unexplained."""
    ev = ledger.evidence_for(subject, as_of)
    keys = sorted({k for e in ev for k, _ in e.context})
    out = []
    for key in keys:
        groups: dict[str, list[Evidence]] = {}
        for e in ev:
            v = dict(e.context).get(key)
            if v is not None:
                groups.setdefault(v, []).append(e)
        groups = {v: g for v, g in groups.items() if len(g) >= min_group}
        if len(groups) < 2:
            continue
        means, ses = [], []
        for v, g in sorted(groups.items()):
            w = np.array([e.quality(ledger.cfg) / max(e.se, ledger.cfg.min_se) ** 2 for e in g])
            x = np.array([e.estimate for e in g])
            means.append(float((w * x).sum() / w.sum()))
            ses.append(float(1.0 / math.sqrt(w.sum())))
        m, s = np.array(means), np.array(ses)
        wgt = 1.0 / s ** 2
        mu = float((wgt * m).sum() / wgt.sum())
        Q = float((wgt * (m - mu) ** 2).sum())
        p = float(sps.chi2.sf(Q, len(m) - 1))
        out.append({"key": key, "groups": {v: {"mean": mm, "se": ss, "n_records": len(groups[v])}
                                            for v, mm, ss in zip(sorted(groups), means, ses)},
                    "Q": Q, "p": p, "explains": p < 0.05})
    return sorted(out, key=lambda r: r["p"])


# ------------------------------------------------------------------------------------------------ consumers
def decision_summary(ledger: BeliefLedger, as_of, level: float = 0.80) -> pd.DataFrame:
    """One row per subject for the layer that USES beliefs: the effect to plan on (conservative bound), the influence cap, the
    epistemic status and why, evidence volume, staleness and contradiction share.  Everything is recomputed at `as_of`."""
    rows = []
    for subj in ledger.subjects():
        st = ledger.view(subj, as_of)
        epi, why = belief_status(st, ledger.cfg)
        stale = staleness(ledger, subj, as_of)
        rows.append({"subject": subj, "plan_effect": conservative_effect(st, 1 if st.mean >= 0 else -1, level),
                     "mean": st.mean, "sd": st.sd, "influence": influence_weight(st, ledger.cfg), "epistemic": str(epi),
                     "why_not_sure": str(why) if why else "", "n_obs": st.n_obs, "n_evidence": st.n_evidence,
                     "contra_mass": st.contradiction_mass, "stale": bool(stale["stale"]), "retracted": len(ledger.retractions(subj))})
    cols = ["subject", "plan_effect", "mean", "sd", "influence", "epistemic", "why_not_sure", "n_obs", "n_evidence", "contra_mass",
            "stale", "retracted"]
    return pd.DataFrame(rows, columns=cols)


def records_to_support(true_effect: float, se: float, quality: float = 1.0, prior_sd: float = 0.05, target: float = 0.95,
                       max_records: int = 200, seed: int = 0, n_sims: int = 200) -> dict:
    """How many evidence records of a given precision and quality does it take, on average, for the belief to reach
    P(sign right) >= target?  A planning tool: it says how long a claim of a given size will stay a hypothesis, and shows that
    low-quality evidence slows learning in proportion.  Deterministic in `seed`."""
    if se <= 0 or not 0 < quality <= 1:
        raise BeliefError("need se > 0 and 0 < quality <= 1")
    rng = np.random.default_rng(seed)
    cfg = BeliefConfig(prior_sd=prior_sd, half_life_days=None, use_random_effects=False)
    need = []
    for _ in range(n_sims):
        x = rng.normal(true_effect, se, max_records)
        prec0, num, hit = 1.0 / prior_sd ** 2, 0.0, None
        for k in range(max_records):
            prec0 += quality / se ** 2
            num += quality * x[k] / se ** 2
            m, sd = num / prec0, 1.0 / math.sqrt(prec0)
            if sps.norm.cdf(abs(m) / sd) >= target:
                hit = k + 1
                break
        need.append(hit if hit is not None else max_records + 1)
    arr = np.asarray(need, float)
    return {"median": float(np.median(arr)), "mean": float(arr.mean()), "share_never": float(np.mean(arr > max_records)),
            "true_effect": true_effect, "se": se, "quality": quality, "cfg_hash": cfg.config_hash()}


def evidence_table(ledger: BeliefLedger, subject: str, as_of=None) -> pd.DataFrame:
    """The evidence behind a belief, one row per record, with the quality it was given and whether it was later retracted."""
    gone = ledger.retractions(subject)
    rows = []
    for e in ledger.evidence_for(subject, as_of, include_retracted=True):
        rows.append({"evidence_id": e.evidence_id, "observed_at": e.observed_at, "kind": str(e.kind), "estimate": e.estimate,
                     "se": e.se, "n": e.n, "n_trials": e.n_trials, "quality": e.quality(ledger.cfg), "source": e.source,
                     "retracted": e.evidence_id in gone, "retraction_reason": gone.get(e.evidence_id, ("", ""))[0]})
    return pd.DataFrame(rows, columns=["evidence_id", "observed_at", "kind", "estimate", "se", "n", "n_trials", "quality", "source",
                                       "retracted", "retraction_reason"])
