"""The +-1 percentage-point calibration target, measured honestly, and the anti-gaming layer (C68 checklists P and M). IMPLEMENTED -
NOT VALIDATED.

Checklist P: the long-run goal is that the realised outcome lands within +-1pp of the ORIGINAL prediction at least 80% of the time,
with enough sample and out-of-sample validation. It is an EVALUATION target. Nothing here is imported by, or feeds, any exit or hold
decision (engine.research.exit_research is target-blind and exit_independence_audit proves it).

Checklist M makes each way of manufacturing the statistic impossible or detected:
    hold longer / delay the exit       every outcome carries the learned policy's own exit date; a later exit, or a fill more than
                                       one session after the decision, is an abuse (HELD_LONGER, DELAYED_EXIT)
    choose a worse exit                an exit not decided by the learned policy, or a realised return that differs from the policy's
                                       and sits closer to the prediction, is an abuse (WORSE_EXIT, NOT_POLICY_EXIT)
    redefine the prediction            the prediction is COMMITTED (value + expectation digest) on an append-only hash chain before the
                                       outcome exists; a later digest or value that differs is REDEFINED
    suppress losers                    the evaluated sample is the commitment book, not the caller's list: a matured commitment
                                       without an outcome is SUPPRESSED and counted as a miss in the headline
    cherry-pick samples                a sample narrower than the book is refused unless it is a cohort pre-registered before its
                                       first commitment (CHERRY_PICKED)
    change the target                  targets are registered on the chain; one registered after the first commitment it grades, or
                                       different from the one the commitment named, is TARGET_CHANGED
If the goal is not reached honestly the report says so, with the sample size and out-of-sample intervals. The prediction record
itself belongs to P01's expectation ledger; this module stores only the commitment (id, value, digest), duck-typed so it does not
import a parallel builder's module. Builds on engine.research.replication.ResearchLane (hash-chain lanes of the archive),
engine.learning.calibration.wilson and engine.learning.promotion.block_bootstrap_ci."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning import promotion as PR
from engine.learning.calibration import wilson
from engine.learning.core import FirewallBreach, _StrEnum, as_date, stable_hash
from engine.research.replication import ResearchLane

SECTION = "C68 checklists M, P"
LEARNED_POLICY = "learned_policy"


class AbuseKind(_StrEnum):
    HELD_LONGER = "HELD_LONGER"
    DELAYED_EXIT = "DELAYED_EXIT"
    WORSE_EXIT = "WORSE_EXIT"
    NOT_POLICY_EXIT = "NOT_POLICY_EXIT"
    REDEFINED = "REDEFINED"
    SUPPRESSED = "SUPPRESSED"
    CHERRY_PICKED = "CHERRY_PICKED"
    TARGET_CHANGED = "TARGET_CHANGED"
    POLICY_MISMATCH = "POLICY_MISMATCH"
    FUTURE_INFORMATION = "FUTURE_INFORMATION"


class Status(_StrEnum):
    ACHIEVED = "ACHIEVED"                          # OOS lower confidence bound >= goal, sample sufficient, no abuse
    NOT_ACHIEVED = "NOT_ACHIEVED"                  # measured honestly and short of the goal (or the interval straddles it)
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    COMPROMISED = "COMPROMISED"                    # an abuse was found: the number is reported but may not be quoted as a result
    EMPTY = "EMPTY"


@dataclass(frozen=True)
class Target:
    """The evaluation target. Frozen and hashed: a different tolerance or goal is a DIFFERENT target with a different digest."""
    tolerance: float = 0.01
    goal_share: float = 0.80
    min_n: int = 200
    min_oos_n: int = 100
    level: float = 0.95
    name: str = "pm1pp_80"

    def validate(self) -> list[str]:
        errs = []
        if not 0 < self.tolerance < 1:
            errs.append("tolerance must be in (0,1)")
        if not 0 < self.goal_share < 1:
            errs.append("goal_share must be in (0,1)")
        if self.min_n < 30 or self.min_oos_n < 30:
            errs.append("min_n and min_oos_n must be at least 30")
        if not 0.5 < self.level < 1:
            errs.append("level must be in (0.5,1)")
        return errs

    def digest(self) -> str:
        return stable_hash(self)


DEFAULT_TARGET = Target()


@dataclass(frozen=True)
class Commitment:
    """A prediction frozen BEFORE its outcome: the value, the digest of the full expectation it came from, the exit policy it will
    be traded under, when the model was last trained (for the OOS split) and when the outcome is due."""
    pred_id: str
    recorded_at: str
    predicted: float
    expectation_digest: str
    exit_policy_id: str
    model_trained_through: str
    matures_by: str
    target_digest: str
    cohort: str = ""
    in_sample: bool = False                         # an explicit research backfill by a model that saw this period: never OOS

    def validate(self) -> list[str]:
        errs = []
        if not self.pred_id:
            errs.append("pred_id missing")
        if not math.isfinite(self.predicted):
            errs.append("prediction is not finite")
        if as_date(self.matures_by) < as_date(self.recorded_at):
            errs.append("matures before it was recorded")
        if not self.in_sample and as_date(self.model_trained_through) >= as_date(self.recorded_at):
            errs.append("model trained on data at/after the prediction's decision date but not flagged in_sample")
        if not self.expectation_digest:
            errs.append("expectation digest missing")
        return errs

    @property
    def oos(self) -> bool:
        return not self.in_sample and as_date(self.model_trained_through) < as_date(self.recorded_at)


@dataclass(frozen=True)
class ExitRecord:
    """What actually happened. `policy_exit_date` / `policy_realised` are what the learned exit policy decided; `exit_date` /
    `realised` are what was traded. In an honest system they are the same."""
    pred_id: str
    exit_date: str
    realised: float
    decided_by: str
    policy_exit_date: str
    policy_realised: float
    fill_lag_sessions: int
    exit_policy_id: str
    predicted_as_reported: float | None = None       # the prediction as the reporter quotes it now (must equal the commitment)
    expectation_digest_now: str | None = None


@dataclass(frozen=True)
class Abuse:
    kind: AbuseKind
    pred_id: str
    detail: str


def expectation_digest(expectation: Any) -> str:
    """Digest of any expectation object: its own digest()/hash if it has one (P01 ledger rows do), else a canonical hash."""
    for attr in ("content_hash", "digest", "record_hash", "hash"):
        v = getattr(expectation, attr, None)
        if callable(v):
            v = v()
        if isinstance(v, str) and v:
            return v
    if dataclasses.is_dataclass(expectation):
        return stable_hash(expectation)
    if isinstance(expectation, Mapping):
        return stable_hash(dict(expectation))
    raise TypeError(f"cannot digest expectation of type {type(expectation).__name__}")


def _predicted_of(expectation: Any) -> float:
    for attr in ("predicted_realisable", "expected_realisable", "predicted_return", "expected_return", "mean"):
        v = expectation.get(attr) if isinstance(expectation, Mapping) else getattr(expectation, attr, None)
        if v is not None:
            return float(v)
    raise ValueError("expectation carries no predicted realisable gain")


class CommitmentBook(ResearchLane):
    """Append-only, hash-chained commitments, target registrations and pre-registered cohorts (one lane of the archive chain)."""
    LANE = "p05_calib"

    def register_target(self, target: Target, now) -> str:
        errs = target.validate()
        if errs:
            raise ValueError("invalid target: " + "; ".join(errs))
        dg = target.digest()
        if not any(r["kind"] == "target" and r["body"]["digest"] == dg for r in self.rows()):
            self.append("target", {"digest": dg, "registered": str(as_date(now)), "target": dataclasses.asdict(target)})
        return dg

    def target_registered(self, digest: str) -> str | None:
        return next((r["body"]["registered"] for r in self.rows() if r["kind"] == "target" and r["body"]["digest"] == digest), None)

    def register_cohort(self, name: str, rule: str, now) -> None:
        """A named sub-sample defined BEFORE its members' outcomes exist (e.g. 'regime=calm'). Only these may be evaluated alone."""
        if any(r["kind"] == "cohort" and r["body"]["name"] == name for r in self.rows()):
            raise FileExistsError(f"cohort {name} already registered: a definition is never rewritten")
        self.append("cohort", {"name": name, "rule": rule, "registered": str(as_date(now))})

    def cohorts(self) -> dict[str, str]:
        return {r["body"]["name"]: r["body"]["registered"] for r in self.rows() if r["kind"] == "cohort"}

    def commit(self, pred_id: str, expectation: Any, exit_policy_id: str, model_trained_through, matures_by, now,
               target: Target = DEFAULT_TARGET, cohort: str = "", in_sample: bool = False) -> Commitment:
        """Freeze a prediction at decision time `now`. Refuses duplicates (no second version of a prediction) and a model trained on
        data dated at/after `now` - unless the caller declares a research backfill (`in_sample=True`), which is then graded but
        never counted out of sample."""
        if not in_sample and as_date(model_trained_through) >= as_date(now):
            raise FirewallBreach(f"model trained through {model_trained_through} is not strictly before the decision {now}")
        if any(r["kind"] == "commit" and r["body"]["pred_id"] == pred_id for r in self.rows()):
            raise FileExistsError(f"prediction {pred_id} already committed: predictions are never redefined")
        if cohort and cohort not in self.cohorts():
            raise ValueError(f"cohort {cohort!r} is not pre-registered: a sub-sample must be defined before its members exist")
        tdg = self.register_target(target, now) if self.target_registered(target.digest()) is None else target.digest()
        c = Commitment(str(pred_id), str(as_date(now)), _predicted_of(expectation), expectation_digest(expectation), str(exit_policy_id),
                       str(as_date(model_trained_through)), str(as_date(matures_by)), tdg, cohort, bool(in_sample))
        errs = c.validate()
        if errs:
            raise ValueError("invalid commitment: " + "; ".join(errs))
        self.append("commit", dataclasses.asdict(c))
        return c

    def commitments(self, now=None) -> list[Commitment]:
        out = [Commitment(**r["body"]) for r in self.rows() if r["kind"] == "commit"]
        return out if now is None else [c for c in out if as_date(c.recorded_at) < as_date(now)]


# ------------------------------------------------------------------------------------------------ anti-gaming audit (checklist M)
def audit(book: CommitmentBook, outcomes: Iterable[ExitRecord], now, target: Target = DEFAULT_TARGET,
          sample: Sequence[str] | None = None, cohort: str | None = None) -> tuple[list[Abuse], list[Commitment], dict[str, ExitRecord]]:
    """Return (abuses, the sample that MUST be graded, resolved outcomes). The sample is decided here from the book, never by the
    caller: every commitment matured before `now` (optionally one pre-registered cohort)."""
    abuses: list[Abuse] = []
    chain = book.verify()
    if chain:
        abuses.append(Abuse(AbuseKind.REDEFINED, "*", f"commitment chain tampered: {chain}"))
    comm = book.commitments(now)
    by_id = {c.pred_id: c for c in comm}
    outs: dict[str, ExitRecord] = {}
    for o in outcomes:
        if o.pred_id in outs:
            abuses.append(Abuse(AbuseKind.REDEFINED, o.pred_id, "two outcomes reported for one prediction"))
        if as_date(o.exit_date) >= as_date(now):
            abuses.append(Abuse(AbuseKind.FUTURE_INFORMATION, o.pred_id, f"outcome dated {o.exit_date} is not before now={now}"))
            continue
        outs[o.pred_id] = o
    if cohort is not None:
        reg = book.cohorts().get(cohort)
        members = [c for c in comm if c.cohort == cohort]
        if reg is None or any(as_date(c.recorded_at) < as_date(reg) for c in members):
            abuses.append(Abuse(AbuseKind.CHERRY_PICKED, "*", f"cohort {cohort!r} was not registered before its first member"))
        comm = members
    matured = [c for c in comm if as_date(c.matures_by) < as_date(now)]
    if sample is not None and set(sample) != {c.pred_id for c in matured}:
        dropped = len({c.pred_id for c in matured} - set(sample))
        abuses.append(Abuse(AbuseKind.CHERRY_PICKED, "*", f"requested sample differs from the committed book ({dropped} matured prediction(s) left out)"))
    reg = book.target_registered(target.digest())
    if reg is None:
        abuses.append(Abuse(AbuseKind.TARGET_CHANGED, "*", "target never registered"))
    elif matured and as_date(reg) > min(as_date(c.recorded_at) for c in matured):
        abuses.append(Abuse(AbuseKind.TARGET_CHANGED, "*", f"target registered {reg}, after predictions it grades were made"))
    for c in matured:
        if c.target_digest != target.digest():
            abuses.append(Abuse(AbuseKind.TARGET_CHANGED, c.pred_id, "graded under a target other than the one committed"))
        o = outs.get(c.pred_id)
        if o is None:
            abuses.append(Abuse(AbuseKind.SUPPRESSED, c.pred_id, f"matured by {c.matures_by} with no reported outcome"))
            continue
        abuses.extend(_outcome_abuses(c, o))
    for pid in outs:
        if pid not in by_id:
            abuses.append(Abuse(AbuseKind.REDEFINED, pid, "outcome for a prediction that was never committed (made up after the fact)"))
    return abuses, matured, outs


def _outcome_abuses(c: Commitment, o: ExitRecord) -> list[Abuse]:
    out = []
    if o.exit_policy_id != c.exit_policy_id:
        out.append(Abuse(AbuseKind.POLICY_MISMATCH, c.pred_id, f"traded under {o.exit_policy_id}, predicted under {c.exit_policy_id}"))
    if o.decided_by != LEARNED_POLICY:
        out.append(Abuse(AbuseKind.NOT_POLICY_EXIT, c.pred_id, f"exit decided by {o.decided_by!r}, not the learned exit policy"))
    if as_date(o.exit_date) > as_date(o.policy_exit_date):
        out.append(Abuse(AbuseKind.HELD_LONGER, c.pred_id, f"held to {o.exit_date} past the policy exit {o.policy_exit_date}"))
    if o.fill_lag_sessions > 1:
        out.append(Abuse(AbuseKind.DELAYED_EXIT, c.pred_id, f"filled {o.fill_lag_sessions} sessions after the decision (max 1: next open)"))
    if not math.isclose(o.realised, o.policy_realised, abs_tol=1e-9):
        closer = abs(o.realised - c.predicted) < abs(o.policy_realised - c.predicted)
        out.append(Abuse(AbuseKind.WORSE_EXIT, c.pred_id, f"realised {o.realised:+.4f} differs from the policy's {o.policy_realised:+.4f}"
                         + (" and moved TOWARD the prediction" if closer else "")))
    if o.predicted_as_reported is not None and not math.isclose(o.predicted_as_reported, c.predicted, abs_tol=1e-12):
        out.append(Abuse(AbuseKind.REDEFINED, c.pred_id, f"prediction reported as {o.predicted_as_reported:+.4f}, committed {c.predicted:+.4f}"))
    if o.expectation_digest_now is not None and o.expectation_digest_now != c.expectation_digest:
        out.append(Abuse(AbuseKind.REDEFINED, c.pred_id, "expectation digest changed after commitment"))
    return out


# ------------------------------------------------------------------------------------------------ honest measurement (checklist P)
@dataclass(frozen=True)
class Share:
    n: int
    hits: int
    share: float
    lo: float
    hi: float
    boot_lo: float
    boot_hi: float


@dataclass(frozen=True)
class CalibrationReport:
    now: str
    target: Target
    status: Status
    n_committed: int
    n_matured: int
    n_resolved: int
    n_unresolved: int
    all_: Share                         # headline: unresolved matured predictions count as MISSES
    oos: Share                          # only predictions whose model never saw data at/after their recording date
    resolved_only: Share                # for diagnosis only; never the headline
    bias: float                         # mean(realised - predicted) over resolved
    mae: float
    naive_share: float | None           # share hit by "predict the median of outcomes matured before" (same sample)
    required_n: int | None              # sample needed to show the goal if the true share were the observed one
    abuses: tuple[Abuse, ...]
    by_cohort: tuple[tuple[str, Share], ...]
    headline: str

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        d["abuses"] = [{"kind": a.kind.value, "pred_id": a.pred_id, "detail": a.detail} for a in self.abuses]
        return d


def _share(hits: np.ndarray, periods: Sequence[str], target: Target, seed: int) -> Share:
    n = len(hits)
    if n == 0:
        return Share(0, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    k = int(hits.sum())
    lo, hi = wilson(k, n, _z(target.level))
    order = np.argsort(np.asarray(periods, dtype="datetime64[D]"), kind="stable")
    blo, bhi = PR.block_bootstrap_ci(hits[order].astype(float), seed, n_boot=600, block=max(2, int(round(n ** (1 / 3)))), level=target.level)
    return Share(n, k, k / n, float(lo), float(hi), float(blo), float(bhi))


def _z(level: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(0.5 + level / 2))


def required_n(p_true: float, goal: float, level: float = 0.95) -> int | None:
    """Smallest n at which the Wilson lower bound of an observed share p_true clears `goal` (None if p_true <= goal)."""
    if not math.isfinite(p_true) or p_true <= goal:
        return None
    z = _z(level)
    n = 10
    while n < 10_000_000:
        if wilson(p_true * n, n, z)[0] >= goal:
            return n
        n = int(n * 1.25) + 1
    return None


def naive_share(matured: Sequence[Commitment], outs: Mapping[str, ExitRecord], tolerance: float) -> float | None:
    """A no-skill comparator on the same sample: predict the median of all outcomes that had ALREADY matured (exit date strictly
    before the commitment's recording date). A system that cannot beat this has no calibration skill to report."""
    hist = sorted((as_date(o.exit_date), o.realised) for o in outs.values())
    hits, n = 0, 0
    for c in matured:
        o = outs.get(c.pred_id)
        if o is None:
            n += 1                                  # unresolved counts as a miss for the comparator too
            continue
        past = [r for d, r in hist if d < as_date(c.recorded_at)]
        if not past:
            continue
        n += 1
        hits += abs(o.realised - float(np.median(past))) <= tolerance
    return hits / n if n else None


def evaluate(book: CommitmentBook, outcomes: Iterable[ExitRecord], now, target: Target = DEFAULT_TARGET, seed: int = 0,
             sample: Sequence[str] | None = None, cohort: str | None = None) -> CalibrationReport:
    """The public entry for checklist P. Grades every matured commitment (never a caller-chosen subset) against the committed
    prediction, reports the real share with Wilson and period-block bootstrap intervals, and the OOS share separately. ACHIEVED
    only when the OOS lower bound clears the goal on at least min_oos_n predictions and nothing was gamed."""
    errs = target.validate()
    if errs:
        raise ValueError("invalid target: " + "; ".join(errs))
    abuses, matured, outs = audit(book, outcomes, now, target, sample, cohort)
    tol = target.tolerance + 1e-12
    hits_all, hits_res, periods_all, periods_res, errs_res = [], [], [], [], []
    oos_hits, oos_periods = [], []
    groups: dict[str, list[tuple[bool, str]]] = {}
    for c in matured:
        o = outs.get(c.pred_id)
        hit = o is not None and abs(o.realised - c.predicted) <= tol
        hits_all.append(hit)
        periods_all.append(c.recorded_at)
        groups.setdefault(c.cohort or "-", []).append((hit, c.recorded_at))
        if c.oos:
            oos_hits.append(hit)
            oos_periods.append(c.recorded_at)
        if o is not None:
            hits_res.append(hit)
            periods_res.append(c.recorded_at)
            errs_res.append(o.realised - c.predicted)
    A = _share(np.array(hits_all, bool), periods_all, target, seed)
    O = _share(np.array(oos_hits, bool), oos_periods, target, seed + 1)
    R = _share(np.array(hits_res, bool), periods_res, target, seed + 2)
    e = np.array(errs_res, float)
    by = tuple((g, _share(np.array([h for h, _ in v], bool), [p for _, p in v], target, seed + 3)) for g, v in sorted(groups.items()))
    n_unres = sum(1 for c in matured if c.pred_id not in outs)
    if not matured:
        status = Status.EMPTY
    elif abuses:
        status = Status.COMPROMISED
    elif A.n < target.min_n or O.n < target.min_oos_n:
        status = Status.INSUFFICIENT_SAMPLE
    elif O.lo >= target.goal_share and O.boot_lo >= target.goal_share:
        status = Status.ACHIEVED
    else:
        status = Status.NOT_ACHIEVED
    rep = CalibrationReport(str(as_date(now)), target, status, len(book.commitments(now)), len(matured), len(outs), n_unres, A, O, R,
                            float(e.mean()) if len(e) else float("nan"), float(np.abs(e).mean()) if len(e) else float("nan"),
                            naive_share(matured, outs, tol), required_n(O.share if O.n else A.share, target.goal_share, target.level),
                            tuple(abuses), by, "")
    return dataclasses.replace(rep, headline=headline(rep))


def headline(r: CalibrationReport) -> str:
    """One honest sentence. It always states the actual share and n; it never says 'achieved' unless the status is ACHIEVED."""
    t = r.target
    if r.status == Status.EMPTY:
        return f"No matured predictions yet: the +-{t.tolerance:.0%} target is unmeasured."
    s = (f"{r.all_.share:.1%} of {r.all_.n} matured predictions landed within +-{t.tolerance * 100:.0f}pp of the committed prediction "
         f"(95% CI {r.all_.lo:.1%}-{r.all_.hi:.1%}); out of sample {r.oos.share:.1%} of {r.oos.n} "
         f"(CI {r.oos.lo:.1%}-{r.oos.hi:.1%}, block bootstrap {r.oos.boot_lo:.1%}-{r.oos.boot_hi:.1%}). Goal {t.goal_share:.0%}: ")
    tail = {Status.ACHIEVED: "achieved out of sample.",
            Status.NOT_ACHIEVED: "NOT achieved.",
            Status.INSUFFICIENT_SAMPLE: f"sample too small to judge (need {t.min_n} overall and {t.min_oos_n} out of sample).",
            Status.COMPROMISED: f"NOT a valid result - {len(r.abuses)} integrity problem(s): "
                                + ", ".join(sorted({a.kind.value for a in r.abuses})) + "."}[r.status]
    if r.n_unresolved:
        tail += f" {r.n_unresolved} matured prediction(s) have no outcome and are counted as misses."
    if r.naive_share is not None:
        tail += f" A no-skill median forecast scores {r.naive_share:.1%} on the same sample."
    return s + tail


def trend(book: CommitmentBook, outcomes: Iterable[ExitRecord], now, target: Target = DEFAULT_TARGET, freq: str = "Q") -> pd.DataFrame:
    """Per-period hit share (unresolved = miss) with Wilson intervals: is the system moving toward the goal, honestly measured?"""
    abuses, matured, outs = audit(book, outcomes, now, target)
    if not matured:
        return pd.DataFrame(columns=["period", "n", "share", "lo", "hi", "oos_n"])
    rows = [(pd.Timestamp(as_date(c.recorded_at)).to_period(freq), (c.pred_id in outs) and abs(outs[c.pred_id].realised - c.predicted) <= target.tolerance + 1e-12, c.oos)
            for c in matured]
    df = pd.DataFrame(rows, columns=["period", "hit", "oos"])
    out = []
    for per, g in df.groupby("period", sort=True):
        k, n = int(g.hit.sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"period": str(per), "n": n, "share": k / n, "lo": lo, "hi": hi, "oos_n": int(g.oos.sum())})
    return pd.DataFrame(out)


def tolerance_profile(book: CommitmentBook, outcomes: Iterable[ExitRecord], now, tolerances: Sequence[float] = (0.005, 0.01, 0.02, 0.03, 0.05)) -> pd.DataFrame:
    """Hit share at several tolerances on the SAME committed sample: where the system actually is relative to +-1pp. Reported for
    context only - it never replaces the registered target in `evaluate`."""
    abuses, matured, outs = audit(book, outcomes, now, DEFAULT_TARGET)
    rows = []
    for t in tolerances:
        hits = [(c.pred_id in outs) and abs(outs[c.pred_id].realised - c.predicted) <= t + 1e-12 for c in matured]
        rows.append({"tolerance": t, "n": len(hits), "share": float(np.mean(hits)) if hits else float("nan")})
    return pd.DataFrame(rows)


def exit_record_from_policy(pred_id: str, exit_date, policy_realised: float, exit_policy_id: str, fill_lag_sessions: int = 1) -> ExitRecord:
    """The honest path: the traded exit IS the learned policy's exit (same date, same return)."""
    d = str(as_date(exit_date))
    return ExitRecord(pred_id, d, float(policy_realised), LEARNED_POLICY, d, float(policy_realised), int(fill_lag_sessions), exit_policy_id)
