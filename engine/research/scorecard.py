"""Research self-improvement scorecard (contract C66 section 36, with sections 43 and 48). IMPLEMENTED - NOT VALIDATED.

Every major research cycle reports five sections, all computed from artefacts (forecast frames, loss records, knowledge events, job
ledger), never typed in:
    VOLATILITY           precision, recall, ranking quality, calibration, coverage, transfer, stability
    DIRECTION            accuracy, Brier score, calibration, coverage, accuracy at coverage levels, transfer, stability, and the honest
                         80%-frontier finding (recorded as a scientific limitation when it is not reachable - never manufactured)
    LOSS                 worst loss, tail loss, false-positive loss, avoidable-loss rate, large-loss frequency
    LEARNING             new knowledge, retired, downgraded, promoted, experiments completed and replicated, failed learners,
                         successful transfers, memorisation failures (counts: context, never a quality claim)
    RESEARCH EFFICIENCY  compute consumed, information gained, decision changes, duplicate experiments avoided, branches abandoned/escalated
A quantity that was not measured is UNTESTED (a `Measured` from engine.learning.scorecard), which is different from zero and is never
drawn as a number. Rates carry a cluster-bootstrap interval (clusters = periods), so a lucky week does not move the card.

"The system must learn from this scorecard": `derive_feedback` turns weak or regressing sections into research-priority shifts over the
section-34 problem hierarchy. Counts (trades, patterns, experiments) can never produce positive feedback (section 43), direction
effort is withheld until volatility has evidence (section 0), and no improvement may be claimed without out-of-sample transfer
(section 48: SAME-YEAR IMPROVEMENT = INTERESTING, OOS TRANSFER = REQUIRED). Builds on engine.learning.scorecard (Measured, the
improvement-claim gate), engine.learning.transfer_score (cluster bootstrap), engine.learning.calibration and the chain log of
engine.research.replication. Artefacts dated at/after `now` are a FirewallBreach: a cycle may only score what has matured."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning import calibration as CAL
from engine.learning import scorecard as LS
from engine.learning import transfer_score as TS
from engine.learning.scorecard import Measured, MStatus
from engine.learning.core import FirewallBreach, ValidationLabel, _StrEnum, as_date, stable_hash
from engine.research import replication as RP
from engine.research.core import Problem

SECTION = "C66 section 36"
COVERAGES = (0.1, 0.25, 0.5, 0.75, 1.0)
EVENT_TYPES = ("knowledge_new", "knowledge_retired", "knowledge_downgraded", "knowledge_promoted", "experiment_completed",
               "experiment_replicated", "learner_failed", "transfer_success", "memorization_failure")
FORBIDDEN_CLAIMS = ("validated", "production ready", "solved", "done", "learning works")
CLAIM_WORDS = ("improved", "improvement", "better", "learning improved")
HIGHER_IS_BETTER = {
    "volatility": {"precision": True, "recall": True, "ranking_quality": True, "calibration": True, "coverage": True, "transfer": True, "stability": True},
    "direction": {"accuracy": True, "brier_skill": True, "calibration": True, "coverage": True, "transfer": True, "stability": True},
    "loss": {"worst_loss": True, "tail_loss": True, "false_positive_loss": True, "avoidable_loss_rate": False, "large_loss_frequency": False},
    "efficiency": {"information_per_compute": True, "decisions_per_compute": True},
}


# ------------------------------------------------------------------------------------------------ policy
@dataclass(frozen=True)
class ScorecardPolicy:
    """Every knob of the scorecard in one hashable place: an operating point chosen after seeing a result must show up as a different
    policy digest."""
    select_top_frac: float = 0.10            # share of each period's forecasts treated as 'the picks'
    mover_threshold: float = 0.05            # |move| that counts as a large move when the artefact has only magnitudes
    large_loss: float = 0.10                 # a position loss at least this large is a large loss
    tail_frac: float = 0.05                  # worst share of positions averaged for the tail loss
    frontier_target: float = 0.80            # the direction accuracy the contract hopes for
    frontier_min_coverage: float = 0.05      # a frontier accuracy on a thinner slice than this is not reported
    min_periods: int = 6                     # fewer clusters than this: rates are measured but flagged thin
    n_boot: int = 400
    level: float = 0.95
    stability_floor: float = 0.5             # a period counts as good when its metric beats this share of the chance/base level
    calibration_bins: int = 10
    min_calibration_n: int = 100
    seed: int = 0
    feedback_step: float = 0.25
    feedback_floor: float = 0.02

    def validate(self) -> list[str]:
        errs = []
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, float) and not math.isfinite(v):
                errs.append(f"{f.name} is not finite")
        for name in ("select_top_frac", "tail_frac", "frontier_target", "level", "feedback_floor"):
            if not 0.0 < getattr(self, name) < 1.0:
                errs.append(f"{name} must be in (0,1)")
        if self.large_loss <= 0 or self.mover_threshold <= 0:
            errs.append("large_loss and mover_threshold must be positive")
        if self.min_periods < 2 or self.n_boot < 50 or self.calibration_bins < 2:
            errs.append("min_periods >= 2, n_boot >= 50, calibration_bins >= 2")
        if self.feedback_step <= 0:
            errs.append("feedback_step must be positive")
        return errs

    def digest(self) -> str:
        return stable_hash(self)


DEFAULT_POLICY = ScorecardPolicy()


# ------------------------------------------------------------------------------------------------ artefacts
@dataclass(frozen=True)
class Artefacts:
    """What a cycle scored. Every frame needs `matured_at` (real date the outcome became known); a frame is optional and its absence
    leaves the matching section UNTESTED. Column contracts (checked by `validate_artefacts`):
      vol        p_move (P of a large move), moved (0/1), abs_move, period; optional considered (0/1), seen_context (0/1)
      direction  p_up, up (0/1), period; optional abstained (0/1), seen_context (0/1)
      losses     ret (position return), period; optional false_positive (0/1), avoidable (0/1)
      jobs       cost_minutes, information_bits, decision_changed (0/1), duplicate_avoided (0/1), branch (''|'abandoned'|'escalated')
      events     records with type in EVENT_TYPES and `at`"""
    vol: pd.DataFrame | None = None
    direction: pd.DataFrame | None = None
    losses: pd.DataFrame | None = None
    jobs: pd.DataFrame | None = None
    events: tuple[Mapping[str, Any], ...] = ()


REQUIRED = {"vol": ("matured_at", "p_move", "moved", "abs_move", "period"), "direction": ("matured_at", "p_up", "up", "period"),
            "losses": ("matured_at", "ret", "period"),
            "jobs": ("matured_at", "cost_minutes", "information_bits", "decision_changed", "duplicate_avoided", "branch")}


def validate_artefacts(a: Artefacts, now) -> list[str]:
    """Structural problems (missing columns, NaN, out-of-range probabilities, unknown event types). Maturity is NOT checked here: an
    artefact from the future is a FirewallBreach raised by `require_matured`, never a soft problem."""
    errs = []
    for name, cols in REQUIRED.items():
        df = getattr(a, name)
        if df is None:
            continue
        miss = [c for c in cols if c not in df.columns]
        if miss:
            errs.append(f"{name}: missing columns {miss}")
            continue
        if df[list(cols)].isna().any().any():
            errs.append(f"{name}: NaN in required columns (a missing outcome is not a zero)")
    if a.vol is not None and not {"p_move", "moved"} - set(a.vol.columns):
        if ((a.vol["p_move"] < 0) | (a.vol["p_move"] > 1)).any() or not a.vol["moved"].isin([0, 1]).all():
            errs.append("vol: p_move must be in [0,1] and moved in {0,1}")
    if a.direction is not None and not {"p_up", "up"} - set(a.direction.columns):
        if ((a.direction["p_up"] < 0) | (a.direction["p_up"] > 1)).any() or not a.direction["up"].isin([0, 1]).all():
            errs.append("direction: p_up must be in [0,1] and up in {0,1}")
    bad = [e.get("type") for e in a.events if e.get("type") not in EVENT_TYPES]
    if bad:
        errs.append(f"events: unknown types {sorted(set(map(str, bad)))}")
    if any("at" not in e for e in a.events):
        errs.append("events: every event needs an `at` date")
    return errs


def require_matured(a: Artefacts, now) -> None:
    """Fail closed on any artefact whose outcome had not matured strictly before `now`."""
    for name in ("vol", "direction", "losses", "jobs"):
        df = getattr(a, name)
        if df is None or df.empty or "matured_at" not in df.columns:
            continue
        late = df[pd.to_datetime(df["matured_at"]).dt.date >= as_date(now)]
        if len(late):
            raise FirewallBreach(f"scorecard {name}: {len(late)} row(s) matured at/after now={as_date(now)} (first {late['matured_at'].iloc[0]})")
    for e in a.events:
        if as_date(e["at"]) >= as_date(now):
            raise FirewallBreach(f"scorecard events: {e.get('type')} dated {e['at']} at/after now={as_date(now)}")


# ------------------------------------------------------------------------------------------------ statistics helpers
def _boot(values, clusters, pol: ScorecardPolicy, salt: str, note: str = "") -> Measured:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return Measured.untested(note or "no data")
    seed = pol.seed + int(stable_hash(salt, 6), 16) % 100_000
    b = TS.cluster_bootstrap_mean(v, None if clusters is None else np.asarray(clusters), pol.n_boot, pol.level, seed)
    m = Measured.from_boot(b, note)
    if b.n_clusters < pol.min_periods:
        return dataclasses.replace(m, note=(note + "; " if note else "") + f"thin: {b.n_clusters} periods < {pol.min_periods}")
    return m


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), average ranks for ties. NaN when one class is missing."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def period_ic(df: pd.DataFrame, score: str, target: str, min_n: int = 5) -> pd.Series:
    """Spearman rank correlation of score with target within each period (a cross-sectional information coefficient)."""
    out = {}
    for k, g in df.groupby("period"):
        if len(g) >= min_n and g[score].nunique() > 1 and g[target].nunique() > 1:
            out[k] = float(g[score].rank().corr(g[target].rank()))
    return pd.Series(out, dtype=float)


def top_picks(df: pd.DataFrame, score: str, frac: float) -> pd.Series:
    """Boolean mask of each period's top `frac` by score (at least one per period). Ties are broken by original order, never by name."""
    rank = df.groupby("period")[score].rank(method="first", ascending=False)
    size = df.groupby("period")[score].transform("size")
    return rank <= np.maximum(1, np.ceil(size * frac))


def _ratio(unseen: Measured, seen: Measured) -> Measured:
    """Transfer: the metric on contexts the system never trained on as a share of the same metric on contexts it did."""
    if not (unseen.measured and seen.measured) or not math.isfinite(seen.value) or seen.value <= 1e-12:
        return Measured.untested("needs both seen and unseen contexts with a positive seen-context metric")
    r = unseen.value / seen.value
    lo = unseen.lo / seen.hi if math.isfinite(unseen.lo) and math.isfinite(seen.hi) and seen.hi > 0 else float("nan")
    hi = unseen.hi / seen.lo if math.isfinite(unseen.hi) and math.isfinite(seen.lo) and seen.lo > 1e-12 else float("nan")
    return Measured(float(r), float(lo), float(hi), min(unseen.n, seen.n), MStatus.MEASURED, "unseen / seen (interval is conservative)")


def _stability(per_period: pd.Series, chance: float) -> Measured:
    """Share of periods in which the metric beat `chance`; the worst period is kept in the note."""
    if len(per_period) < 2:
        return Measured.untested("fewer than two periods")
    share = float((per_period > chance).mean())
    return Measured.point(share, len(per_period), f"worst period {float(per_period.min()):+.3f}, median {float(per_period.median()):+.3f}")


# ------------------------------------------------------------------------------------------------ the five sections
@dataclass(frozen=True)
class VolatilityCard:
    precision: Measured = field(default_factory=Measured)
    lift: Measured = field(default_factory=Measured)
    recall: Measured = field(default_factory=Measured)
    ranking_quality: Measured = field(default_factory=Measured)
    auc: Measured = field(default_factory=Measured)
    calibration: Measured = field(default_factory=Measured)
    brier_skill: Measured = field(default_factory=Measured)
    coverage: Measured = field(default_factory=Measured)
    transfer: Measured = field(default_factory=Measured)
    stability: Measured = field(default_factory=Measured)
    base_rate: Measured = field(default_factory=Measured)


@dataclass(frozen=True)
class DirectionCard:
    accuracy: Measured = field(default_factory=Measured)
    brier: Measured = field(default_factory=Measured)
    brier_skill: Measured = field(default_factory=Measured)
    calibration: Measured = field(default_factory=Measured)
    coverage: Measured = field(default_factory=Measured)
    accuracy_at_coverage: Mapping[str, Measured] = field(default_factory=dict)
    frontier: Mapping[str, Any] = field(default_factory=dict)
    transfer: Measured = field(default_factory=Measured)
    stability: Measured = field(default_factory=Measured)


@dataclass(frozen=True)
class LossCard:
    worst_loss: Measured = field(default_factory=Measured)
    tail_loss: Measured = field(default_factory=Measured)
    false_positive_loss: Measured = field(default_factory=Measured)
    avoidable_loss_rate: Measured = field(default_factory=Measured)
    large_loss_frequency: Measured = field(default_factory=Measured)


@dataclass(frozen=True)
class LearningCard:
    counts: Mapping[str, int] = field(default_factory=dict)
    promotion_yield: Measured = field(default_factory=Measured)
    replication_share: Measured = field(default_factory=Measured)
    since: str = ""


@dataclass(frozen=True)
class EfficiencyCard:
    compute_minutes: Measured = field(default_factory=Measured)
    information_bits: Measured = field(default_factory=Measured)
    information_per_compute: Measured = field(default_factory=Measured)
    decision_changes: Measured = field(default_factory=Measured)
    decisions_per_compute: Measured = field(default_factory=Measured)
    duplicates_avoided: Measured = field(default_factory=Measured)
    branches_abandoned: Measured = field(default_factory=Measured)
    branches_escalated: Measured = field(default_factory=Measured)


def _calibration_pair(p: np.ndarray, y: np.ndarray, pol: ScorecardPolicy) -> tuple[Measured, Measured]:
    """(ECE, Brier skill over the base rate). UNTESTED when there are too few forecasts or a constant outcome."""
    if len(p) < pol.min_calibration_n or y.min() == y.max():
        why = f"{len(p)} forecasts < {pol.min_calibration_n}" if len(p) < pol.min_calibration_n else "constant outcomes"
        return Measured.untested(why), Measured.untested(why)
    base = float(y.mean())
    skill = 1.0 - CAL.brier(p, y) / max(base * (1 - base), 1e-12)
    return Measured.point(CAL.ece_equal_mass(p, y, pol.calibration_bins), len(p), "lower is better"), Measured.point(skill, len(p), "over the constant base-rate forecast")


def volatility_section(vol: pd.DataFrame | None, pol: ScorecardPolicy = DEFAULT_POLICY) -> VolatilityCard:
    if vol is None or vol.empty:
        return VolatilityCard()
    df = vol.reset_index(drop=True)
    picked = top_picks(df, "p_move", pol.select_top_frac)
    base = float(df["moved"].mean())
    prec_rows = df.loc[picked]
    precision = _boot(prec_rows["moved"], prec_rows["period"], pol, "vol.precision", "precision of each period's top picks")
    lift = Measured(precision.value / base, precision.lo / base, precision.hi / base, precision.n, precision.status, f"precision over base rate {base:.3f}") \
        if precision.measured and base > 0 else Measured.untested("no base rate")
    per_period = df.assign(hit=picked & (df["moved"] == 1)).groupby("period").agg(hits=("hit", "sum"), moved=("moved", "sum"))
    rec_ok = per_period[per_period["moved"] > 0]
    recall = _boot((rec_ok["hits"] / rec_ok["moved"]).to_numpy(), rec_ok.index.to_numpy(), pol, "vol.recall", "share of each period's movers that were picked")
    ic = period_ic(df, "p_move", "abs_move")
    ranking = _boot(ic.to_numpy(), ic.index.to_numpy(), pol, "vol.rank", "period rank IC of p_move against |move|") if len(ic) else Measured.untested("no period had enough spread")
    a = auc(df["p_move"].to_numpy(), df["moved"].to_numpy())
    ece, skill = _calibration_pair(df["p_move"].to_numpy(), df["moved"].to_numpy().astype(float), pol)
    if "considered" in df.columns:
        movers = df[df["moved"] == 1]
        coverage = _boot(movers["considered"].astype(float), movers["period"], pol, "vol.coverage", "share of realised movers the system had considered") \
            if len(movers) else Measured.untested("no realised movers")
    else:
        coverage = Measured.untested("no `considered` column: coverage cannot be measured")
    transfer = Measured.untested("no seen_context column")
    if "seen_context" in df.columns:
        parts = {}
        for flag in (0, 1):
            sub = df[df["seen_context"] == flag]
            pk = top_picks(sub, "p_move", pol.select_top_frac) if len(sub) else pd.Series(dtype=bool)
            rows = sub.loc[pk[pk].index] if len(sub) else sub
            parts[flag] = _boot(rows["moved"], rows["period"], pol, f"vol.transfer{flag}") if len(rows) else Measured.untested("no rows")
        transfer = _ratio(parts[0], parts[1])
    per_prec = prec_rows.groupby("period")["moved"].mean()
    return VolatilityCard(precision, lift, recall, ranking, Measured.point(a, len(df), "0.5 = chance") if math.isfinite(a) else Measured.untested("one class only"),
                          ece, skill, coverage, transfer, _stability(per_prec, base), Measured.point(base, len(df), "share of forecasts that moved"))


def direction_section(direction: pd.DataFrame | None, pol: ScorecardPolicy = DEFAULT_POLICY) -> DirectionCard:
    if direction is None or direction.empty:
        return DirectionCard()
    df = direction.reset_index(drop=True)
    made = df[df["abstained"] == 0] if "abstained" in df.columns else df
    coverage = Measured.point(len(made) / len(df), len(df), "share of cases the system was willing to call")
    if made.empty:
        return DirectionCard(coverage=coverage, frontier=frontier_finding(made, pol))
    correct = ((made["p_up"] >= 0.5).astype(int) == made["up"]).astype(float)
    accuracy = _boot(correct, made["period"], pol, "dir.accuracy")
    p, y = made["p_up"].to_numpy(), made["up"].to_numpy().astype(float)
    brier = Measured.point(CAL.brier(p, y), len(p), "lower is better")
    ece, skill = _calibration_pair(p, y, pol)
    conf = (made["p_up"] - 0.5).abs()
    acc_cov: dict[str, Measured] = {}
    for c in COVERAGES:
        k = max(1, int(math.ceil(c * len(made))))
        idx = conf.sort_values(ascending=False, kind="mergesort").index[:k]
        acc_cov[f"{c:.2f}"] = _boot(correct.loc[idx], made.loc[idx, "period"], pol, f"dir.cov{c}", f"most confident {c:.0%} of calls")
    transfer = Measured.untested("no seen_context column")
    if "seen_context" in made.columns:
        parts = {f: (_boot(correct[made["seen_context"] == f], made.loc[made["seen_context"] == f, "period"], pol, f"dir.tr{f}")
                     if (made["seen_context"] == f).any() else Measured.untested("no rows")) for f in (0, 1)}
        excess = {f: (Measured(m.value - 0.5, m.lo - 0.5, m.hi - 0.5, m.n, m.status, "accuracy above chance") if m.measured else m) for f, m in parts.items()}
        transfer = _ratio(excess[0], excess[1])
    per_acc = correct.groupby(made["period"]).mean()
    return DirectionCard(accuracy, brier, skill, ece, coverage, acc_cov, frontier_finding(made, pol), transfer, _stability(per_acc, 0.5))


def frontier_finding(made: pd.DataFrame, pol: ScorecardPolicy = DEFAULT_POLICY) -> dict:
    """The conditional 80% frontier, honestly: the largest coverage at which accuracy reaches the target with a lower bound above chance,
    or the recorded finding that no such slice exists. A slice thinner than frontier_min_coverage is not evidence of anything."""
    if made.empty:
        return {"reachable": None, "note": "no calls made"}
    conf = (made["p_up"] - 0.5).abs().to_numpy()
    correct = ((made["p_up"].to_numpy() >= 0.5).astype(int) == made["up"].to_numpy()).astype(float)
    order = np.argsort(-conf, kind="mergesort")
    c_sorted = correct[order]
    cum = np.cumsum(c_sorted) / np.arange(1, len(c_sorted) + 1)
    min_n = max(int(math.ceil(pol.frontier_min_coverage * len(made))), 20)
    best = None
    for k in range(len(cum), min_n - 1, -1):
        if cum[k - 1] >= pol.frontier_target:
            se = math.sqrt(cum[k - 1] * (1 - cum[k - 1]) / k)
            if cum[k - 1] - 3.0 * se > 0.5:      # z=3: the best prefix of many is picked, so the bar is higher than 1.96
                best = (k, float(cum[k - 1]))
                break
    if best is None:
        top = float(cum[min_n - 1]) if len(cum) >= min_n else float("nan")
        return {"reachable": False, "target": pol.frontier_target, "best_accuracy_at_min_coverage": top, "min_calls": min_n,
                "note": f"no slice of at least {min_n} calls reaches {pol.frontier_target:.0%}: recorded as a scientific limitation of the available information"}
    return {"reachable": True, "target": pol.frontier_target, "coverage": best[0] / len(made), "accuracy": best[1], "calls": best[0],
            "note": "a conditional frontier exists on this sample; it must still replicate out of sample before it is believed"}


def loss_section(losses: pd.DataFrame | None, pol: ScorecardPolicy = DEFAULT_POLICY) -> LossCard:
    if losses is None or losses.empty:
        return LossCard()
    df = losses.reset_index(drop=True)
    r = df["ret"].to_numpy(dtype=float)
    k = max(1, int(math.ceil(pol.tail_frac * len(r))))
    worst = Measured.point(float(r.min()), len(r), "single worst position return")
    tail = Measured.point(float(np.sort(r)[:k].mean()), k, f"mean of the worst {pol.tail_frac:.0%}")
    fp = Measured.untested("no false_positive column")
    if "false_positive" in df.columns:
        rows = df[df["false_positive"] == 1]
        fp = _boot(rows["ret"], rows["period"], pol, "loss.fp", "mean return of positions the system picked as movers that did not move") if len(rows) \
            else Measured.untested("no false positives in the cycle")
    avoid = Measured.untested("no avoidable column")
    if "avoidable" in df.columns:
        lose = df[df["ret"] < 0]
        total = float(-lose["ret"].sum())
        avoid = Measured.point(float(-lose.loc[lose["avoidable"] == 1, "ret"].sum()) / total, len(lose), "share of loss magnitude the system's own signals could have avoided") \
            if total > 0 else Measured.undefined("no losses in the cycle")
    large = _boot((df["ret"] <= -pol.large_loss).astype(float), df["period"], pol, "loss.large", f"share of positions losing at least {pol.large_loss:.0%}")
    return LossCard(worst, tail, fp, avoid, large)


def learning_section(events: Sequence[Mapping[str, Any]], now, since=None) -> LearningCard:
    """Counts of what the cycle learned, from dated events inside [since, now). Counts are context, never a quality claim."""
    lo = as_date(since) if since is not None else None
    inside = [e for e in events if as_date(e["at"]) < as_date(now) and (lo is None or as_date(e["at"]) >= lo)]
    counts = {t: sum(1 for e in inside if e["type"] == t) for t in EVENT_TYPES}
    done, promoted, repl = counts["experiment_completed"], counts["knowledge_promoted"], counts["experiment_replicated"]
    return LearningCard(counts, Measured.point(promoted / done, done, "promoted per completed experiment") if done else Measured.untested("no experiments completed"),
                        Measured.point(repl / done, done, "replicated per completed experiment") if done else Measured.untested("no experiments completed"),
                        str(lo) if lo else "")


def efficiency_section(jobs: pd.DataFrame | None) -> EfficiencyCard:
    if jobs is None or jobs.empty:
        return EfficiencyCard()
    cost, bits = float(jobs["cost_minutes"].sum()), float(jobs["information_bits"].sum())
    changes = int(jobs["decision_changed"].sum())
    per = lambda v: Measured.point(v / cost, len(jobs), "per CPU-minute") if cost > 0 else Measured.undefined("no compute was consumed")      # noqa: E731
    return EfficiencyCard(Measured.point(cost, len(jobs), "CPU-minutes"), Measured.point(bits, len(jobs), "bits of information"), per(bits),
                          Measured.point(float(changes), len(jobs), "jobs whose result changed a decision"), per(float(changes)),
                          Measured.point(float(jobs["duplicate_avoided"].sum()), len(jobs)),
                          Measured.point(float((jobs["branch"] == "abandoned").sum()), len(jobs)), Measured.point(float((jobs["branch"] == "escalated").sum()), len(jobs)))


# ------------------------------------------------------------------------------------------------ the card
@dataclass(frozen=True)
class ResearchScorecard:
    cycle: str
    now: str
    code_hash: str
    seed: int
    policy_digest: str
    volatility: VolatilityCard
    direction: DirectionCard
    loss: LossCard
    learning: LearningCard
    efficiency: EfficiencyCard
    data_hash: str = ""
    label: ValidationLabel = ValidationLabel.NOT_VALIDATED

    def check(self) -> list[str]:
        errs = []
        if not self.cycle:
            errs.append("cycle id missing")
        if not self.code_hash:
            errs.append("code_hash missing: a scorecard must say which code produced it")
        if self.label == ValidationLabel.VALIDATED:
            errs.append("a scorecard may never carry the VALIDATED label; validation needs a sealed window")
        for sec, name, m in self.measured_fields():
            if not m.measured:
                continue
            rate = name in ("precision", "recall", "coverage", "accuracy", "stability", "avoidable_loss_rate", "large_loss_frequency", "auc", "base_rate")
            if rate and not -1e-9 <= m.value <= 1 + 1e-9 and sec != "loss":
                errs.append(f"{sec}.{name}={m.value:.4g} outside [0,1]")
            if math.isfinite(m.lo) and math.isfinite(m.hi) and not m.lo - 1e-9 <= m.value <= m.hi + 1e-9 and sec != "efficiency":
                errs.append(f"{sec}.{name}: value {m.value:.4g} outside its interval [{m.lo:.4g}, {m.hi:.4g}]")
        for k, v in self.learning.counts.items():
            if v < 0 or int(v) != v:
                errs.append(f"learning.{k} must be a non-negative integer")
        return errs

    def measured_fields(self) -> list[tuple[str, str, Measured]]:
        out = []
        for sec_name, sec in (("volatility", self.volatility), ("direction", self.direction), ("loss", self.loss), ("efficiency", self.efficiency),
                              ("learning", self.learning)):
            for f in dataclasses.fields(sec):
                v = getattr(sec, f.name)
                if isinstance(v, Measured):
                    out.append((sec_name, f.name, v))
                elif isinstance(v, Mapping) and v and all(isinstance(x, Measured) for x in v.values()):
                    out.extend((sec_name, f"{f.name}[{k}]", x) for k, x in v.items())
        return out

    def untested_fields(self) -> list[str]:
        return [f"{s}.{n}" for s, n, m in self.measured_fields() if not m.measured and m.status != MStatus.NOT_APPLICABLE]

    def as_record(self) -> dict:
        rec: dict[str, Any] = {"cycle": self.cycle, "now": self.now, "code_hash": self.code_hash, "seed": self.seed, "policy": self.policy_digest,
                               "data_hash": self.data_hash, "label": self.label.value, "counts": dict(self.learning.counts),
                               "frontier": {k: v for k, v in self.direction.frontier.items()}}
        for s, n, m in self.measured_fields():
            rec[f"{s}.{n}"] = m.as_dict()
        rec["card_id"] = stable_hash(rec)
        return rec

    @property
    def card_id(self) -> str:
        return self.as_record()["card_id"]

    def field(self, path: str) -> Measured:
        sec, name = path.split(".", 1)
        v = getattr(getattr(self, sec), name)
        if not isinstance(v, Measured):
            raise KeyError(path)
        return v


def build_scorecard(a: Artefacts, now, code_hash: str, pol: ScorecardPolicy = DEFAULT_POLICY, since=None, data_hash: str = "") -> ResearchScorecard:
    """Compute every section from the artefacts. Raises FirewallBreach on anything not yet matured and ValueError on malformed frames."""
    errs = pol.validate() + validate_artefacts(a, now)
    if errs:
        raise ValueError(f"cannot build scorecard: {errs}")
    require_matured(a, now)
    parts = [len(a.vol) if a.vol is not None else 0, len(a.direction) if a.direction is not None else 0, len(a.losses) if a.losses is not None else 0,
             len(a.jobs) if a.jobs is not None else 0, len(a.events)]
    dh = data_hash or stable_hash(parts + [str(as_date(now))], 12)
    card = ResearchScorecard("", str(as_date(now)), code_hash, pol.seed, pol.digest(), volatility_section(a.vol, pol), direction_section(a.direction, pol),
                             loss_section(a.losses, pol), learning_section(a.events, now, since), efficiency_section(a.jobs), dh)
    return dataclasses.replace(card, cycle="CYC-" + stable_hash([card.now, card.policy_digest, card.data_hash, code_hash], 10))


# ------------------------------------------------------------------------------------------------ comparing cycles
@dataclass(frozen=True)
class FieldChange:
    path: str
    old: float
    new: float
    verdict: str                 # IMPROVED | WORSE | UNCHANGED | NOT_COMPARABLE
    detail: str


def compare(old: ResearchScorecard, new: ResearchScorecard) -> list[FieldChange]:
    """Field by field, in the direction that is 'better'. A change counts only when the intervals separate; a move inside the noise is
    UNCHANGED (a lucky cycle does not become progress)."""
    out = []
    olds = {f"{s}.{n}": m for s, n, m in old.measured_fields()}
    for s, n, m in new.measured_fields():
        path = f"{s}.{n}"
        base = n.split("[")[0]
        better = HIGHER_IS_BETTER.get(s, {}).get(base)
        o = olds.get(path)
        if better is None or o is None:
            continue
        if not (o.measured and m.measured):
            out.append(FieldChange(path, o.value, m.value, "NOT_COMPARABLE", "one side is UNTESTED"))
            continue
        sep_up = math.isfinite(m.lo) and math.isfinite(o.hi) and m.lo > o.hi
        sep_dn = math.isfinite(m.hi) and math.isfinite(o.lo) and m.hi < o.lo
        if not (math.isfinite(m.lo) and math.isfinite(o.lo)):
            out.append(FieldChange(path, o.value, m.value, "NOT_COMPARABLE", "a point estimate without an interval cannot separate"))
        elif (sep_up and better) or (sep_dn and not better):
            out.append(FieldChange(path, o.value, m.value, "IMPROVED", "interval clears the old one"))
        elif (sep_dn and better) or (sep_up and not better):
            out.append(FieldChange(path, o.value, m.value, "WORSE", "interval clears the old one, in the wrong direction"))
        else:
            out.append(FieldChange(path, o.value, m.value, "UNCHANGED", "within noise"))
    return out


# ------------------------------------------------------------------------------------------------ learning from the scorecard
@dataclass(frozen=True)
class Feedback:
    problem: Problem
    delta: float                 # +raise / -lower the research priority of this problem
    reason: str
    evidence: str                # the scorecard field(s) that triggered it


def derive_feedback(card: ResearchScorecard, prev: ResearchScorecard | None = None) -> list[Feedback]:
    """Research-priority shifts from a scorecard (section 36: the system must learn from it). Rates and intervals only: a count of
    trades, patterns or experiments never produces positive feedback (section 43). Direction is not pushed until volatility has
    evidence (section 0), and a regression against the previous cycle raises the regressed problem."""
    fb: list[Feedback] = []
    v, d, lo, ef = card.volatility, card.direction, card.loss, card.efficiency
    vol_evidence = v.lift.measured and math.isfinite(v.lift.lo) and v.lift.lo > 1.0
    if not v.lift.measured:
        fb.append(Feedback(Problem.VOLATILITY, +1.0, "volatility precision is UNTESTED: the first problem has no measurement", "volatility.lift"))
    elif not vol_evidence:
        fb.append(Feedback(Problem.VOLATILITY, +1.0, f"top-pick lift {v.lift.value:.2f} (lower bound {v.lift.lo:.2f}) is not above 1: no proven mover skill", "volatility.lift"))
    if v.transfer.measured and v.transfer.value < 0.5:
        fb.append(Feedback(Problem.CONSISTENCY, +0.5, f"volatility skill on unseen contexts is {v.transfer.value:.0%} of seen: it does not transfer", "volatility.transfer"))
    if v.calibration.measured and v.brier_skill.measured and v.brier_skill.value <= 0:
        fb.append(Feedback(Problem.DATA_QUALITY, +0.5, "volatility probabilities are no better than the base rate", "volatility.brier_skill"))
    if not vol_evidence:
        fb.append(Feedback(Problem.DIRECTION, -0.5, "direction effort withheld: volatility has no evidence yet (section 0)", "volatility.lift"))
    elif d.accuracy.measured and math.isfinite(d.accuracy.lo) and d.accuracy.lo <= 0.5:
        fb.append(Feedback(Problem.DIRECTION, +0.5, f"direction accuracy {d.accuracy.value:.3f} is not above chance (lower bound {d.accuracy.lo:.3f})", "direction.accuracy"))
    if d.frontier.get("reachable") is False:
        fb.append(Feedback(Problem.COVERAGE, +0.25, "no conditional 80% slice exists: widen what is knowable before pushing accuracy", "direction.frontier"))
    if lo.avoidable_loss_rate.measured and lo.avoidable_loss_rate.value > 0.3:
        fb.append(Feedback(Problem.LOSS_AVOIDANCE, +1.0, f"{lo.avoidable_loss_rate.value:.0%} of loss magnitude was avoidable", "loss.avoidable_loss_rate"))
    if lo.large_loss_frequency.measured and math.isfinite(lo.large_loss_frequency.lo) and lo.large_loss_frequency.lo > 0.02:
        fb.append(Feedback(Problem.LOSS_AVOIDANCE, +0.5, f"large losses in {lo.large_loss_frequency.value:.1%} of positions", "loss.large_loss_frequency"))
    if ef.information_per_compute.measured and ef.decision_changes.measured and ef.decision_changes.value == 0 and ef.compute_minutes.value > 0:
        fb.append(Feedback(Problem.RESEARCH_PROCESS, +1.0, "compute was spent and no decision changed", "efficiency.decision_changes"))
    if card.learning.counts.get("memorization_failure", 0) > 0:
        fb.append(Feedback(Problem.DATA_QUALITY, +0.5, f"{card.learning.counts['memorization_failure']} memorisation failure(s) this cycle", "learning.counts"))
    if prev is not None:
        for ch in compare(prev, card):
            if ch.verdict == "WORSE":
                sec = ch.path.split(".")[0]
                prob = {"volatility": Problem.VOLATILITY, "direction": Problem.DIRECTION, "loss": Problem.LOSS_AVOIDANCE, "efficiency": Problem.EFFICIENCY}[sec]
                if prob == Problem.DIRECTION and not vol_evidence:
                    continue
                fb.append(Feedback(prob, +0.5, f"{ch.path} regressed ({ch.old:.4g} -> {ch.new:.4g})", ch.path))
    return fb


def check_no_count_rewards(fb: Sequence[Feedback]) -> list[str]:
    """Section 43 guard: no feedback may cite a raw count as its evidence."""
    return [f"feedback for {f.problem.value} cites a count: {f.evidence}" for f in fb if f.evidence.startswith("learning.counts") and f.delta > 0 and "memorisation" not in f.reason]


def apply_feedback(weights: Mapping[Problem, float], fb: Sequence[Feedback], pol: ScorecardPolicy = DEFAULT_POLICY) -> dict[Problem, float]:
    """New priority weights: each feedback moves its problem by delta * feedback_step (multiplicatively), floors keep every problem
    alive, and the result is renormalised. The section-34 order is a prior; feedback refines it and never removes a problem."""
    w = {p: max(float(weights.get(p, 0.0)), pol.feedback_floor) for p in weights}
    for f in fb:
        if f.problem in w:
            w[f.problem] = max(w[f.problem] * math.exp(f.delta * pol.feedback_step), pol.feedback_floor)
    total = sum(w.values())
    return {p: v / total for p, v in w.items()} if total > 0 else dict(weights)


# ------------------------------------------------------------------------------------------------ claims
@dataclass(frozen=True)
class ClaimVerdict:
    allowed: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    learning_gate: str = ""


def gate_claim(card: ResearchScorecard, learning_card: LS.LearningScorecard | None = None) -> ClaimVerdict:
    """May 'the system improved' be said about this cycle? Only with out-of-sample transfer measured and not collapsing in every
    predictive section, a measured stability, no unexplained loss deterioration - and, when a C62 learning scorecard is supplied, its
    own improvement-claim gate (controls A-E). Same-year gains alone never suffice (section 48)."""
    block, warn = [], []
    for sec, m in (("volatility", card.volatility.transfer), ("direction", card.direction.transfer)):
        if not m.measured:
            block.append(f"{sec} transfer to unseen contexts is UNTESTED")
        elif m.value < 0.5:
            block.append(f"{sec} keeps only {m.value:.0%} of its skill on unseen contexts")
    for sec, m in (("volatility", card.volatility.stability), ("direction", card.direction.stability)):
        if not m.measured:
            block.append(f"{sec} stability is UNTESTED")
        elif m.value < 0.5:
            block.append(f"{sec} beats its baseline in only {m.value:.0%} of periods")
    if not card.volatility.lift.measured or card.volatility.lift.lo <= 1.0:
        block.append("volatility lift over the base rate is not established")
    if not card.loss.tail_loss.measured:
        block.append("loss section is UNTESTED")
    if card.direction.frontier.get("reachable") is False:
        warn.append("the conditional 80% direction frontier is not reachable on this sample")
    warn += [f"{f} is UNTESTED" for f in card.untested_fields() if f.startswith(("volatility.calibration", "direction.calibration"))]
    gate = ""
    if learning_card is not None:
        dec = LS.gate_improvement_claim(learning_card)
        gate = "allowed" if dec.allowed else "refused"
        block.extend(f"learning gate: {b}" for b in dec.blockers)
    return ClaimVerdict(not block, tuple(block), tuple(warn), gate)


def claim_violations(text: str, verdict: ClaimVerdict) -> list[str]:
    """Words a report may not use given the verdict: 'validated'-type words never; improvement words unless the claim gate allowed."""
    low = text.lower()
    bad = [f"'{w}' is never permitted (the label stays {ValidationLabel.NOT_VALIDATED.value})" for w in FORBIDDEN_CLAIMS if w in low]
    if not verdict.allowed:
        bad += [f"'{w}' used but the claim gate refused ({verdict.blockers[0] if verdict.blockers else 'blocked'})" for w in CLAIM_WORDS if w in low]
    return bad


# ------------------------------------------------------------------------------------------------ persistence
class ScorecardLog(RP.ResearchLane):
    LANE = "rscore"

    """Append-only, hash-chained log of cycle scorecards: a cycle is written once, history is never overwritten."""

    def add(self, card: ResearchScorecard) -> str:
        errs = card.check()
        if errs:
            raise ValueError("refusing to store an invalid scorecard: " + "; ".join(errs))
        if any(r["body"]["cycle"] == card.cycle for r in self.rows()):
            raise FileExistsError(f"cycle {card.cycle} already stored: history is never overwritten")
        rec = card.as_record()
        self.append("scorecard", rec)
        return rec["card_id"]

    def records(self) -> list[dict]:
        return [r["body"] for r in self.rows() if r["kind"] == "scorecard"]

    def trajectory(self, path: str) -> list[tuple[str, str, float, float, float]]:
        """[(cycle, now, value, lo, hi)] of one field across stored cycles."""
        out = []
        for rec in self.records():
            m = rec.get(path)
            if isinstance(m, dict) and m.get("status") == "MEASURED":
                num = lambda x: float(x) if x not in (None, "nan") else float("nan")      # noqa: E731
                out.append((rec["cycle"], rec["now"], num(m["value"]), num(m["lo"]), num(m["hi"])))
        return out

    def regression_alerts(self, paths: Sequence[str] = ("volatility.lift", "direction.accuracy", "volatility.transfer", "volatility.stability")) -> list[str]:
        """Consecutive cycles where a higher-is-better field fell with separated intervals. A prompt to look, not a verdict."""
        out = []
        for p in paths:
            tr = self.trajectory(p)
            for a, b in zip(tr, tr[1:]):
                if all(math.isfinite(x) for x in (a[3], b[4])) and b[4] < a[3]:
                    out.append(f"{b[0]} is significantly worse than {a[0]} on {p}")
        return out


# ------------------------------------------------------------------------------------------------ reporting
def _fmt(m: Measured) -> str:
    if not m.measured:
        return "UNTESTED" if m.status == MStatus.UNTESTED else "n/a"
    ci = f" [{m.lo:.3g}, {m.hi:.3g}]" if math.isfinite(m.lo) and math.isfinite(m.hi) else ""
    return f"{m.value:.4g}{ci}"


def render(card: ResearchScorecard, verdict: ClaimVerdict | None = None) -> str:
    """The cycle report. UNTESTED is printed as UNTESTED. Ends with the claim statement, never with 'validated'."""
    lines = [f"# Research cycle {card.cycle} ({card.now})", f"code {card.code_hash}, seed {card.seed}, policy {card.policy_digest}", ""]
    for title, sec in (("VOLATILITY", card.volatility), ("DIRECTION", card.direction), ("LOSS", card.loss), ("RESEARCH EFFICIENCY", card.efficiency)):
        lines += [f"## {title}", "| field | value (95% interval) | note |", "|---|---|---|"]
        for f in dataclasses.fields(sec):
            v = getattr(sec, f.name)
            if isinstance(v, Measured):
                lines.append(f"| {f.name} | {_fmt(v)} | {v.note} |")
            elif isinstance(v, Mapping) and v and all(isinstance(x, Measured) for x in v.values()):
                lines += [f"| {f.name}[{k}] | {_fmt(x)} | {x.note} |" for k, x in v.items()]
        lines.append("")
    if card.direction.frontier:
        lines += ["Direction frontier: " + str(card.direction.frontier.get("note", "")), ""]
    lines += ["## LEARNING (counts are context, not a score)"] + [f"- {k}: {v}" for k, v in card.learning.counts.items()]
    lines += [f"- promotion yield: {_fmt(card.learning.promotion_yield)}", f"- replication share: {_fmt(card.learning.replication_share)}", ""]
    if verdict is not None:
        lines.append("Claim: " + ("improvement may be reported, label stays IMPLEMENTED - NOT VALIDATED." if verdict.allowed
                                  else "no claim of improvement. Blocked by: " + "; ".join(verdict.blockers[:5])))
    lines.append("IMPLEMENTED - NOT VALIDATED.")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ public entry
@dataclass(frozen=True)
class CycleReport:
    card: ResearchScorecard
    claim: ClaimVerdict
    changes: tuple[FieldChange, ...]
    feedback: tuple[Feedback, ...]
    weights: Mapping[Problem, float]
    alerts: tuple[str, ...]
    markdown: str


def step(a: Artefacts, now, code_hash: str, weights: Mapping[Problem, float], log: ScorecardLog | None = None, prev: ResearchScorecard | None = None,
         pol: ScorecardPolicy = DEFAULT_POLICY, since=None, learning_card: LS.LearningScorecard | None = None) -> CycleReport:
    """The wave-2 research loop's entry: score the matured artefacts of one cycle, compare with the previous cycle, derive the
    research-priority feedback, store the card once, and return the report with the updated priority weights."""
    card = build_scorecard(a, now, code_hash, pol, since)
    claim = gate_claim(card, learning_card)
    changes = tuple(compare(prev, card)) if prev is not None else ()
    fb = derive_feedback(card, prev)
    guard = check_no_count_rewards(fb)
    if guard:
        raise ValueError("; ".join(guard))
    new_w = apply_feedback(weights, fb, pol)
    alerts: tuple[str, ...] = ()
    if log is not None:
        log.add(card)
        alerts = tuple(log.regression_alerts())
    return CycleReport(card, claim, changes, tuple(fb), new_w, alerts, render(card, claim))


# ------------------------------------------------------------------------------------------------ breakdowns: which era, regime or type carries the number
def breakdown(df: pd.DataFrame | None, by: str, metric: str, pol: ScorecardPolicy = DEFAULT_POLICY, min_n: int = 30) -> pd.DataFrame:
    """One row per value of `by` (era, regime, sector, stock type...) with the section's headline metric, its interval and its size.
    `metric` is 'vol' (precision of the top picks) or 'dir' (accuracy of calls). A group thinner than min_n is listed, marked thin,
    and given no number: a headline that lives in one small group must be visible as such."""
    cols = ["group", "n", "value", "lo", "hi", "thin"]
    if df is None or df.empty or by not in df.columns:
        return pd.DataFrame(columns=cols)
    rows = []
    for key, g in df.groupby(by, sort=True):
        g = g.reset_index(drop=True)
        if len(g) < min_n:
            rows.append((key, len(g), float("nan"), float("nan"), float("nan"), True))
            continue
        if metric == "vol":
            pk = g[top_picks(g, "p_move", pol.select_top_frac)]
            m = _boot(pk["moved"], pk["period"], pol, f"bd.vol.{key}")
        else:
            correct = ((g["p_up"] >= 0.5).astype(int) == g["up"]).astype(float)
            m = _boot(correct, g["period"], pol, f"bd.dir.{key}")
        rows.append((key, len(g), m.value, m.lo, m.hi, False))
    return pd.DataFrame(rows, columns=cols)


def concentration(table: pd.DataFrame) -> dict:
    """Is the headline number carried by one group? Share of total rows and the best-versus-rest gap over the groups that can speak."""
    ok = table[~table["thin"] & table["value"].notna()]
    if len(ok) < 2:
        return {"groups": int(len(ok)), "concentrated": None, "note": "fewer than two groups can speak"}
    best = ok.loc[ok["value"].idxmax()]
    rest = ok.drop(ok["value"].idxmax())
    gap = float(best["value"] - np.average(rest["value"], weights=rest["n"]))
    return {"groups": int(len(ok)), "best": best["group"], "best_share_of_rows": float(best["n"] / ok["n"].sum()), "gap_to_rest": gap,
            "concentrated": bool(best["lo"] > rest["hi"].max()), "worst": ok.loc[ok["value"].idxmin(), "group"]}


def precision_curve(vol: pd.DataFrame | None, fracs: Sequence[float] = (0.02, 0.05, 0.1, 0.2, 0.4)) -> pd.DataFrame:
    """Precision and recall as the pick size grows: a card that reports only one operating point hides a ranking that collapses
    just beyond it. Lift is precision over the base rate."""
    if vol is None or vol.empty:
        return pd.DataFrame(columns=["frac", "precision", "recall", "lift"])
    df = vol.reset_index(drop=True)
    base, movers = float(df["moved"].mean()), int(df["moved"].sum())
    rows = []
    for f in fracs:
        pk = top_picks(df, "p_move", f)
        hits = int(df.loc[pk, "moved"].sum())
        prec = hits / max(int(pk.sum()), 1)
        rows.append((f, prec, hits / movers if movers else float("nan"), prec / base if base > 0 else float("nan")))
    return pd.DataFrame(rows, columns=["frac", "precision", "recall", "lift"])


def ndcg_at(vol: pd.DataFrame | None, frac: float = 0.1, pol: ScorecardPolicy = DEFAULT_POLICY) -> Measured:
    """Normalised discounted cumulative gain of each period's top picks with |move| as the gain: rewards putting the BIGGEST movers
    first, not just any mover. 1.0 = perfect ordering; what random order would score is reported in the note."""
    if vol is None or vol.empty:
        return Measured.untested("no volatility artefacts")
    vals, periods, rand = [], [], []
    for k, g in vol.groupby("period"):
        n = len(g)
        m = max(1, int(math.ceil(n * frac)))
        if n < 5 or g["abs_move"].sum() <= 0:
            continue
        disc = 1.0 / np.log2(np.arange(2, m + 2))
        got = g.sort_values("p_move", ascending=False, kind="mergesort")["abs_move"].to_numpy()[:m]
        ideal = np.sort(g["abs_move"].to_numpy())[::-1][:m]
        vals.append(float((got * disc).sum() / (ideal * disc).sum()))
        periods.append(k)
        rand.append(float((np.full(m, g["abs_move"].mean()) * disc).sum() / (ideal * disc).sum()))
    if not vals:
        return Measured.untested("no period had enough rows and movement")
    m = _boot(vals, periods, pol, "vol.ndcg")
    return dataclasses.replace(m, note=f"random order would score {float(np.mean(rand)):.3f}")


def reliability_table(p, y, n_bins: int = 10) -> pd.DataFrame:
    """The reliability diagram as a table (bin, n, mean forecast, observed frequency, gap): where the forecasts lie."""
    bins = CAL.reliability_diagram(np.asarray(p, float), np.asarray(y, float), n_bins)
    return pd.DataFrame([(b.lo, b.hi, b.n, b.mean_p, b.freq, b.gap) for b in bins], columns=["lo", "hi", "n", "mean_p", "freq", "gap"])


def calibration_drift(df: pd.DataFrame | None, p_col: str, y_col: str, dates_col: str = "matured_at", window: int = 60) -> dict:
    """Has calibration moved over time? engine.learning.calibration.confidence_drift on the forecasts in maturity order. A drift
    flagged here says the scorecard's calibration number is an average over two different regimes."""
    if df is None or len(df) < 2 * window:
        return {"drift": None, "note": f"fewer than {2 * window} forecasts"}
    d = df.sort_values(dates_col, kind="mergesort")
    rep = CAL.confidence_drift(d[p_col].to_numpy(float), d[y_col].to_numpy(float), d[dates_col].tolist(), window=window, step=max(window // 2, 1))
    return {"drift": bool(getattr(rep, "drift", False)), "report": rep}


# ------------------------------------------------------------------------------------------------ losses: where they come from
def loss_attribution(losses: pd.DataFrame | None, pol: ScorecardPolicy = DEFAULT_POLICY, top_k: int = 5) -> dict:
    """Loss magnitude by `cause` (a FailureCause value, when the loss autopsy recorded one). UNKNOWN is a legitimate cause and its share
    is reported: a scorecard whose losses are all 'explained' is more suspicious than one that admits what it does not know."""
    if losses is None or losses.empty:
        return {"n_losses": 0, "total": 0.0, "by_cause": {}, "unknown_share": float("nan"), "worst": []}
    lose = losses[losses["ret"] < 0]
    total = float(-lose["ret"].sum())
    if "cause" in lose.columns:
        cause = lose["cause"].fillna("UNKNOWN").astype(str)
        by = {c: float(-g["ret"].sum()) for c, g in lose.groupby(cause)}
    else:
        by = {"UNKNOWN": total}
    share = {c: v / total for c, v in by.items()} if total > 0 else {}
    worst = lose.nsmallest(top_k, "ret")
    cols = [c for c in ("period", "ret", "cause", "avoidable") if c in worst.columns]
    return {"n_losses": int(len(lose)), "total": total, "by_cause": dict(sorted(share.items(), key=lambda kv: -kv[1])),
            "unknown_share": share.get("UNKNOWN", 0.0), "worst": worst[cols].to_dict("records"),
            "large": int((lose["ret"] <= -pol.large_loss).sum())}


def loss_asymmetry(losses: pd.DataFrame | None) -> dict:
    """Winner/loser symmetry check (section 12): average gain of winners versus average loss of losers, and the payoff ratio. A high
    hit-rate with a payoff ratio far below 1 is the classic way to look good until one bad week."""
    if losses is None or losses.empty:
        return {"n": 0}
    r = losses["ret"].to_numpy(float)
    w, lo = r[r > 0], r[r < 0]
    if w.size == 0 or lo.size == 0:
        return {"n": int(r.size), "hit_rate": float((r > 0).mean()), "payoff_ratio": float("nan"), "expectancy": float(r.mean())}
    return {"n": int(r.size), "hit_rate": float((r > 0).mean()), "avg_win": float(w.mean()), "avg_loss": float(lo.mean()),
            "payoff_ratio": float(w.mean() / -lo.mean()), "expectancy": float(r.mean())}


# ------------------------------------------------------------------------------------------------ learning and efficiency detail
def knowledge_flow(events: Sequence[Mapping[str, Any]], now, freq: str = "M") -> pd.DataFrame:
    """Events per period and type, up to (not including) `now`. Shows whether promotion and retirement keep pace with discovery; a
    store that only ever grows is not learning, it is hoarding."""
    inside = [e for e in events if as_date(e["at"]) < as_date(now)]
    if not inside:
        return pd.DataFrame(columns=list(EVENT_TYPES))
    df = pd.DataFrame({"at": pd.to_datetime([str(as_date(e["at"])) for e in inside]), "type": [e["type"] for e in inside]})
    t = df.groupby([df["at"].dt.to_period(freq), "type"]).size().unstack(fill_value=0)
    return t.reindex(columns=list(EVENT_TYPES), fill_value=0)


def check_events(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Inconsistencies in the event log: a subject retired or promoted before it existed, or promoted twice. Events without a
    `subject` are only counted."""
    problems, born, seen_promoted = [], {}, set()
    for e in sorted(events, key=lambda e: (str(as_date(e["at"])), EVENT_TYPES.index(e["type"]) if e["type"] in EVENT_TYPES else 99)):
        s = e.get("subject")
        if s is None:
            continue
        t, at = e["type"], str(as_date(e["at"]))
        if t in ("experiment_completed", "knowledge_new"):
            born.setdefault(s, at)
        elif t in ("knowledge_retired", "knowledge_downgraded", "knowledge_promoted"):
            if s not in born:
                problems.append(f"{t} of {s} on {at} has no earlier knowledge_new/experiment_completed")
            if t == "knowledge_promoted":
                if s in seen_promoted:
                    problems.append(f"{s} promoted twice (second on {at})")
                seen_promoted.add(s)
    return problems


def diminishing_returns(jobs: pd.DataFrame | None, window: int = 10) -> dict:
    """Is more compute still buying information? Information per CPU-minute of the latest `window` jobs against the earlier ones
    (jobs in maturity order). A ratio well under 1 is the signal for the compute manager to redirect effort (section 20)."""
    if jobs is None or len(jobs) < 2 * window:
        return {"ratio": float("nan"), "diminishing": None, "note": f"fewer than {2 * window} jobs"}
    j = jobs.sort_values("matured_at", kind="mergesort")
    late, early = j.tail(window), j.iloc[:-window]

    def rate(x: pd.DataFrame) -> float:
        c = float(x["cost_minutes"].sum())
        return float(x["information_bits"].sum()) / c if c > 0 else float("nan")

    r_late, r_early = rate(late), rate(early)
    ratio = r_late / r_early if math.isfinite(r_early) and r_early > 0 else float("nan")
    return {"ratio": ratio, "late_rate": r_late, "early_rate": r_early, "diminishing": bool(ratio < 0.5) if math.isfinite(ratio) else None,
            "note": "latest jobs yield less than half the information per minute of the earlier ones" if math.isfinite(ratio) and ratio < 0.5 else ""}


def compute_waste(jobs: pd.DataFrame | None, min_bits: float = 1e-9) -> dict:
    """Share of compute spent on jobs that gained no information and changed no decision, and the families that account for it."""
    if jobs is None or jobs.empty:
        return {"wasted_share": float("nan"), "by_family": {}}
    dead = (jobs["information_bits"] <= min_bits) & (jobs["decision_changed"] == 0)
    total = float(jobs["cost_minutes"].sum())
    by = {}
    if "family" in jobs.columns and total > 0:
        for fam, g in jobs[dead].groupby("family"):
            by[str(fam)] = float(g["cost_minutes"].sum()) / total
    return {"wasted_share": float(jobs.loc[dead, "cost_minutes"].sum()) / total if total > 0 else float("nan"),
            "by_family": dict(sorted(by.items(), key=lambda kv: -kv[1])), "dead_jobs": int(dead.sum())}


# ------------------------------------------------------------------------------------------------ sanity: can the artefacts be believed?
def sanity_checks(a: Artefacts) -> list[str]:
    """Signs that an artefact frame is degenerate or counterfeit before any number is computed from it: constant forecasts, forecasts
    that are only 0 or 1, a single period, duplicated rows, a perfect record. Each finding is a reason to distrust the card."""
    out = []
    for name, pcol, ycol in (("vol", "p_move", "moved"), ("direction", "p_up", "up")):
        df = getattr(a, name)
        if df is None or df.empty:
            continue
        if df[pcol].nunique() == 1:
            out.append(f"{name}: every forecast is the same value ({float(df[pcol].iloc[0]):.3f}): no ranking exists")
        if df[pcol].isin([0.0, 1.0]).all():
            out.append(f"{name}: every forecast is exactly 0 or 1: probabilities were replaced by labels or answers")
        if df["period"].nunique() < 2:
            out.append(f"{name}: a single period: no bootstrap or stability is possible")
        if df.duplicated().any():
            out.append(f"{name}: {int(df.duplicated().sum())} duplicated row(s) inflate n")
        if len(df) >= 30 and ((df[pcol] >= 0.5).astype(int) == df[ycol]).all():
            out.append(f"{name}: forecasts match every outcome: a perfect record is a leak until shown otherwise")
        if df[ycol].nunique() < 2:
            out.append(f"{name}: the outcome never varies")
    if a.losses is not None and not a.losses.empty and (a.losses["ret"] > 0).all():
        out.append("losses: the loss frame contains no loss")
    if a.jobs is not None and not a.jobs.empty:
        if (a.jobs["cost_minutes"] < 0).any() or (a.jobs["information_bits"] < 0).any():
            out.append("jobs: negative cost or information")
    return out


def cycle_signature(card: ResearchScorecard) -> dict[str, str]:
    """One word per section (STRONG / WEAK / UNTESTED / WITHHELD): a compact answer to 'where do we stand' that never hides an UNTESTED
    section behind a number. Volatility is judged on lift and transfer, direction only when volatility has evidence."""
    v, d = card.volatility, card.direction
    vol = "UNTESTED" if not v.lift.measured else "STRONG" if v.lift.lo > 1.0 and (not v.transfer.measured or v.transfer.value >= 0.5) else "WEAK"
    if vol != "STRONG":
        dirn = "WITHHELD"
    elif not d.accuracy.measured:
        dirn = "UNTESTED"
    else:
        dirn = "STRONG" if d.accuracy.lo > 0.5 else "WEAK"
    loss = "UNTESTED" if not card.loss.tail_loss.measured else "WEAK" if card.loss.avoidable_loss_rate.measured and card.loss.avoidable_loss_rate.value > 0.3 else "STRONG"
    eff = "UNTESTED" if not card.efficiency.compute_minutes.measured else "WEAK" if card.efficiency.decision_changes.value == 0 else "STRONG"
    return {"volatility": vol, "direction": dirn, "loss": loss, "efficiency": eff}


# ------------------------------------------------------------------------------------------------ contract coverage and internal consistency
CONTRACT_FIELDS = {
    "volatility": ("precision", "recall", "ranking_quality", "calibration", "coverage", "transfer", "stability"),
    "direction": ("accuracy", "brier", "calibration", "coverage", "accuracy_at_coverage", "transfer", "stability"),
    "loss": ("worst_loss", "tail_loss", "false_positive_loss", "avoidable_loss_rate", "large_loss_frequency"),
    "efficiency": ("compute_minutes", "information_bits", "decision_changes", "duplicates_avoided", "branches_abandoned", "branches_escalated"),
}


def missing_contract_fields(card: ResearchScorecard) -> list[str]:
    """Section-36 fields the card has no home for. Every listed quantity must exist as a field, even when its value is UNTESTED: a
    missing field would silently drop a required measurement. The learning section is checked against EVENT_TYPES."""
    out = []
    for sec, names in CONTRACT_FIELDS.items():
        obj = getattr(card, sec)
        out += [f"{sec}.{n}" for n in names if not hasattr(obj, n)]
    out += [f"learning.counts[{t}]" for t in EVENT_TYPES if t not in card.learning.counts]
    return out


def internal_consistency(card: ResearchScorecard) -> list[str]:
    """Relations between fields that must hold if the card was computed from one set of artefacts. A violation means two sections were
    computed from different data, or one is wrong."""
    out = []
    v, d, lo = card.volatility, card.direction, card.loss
    if v.precision.measured and v.base_rate.measured and v.lift.measured:
        implied = v.precision.value / v.base_rate.value if v.base_rate.value > 0 else float("nan")
        if math.isfinite(implied) and abs(implied - v.lift.value) > 1e-6 * max(1.0, implied):
            out.append(f"volatility lift {v.lift.value:.4f} is not precision/base_rate = {implied:.4f}")
    if v.auc.measured and v.lift.measured and v.auc.value < 0.5 and v.lift.value > 1.5:
        out.append("volatility AUC below chance while the top picks show a large lift: the picks are not what the ranking says")
    if lo.worst_loss.measured and lo.tail_loss.measured and lo.tail_loss.value < lo.worst_loss.value - 1e-12:
        out.append("tail loss is worse than the worst loss")
    if d.accuracy_at_coverage:
        full = d.accuracy_at_coverage.get("1.00")
        if full is not None and full.measured and d.accuracy.measured and abs(full.value - d.accuracy.value) > 1e-9:
            out.append("accuracy at 100% coverage differs from the headline accuracy")
        vals = [m.value for m in d.accuracy_at_coverage.values() if m.measured]
        if len(vals) >= 3 and vals[0] < vals[-1] - 0.05:
            out.append("accuracy falls as the system becomes MORE selective: confidence carries no information")
    if d.coverage.measured and not 0.0 <= d.coverage.value <= 1.0:
        out.append("direction coverage outside [0,1]")
    ef = card.efficiency
    if ef.information_per_compute.measured and ef.compute_minutes.measured and ef.information_bits.measured and ef.compute_minutes.value > 0:
        if abs(ef.information_per_compute.value - ef.information_bits.value / ef.compute_minutes.value) > 1e-9 * max(1.0, ef.information_bits.value):
            out.append("information per compute is not information / compute")
    return out


def thin_fields(card: ResearchScorecard) -> list[str]:
    """Fields measured on fewer periods than the policy minimum (flagged 'thin' when computed): reported, never hidden."""
    return [f"{s}.{n}" for s, n, m in card.measured_fields() if m.measured and "thin" in m.note]


def rolling_precision(vol: pd.DataFrame | None, pol: ScorecardPolicy = DEFAULT_POLICY, window: int = 8) -> pd.Series:
    """Top-pick precision per period, then its rolling mean: the series a reader looks at to see whether skill is stable, decaying or
    lives in one stretch. Index is the period label in first-seen order."""
    if vol is None or vol.empty:
        return pd.Series(dtype=float)
    df = vol.reset_index(drop=True)
    pk = df[top_picks(df, "p_move", pol.select_top_frac)]
    order = pd.Index(df["period"].drop_duplicates())
    per = pk.groupby("period")["moved"].mean().reindex(order)
    return per.rolling(window, min_periods=max(2, window // 2)).mean()


def decay_check(series: pd.Series, min_points: int = 8) -> dict:
    """Trend of a per-period series: OLS slope, its t-statistic, and whether the second half is significantly below the first. A
    volatility signal that decays is the section-0 'signals that decay' item, found from the scorecard rather than from the pattern."""
    s = series.dropna()
    if len(s) < min_points:
        return {"n": int(len(s)), "decaying": None, "note": f"fewer than {min_points} periods"}
    x = np.arange(len(s), dtype=float)
    slope, icpt = np.polyfit(x, s.to_numpy(float), 1)
    resid = s.to_numpy(float) - (slope * x + icpt)
    se = math.sqrt(float((resid ** 2).sum()) / (len(s) - 2) / float(((x - x.mean()) ** 2).sum())) if len(s) > 2 else float("nan")
    t = slope / se if se and math.isfinite(se) and se > 0 else float("nan")
    half = len(s) // 2
    return {"n": int(len(s)), "slope": float(slope), "t": float(t), "first_half": float(s.iloc[:half].mean()), "second_half": float(s.iloc[half:].mean()),
            "decaying": bool(math.isfinite(t) and t < -2.0)}


def reproducible(a: Artefacts, now, code_hash: str, pol: ScorecardPolicy = DEFAULT_POLICY) -> dict:
    """Build the card twice: point estimates must be identical (they involve no randomness) and the interval bounds must be identical
    (same seed). Then build it with another seed: the point estimates must still match and the intervals must overlap. A card that
    changes when nothing changed is measuring the random number generator."""
    c1, c2 = build_scorecard(a, now, code_hash, pol), build_scorecard(a, now, code_hash, pol)
    c3 = build_scorecard(a, now, code_hash, dataclasses.replace(pol, seed=pol.seed + 1))
    same_exact = c1.card_id == c2.card_id                # hash compare: NaN != NaN in dict equality
    bad = []
    for (s, n, m1), (_, _, m3) in zip(c1.measured_fields(), c3.measured_fields()):
        if m1.measured != m3.measured:
            bad.append(f"{s}.{n}: measured status differs between seeds")
        elif m1.measured and abs(m1.value - m3.value) > 1e-9 * max(1.0, abs(m1.value)):
            bad.append(f"{s}.{n}: point estimate depends on the seed")
        elif m1.measured and all(math.isfinite(x) for x in (m1.lo, m1.hi, m3.lo, m3.hi)) and (m1.hi < m3.lo or m3.hi < m1.lo):
            bad.append(f"{s}.{n}: intervals from two seeds do not overlap")
    return {"deterministic": bool(same_exact), "seed_sensitive": bad, "ok": bool(same_exact and not bad)}


# ------------------------------------------------------------------------------------------------ history
def initial_weights(order: Sequence[Problem] | None = None, floor: float = 0.02) -> dict[Problem, float]:
    """Starting research-priority weights from the section-34 objective hierarchy (geometric decay down the order), normalised."""
    from engine.research.core import OBJECTIVE_ORDER
    seq = list(order or OBJECTIVE_ORDER)
    w = {p: max(0.5 ** i, floor) for i, p in enumerate(seq)}
    total = sum(w.values())
    return {p: v / total for p, v in w.items()}


def compare_records(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[FieldChange]:
    """`compare` for stored records (dicts from ScorecardLog), so a cycle can be compared with one written in an earlier session."""
    out = []
    for key, m in new.items():
        if not (isinstance(m, dict) and "status" in m and "." in key):
            continue
        sec, name = key.split(".", 1)
        better = HIGHER_IS_BETTER.get(sec, {}).get(name.split("[")[0])
        o = old.get(key)
        if better is None or not isinstance(o, dict):
            continue
        num = lambda x: float(x) if x not in (None, "nan") else float("nan")      # noqa: E731
        if o["status"] != "MEASURED" or m["status"] != "MEASURED":
            out.append(FieldChange(key, num(o.get("value")), num(m.get("value")), "NOT_COMPARABLE", "one side is UNTESTED"))
            continue
        ov, nv, olo, ohi, nlo, nhi = (num(o["value"]), num(m["value"]), num(o["lo"]), num(o["hi"]), num(m["lo"]), num(m["hi"]))
        if not all(math.isfinite(x) for x in (olo, ohi, nlo, nhi)):
            out.append(FieldChange(key, ov, nv, "NOT_COMPARABLE", "a point estimate without an interval cannot separate"))
        elif (nlo > ohi and better) or (nhi < olo and not better):
            out.append(FieldChange(key, ov, nv, "IMPROVED", "interval clears the old one"))
        elif (nhi < olo and better) or (nlo > ohi and not better):
            out.append(FieldChange(key, ov, nv, "WORSE", "interval clears the old one, in the wrong direction"))
        else:
            out.append(FieldChange(key, ov, nv, "UNCHANGED", "within noise"))
    return out


def history_table(log: ScorecardLog, paths: Sequence[str] = ("volatility.lift", "volatility.transfer", "direction.accuracy", "loss.tail_loss",
                                                             "efficiency.information_per_compute")) -> pd.DataFrame:
    """One row per stored cycle and one column per path (NaN where UNTESTED): the trajectory a reader scans for regressions."""
    rows = []
    for rec in log.records():
        row: dict[str, Any] = {"cycle": rec["cycle"], "now": rec["now"]}
        for p in paths:
            m = rec.get(p)
            row[p] = float(m["value"]) if isinstance(m, dict) and m.get("status") == "MEASURED" and m["value"] not in (None, "nan") else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=["cycle", "now", *paths])
