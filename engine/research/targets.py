"""Daily research target generator (C66 section 22; also 4, 21, 33, 34, 40, 41; canon C66, C63).

Section 22: every day generate research candidates from surprises, losses, wins, missed winners, missed losers, new volatility
clusters, pattern breaks, contradictions, new regimes, data anomalies, research failures, new combinations, understudied sectors,
understudied market conditions and unknown areas -- then TEST them; promote what survives out of sample, record what does not.

The contract's own example is the model case and is built exactly:

    Pattern X predicted 17 volatile stocks; 12 correct, 5 wrong; all 5 wrong in high market dispersion
    -> hypothesis "Pattern X loses directional reliability when dispersion > D" -> test out of sample -> promote or record failure.

Pieces
* `PredictionRow` / `DayInput`: the matured day, identity free (pattern ids and feature names, never tickers or dates).
* `find_failure_conditions`: scans every numeric feature for a threshold that concentrates the day's failures, with a multiplicity
  correction over features x thresholds and a binomial tail p-value; refuses to invent a condition when failures are spread.
* one generator per section-22 source, each returning `Target` records with a hypothesis, a test, and an expected-value vector.
* `TargetLedger` / `evaluate_condition`: the out-of-sample test with a fixed threshold, a placebo control, the section-42 verdict
  (PROMOTE / FAILED / NEEDS_MORE_EVIDENCE) and a permanent record of failures so a dead idea is not re-proposed.
* `rank_targets`: targets become section-2 items and are ranked by the learned priority model (engine.research.priority).

Blind-trader rule (C64/C66 sections 29-31): targets are built from matured outcomes and live in MATURED_RESEARCH_STATE; input is
refused unless it matured strictly before `now`. Public entry: `run_day(day, now, ledger)`. Builds on engine.research.priority,
engine.research.questions, engine.learning.research_priority. IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binom

from engine.learning.experiment_memory import question_key, to_ts
from engine.learning.research_priority import identity_leak
from engine.research import priority as PRI
from engine.research import questions as QST
from engine.research.core import (ExperimentValue, FirewallBreach, GateVerdict, Namespace, Problem, require_past, stable_hash)

LABEL = "IMPLEMENTED - NOT VALIDATED"
EPS = 1e-9

TARGET_SOURCES = ("surprise", "loss", "win", "missed_winner", "missed_loser", "volatility_cluster", "pattern_break", "contradiction",
                  "new_regime", "data_anomaly", "research_failure", "new_combination", "understudied_sector",
                  "understudied_condition", "unknown_area", "failure_condition")


class TargetError(ValueError):
    """A malformed input row or target."""


# ---------------------------------------------------------------------------------------------------------- day input

@dataclass(frozen=True)
class PredictionRow:
    """One prediction of the day after its outcome matured. `features` are the market/name context at decision time (numeric,
    e.g. dispersion, vix_level, breadth); `group` is a coarse identity-free bucket such as a sector code."""
    pattern: str
    correct: bool
    features: Mapping
    group: str = ""
    move: float = 0.0                               # realised move; sign matters for loss vs win
    predicted_move: bool = True

    def check(self) -> list:
        errs = []
        if not self.pattern:
            errs.append("prediction without pattern id")
        for k, v in self.features.items():
            if not isinstance(v, (int, float)) or math.isnan(float(v)):
                errs.append(f"feature {k}={v!r} is not a finite number")
        for t in (self.pattern, self.group, *self.features.keys()):
            leak = identity_leak(str(t))
            if leak:
                errs.append(f"{leak} in {str(t)!r}")
        return errs


@dataclass(frozen=True)
class DayInput:
    """Everything the matured day offers. Each optional list is a source in section 22; an empty list simply yields no targets."""
    matured_through: str
    predictions: Sequence[PredictionRow] = ()
    surprises: Sequence[Mapping] = ()               # {subject, z, stake?}
    missed_winners: Sequence[Mapping] = ()          # {subject, gain_share}
    missed_losers: Sequence[Mapping] = ()           # {subject, loss_share}
    clusters: Sequence[Mapping] = ()                # {subject, size, base_size}
    breaks: Sequence[Mapping] = ()                  # {subject, drop}
    contradictions: Sequence[Mapping] = ()          # {subject, counterpart, strength}
    regimes: Sequence[Mapping] = ()                 # {subject, shift}
    anomalies: Sequence[Mapping] = ()               # {subject, severity}
    failed_lines: Sequence[Mapping] = ()            # {subject, barren_streak}
    combinations: Sequence[Mapping] = ()            # {a, b, lift_a, lift_b, lift_joint}
    coverage: Mapping = field(default_factory=dict) # {"sector": {code: n_studied}, "condition": {name: n_studied}}
    unknown_areas: Sequence[Mapping] = ()           # {subject, gap}
    pattern_history: Sequence[PredictionRow] = ()   # earlier matured predictions (for the out-of-sample test)


@dataclass(frozen=True)
class Condition:
    """A candidate explanation of a failure cluster: feature > threshold (or < threshold)."""
    feature: str
    op: str                                         # ">" or "<"
    threshold: float
    n_in: int
    fail_in: int
    n_out: int
    fail_out: int
    p_value: float                                  # multiplicity-corrected

    def holds(self, features: Mapping) -> bool:
        if self.feature not in features:
            return False
        return features[self.feature] > self.threshold if self.op == ">" else features[self.feature] < self.threshold

    @property
    def rate_in(self) -> float:
        return self.fail_in / self.n_in if self.n_in else 0.0

    @property
    def rate_out(self) -> float:
        return self.fail_out / self.n_out if self.n_out else 0.0

    def text(self, pattern: str) -> str:
        word = "above" if self.op == ">" else "below"
        return f"Pattern {pattern} loses reliability when {self.feature} is {word} {self.threshold:.3g}"


@dataclass(frozen=True)
class Target:
    """A research candidate generated today."""
    target_id: str
    source: str
    text: str
    subject: str
    problem: Problem
    evidence_through: str
    magnitude: float
    stake: float
    hypothesis: str
    test: str
    value: ExperimentValue
    condition: Condition | None = None
    pattern: str = ""
    priority: float = 0.0

    def check(self) -> list:
        errs = []
        if self.source not in TARGET_SOURCES:
            errs.append(f"unknown source {self.source!r}")
        for f in ("text", "hypothesis", "test"):
            leak = identity_leak(getattr(self, f))
            if leak:
                errs.append(f"{self.target_id}: {f} {leak}")
        if not (0 <= self.magnitude <= 1 and 0 <= self.stake <= 1):
            errs.append(f"{self.target_id}: magnitude/stake outside [0,1]")
        return errs

    def to_item(self, created: str) -> PRI.ResearchItem:
        return PRI.ResearchItem(self.target_id, self.text, self.problem, f"target:{self.source}", self.value, created, question_id=self.target_id,
                                real_data=self.source in ("understudied_sector", "understudied_condition", "unknown_area", "new_combination"),
                                tags=(self.source,))


def _mk(source: str, subject: str, text: str, hypothesis: str, test: str, problem: Problem, evidence_through: str, magnitude: float, stake: float,
        cost: float, bits: float, loss: float | None = None, condition: Condition | None = None, pattern: str = "") -> Target:
    magnitude = min(max(magnitude, 0.0), 1.0)
    stake = min(max(stake, 0.0), 1.0)
    vol = stake * magnitude * 0.5 if problem == Problem.VOLATILITY else None
    val = PRI.make_value(cost, information_bits=bits, decision=max(stake * magnitude * 0.5, loss or 0.0), uncertainty=min(1.0, 0.2 + bits / 1.5),
                         transfer=0.4, loss=loss, volatility=vol, failure=(stake * magnitude if source in ("loss", "pattern_break", "failure_condition", "missed_loser") else None),
                         overfit=0.1)
    tid = "g_" + stable_hash([source, subject, text, evidence_through], 10)
    return Target(tid, source, text, subject, problem, evidence_through, magnitude, stake, hypothesis, test, val, condition, pattern)


# ---------------------------------------------------------------------------------------------------------- the dispersion example

def failure_rate(rows: Sequence[PredictionRow]) -> float:
    return sum(1 for r in rows if not r.correct) / len(rows) if rows else 0.0


def find_failure_conditions(rows: Sequence[PredictionRow], min_fail: int = 3, alpha: float = 0.05, n_thresholds: int = 7,
                            min_side: int = 3) -> list:
    """Which observable concentrates the failures? For each numeric feature, thresholds at interior quantiles; for each, the failure
    rate inside the region against the rate outside; a one-sided binomial tail p-value (inside failures vs the OUTSIDE rate);
    Bonferroni over every (feature, direction, threshold) tried. Returns the significant conditions best first, or [] -- with
    spread-out failures the honest answer is 'no condition'. The contract example (5 failures, all at high dispersion) passes;
    5 failures scattered across dispersion do not."""
    rows = list(rows)
    fails = [r for r in rows if not r.correct]
    if len(fails) < min_fail or len(rows) - len(fails) < 1:
        return []
    feats = sorted(set.intersection(*[set(r.features) for r in rows])) if rows else []
    tried = max(1, len(feats) * 2 * n_thresholds)
    out = []
    for f in feats:
        vals = np.array([r.features[f] for r in rows], float)
        qs = np.unique(np.quantile(vals, np.linspace(0.2, 0.8, n_thresholds)))
        for thr in qs:
            for op in (">", "<"):
                inside = [(r.features[f] > thr) if op == ">" else (r.features[f] < thr) for r in rows]
                n_in = sum(inside)
                fail_in = sum(1 for r, i in zip(rows, inside) if i and not r.correct)
                n_out = len(rows) - n_in
                fail_out = len(fails) - fail_in
                if n_in < min_side or n_out < min_side or fail_in < min_fail:
                    continue
                base = max(fail_out / n_out, 1.0 / (n_out + 2.0))          # never a zero null rate
                p = float(binom.sf(fail_in - 1, n_in, base))
                if p * tried <= alpha and fail_in / n_in > fail_out / n_out:
                    out.append(Condition(f, op, float(thr), int(n_in), int(fail_in), int(n_out), int(fail_out), float(min(1.0, p * tried))))
    return sorted(out, key=lambda c: (c.p_value, -c.rate_in, c.feature, c.op, c.threshold))


def condition_target(pattern: str, cond: Condition, evidence_through: str, loss_share: float = 0.0) -> Target:
    text = f"Does {cond.text(pattern)}, and does that hold out of sample?"
    loss = PRI.expected_loss_avoided(min(loss_share, 1.0), 0.6, 0.6, 0.6) if loss_share > 0 else None
    return _mk("failure_condition", pattern, text, cond.text(pattern),
               f"fix threshold {cond.threshold:.3g} on {cond.feature}; measure failure rate inside vs outside on unseen days and against a placebo feature",
               Problem.LOSS_AVOIDANCE, evidence_through, cond.rate_in, 0.7, 15.0, 1.0, loss, cond, pattern)


# ---------------------------------------------------------------------------------------------------------- testing a condition

@dataclass(frozen=True)
class ConditionVerdict:
    verdict: GateVerdict
    n_in: int
    n_out: int
    rate_in: float
    rate_out: float
    p_value: float
    placebo_p: float
    reason: str


def evaluate_condition(cond: Condition, pattern: str, history: Sequence[PredictionRow], now, min_in: int = 8, min_out: int = 8, alpha: float = 0.05,
                       seed: int = 0, n_placebo: int = 200) -> ConditionVerdict:
    """Out-of-sample test with the threshold FIXED from the discovery day. `history` must all be strictly before `now` and must not
    include the discovery day (the caller passes only later-matured or earlier-held-out rows; rows already used to find the
    condition would make this circular, so the caller's `matured_through` filter is the guard). Verdicts: PROMOTE when the failure
    rate inside is significantly higher than outside AND a placebo (random split of the same size) does not do as well;
    FAILED when there is enough data and no effect; NEEDS_MORE_EVIDENCE when there is not enough data either way."""
    mine = [r for r in history if r.pattern == pattern and cond.feature in r.features]
    inside = [r for r in mine if cond.holds(r.features)]
    outside = [r for r in mine if not cond.holds(r.features)]
    if len(inside) < min_in or len(outside) < min_out:
        return ConditionVerdict(GateVerdict.NEEDS_MORE_EVIDENCE, len(inside), len(outside), failure_rate(inside), failure_rate(outside), 1.0, 1.0,
                                f"{len(inside)} inside / {len(outside)} outside, need {min_in}/{min_out}")
    fi = sum(1 for r in inside if not r.correct)
    base = max(failure_rate(outside), 1.0 / (len(outside) + 2.0))
    p = float(binom.sf(fi - 1, len(inside), base))
    rng = np.random.default_rng(seed)
    flags = np.array([not r.correct for r in mine])
    n_in = len(inside)
    obs = fi / n_in - failure_rate(outside)
    hits = 0
    for _ in range(n_placebo):
        idx = rng.permutation(len(mine))
        a, b = flags[idx[:n_in]].mean(), flags[idx[n_in:]].mean()
        hits += int(a - b >= obs - 1e-12)
    pp = (hits + 1) / (n_placebo + 1)
    if p <= alpha and pp <= alpha and failure_rate(inside) > failure_rate(outside):
        return ConditionVerdict(GateVerdict.PROMOTE, len(inside), len(outside), failure_rate(inside), failure_rate(outside), p, pp,
                                "failure concentration replicated out of sample and beat the placebo split")
    if p > 0.30 or failure_rate(inside) <= failure_rate(outside):
        return ConditionVerdict(GateVerdict.FAILED, len(inside), len(outside), failure_rate(inside), failure_rate(outside), p, pp,
                                "no failure concentration out of sample")
    return ConditionVerdict(GateVerdict.NEEDS_MORE_EVIDENCE, len(inside), len(outside), failure_rate(inside), failure_rate(outside), p, pp,
                            "suggestive but not significant")


@dataclass
class TargetLedger:
    """Permanent record of what was proposed and what became of it. Failures are kept: a dead idea is not re-proposed until its
    pattern gains at least `retest_after` new observations."""
    rows: list = field(default_factory=list)        # {target_id, key, source, text, status, at, evidence_through, detail, n_obs}
    namespace: Namespace = Namespace.MATURED_RESEARCH

    def _last(self) -> dict:
        d: dict = {}
        for r in self.rows:
            d[r["key"]] = r
        return d

    def status(self, key: str) -> str:
        r = self._last().get(key)
        return r["status"] if r else "NEW"

    def record(self, t: Target, status: str, now, detail: str = "", n_obs: int = 0) -> None:
        if status not in ("PROPOSED", "PROMOTED", "FAILED", "NEEDS_MORE_EVIDENCE"):
            raise TargetError(f"unknown status {status!r}")
        self.rows.append({"target_id": t.target_id, "key": _key(t), "source": t.source, "text": t.text, "status": status, "at": str(now),
                          "evidence_through": t.evidence_through, "detail": detail, "n_obs": n_obs})

    def blocked(self, t: Target, n_obs_now: int = 0, retest_after: int = 40) -> str:
        r = self._last().get(_key(t))
        if r is None:
            return ""
        if r["status"] == "FAILED" and n_obs_now - r["n_obs"] < retest_after:
            return f"already tested and failed ({r['detail']}); needs {retest_after} new observations"
        if r["status"] in ("PROPOSED", "NEEDS_MORE_EVIDENCE"):
            return "already open"
        return ""

    def failures(self) -> list:
        return [r for r in self._last().values() if r["status"] == "FAILED"]

    def promoted(self) -> list:
        return [r for r in self._last().values() if r["status"] == "PROMOTED"]

    def to_json(self) -> str:
        return json.dumps(self.rows, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TargetLedger":
        return cls(rows=json.loads(text))


def _key(t: Target) -> str:
    return stable_hash([t.source, t.pattern or t.subject, question_key(t.hypothesis)], 12)


# ---------------------------------------------------------------------------------------------------------- one generator per source

def _rows(day: DayInput, now) -> list:
    require_past(day.matured_through, now, "day input")
    out = list(day.predictions)
    for r in out:
        errs = r.check()
        if errs:
            raise TargetError("; ".join(errs))
    return out


def targets_from_predictions(day: DayInput, now, min_rows: int = 6) -> list:
    """Losses, wins and dispersion-style failure conditions from the day's predictions. Per pattern: the failure rate, and if the
    failures concentrate in a condition, a failure_condition target with the contract's hypothesis wording."""
    rows = _rows(day, now)
    out = []
    by: dict = {}
    for r in rows:
        by.setdefault(r.pattern, []).append(r)
    total_loss = sum(abs(r.move) for r in rows if not r.correct and r.move < 0) or 0.0
    for pat, rs in sorted(by.items()):
        if len(rs) < min_rows:
            continue
        fails = [r for r in rs if not r.correct]
        loss_here = sum(abs(r.move) for r in fails if r.move < 0)
        share = loss_here / total_loss if total_loss > 0 else 0.0
        if fails:
            out.append(_mk("loss", pat, f"Why did {pat} fail on {len(fails)} of {len(rs)} predictions?",
                           f"{pat} failures share a cause", "split failures by context and era against successes", Problem.LOSS_AVOIDANCE,
                           day.matured_through, len(fails) / len(rs), 0.6, 20.0, 0.8, PRI.expected_loss_avoided(min(share, 1.0), 0.5, 0.5, 0.6), pattern=pat))
        for c in find_failure_conditions(rs)[:2]:
            out.append(condition_target(pat, c, day.matured_through, share))
        wins = [r for r in rs if r.correct]
        if wins and len(wins) / len(rs) >= 0.7:
            out.append(_mk("win", pat, f"Are the {len(wins)} wins of {pat} explained by a condition that will recur?",
                           f"{pat} wins concentrate in a recurring condition", "compare wins against failures on every recorded feature", Problem.VOLATILITY,
                           day.matured_through, len(wins) / len(rs) - 0.5, 0.4, 20.0, 0.6, pattern=pat))
    return out


def _simple(source: str, rows: Iterable[Mapping], day: DayInput, now, text_fn, hyp_fn, test: str, problem: Problem, key: str, cost: float, bits: float,
            stake: float = 0.5, loss_key: str = "") -> list:
    require_past(day.matured_through, now, source)
    out = []
    for r in rows:
        subj = str(r["subject"])
        leak = identity_leak(subj)
        if leak:
            raise FirewallBreach(f"{source} subject {subj!r} {leak}")
        mag = float(r.get(key, 0.0))
        if mag <= 0:
            continue
        loss = PRI.expected_loss_avoided(min(float(r[loss_key]), 1.0), 0.5, 0.5, 0.6) if loss_key and r.get(loss_key) else None
        out.append(_mk(source, subj, text_fn(subj, r), hyp_fn(subj, r), test, problem, day.matured_through, min(mag, 1.0), float(r.get("stake", stake)), cost, bits, loss))
    return out


def all_targets(day: DayInput, now) -> list:
    """Every section-22 source, one call. Sources with no input yield nothing."""
    t = targets_from_predictions(day, now)
    t += _surprises(day, now)
    t += _simple("missed_winner", day.missed_winners, day, now, lambda s, r: f"What information existed before the move that {s} missed?",
                 lambda s, r: f"a signal available before the move separated {s}", "date every candidate signal; keep only those before the move",
                 Problem.VOLATILITY, "gain_share", 45.0, 1.0)
    t += _simple("missed_loser", day.missed_losers, day, now, lambda s, r: f"What warned of the loss in {s} that we failed to avoid?",
                 lambda s, r: f"a signal available before the loss separated {s}", "date every candidate signal; test avoidance out of sample",
                 Problem.LOSS_AVOIDANCE, "loss_share", 35.0, 1.0, 0.7, "loss_share")
    t += _simple("volatility_cluster", day.clusters, day, now, lambda s, r: f"Is the cluster of unusual movers in {s} a new volatility structure?",
                 lambda s, r: f"{s} is a persistent structure and not a one-day coincidence", "test persistence over the next windows against shuffled dates",
                 Problem.VOLATILITY, "excess", 40.0, 0.9)
    t += _simple("pattern_break", day.breaks, day, now, lambda s, r: f"What changed before {s} stopped working?",
                 lambda s, r: f"a measurable change precedes the break of {s}", "find change points; search what changed before them",
                 Problem.LOSS_AVOIDANCE, "drop", 25.0, 1.0, 0.7, "drop")
    t += _simple("contradiction", day.contradictions, day, now, lambda s, r: f"Why do {s} and {r.get('counterpart', 'its twin')} disagree?",
                 lambda s, r: f"a condition separates {s} from {r.get('counterpart', 'its twin')}", "align on all features; find separating conditions",
                 Problem.VOLATILITY, "strength", 30.0, 1.0)
    t += _simple("new_regime", day.regimes, day, now, lambda s, r: f"Which knowledge survives the new regime {s}?",
                 lambda s, r: f"knowledge splits into survivors and casualties in {s}", "re-measure each item inside the regime",
                 Problem.CONSISTENCY, "shift", 30.0, 1.1)
    t += _simple("data_anomaly", day.anomalies, day, now, lambda s, r: f"Is the anomaly in {s} a data fault or a real change?",
                 lambda s, r: f"the {s} anomaly is a data fault", "compare against an independent source", Problem.DATA_QUALITY, "severity", 10.0, 0.8, 0.8)
    t += _simple("research_failure", day.failed_lines, day, now, lambda s, r: f"Why did repeated research on {s} produce nothing?",
                 lambda s, r: f"{s} needs a different representation, not more tuning", "screen new representations against the barren streak",
                 Problem.RESEARCH_PROCESS, "barren_norm", 60.0, 1.0)
    t += _simple("unknown_area", day.unknown_areas, day, now, lambda s, r: f"Can the gap '{r.get('gap', s)}' in {s} be closed with available data?",
                 lambda s, r: f"there is structure in {s}", "coverage probe with controls", Problem.COVERAGE, "size", 60.0, 1.0, 0.4)
    t += combination_targets(day, now)
    t += understudied_targets(day, now)
    return t


def _surprises(day: DayInput, now, z_min: float = 2.5) -> list:
    require_past(day.matured_through, now, "surprises")
    out = []
    for r in day.surprises:
        z = abs(float(r["z"]))
        if z < z_min:
            continue
        s = str(r["subject"])
        if identity_leak(s):
            raise FirewallBreach(f"surprise subject {s!r} carries an identity")
        out.append(_mk("surprise", s, f"Why did {s} deviate {z:.1f} standard deviations from expectation?",
                       f"the deviation of {s} has an observable cause known beforehand", "search observables known beforehand; shuffled control",
                       Problem.VOLATILITY, day.matured_through, 1 - math.exp(-z / 3.0), float(r.get("stake", min(1.0, z / 6.0))), 25.0, 0.9))
    return out


def combination_targets(day: DayInput, now, min_gain: float = 0.02) -> list:
    """New combinations: two signals each with a lift, and a joint lift above the better of the two (super-additive) -- or a joint
    lift well BELOW the sum (redundant), which is also worth knowing. Only the surprising ones become targets."""
    require_past(day.matured_through, now, "combinations")
    out = []
    for r in day.combinations:
        a, b, j = str(r["a"]), str(r["b"]), float(r["lift_joint"])
        best = max(float(r["lift_a"]), float(r["lift_b"]))
        add = float(r["lift_a"]) + float(r["lift_b"])
        if j - best >= min_gain:
            kind, mag = "super-additive", min(1.0, (j - best) / max(best, 0.05))
        elif add - j >= 2 * min_gain:
            kind, mag = "redundant", min(1.0, (add - j) / max(add, 0.05))
        else:
            continue
        out.append(_mk("new_combination", f"{a}+{b}", f"Is the {kind} combination of {a} and {b} real?", f"{a} and {b} interact ({kind})",
                       "test the interaction out of sample against the additive model", Problem.VOLATILITY, day.matured_through, mag, 0.5, 40.0, 0.9))
    return out


def understudied_targets(day: DayInput, now, floor_ratio: float = 0.34) -> list:
    """Understudied sectors and market conditions: a bucket whose number of past studies is below `floor_ratio` of the median across
    buckets (and below the median by at least 2) becomes a target, the emptier the more so. Buckets never studied are the strongest."""
    require_past(day.matured_through, now, "coverage")
    out = []
    for kind, source in (("sector", "understudied_sector"), ("condition", "understudied_condition")):
        table = dict(day.coverage.get(kind, {}))
        if len(table) < 3:
            continue
        med = float(np.median(list(table.values())))
        for name, n in sorted(table.items()):
            if identity_leak(name):
                raise FirewallBreach(f"coverage bucket {name!r} carries an identity")
            if n <= floor_ratio * med and med - n >= 2:
                gap = 1.0 - n / max(med, 1.0)
                out.append(_mk(source, name, f"What is known about {kind} {name}, which has been studied {int(n)} times against a median of {med:.0f}?",
                               f"{kind} {name} holds structure the studied buckets do not", "run the standard probe there with controls",
                               Problem.COVERAGE, day.matured_through, gap, 0.4, 45.0, 0.8))
    return out


# ---------------------------------------------------------------------------------------------------------- the daily run

@dataclass(frozen=True)
class DayReport:
    now: str
    targets: tuple                                  # Target, ranked, blocked ones removed
    blocked: tuple                                  # (target id, reason)
    by_source: Mapping
    promoted: tuple                                 # (target id, ConditionVerdict) from tests run today
    failed: tuple
    open: tuple


def rank_targets(targets: Sequence[Target], now, state: PRI.PriorityState | None = None) -> list:
    """Order targets by the learned priority model (or the conceptual formula until one exists). Ties break by id."""
    if not targets:
        return []
    state = state or PRI.new_state()
    items = [t.to_item(t.evidence_through) for t in targets]
    ranked = PRI.rank(state, items, now)
    by_id = {t.target_id: t for t in targets}
    return [replace(by_id[r.item.item_id], priority=r.rate) for r in ranked if not r.blocked]


def run_day(day: DayInput, now, ledger: TargetLedger | None = None, state: PRI.PriorityState | None = None, seed: int = 0) -> DayReport:
    """THE public entry. Generate every target from the matured day, drop those the ledger says are dead or already open, TEST the
    failure-condition targets against the pattern history (out of sample, fixed threshold), record promote / fail / open, and
    rank what remains by priority. A promoted condition is the section-22 'promote knowledge'; a failed one is 'record failure'."""
    ledger = ledger if ledger is not None else TargetLedger()
    made = all_targets(day, now)
    for t in made:
        errs = t.check()
        if errs:
            raise TargetError("; ".join(errs))
    n_obs = {}
    for r in day.pattern_history:
        n_obs[r.pattern] = n_obs.get(r.pattern, 0) + 1
    keep, blocked, promoted, failed, opened = [], [], [], [], []
    for t in made:
        why = ledger.blocked(t, n_obs.get(t.pattern, 0))
        if why:
            blocked.append((t.target_id, why))
            continue
        if t.condition is not None and t.pattern:
            v = evaluate_condition(t.condition, t.pattern, day.pattern_history, now, seed=seed)
            if v.verdict == GateVerdict.PROMOTE:
                ledger.record(t, "PROMOTED", now, v.reason, n_obs.get(t.pattern, 0))
                promoted.append((t.target_id, v))
                continue
            if v.verdict == GateVerdict.FAILED:
                ledger.record(t, "FAILED", now, v.reason, n_obs.get(t.pattern, 0))
                failed.append((t.target_id, v))
                continue
            ledger.record(t, "NEEDS_MORE_EVIDENCE", now, v.reason, n_obs.get(t.pattern, 0))
            opened.append(t.target_id)
        else:
            ledger.record(t, "PROPOSED", now, "", n_obs.get(t.pattern, 0))
            opened.append(t.target_id)
        keep.append(t)
    ranked = rank_targets(keep, now, state)
    mix: dict = {}
    for t in made:
        mix[t.source] = mix.get(t.source, 0) + 1
    return DayReport(str(now), tuple(ranked), tuple(blocked), mix, tuple(promoted), tuple(failed), tuple(opened))


def source_coverage(reports: Sequence[DayReport]) -> dict:
    """Over many days, which section-22 sources ever produced a target? A source that never fires is either empty of events or
    disconnected from its input; the health system should know which."""
    seen = {s: 0 for s in TARGET_SOURCES}
    for r in reports:
        for s, n in r.by_source.items():
            seen[s] = seen.get(s, 0) + n
    return {"per_source": seen, "silent": sorted(s for s, n in seen.items() if n == 0)}


def dispersion_example(seed: int = 0, n_correct: int = 12, n_wrong: int = 5, cutoff: float = 1.5) -> DayInput:
    """The contract's example as data: pattern X predicted 17, 12 correct, 5 wrong, all 5 wrong when dispersion was high, with a
    matured history in which the same rule holds (so the out-of-sample test can promote it)."""
    rng = np.random.default_rng(seed)

    def make(n_c, n_w):
        rows = [PredictionRow("pattern_x", True, {"dispersion": float(rng.uniform(0.3, cutoff * 0.9)), "breadth": float(rng.uniform(0, 1))}, "g1", 0.06) for _ in range(n_c)]
        rows += [PredictionRow("pattern_x", False, {"dispersion": float(rng.uniform(cutoff * 1.1, cutoff * 1.6)), "breadth": float(rng.uniform(0, 1))}, "g1", -0.07) for _ in range(n_w)]
        return rows
    hist = []
    for _ in range(12):
        hist += make(int(rng.integers(8, 14)), int(rng.integers(3, 6)))
    return DayInput("2003-05-01", predictions=make(n_correct, n_wrong), pattern_history=hist)


# ---------------------------------------------------------------------------------------------------------- per-source scoring

@dataclass(frozen=True)
class SourceScore:
    """How big and how valuable a raw row from one section-22 source is, on the source's OWN scale. Each source has its own
    normalisation because a 'magnitude' means different things: a z-score, a loss share, a Poisson excess."""
    source: str
    magnitude: float                                # [0,1] how strong the event is
    stake: float                                    # [0,1] decision value riding on the answer
    bits: float                                     # expected information of the standard test
    cost: float                                     # cpu-minutes of the standard test
    problem: Problem
    confidence: float                               # [0,1] how sure we are the event is not noise
    reasons: tuple = ()


def _sat(x: float, scale: float) -> float:
    return 1.0 - math.exp(-max(x, 0.0) / scale)


def _pois_tail_conf(size: float, base: float) -> float:
    """1 - P(Poisson(base) >= size): how surprising a cluster of `size` is when `base` was expected."""
    from scipy.stats import poisson
    if base <= 0:
        return 1.0 if size > 0 else 0.0
    return float(1.0 - poisson.sf(max(size - 1, 0), base))


def score_surprise(row: Mapping) -> SourceScore:
    z = abs(float(row["z"]))
    conf = 1.0 - 2.0 * (1.0 - _norm_cdf(z))                   # two-sided tail: how unlikely under the null
    return SourceScore("surprise", _sat(z, 3.0), float(row.get("stake", min(1.0, z / 6.0))), 0.9, 25.0, Problem.VOLATILITY, max(0.0, conf),
                       (f"z={z:.1f}",))


def score_loss(row: Mapping) -> SourceScore:
    share = float(row.get("loss_share", 0.0))
    conf = float(row.get("model_confidence", 0.5))
    return SourceScore("loss", min(1.0, 2 * share), min(1.0, 0.4 + 0.6 * conf), 0.8, 20.0, Problem.LOSS_AVOIDANCE, min(1.0, 0.3 + share),
                       (f"loss share {share:.0%}", f"model confidence {conf:.0%}"))


def score_win(row: Mapping) -> SourceScore:
    """A win is worth studying when the win RATE is high relative to base and the sample is real, not when a single trade did well."""
    n, k = int(row.get("n", 0)), int(row.get("wins", 0))
    base = float(row.get("base_rate", 0.5))
    if n < 5:
        return SourceScore("win", 0.0, 0.2, 0.3, 15.0, Problem.VOLATILITY, 0.0, ("too few to study",))
    from scipy.stats import binom
    p = float(binom.sf(k - 1, n, base))
    return SourceScore("win", min(1.0, max(0.0, k / n - base) * 2), 0.4, 0.6, 20.0, Problem.VOLATILITY, 1.0 - p, (f"{k}/{n} vs base {base:.2f}",))


def score_missed_winner(row: Mapping) -> SourceScore:
    gain = float(row.get("gain_share", 0.0))
    know = float(row.get("knowable_before", 0.5))
    return SourceScore("missed_winner", min(1.0, gain) * (0.2 + 0.8 * know), 0.3 + 0.5 * know, 1.0, 45.0, Problem.VOLATILITY, know,
                       (f"gain share {gain:.0%}", f"knowable {know:.0%}"))


def score_missed_loser(row: Mapping) -> SourceScore:
    loss = float(row.get("loss_share", 0.0))
    know = float(row.get("knowable_before", 0.5))
    return SourceScore("missed_loser", min(1.0, 2 * loss) * (0.2 + 0.8 * know), 0.5 + 0.4 * know, 1.0, 35.0, Problem.LOSS_AVOIDANCE, know,
                       (f"loss share {loss:.0%}", f"knowable {know:.0%}"))


def score_cluster(row: Mapping) -> SourceScore:
    size, base = float(row.get("size", 0)), float(row.get("base_size", 1.0))
    conf = _pois_tail_conf(size, base)
    return SourceScore("volatility_cluster", min(1.0, (size - base) / max(size, 1.0)), 0.6, 0.9, 40.0, Problem.VOLATILITY, conf, (f"{size:.0f} movers vs {base:.1f} expected",))


def score_break(row: Mapping) -> SourceScore:
    drop, before, n = float(row.get("drop", 0.0)), float(row.get("before", 0.6)), int(row.get("n_after", 0))
    conf = min(1.0, n / 40.0) * min(1.0, drop / 0.2)
    return SourceScore("pattern_break", min(1.0, drop / max(before, 0.1)), min(1.0, 0.5 + drop), 1.0, 25.0, Problem.LOSS_AVOIDANCE, conf,
                       (f"reliability fell {drop:.2f}", f"n after {n}"))


def score_contradiction(row: Mapping) -> SourceScore:
    s = float(row.get("strength", 0.0))
    n = int(row.get("n", 20))
    return SourceScore("contradiction", min(1.0, s), 0.5, 1.0, 30.0, Problem.VOLATILITY, min(1.0, n / 40.0) * min(1.0, s / 0.2), (f"disagreement {s:.2f}",))


def score_regime(row: Mapping) -> SourceScore:
    sh = float(row.get("shift", 0.0))
    exposed = int(row.get("n_items", 1))
    return SourceScore("new_regime", min(1.0, sh), min(1.0, 0.3 + 0.1 * exposed), 1.1, 30.0, Problem.CONSISTENCY, min(1.0, sh * 1.5), (f"shift {sh:.2f}", f"{exposed} items exposed"))


def score_anomaly(row: Mapping) -> SourceScore:
    sev = float(row.get("severity", 0.0))
    return SourceScore("data_anomaly", sev, 0.8, 0.8, 10.0, Problem.DATA_QUALITY, min(1.0, 0.5 + sev / 2), (f"severity {sev:.2f}",))


def score_research_failure(row: Mapping) -> SourceScore:
    streak = int(row.get("barren_streak", 0))
    return SourceScore("research_failure", _sat(streak, 4.0), 0.5, 1.0, 60.0, Problem.RESEARCH_PROCESS, min(1.0, streak / 6.0), (f"{streak} barren results",))


def score_combination(row: Mapping) -> SourceScore:
    a, b, j = float(row["lift_a"]), float(row["lift_b"]), float(row["lift_joint"])
    best = max(a, b)
    if j - best >= 0.02:
        mag, why = min(1.0, (j - best) / max(best, 0.05)), "super-additive"
    elif a + b - j >= 0.04:
        mag, why = min(1.0, (a + b - j) / max(a + b, 0.05)), "redundant"
    else:
        mag, why = 0.0, "additive"
    return SourceScore("new_combination", mag, 0.5, 0.9, 40.0, Problem.VOLATILITY, 0.5 if mag else 0.0, (why,))


def score_understudied(kind: str, n: float, median: float) -> SourceScore:
    gap = 1.0 - n / max(median, 1.0)
    return SourceScore(f"understudied_{kind}", max(0.0, gap), 0.4, 0.8, 45.0, Problem.COVERAGE, min(1.0, median / 20.0), (f"{n:.0f} studies vs median {median:.0f}",))


def score_unknown(row: Mapping) -> SourceScore:
    size = float(row.get("size", 0.0))
    return SourceScore("unknown_area", min(1.0, size), 0.4, 1.0, 60.0, Problem.COVERAGE, 0.3, (str(row.get("gap", "unresolved gap")),))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


SCORERS = {"surprise": score_surprise, "loss": score_loss, "win": score_win, "missed_winner": score_missed_winner, "missed_loser": score_missed_loser,
           "volatility_cluster": score_cluster, "pattern_break": score_break, "contradiction": score_contradiction, "new_regime": score_regime,
           "data_anomaly": score_anomaly, "research_failure": score_research_failure, "new_combination": score_combination, "unknown_area": score_unknown}


def score_row(source: str, row: Mapping) -> SourceScore:
    """Route a raw row to its source's scorer. A source with no scorer (failure_condition, understudied_*) is scored where it is made."""
    if source not in SCORERS:
        raise TargetError(f"no row scorer for source {source!r}")
    sc = SCORERS[source](row)
    for name in ("magnitude", "stake", "confidence"):
        v = getattr(sc, name)
        if not (0.0 <= v <= 1.0) or math.isnan(v):
            raise TargetError(f"{source}: score {name}={v!r} outside [0,1]")
    return sc


def target_from_score(sc: SourceScore, subject: str, text: str, hypothesis: str, test: str, evidence_through: str) -> Target:
    """Turn a scored row into a Target, with confidence discounting the magnitude: an event we are unsure is real is worth less."""
    return _mk(sc.source, subject, text, hypothesis, test, sc.problem, evidence_through, sc.magnitude * (0.4 + 0.6 * sc.confidence), sc.stake, sc.cost, sc.bits)


# ---------------------------------------------------------------------------------------------------------- learning which sources pay

@dataclass
class SourceYield:
    """Per-source yield learning: of the targets a source produced, how many led to promoted knowledge or a decision change? Beta
    posteriors with a pessimistic prior; the multiplier is bounded so the section-22 breadth is kept (every source keeps a floor)."""
    ok: dict = field(default_factory=dict)
    tried: dict = field(default_factory=dict)
    log: list = field(default_factory=list)          # (at, source, useful)
    prior_a: float = 1.0
    prior_b: float = 3.0

    def observe(self, source: str, useful: bool, at, now) -> None:
        if source not in TARGET_SOURCES:
            raise TargetError(f"unknown source {source!r}")
        require_past(at, now, f"yield of {source}")
        self.tried[source] = self.tried.get(source, 0) + 1
        self.ok[source] = self.ok.get(source, 0) + int(useful)
        self.log.append((str(at), source, bool(useful)))

    def rate(self, source: str) -> float:
        return (self.prior_a + self.ok.get(source, 0)) / (self.prior_a + self.prior_b + self.tried.get(source, 0))

    def mean_rate(self) -> float:
        return float(np.mean([self.rate(s) for s in TARGET_SOURCES]))

    def multiplier(self, source: str, floor: float = 0.5, cap: float = 2.0) -> float:
        n = self.tried.get(source, 0)
        w = n / (n + 8.0)
        return float(min(cap, max(floor, (1 - w) + w * self.rate(source) / self.mean_rate())))

    def report(self) -> dict:
        return {s: {"tried": self.tried.get(s, 0), "useful": self.ok.get(s, 0), "rate": self.rate(s), "multiplier": self.multiplier(s)} for s in TARGET_SOURCES}

    def to_json(self) -> str:
        return json.dumps({"ok": self.ok, "tried": self.tried, "log": self.log}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "SourceYield":
        d = json.loads(text)
        return cls(ok=d["ok"], tried=d["tried"], log=[tuple(x) for x in d["log"]])


def apply_source_yield(targets: Sequence[Target], sy: SourceYield) -> list:
    """Scale each target's priority by its source's learned yield and re-sort."""
    out = [replace(t, priority=t.priority * sy.multiplier(t.source)) for t in targets]
    return sorted(out, key=lambda t: (-t.priority, t.target_id))


# ---------------------------------------------------------------------------------------------------------- coverage models

def condition_bucket(features: Mapping, edges: Mapping) -> str:
    """Name a market condition from numeric features and per-feature quantile edges {feature: (lo_edge, hi_edge)}: e.g.
    'dispersion=high|breadth=low'. Identity free, so it can be stored and queried."""
    parts = []
    for f in sorted(edges):
        if f not in features:
            continue
        lo, hi = edges[f]
        v = features[f]
        parts.append(f"{f}={'low' if v < lo else 'high' if v > hi else 'mid'}")
    return "|".join(parts) or "unconditioned"


def quantile_edges(history: Sequence[Mapping], features: Sequence[str], q: tuple = (1 / 3, 2 / 3)) -> dict:
    """Terciles of each feature from PAST rows only (the caller passes rows matured before now)."""
    out = {}
    for f in features:
        v = np.array([h[f] for h in history if f in h], float)
        if v.size >= 9:
            out[f] = (float(np.quantile(v, q[0])), float(np.quantile(v, q[1])))
    return out


@dataclass
class CoverageModel:
    """What has been studied. Each study is a (date, bucket) event; counts decay with `half_life_days` so a bucket studied years ago
    counts as partly unstudied again. A bucket's need = how far its decayed count is below the fair share, and unseen buckets are
    always needy. This replaces the flat median rule with a model that knows about age, unequal opportunity (`exposure`) and total
    effort (`gini`)."""
    kind: str
    half_life_days: float = 365.0
    studies: list = field(default_factory=list)       # (date, bucket, weight)
    exposure: dict = field(default_factory=dict)      # bucket -> how often it occurs in the market (share); missing = equal
    known: set = field(default_factory=set)

    def register(self, buckets: Iterable[str], exposure: Mapping | None = None) -> None:
        for b in buckets:
            if identity_leak(b):
                raise FirewallBreach(f"coverage bucket {b!r} carries an identity")
            self.known.add(b)
        if exposure:
            self.exposure.update({k: float(v) for k, v in exposure.items()})

    def study(self, bucket: str, when, now, weight: float = 1.0) -> None:
        require_past(when, now, f"study of {bucket}")
        self.register([bucket])
        self.studies.append((str(when), bucket, float(weight)))

    def decayed_counts(self, now) -> dict:
        out = {b: 0.0 for b in self.known}
        for d, b, w in self.studies:
            age = (to_ts(now) - to_ts(d)).total_seconds() / 86400.0
            if age < 0:
                continue
            out[b] += w * 0.5 ** (age / self.half_life_days)
        return out

    def fair_share(self) -> dict:
        tot = sum(self.exposure.get(b, 1.0) for b in self.known) or 1.0
        return {b: self.exposure.get(b, 1.0) / tot for b in self.known}

    def need(self, now) -> dict:
        """Per bucket, in [0,1]: 1 = completely unstudied for its fair share, 0 = at or above it."""
        c = self.decayed_counts(now)
        tot = sum(c.values())
        fair = self.fair_share()
        if tot <= 0:
            return {b: 1.0 for b in self.known}
        return {b: max(0.0, 1.0 - (c[b] / tot) / max(fair[b], 1e-9)) for b in sorted(self.known)}

    def understudied(self, now, min_need: float = 0.6) -> list:
        n = self.need(now)
        return sorted(((b, v) for b, v in n.items() if v >= min_need), key=lambda kv: (-kv[1], kv[0]))

    def gini(self, now) -> float:
        c = np.sort(np.array(list(self.decayed_counts(now).values())))
        if c.size == 0 or c.sum() <= 0:
            return 0.0
        n = c.size
        return float((2 * np.sum((np.arange(1, n + 1)) * c) / (n * c.sum())) - (n + 1) / n)

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "hl": self.half_life_days, "studies": self.studies, "exposure": self.exposure, "known": sorted(self.known)}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "CoverageModel":
        d = json.loads(text)
        return cls(d["kind"], d["hl"], [tuple(x) for x in d["studies"]], d["exposure"], set(d["known"]))


def coverage_targets(model: CoverageModel, now, evidence_through: str, min_need: float = 0.6) -> list:
    """Targets for the needy buckets. Source name follows the model's kind (sector / condition)."""
    require_past(evidence_through, now, "coverage")
    src = f"understudied_{model.kind}"
    if src not in TARGET_SOURCES:
        raise TargetError(f"coverage kind {model.kind!r} has no target source")
    out = []
    for b, need in model.understudied(now, min_need):
        sc = SourceScore(src, need, 0.4, 0.8, 45.0, Problem.COVERAGE, min(1.0, len(model.studies) / 20.0), (f"need {need:.2f}",))
        out.append(target_from_score(sc, b, f"What is known about {model.kind} {b}, which is under-studied for its share of the market?",
                                     f"{model.kind} {b} holds structure the studied buckets do not", "run the standard probe there with controls", evidence_through))
    return out


# ---------------------------------------------------------------------------------------------------------- multi-day tracking

@dataclass(frozen=True)
class DatedRow:
    """A prediction row with the day its outcome matured (the tracker needs it to use each day once and never early)."""
    day: str
    row: PredictionRow


class TrackState:
    TRACKING = "TRACKING"
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    EXPIRED = "EXPIRED"


@dataclass
class TrackedCondition:
    """A condition seen on day 1 and followed as evidence accumulates. The threshold is FROZEN at discovery. Evidence is a sequential
    probability ratio test on the failures INSIDE the region: H1 = failure rate stays at the (shrunk) discovery rate, H0 = it is the
    outside rate. Confirming needs the log-likelihood ratio above ln((1-beta)/alpha); refuting needs it below ln(beta/(1-alpha));
    otherwise it keeps waiting until `max_days`, then EXPIRES as undecided (which is not a failure)."""
    tid: str
    pattern: str
    cond: Condition
    opened: str
    p1: float
    max_days: int = 120
    alpha: float = 0.05
    beta: float = 0.20
    n_in: int = 0
    fail_in: int = 0
    n_out: int = 0
    fail_out: int = 0
    llr: float = 0.0
    days_seen: set = field(default_factory=set)
    state: str = TrackState.TRACKING
    closed: str = ""
    history: list = field(default_factory=list)      # (day, llr, n_in)

    min_out: int = 10

    def p0(self) -> float:
        """Null failure rate: the outside rate measured DURING tracking, with the discovery day's outside counts as a weak prior
        (weight 0.2: the discovery sample was chosen because it looked extreme, so it is not trusted at face value)."""
        k = self.fail_out + 0.2 * self.cond.fail_out + 1.0
        n = self.n_out + 0.2 * self.cond.n_out + 2.0
        return min(max(k / n, 0.02), 0.98)

    def _bounds(self) -> tuple:
        return math.log((1 - self.beta) / self.alpha), math.log(self.beta / (1 - self.alpha))

    def recompute(self) -> float:
        """LLR of H1 (inside failure rate = p1) against H0 (= current outside rate), from the running counts. Recomputed from counts
        each day so an early, poorly-estimated null rate cannot leave a permanent mark on the evidence."""
        p0, p1 = self.p0(), min(max(self.p1, 0.02), 0.98)
        k, n = self.fail_in, self.n_in
        return k * math.log(p1 / p0) + (n - k) * math.log((1 - p1) / (1 - p0))

    def add_day(self, day: str, rows: Sequence[PredictionRow]) -> None:
        if self.state != TrackState.TRACKING or day in self.days_seen:
            return
        self.days_seen.add(day)
        for r in rows:
            if r.pattern != self.pattern or self.cond.feature not in r.features:
                continue
            fail = not r.correct
            if self.cond.holds(r.features):
                self.n_in += 1
                self.fail_in += int(fail)
            else:
                self.n_out += 1
                self.fail_out += int(fail)
        self.llr = self.recompute()
        self.history.append((day, self.llr, self.n_in))
        hi, lo = self._bounds()
        decidable = self.n_out >= self.min_out and self.p1 > self.p0()
        if decidable and self.llr >= hi:
            self.state, self.closed = TrackState.CONFIRMED, day
        elif self.n_out >= self.min_out and self.llr <= lo:
            self.state, self.closed = TrackState.REFUTED, day
        elif (to_ts(day) - to_ts(self.opened)).days >= self.max_days:
            self.state, self.closed = TrackState.EXPIRED, day

    def progress(self) -> float:
        """How far the LLR is toward confirmation (0 at start, 1 at the upper bound, negative toward refutation)."""
        hi, _ = self._bounds()
        return self.llr / hi


class OpenConditionTracker:
    """Follows every open failure-condition across days. `update(day_rows, now)` feeds each new matured day to every tracked
    condition exactly once; conditions close on evidence, never by the calendar alone (expiry is UNDECIDED)."""

    def __init__(self, max_days: int = 120):
        self.max_days = max_days
        self.items: dict = {}
        self.events: list = []                       # (day, tid, event)

    def open(self, t: Target, day: str, now) -> TrackedCondition | None:
        if t.condition is None or not t.pattern:
            return None
        require_past(day, now, "tracked condition")
        tid = "tc_" + stable_hash([t.pattern, t.condition.feature, t.condition.op, round(t.condition.threshold, 6)], 10)
        if tid in self.items:
            return self.items[tid]
        c = t.condition
        p1 = 0.5 * (c.rate_in + max(c.rate_out, 0.0)) if c.rate_in > c.rate_out else c.rate_in
        tc = TrackedCondition(tid, t.pattern, c, str(day), p1, self.max_days)
        self.items[tid] = tc
        self.events.append((str(day), tid, "opened"))
        return tc

    def update(self, rows_by_day: Sequence[DatedRow], now) -> list:
        """Feed matured days to every open condition. Rows dated on/after `now` or on/before the condition's opening day are refused
        for that condition (the opening day was the discovery sample: reusing it would be circular)."""
        by_day: dict = {}
        for dr in rows_by_day:
            require_past(dr.day, now, "tracked row")
            by_day.setdefault(dr.day, []).append(dr.row)
        changed = []
        for tid, tc in sorted(self.items.items()):
            if tc.state != TrackState.TRACKING:
                continue
            for day in sorted(by_day):
                if to_ts(day) <= to_ts(tc.opened):
                    continue
                before = tc.state
                tc.add_day(day, by_day[day])
                if tc.state != before:
                    self.events.append((day, tid, tc.state))
                    changed.append(tid)
                    break
        return changed

    def by_state(self, state: str) -> list:
        return sorted(t.tid for t in self.items.values() if t.state == state)

    def summary(self) -> dict:
        return {s: len(self.by_state(s)) for s in (TrackState.TRACKING, TrackState.CONFIRMED, TrackState.REFUTED, TrackState.EXPIRED)}

    def stale(self, now, days: int = 60) -> list:
        """Tracked conditions with no new inside observation for `days` days: waiting on data that is not arriving."""
        out = []
        for tc in self.items.values():
            if tc.state == TrackState.TRACKING:
                last = tc.history[-1][0] if tc.history else tc.opened
                if (to_ts(now) - to_ts(last)).days >= days:
                    out.append(tc.tid)
        return sorted(out)

    def to_json(self) -> str:
        rows = []
        for tid, t in sorted(self.items.items()):
            d = {k: getattr(t, k) for k in ("tid", "pattern", "opened", "p1", "max_days", "alpha", "beta", "n_in", "fail_in", "n_out", "fail_out", "llr",
                                           "state", "closed", "history")}
            d["cond"] = [t.cond.feature, t.cond.op, t.cond.threshold, t.cond.n_in, t.cond.fail_in, t.cond.n_out, t.cond.fail_out, t.cond.p_value]
            d["days_seen"] = sorted(t.days_seen)
            rows.append(d)
        return json.dumps({"max_days": self.max_days, "items": rows, "events": self.events}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "OpenConditionTracker":
        d = json.loads(text)
        tr = cls(d["max_days"])
        for r in d["items"]:
            cond = Condition(*r["cond"])
            tc = TrackedCondition(r["tid"], r["pattern"], cond, r["opened"], r["p1"], r["max_days"], r["alpha"], r["beta"], r["n_in"], r["fail_in"],
                                  r["n_out"], r["fail_out"], r["llr"], set(r["days_seen"]), r["state"], r["closed"], [tuple(h) for h in r["history"]])
            tr.items[tc.tid] = tc
        tr.events = [tuple(e) for e in d["events"]]
        return tr


def expected_days_to_confirm(cond: Condition, inside_per_day: float, alpha: float = 0.05, beta: float = 0.20) -> float:
    """Planning number: expected days until the SPRT confirms a TRUE condition, from the expected LLR drift per inside observation
    (Kullback-Leibler between the discovery-rate and the outside-rate Bernoulli). infinity when there is no drift."""
    p0 = min(max(cond.rate_out, 0.02), 0.98)
    p1 = min(max(0.5 * (cond.rate_in + cond.rate_out), 0.02), 0.98)
    if p1 <= p0 or inside_per_day <= 0:
        return math.inf
    kl = p1 * math.log(p1 / p0) + (1 - p1) * math.log((1 - p1) / (1 - p0))
    return math.log((1 - beta) / alpha) / (kl * inside_per_day)


# ---------------------------------------------------------------------------------------------------------- autopsy record (R04 feed)

@dataclass(frozen=True)
class AutopsyEntry:
    """One line of the daily market autopsy. `category` is the section-4 MoveCategory; `subject` an identity-free situation key."""
    day: str
    category: str
    subject: str
    move: float
    pattern: str = ""
    features: Mapping = field(default_factory=dict)
    predicted: bool = False
    correct: bool = False
    sd: float = 0.0                                  # expected move sd, for the surprise z
    knowable_before: float = 0.5
    group: str = ""

    def check(self) -> list:
        errs = []
        from engine.research.core import MoveCategory
        try:
            MoveCategory(self.category)
        except ValueError:
            errs.append(f"unknown category {self.category!r}")
        leak = identity_leak(self.subject + " " + self.pattern)
        if leak:
            errs.append(f"{leak} in subject/pattern")
        return errs


def day_input_from_autopsy(entries: Sequence[AutopsyEntry], now, history: Sequence[PredictionRow] = (), coverage: Mapping | None = None,
                           z_min: float = 2.5) -> DayInput:
    """Map the autopsy's categories onto the section-22 sources. Held-and-wrong (FALSE_POSITIVE, or LOSER we predicted) become failed
    predictions; predicted winners become correct ones; FALSE_NEGATIVE with a positive move is a missed winner (gain share among all
    missed winners); a negative-move entry we did not avoid is a missed loser; UNPREDICTABLE_MOVER with an sd gives a surprise z;
    NEAR_MISS entries are counted but make no target (nothing failed). All entries must be strictly before `now`."""
    entries = list(entries)
    if not entries:
        raise TargetError("an autopsy with no entries is not a day")
    days = {e.day for e in entries}
    for e in entries:
        errs = e.check()
        if errs:
            raise TargetError("; ".join(errs))
        require_past(e.day, now, f"autopsy entry {e.subject}")
    through = max(days)
    preds, surprises, mw, ml = [], [], [], []
    mw_total = sum(max(e.move, 0.0) for e in entries if e.category == "FALSE_NEGATIVE") or 1.0
    ml_total = sum(abs(min(e.move, 0.0)) for e in entries if e.category in ("LOSER", "EXTREME_DOWN") and not e.predicted) or 1.0
    for e in entries:
        if e.predicted and e.pattern:
            preds.append(PredictionRow(e.pattern, e.correct and e.category != "FALSE_POSITIVE", dict(e.features), e.group, e.move))
        if e.category == "FALSE_NEGATIVE" and e.move > 0:
            mw.append({"subject": e.subject, "gain_share": e.move / mw_total, "knowable_before": e.knowable_before})
        if e.category in ("LOSER", "EXTREME_DOWN") and not e.predicted and e.move < 0:
            ml.append({"subject": e.subject, "loss_share": abs(e.move) / ml_total, "knowable_before": e.knowable_before})
        if e.category == "UNPREDICTABLE_MOVER" and e.sd > 0 and abs(e.move) / e.sd >= z_min:
            surprises.append({"subject": e.subject, "z": abs(e.move) / e.sd})
    return DayInput(through, predictions=preds, surprises=surprises, missed_winners=mw, missed_losers=ml, coverage=dict(coverage or {}),
                    pattern_history=list(history))


def autopsy_counts(entries: Sequence[AutopsyEntry]) -> dict:
    """Entries per section-4 category, for the day-to-target record."""
    out: dict = {}
    for e in entries:
        out[e.category] = out.get(e.category, 0) + 1
    return dict(sorted(out.items()))


@dataclass
class DayTargetBook:
    """The day-to-target record: for every day, what the autopsy contained, which targets it produced, what became of each, and the
    lineage target -> question -> experiment -> verdict so any finding can be traced to the day that raised it. Append-only."""
    days: dict = field(default_factory=dict)         # day -> {"counts": {...}, "targets": [ids], "sources": {...}}
    lineage: dict = field(default_factory=dict)      # target_id -> {"day", "source", "text", "question_id", "experiment_id", "verdict"}

    def record_day(self, day: str, counts: Mapping, report: DayReport) -> None:
        if day in self.days:
            raise TargetError(f"day {day} already recorded: the record is append-only")
        ids = [t.target_id for t in report.targets] + [i for i, _ in report.promoted] + [i for i, _ in report.failed]
        self.days[day] = {"counts": dict(counts), "targets": ids, "sources": dict(report.by_source), "promoted": [i for i, _ in report.promoted],
                          "failed": [i for i, _ in report.failed], "blocked": len(report.blocked)}
        for t in report.targets:
            self.lineage[t.target_id] = {"day": day, "source": t.source, "text": t.text, "question_id": "", "experiment_id": "", "verdict": "OPEN"}

    def link(self, target_id: str, question_id: str = "", experiment_id: str = "", verdict: str = "") -> None:
        if target_id not in self.lineage:
            raise TargetError(f"unknown target {target_id}")
        row = self.lineage[target_id]
        for k, v in (("question_id", question_id), ("experiment_id", experiment_id), ("verdict", verdict)):
            if v:
                row[k] = v

    def trace(self, target_id: str) -> dict:
        return dict(self.lineage.get(target_id, {}))

    def open_targets(self) -> list:
        return sorted(t for t, r in self.lineage.items() if r["verdict"] == "OPEN")

    def yield_by_source(self) -> dict:
        out: dict = {}
        for r in self.lineage.values():
            d = out.setdefault(r["source"], {"n": 0, "useful": 0})
            d["n"] += 1
            d["useful"] += int(r["verdict"] in ("PROMOTED", "SUCCESS"))
        return out

    def to_json(self) -> str:
        return json.dumps({"days": self.days, "lineage": self.lineage}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DayTargetBook":
        d = json.loads(text)
        return cls(d["days"], d["lineage"])


def run_autopsy_day(entries: Sequence[AutopsyEntry], now, ledger: TargetLedger, tracker: OpenConditionTracker, book: DayTargetBook,
                    history: Sequence[DatedRow] = (), state: PRI.PriorityState | None = None, seed: int = 0) -> DayReport:
    """One full day: autopsy -> day input -> targets (out-of-sample tested against history) -> open trackers for undecided conditions ->
    advance every open tracker with the history rows -> write the day-to-target record. Returns the DayReport."""
    day_rows = [dr.row for dr in history if to_ts(dr.day) < to_ts(now)]
    day = day_input_from_autopsy(entries, now, history=day_rows)
    rep = run_day(day, now, ledger, state, seed)
    through = day.matured_through
    for t in rep.targets:
        if t.condition is not None:
            tracker.open(t, through, now)
    tracker.update([dr for dr in history], now)
    for tid in tracker.by_state(TrackState.CONFIRMED):
        ledger.rows.append({"target_id": tid, "key": tid, "source": "failure_condition", "text": "tracked condition", "status": "PROMOTED",
                            "at": str(now), "evidence_through": through, "detail": "sequential test confirmed", "n_obs": 0})
    book.record_day(through, autopsy_counts(entries), rep)
    return rep


# ---------------------------------------------------------------------------------------------------------- source detectors (raw data -> rows)

def detect_clusters(counts: Mapping, baseline: Mapping, alpha: float = 0.01, min_size: int = 4) -> list:
    """New volatility clusters: groups (sector code, condition bucket) whose count of unusual movers today is a Poisson upper-tail
    surprise against that group's own baseline rate, Bonferroni-corrected over the groups. Returns rows for `DayInput.clusters`."""
    from scipy.stats import poisson
    groups = sorted(counts)
    m = max(len(groups), 1)
    rows = []
    for g in groups:
        size, base = float(counts[g]), max(float(baseline.get(g, 0.0)), 0.05)
        if size < min_size:
            continue
        p = float(poisson.sf(size - 1, base))
        if p * m <= alpha:
            rows.append({"subject": g, "size": size, "base_size": base, "excess": min(1.0, (size - base) / size), "p_value": min(1.0, p * m)})
    return rows


def detect_breaks(series: Mapping, window: int = 20, min_drop: float = 0.15, k: float = 0.5, h: float = 4.0) -> list:
    """Pattern breaks from per-pattern hit series (0/1 per matured prediction, oldest first). A one-sided CUSUM on the DROP of the hit
    rate against the pattern's own earlier mean; a break needs the CUSUM alarm AND a recent-window mean at least `min_drop` under
    the earlier mean. Returns rows for `DayInput.breaks` with the before/after reliability and the number of post-break observations."""
    rows = []
    for name in sorted(series):
        x = np.asarray(series[name], float)
        if x.size < 2 * window:
            continue
        ref = x[:-window]
        mu, sd = float(ref.mean()), max(float(ref.std()), 0.05)
        s, alarm_at = 0.0, None
        for i, v in enumerate(x[len(ref):]):
            s = max(0.0, s + (mu - v) / sd - k)
            if s >= h and alarm_at is None:
                alarm_at = i
        recent = float(x[-window:].mean())
        if alarm_at is not None and mu - recent >= min_drop:
            rows.append({"subject": name, "drop": mu - recent, "before": mu, "n_after": int(window - alarm_at)})
    return rows


def detect_regime_shift(feature: Sequence[float], recent: int = 20, z_min: float = 2.5) -> dict | None:
    """Has the market context moved to a new regime? Mean of the last `recent` observations against the earlier history, in standard
    errors of the earlier history (with autocorrelation-inflated variance). Returns a row for `DayInput.regimes`, or None."""
    x = np.asarray(feature, float)
    if x.size < 3 * recent:
        return None
    ref, cur = x[:-recent], x[-recent:]
    r1 = float(np.corrcoef(ref[:-1], ref[1:])[0, 1]) if ref.std() > 0 else 0.0
    infl = (1 + max(r1, 0.0)) / (1 - max(r1, 0.0) + 1e-9)
    se = float(ref.std()) * math.sqrt(infl / recent)
    z = abs(float(cur.mean() - ref.mean())) / max(se, 1e-9)
    return {"z": z, "shift": min(1.0, z / 6.0)} if z >= z_min else None


def detect_contradictions(setups: Mapping, min_n: int = 10, z_min: float = 2.0) -> list:
    """Setups with the same feature signature but different hit rates. `setups` maps signature -> {name: (hits, n)}; any pair inside a
    signature separated by a two-proportion z above `z_min` is a contradiction row (strength = the gap)."""
    rows = []
    for sig in sorted(setups):
        names = sorted(setups[sig])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                (ha, na), (hb, nb) = setups[sig][a], setups[sig][b]
                if min(na, nb) < min_n:
                    continue
                pool = (ha + hb) / (na + nb)
                se = math.sqrt(max(pool * (1 - pool) * (1 / na + 1 / nb), 1e-12))
                z = abs(ha / na - hb / nb) / se
                if z >= z_min:
                    rows.append({"subject": a, "counterpart": b, "strength": abs(ha / na - hb / nb), "n": na + nb, "z": z})
    return rows


# ---------------------------------------------------------------------------------------------------------- multiplicity and interaction

def bh_select(pvalues: Sequence[float], q: float = 0.10) -> list:
    """Benjamini-Hochberg: indices of hypotheses to keep at false-discovery rate q. Used across the day's candidate conditions so that
    generating many targets does not itself manufacture false ones (section 32)."""
    p = np.asarray(pvalues, float)
    if p.size == 0:
        return []
    order = np.argsort(p)
    thr = q * (np.arange(1, p.size + 1) / p.size)
    ok = np.where(p[order] <= thr)[0]
    return sorted(int(i) for i in order[: ok.max() + 1]) if ok.size else []


def find_conjunction_conditions(rows: Sequence[PredictionRow], base: Sequence[Condition], min_fail: int = 3, alpha: float = 0.05) -> list:
    """Two-feature conditions: a base condition AND a second single-feature condition, kept only if the conjunction concentrates
    failures BETTER than either alone (higher inside failure rate and a smaller p-value). Returns Condition objects whose feature name is
    'a&b' and whose threshold is the first's (holds() is not usable for them; they are reported, and tested via `conjunction_holds`)."""
    out = []
    for c1 in base[:3]:
        for c2 in find_failure_conditions([r for r in rows if c1.holds(r.features)], min_fail=min_fail, alpha=alpha):
            if c2.feature == c1.feature:
                continue
            if c2.rate_in > c1.rate_in and c2.p_value < c1.p_value:
                out.append(Condition(f"{c1.feature}&{c2.feature}", c1.op + c2.op, c1.threshold, c2.n_in, c2.fail_in, c2.n_out, c2.fail_out, c2.p_value))
    return out


def threshold_stability(rows: Sequence[PredictionRow], feature: str, n_boot: int = 100, seed: int = 0) -> dict:
    """Would a re-draw of the day's data pick the same threshold? Bootstrap the rows, take the best condition on `feature` each time;
    report how often one is found at all and the spread of the chosen threshold. A condition whose threshold wanders over the whole
    range is describing noise, however small its p-value looked."""
    rng = np.random.default_rng(seed)
    rows = list(rows)
    ths = []
    for _ in range(n_boot):
        samp = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        cs = [c for c in find_failure_conditions(samp) if c.feature == feature]
        if cs:
            ths.append(cs[0].threshold)
    vals = np.array([r.features[feature] for r in rows if feature in r.features], float)
    rng_width = float(vals.max() - vals.min()) if vals.size else 1.0
    if len(ths) < 5:
        return {"found_share": len(ths) / n_boot, "verdict": "UNSTABLE"}
    spread = float(np.std(ths))
    return {"found_share": len(ths) / n_boot, "threshold_sd": spread, "relative_sd": spread / max(rng_width, 1e-9),
            "verdict": "STABLE" if spread / max(rng_width, 1e-9) < 0.15 and len(ths) / n_boot >= 0.6 else "UNSTABLE"}


# ---------------------------------------------------------------------------------------------------------- daily selection and multi-day driving

def select_daily(targets: Sequence[Target], budget_minutes: float, max_per_source: int = 3, min_sources: int = 3) -> tuple:
    """Choose today's targets under a compute budget with breadth: best-priority first, at most `max_per_source` from one source, and
    the first pass takes the best of each source so no section-22 source is starved while others fill the day. Returns (chosen, deferred
    with reasons). Nothing is dropped silently."""
    ranked = sorted(targets, key=lambda t: (-t.priority, t.target_id))
    chosen, used, per = [], 0.0, {}
    reasons: dict = {}
    for src in sorted({t.source for t in ranked}):
        best = next(t for t in ranked if t.source == src)
        cost = best.value.compute_cost or 1.0
        if used + cost <= budget_minutes:
            chosen.append(best)
            used += cost
            per[src] = 1
    for t in ranked:
        if t in chosen:
            continue
        cost = t.value.compute_cost or 1.0
        if per.get(t.source, 0) >= max_per_source:
            reasons[t.target_id] = f"source {t.source} already has {max_per_source} today"
        elif used + cost > budget_minutes:
            reasons[t.target_id] = f"needs {cost:.0f} cpu-min, {max(budget_minutes - used, 0):.0f} left"
        else:
            chosen.append(t)
            used += cost
            per[t.source] = per.get(t.source, 0) + 1
    return tuple(sorted(chosen, key=lambda t: (-t.priority, t.target_id))), tuple(sorted(reasons.items()))


def age_targets(targets: Sequence[Target], now, half_life_days: float = 30.0, drop_below: float = 0.1) -> list:
    """Yesterday's unrun target is worth less today: priority halves every half-life; targets under `drop_below` of their original are
    dropped (they will be regenerated if the evidence persists)."""
    out = []
    for t in targets:
        f = 0.5 ** (max(0.0, (to_ts(now) - to_ts(t.evidence_through)).total_seconds() / 86400.0) / half_life_days)
        if f >= drop_below:
            out.append(replace(t, priority=t.priority * f))
    return sorted(out, key=lambda t: (-t.priority, t.target_id))


def merge_across_days(batches: Sequence[Sequence[Target]]) -> list:
    """Same hypothesis raised on several days is ONE target with the newest evidence and the largest magnitude; its repeat count is
    kept in the id-independent key so a persistent problem can be told from a one-day event."""
    best: dict = {}
    count: dict = {}
    for batch in batches:
        for t in batch:
            k = _key(t)
            count[k] = count.get(k, 0) + 1
            cur = best.get(k)
            if cur is None or t.evidence_through > cur.evidence_through or (t.evidence_through == cur.evidence_through and t.magnitude > cur.magnitude):
                best[k] = t
    out = []
    for k, t in best.items():
        boost = min(1.5, 1.0 + 0.1 * (count[k] - 1))
        out.append(replace(t, magnitude=min(1.0, t.magnitude * boost)))
    return sorted(out, key=lambda t: (-t.magnitude, t.target_id))


@dataclass(frozen=True)
class MultiDayReport:
    days: tuple
    n_targets: int
    promoted: int
    failed: int
    confirmed_by_tracking: int
    silent_sources: tuple
    digest: str


def run_days(days: Mapping, ledger: TargetLedger | None = None, tracker: OpenConditionTracker | None = None, book: DayTargetBook | None = None,
             history: Sequence[DatedRow] = (), seed: int = 0) -> MultiDayReport:
    """Drive several autopsy days in date order. `days` maps day -> list[AutopsyEntry]; each day is processed at `now` = the next calendar
    day (its outcomes matured strictly before). History rows are only those dated before each day's `now`. Deterministic: the digest
    hashes every day's target ids, so two runs over the same input can be compared."""
    import datetime as _dt
    ledger = ledger if ledger is not None else TargetLedger()
    tracker = tracker if tracker is not None else OpenConditionTracker()
    book = book if book is not None else DayTargetBook()
    reports = []
    for day in sorted(days):
        now = (_dt.date.fromisoformat(day) + _dt.timedelta(days=1)).isoformat()
        hist = [dr for dr in history if to_ts(dr.day) < to_ts(now)]
        reports.append(run_autopsy_day(days[day], now, ledger, tracker, book, hist, None, seed))
    cov = source_coverage(reports)
    return MultiDayReport(tuple(sorted(days)), sum(len(r.targets) for r in reports), sum(len(r.promoted) for r in reports),
                          sum(len(r.failed) for r in reports), len(tracker.by_state(TrackState.CONFIRMED)), tuple(cov["silent"]),
                          stable_hash([[t.target_id for t in r.targets] for r in reports], 16))


def target_to_event(t: Target, now) -> QST.QuestionEvent:
    """Bridge to the question generator: a target becomes the event of the matching section-40 source, so the two modules stay one
    pipeline (target -> question -> tree). Sources without a section-40 twin map to the closest one."""
    require_past(t.evidence_through, now, f"target {t.target_id}")
    src = {"surprise": "surprise", "loss": "loss", "win": "new_discovery", "missed_winner": "missed_winner", "missed_loser": "missed_winner",
           "volatility_cluster": "new_discovery", "pattern_break": "pattern_break", "contradiction": "contradiction", "new_regime": "regime_change",
           "data_anomaly": "data_anomaly", "research_failure": "research_failure", "new_combination": "new_discovery",
           "understudied_sector": "coverage_gap", "understudied_condition": "coverage_gap", "unknown_area": "coverage_gap",
           "failure_condition": "loss"}[t.source]
    loss = t.value.loss_reduction_value or 0.0
    return QST.QuestionEvent(src, t.subject if not identity_leak(t.subject) else "situation", t.evidence_through, t.magnitude, stake=t.stake, problem=t.problem,
                             loss_share=min(1.0, loss * 2.0))
