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
                    out.append(Condition(f, op, float(thr), n_in, fail_in, n_out, fail_out, min(1.0, p * tried)))
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
