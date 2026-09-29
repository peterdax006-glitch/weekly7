"""Meta-learning about RESEARCH itself (contract C66 section 16; builds on C62 section 37; checklist R13; Bible canon C58-C66).

meta_learning.py (C62) asks which DISCOVERIES survive. This module asks the same question one level up, about the research
process that produced them (C66 section 16, the twelve questions):

  Q01 which experiment types produce durable knowledge        Q07 which research questions change decisions
  Q02 which experiment types usually overfit                  Q08 which failed experiments were informative
  Q03 which pattern families transfer                         Q09 which research paths repeatedly waste compute
  Q04 which representations repeatedly fail                   Q10 which discoveries survive new eras
  Q05 which datasets create false discoveries                 Q11 which discoveries survive new stocks
  Q06 which validation methods catch the most errors          Q12 which discoveries survive new regimes

Every answer is (1) an empirical-Bayes shrunk rate with a Wilson interval, (2) corrected for the number of groups examined
(Benjamini-Hochberg) so twelve questions over dozens of groups do not manufacture findings, and (3) EVALUATED OUT OF SAMPLE
by walk-forward: a question only reaches the scheduler if its group-rate predictor beat the pooled base rate on runs it never
saw (`walk_forward_groups`, reused from engine.learning.meta_learning). Unknown stays unknown: a group with too little data, or
a question whose predictor failed out of sample, is OMITTED from the advice, never filled with a neutral number.

The scheduler is never allowed to train on its own future evaluation results (C66 section 16). Three mechanisms, all tested:
  * time: a run is visible at `now` only if it resolved strictly before `now`;
  * quarantine: rows that ARE evaluation output (origin != "research", or derived_from a sealed meta-evaluation) are refused
    as training data by `SelfTrainingGuard`, and a dependency on an evaluation sealed at/after `now` is a FirewallBreach;
  * selection: a run the scheduler chose BECAUSE of advice X is excluded when X itself is scored (`chosen_by`), because that
    selection is informed by X's own predictions.

Namespaces (C66 sections 29-31, C64): everything here is built from matured outcomes, so it lives in MATURED_RESEARCH_STATE.
It reaches the scheduler (trusted side) as `SchedulerAdvice`; it may reach the blind trader only through
`SchedulerAdvice.matured_record(...).gate(now)`. Advice carries no ticker, date or year (`assert_identity_free`), and research
filed under a real year is withheld while that same year is replayed in disguise (`release_view`, the same-year rerun leak).

Extends: engine.learning.meta_learning (rates, walk-forward, leak audit, drift, bootstrap), engine.learning.research_policy
(MetaAdvice, Candidate, meta_adjusted_factors), engine.learning.research_priority (identity_leak, barren_streak),
engine.learning.experiment_memory (ExperimentLedger records), engine.learning.failed_learners (FailedLearnerRegistry).
Public entry: `step(state, now, seed)`.  IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import enum
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as sps

from engine.learning.core import ValidationLabel, canonical_json, current_code_hash, stable_hash
from engine.learning.experiment_memory import to_ts
from engine.learning.meta_learning import (DiscoveryRecord, GroupRate, OosResult, SurvivalDrift, brier, fit_beta_prior, group_rates,
                                           leak_audit, paired_bootstrap, shrunk_rate, spearman, walk_forward_groups,
                                           walk_forward_survival, wilson)
from engine.learning.research_policy import Candidate, MetaAdvice, ResearchTarget, meta_adjusted_factors
from engine.learning.research_priority import identity_leak
from engine.research.core import FirewallBreach, GateVerdict, MaturedRecord, Namespace, Provenance

LABEL = ValidationLabel.NOT_VALIDATED.value
ALLOWED_ORIGINS = ("research",)                        # the only origin that may be TRAINED on
EVAL_ORIGINS = ("meta_evaluation", "scheduler_score", "advice_replay")   # outputs of evaluating the scheduler: never training data


class Q16(str, enum.Enum):
    """The twelve section-16 questions."""
    DURABLE_EXPERIMENTS = "Q01_durable_experiment_types"
    OVERFIT_TYPES = "Q02_overfitting_types"
    TRANSFERRING_FAMILIES = "Q03_transferring_families"
    FAILING_REPRESENTATIONS = "Q04_failing_representations"
    FALSE_DISCOVERY_DATASETS = "Q05_false_discovery_datasets"
    VALIDATION_METHODS = "Q06_validation_methods"
    DECISION_QUESTIONS = "Q07_decision_changing_questions"
    INFORMATIVE_FAILURES = "Q08_informative_failures"
    WASTED_COMPUTE = "Q09_compute_wasting_paths"
    ERA_SURVIVAL = "Q10_survives_new_eras"
    STOCK_SURVIVAL = "Q11_survives_new_stocks"
    REGIME_SURVIVAL = "Q12_survives_new_regimes"

    def __str__(self):
        return self.value


# ------------------------------------------------------------------------------------------------ the record

_TEXT_FIELDS = ("exp_type", "family", "representation", "dataset", "validation_method", "question_source", "target",
                "era", "stock_group", "regime")


@dataclass(frozen=True)
class ResearchOutcome:
    """One finished research run, described by HOW it was done and WHAT later became of it. Descriptive fields are frozen when
    the run starts; outcome fields (durable, overfit, ...) become known at `resolved_at`. None = not (yet) known, which is
    different from False: every analyser skips None for that question. Text fields must be identity-free (section 29)."""
    run_id: str
    exp_type: str                                   # kind of experiment: pattern_search, ablation, break_study, transfer_test, ...
    family: str                                     # pattern / learner family the run studied
    representation: str                             # how the data was represented: raw_returns, rank, zscore_context, ...
    dataset: str                                    # which data view: panel_survivor, panel_delisted, movers_only, ...
    validation_method: str                          # how the result was checked: walk_forward, holdout, permutation_null, ...
    question_source: str                            # what created the question: surprise, loss, missed_winner, break, ...
    target: str                                     # ResearchTarget value
    started_at: str
    resolved_at: str
    cost_minutes: float
    failed: bool = False                            # the run's own hypothesis was rejected
    info_bits: float = 0.0                          # realised information gain (H(prior) - H(posterior))
    n_tests: int = 1                                # comparisons the run itself made (multiple-testing burden)
    free_parameters: int = 0
    durable: bool | None = None                     # the knowledge it produced still stood when re-checked later
    overfit: bool | None = None                     # in-sample gain, out-of-sample loss
    false_discovery: bool | None = None             # a finding it reported was later shown to be noise
    decision_changed: bool | None = None            # a belief, gate, threshold or allocation moved because of it
    followups: int = 0                              # questions its result spawned
    error_present: bool | None = None               # the thing validated really contained a defect (planted or later proved)
    validator_flagged: bool | None = None           # the validation method raised an alarm
    error_id: str = ""                              # links the same defect across validation methods
    new_era_survived: bool | None = None
    new_stock_survived: bool | None = None
    new_regime_survived: bool | None = None
    era: str = ""                                   # identity-free era label ("E1"), never a year
    stock_group: str = ""
    regime: str = ""
    evidence_year: str = ""                         # TRUSTED SIDE ONLY: real year the run was filed under (same-year rerun leak)
    origin: str = "research"
    chosen_by: str = ""                             # advice_id whose ranking made the scheduler run this ("" = not advice-driven)
    derived_from: tuple = ()                        # ids of meta-evaluations this row's values were computed from

    def check(self) -> list:
        errs = []
        for f in ("run_id", "exp_type", "started_at", "resolved_at"):
            if not getattr(self, f):
                errs.append(f"{self.run_id or '?'}: {f} missing")
        if errs:
            return errs
        try:
            if to_ts(self.resolved_at) <= to_ts(self.started_at):
                errs.append(f"{self.run_id}: resolved_at not after started_at")
        except (ValueError, FirewallBreach):
            errs.append(f"{self.run_id}: unparseable timestamp")
        if not (math.isfinite(self.cost_minutes) and self.cost_minutes >= 0):
            errs.append(f"{self.run_id}: cost_minutes {self.cost_minutes!r} must be finite and >= 0")
        if not (math.isfinite(self.info_bits) and self.info_bits >= 0):
            errs.append(f"{self.run_id}: info_bits {self.info_bits!r} must be finite and >= 0")
        if self.n_tests < 1 or self.free_parameters < 0 or self.followups < 0:
            errs.append(f"{self.run_id}: n_tests>=1, free_parameters>=0, followups>=0 required")
        for f in _TEXT_FIELDS:
            leak = identity_leak(str(getattr(self, f)))
            if leak:
                errs.append(f"{self.run_id}: {f} {leak} (identity firewall)")
        if self.validator_flagged is not None and self.error_present is None:
            errs.append(f"{self.run_id}: validator_flagged without error_present has no ground truth")
        if self.durable and self.overfit:
            errs.append(f"{self.run_id}: cannot be both durable and overfit")
        if self.durable and self.false_discovery:
            errs.append(f"{self.run_id}: cannot be both durable and a false discovery")
        if self.origin not in ALLOWED_ORIGINS + EVAL_ORIGINS:
            errs.append(f"{self.run_id}: unknown origin {self.origin!r}")
        if self.evidence_year and not self.evidence_year.isdigit():
            errs.append(f"{self.run_id}: evidence_year must be a 4-digit year string")
        return errs

    @property
    def path(self) -> str:
        """The research path: the tuple of choices a scheduler can repeat or avoid (Q09)."""
        return f"{self.exp_type}|{self.family}|{self.representation}"


def outcome_to_dict(o: ResearchOutcome) -> dict:
    d = {k: getattr(o, k) for k in o.__dataclass_fields__}
    d["derived_from"] = list(o.derived_from)
    return d


def outcome_from_dict(d: Mapping) -> ResearchOutcome:
    kw = {k: d[k] for k in ResearchOutcome.__dataclass_fields__ if k in d}
    kw["derived_from"] = tuple(kw.get("derived_from", ()))
    return ResearchOutcome(**kw)


# ------------------------------------------------------------------------------------------------ the store

class OutcomeStore:
    """Append-only research-outcome ledger. Reads are always `as_of(now)`: rows that resolved strictly before `now`."""

    def __init__(self, rows: Iterable[ResearchOutcome] = ()):
        self._rows: list = []
        self._ids: set = set()
        for r in rows:
            self.add(r)

    def add(self, o: ResearchOutcome) -> None:
        errs = o.check()
        if errs:
            raise ValueError("; ".join(errs))
        if o.run_id in self._ids:
            raise ValueError(f"duplicate run_id {o.run_id}: a re-run is a new run, history is never overwritten")
        self._ids.add(o.run_id)
        self._rows.append(o)

    def extend(self, rows: Iterable[ResearchOutcome]) -> int:
        n = 0
        for r in rows:
            self.add(r)
            n += 1
        return n

    def __len__(self) -> int:
        return len(self._rows)

    def all(self) -> list:
        return list(self._rows)

    def as_of(self, now) -> list:
        cut = to_ts(now)
        return sorted((r for r in self._rows if to_ts(r.resolved_at) < cut), key=lambda r: (to_ts(r.started_at), r.run_id))

    def content_hash(self) -> str:
        return stable_hash([canonical_json(outcome_to_dict(r)) for r in self._rows], 12)

    def counts(self, now=None) -> dict:
        rows = self._rows if now is None else self.as_of(now)
        out: dict = defaultdict(int)
        for r in rows:
            out[r.origin] += 1
        out["total"] = len(rows)
        return dict(out)

    def save(self, path) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in self._rows:
                f.write(json.dumps(outcome_to_dict(r), sort_keys=True) + "\n")
        return len(self._rows)

    @classmethod
    def load(cls, path) -> "OutcomeStore":
        st = cls()
        p = Path(path)
        if not p.exists():
            return st
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                st.add(outcome_from_dict(json.loads(line)))
        return st


# ------------------------------------------------------------------------------------------------ never train on own evaluation

@dataclass(frozen=True)
class EvalSeal:
    """A sealed meta-evaluation result: what the scheduler advice scored on runs it had not seen. Once sealed it may be read
    by reports and by humans but never by a training set."""
    eval_id: str
    question: str
    fitted_through: str                             # the advice was fitted on runs resolved before this
    evaluated_at: str                               # when the score became known
    score: float
    run_ids: tuple = ()                             # the held-out runs it was scored on


class EvaluationVault:
    """Registry of meta-evaluation results. Its ids are the 'poison list' the guard refuses to let into training data."""

    def __init__(self):
        self._seals: dict = {}

    def seal(self, question: str, fitted_through, evaluated_at, score: float, run_ids: Iterable[str] = ()) -> EvalSeal:
        if to_ts(evaluated_at) <= to_ts(fitted_through):
            raise FirewallBreach("an evaluation cannot be known before the advice it scores was fitted")
        if not math.isfinite(score):
            raise ValueError("evaluation score must be finite")
        ids = tuple(sorted(run_ids))
        eid = "EV" + stable_hash([question, str(fitted_through), str(evaluated_at), round(score, 9), ids], 10)
        seal = EvalSeal(eid, question, str(fitted_through), str(evaluated_at), float(score), ids)
        self._seals[eid] = seal
        return seal

    def get(self, eval_id: str) -> EvalSeal | None:
        return self._seals.get(eval_id)

    def ids(self) -> frozenset:
        return frozenset(self._seals)

    def visible(self, now) -> list:
        cut = to_ts(now)
        return sorted((s for s in self._seals.values() if to_ts(s.evaluated_at) < cut), key=lambda s: s.eval_id)

    def future_of(self, now) -> frozenset:
        cut = to_ts(now)
        return frozenset(s.eval_id for s in self._seals.values() if to_ts(s.evaluated_at) >= cut)

    def __len__(self) -> int:
        return len(self._seals)


@dataclass(frozen=True)
class GuardReport:
    kept: int
    refused: Mapping                                # run_id -> reason
    future_dependencies: tuple                      # run_ids whose values derive from an evaluation sealed at/after now

    @property
    def clean(self) -> bool:
        return not self.refused and not self.future_dependencies


class SelfTrainingGuard:
    """Refuses to let the scheduler's own evaluation results become its training data. `strict=True` turns a dependency on a
    FUTURE evaluation into a FirewallBreach instead of a refusal, because that can only come from a time-travelling pipeline."""

    def __init__(self, vault: EvaluationVault | None = None, strict: bool = True):
        self.vault = EvaluationVault() if vault is None else vault
        self.strict = strict

    def screen(self, rows: Sequence[ResearchOutcome], now, evaluating_advice: str = "") -> tuple:
        poison = self.vault.ids()
        future = self.vault.future_of(now)
        cut = to_ts(now)
        keep, refused, futdep = [], {}, []
        for r in rows:
            dep = [e for e in r.derived_from if e in future]
            if dep:
                futdep.append(r.run_id)
                if self.strict:
                    raise FirewallBreach(f"run {r.run_id} depends on evaluation(s) {dep} sealed at/after {now}")
                refused[r.run_id] = "depends on a future evaluation result"
            elif r.origin in EVAL_ORIGINS:
                refused[r.run_id] = f"origin {r.origin} is evaluation output, not research"
            elif poison.intersection(r.derived_from):
                refused[r.run_id] = "derived from a sealed meta-evaluation"
            elif to_ts(r.resolved_at) >= cut:
                refused[r.run_id] = "not yet resolved at now"
            elif evaluating_advice and r.chosen_by == evaluating_advice:
                refused[r.run_id] = "chosen by the advice being evaluated (selection feedback)"
            else:
                keep.append(r)
        return keep, GuardReport(len(keep), refused, tuple(futdep))

    def training_rows(self, store: OutcomeStore, now, evaluating_advice: str = "") -> list:
        keep, _ = self.screen(store.as_of(now), now, evaluating_advice)
        return keep

    def assert_clean(self, rows: Sequence[ResearchOutcome], now) -> None:
        """Raise unless every row is legitimate training data at `now`."""
        _, rep = self.screen(rows, now)
        if rep.refused:
            first = sorted(rep.refused.items())[0]
            raise FirewallBreach(f"{len(rep.refused)} row(s) are not legitimate training data, e.g. {first[0]}: {first[1]}")


# ------------------------------------------------------------------------------------------------ statistics shared by all twelve questions

def benjamini_hochberg(pvals: Mapping[str, float], q: float = 0.10) -> dict:
    """key -> (adjusted p, significant at FDR q). Twelve questions over dozens of groups is a multiple-comparison problem;
    without this a meta-learner finds 'families that overfit' in pure noise."""
    if not (0.0 < q < 1.0):
        raise ValueError("FDR level must lie in (0, 1)")
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    adj: dict = {}
    running = 1.0
    for rank in range(m, 0, -1):
        key, p = items[rank - 1]
        running = min(running, min(1.0, p * m / rank))
        adj[key] = running
    return {k: (adj[k], adj[k] <= q) for k in pvals}


def binomial_p(successes: int, n: int, p0: float) -> float:
    """Two-sided exact binomial p-value of `successes/n` against reference rate p0. Degenerate references give 1.0: a group
    cannot differ from a rate of exactly 0 or 1 in a way this test can price."""
    if n <= 0 or not (0.0 < p0 < 1.0):
        return 1.0
    return float(sps.binomtest(int(successes), int(n), float(p0)).pvalue)


def posterior_rate_above(successes: int, n: int, floor: float, a0: float = 1.0, b0: float = 1.0) -> float:
    """P(true rate >= floor) under a Beta(a0 + s, b0 + n - s) posterior: the sequential-stopping quantity for wasted paths."""
    return float(1.0 - sps.beta.cdf(floor, a0 + successes, b0 + n - successes))


def _streak(flags: Sequence[bool], want: bool) -> int:
    """Length of the current run of `want` at the END of a chronologically ordered list."""
    n = 0
    for f in reversed(flags):
        if f != want:
            break
        n += 1
    return n


# ------------------------------------------------------------------------------------------------ one rate table

@dataclass(frozen=True)
class GroupFinding:
    key: str
    n: int
    successes: int
    raw: float
    shrunk: float
    lo: float
    hi: float
    reference: float                                # pooled rate of all OTHER groups (leave-group-out)
    lift: float                                     # shrunk - reference
    p_value: float
    q_value: float
    significant: bool
    direction: str                                  # ABOVE / BELOW / NEUTRAL
    desirable: bool | None                          # is that direction good for research? None when NEUTRAL


@dataclass(frozen=True)
class RateTable:
    outcome: str
    higher_is_better: bool
    n_rows: int
    pooled: float
    prior: tuple
    groups: Mapping                                 # key -> GroupFinding (only groups with n >= min_n)
    unknown: tuple                                  # keys seen but with too little data: unknown, not neutral
    fdr_q: float
    min_n: int

    def keys_by(self, direction: str, desirable: bool | None = None) -> list:
        out = [g for g in self.groups.values() if g.direction == direction and (desirable is None or g.desirable == desirable)]
        return [g.key for g in sorted(out, key=lambda g: (-abs(g.lift), g.key))]

    def best(self, k: int = 3) -> list:
        """Groups best for research on this outcome (highest shrunk rate if higher_is_better else lowest)."""
        s = sorted(self.groups.values(), key=lambda g: (g.shrunk if not self.higher_is_better else -g.shrunk, g.key))
        return [g.key for g in s[:k]]

    def worst(self, k: int = 3) -> list:
        s = sorted(self.groups.values(), key=lambda g: (-g.shrunk if not self.higher_is_better else g.shrunk, g.key))
        return [g.key for g in s[:k]]

    def rate_of(self, key: str) -> float | None:
        g = self.groups.get(key)
        return None if g is None else g.shrunk


def rate_table(pairs: Iterable[tuple], outcome: str, higher_is_better: bool, min_n: int = 5, fdr_q: float = 0.10) -> RateTable:
    """Group -> shrunk rate with Wilson interval and BH-corrected significance against the rate of the OTHER groups."""
    tally: dict = {}
    for key, ok in pairs:
        s, n = tally.get(str(key), (0, 0))
        tally[str(key)] = (s + int(bool(ok)), n + 1)
    n_all = sum(v[1] for v in tally.values())
    s_all = sum(v[0] for v in tally.values())
    pooled = s_all / n_all if n_all else 0.0
    prior = fit_beta_prior([v[0] for v in tally.values()], [v[1] for v in tally.values()]) if tally else (1.0, 1.0)
    eligible = {k: v for k, v in tally.items() if v[1] >= min_n}
    unknown = tuple(sorted(k for k in tally if k not in eligible))
    refs, pv = {}, {}
    for k, (s, n) in eligible.items():
        rest_n = n_all - n
        refs[k] = (s_all - s) / rest_n if rest_n >= min_n else pooled
        pv[k] = binomial_p(s, n, refs[k])
    adj = benjamini_hochberg(pv, fdr_q) if pv else {}
    groups = {}
    for k in sorted(eligible):
        s, n = eligible[k]
        lo, hi = wilson(s, n)
        sh = shrunk_rate(s, n, prior)
        q, sig = adj[k]
        lift = sh - refs[k]
        direction = "NEUTRAL" if not sig else ("ABOVE" if lift > 0 else "BELOW")
        desirable = None if direction == "NEUTRAL" else ((direction == "ABOVE") == higher_is_better)
        groups[k] = GroupFinding(k, n, s, s / n, sh, lo, hi, refs[k], lift, pv[k], q, sig, direction, desirable)
    return RateTable(outcome, higher_is_better, n_all, pooled, tuple(prior), groups, unknown, fdr_q, min_n)


# ------------------------------------------------------------------------------------------------ the question specifications

def informative(o: ResearchOutcome, min_bits: float = 0.05) -> bool:
    """A run was informative if it narrowed the hypothesis set OR spawned a follow-up question (section 15/16)."""
    return o.info_bits >= min_bits or o.followups > 0


def transferred(o: ResearchOutcome) -> bool | None:
    """Survived EVERY new-context test it was subjected to; None when it was never subjected to one."""
    flags = [o.new_era_survived, o.new_stock_survived, o.new_regime_survived]
    known = [f for f in flags if f is not None]
    return None if not known else all(known)


def wasted(o: ResearchOutcome, min_bits: float = 0.05) -> bool | None:
    """A run wasted its compute if nothing durable, no decision change and (almost) no information came out of it.
    None while nothing is known about its durability: unresolved is not wasted."""
    if o.durable is None:
        return None
    return (not o.durable) and (not o.decision_changed) and o.info_bits < min_bits and o.followups == 0


@dataclass(frozen=True)
class QuestionSpec:
    question: Q16
    text: str
    outcome_name: str
    higher_is_better: bool                          # is a True outcome good for the research programme?
    primary: str                                    # name of the grouping dimension
    outcome: Callable[[ResearchOutcome], bool | None]
    groupers: Mapping                               # dimension name -> fn(record) -> key (str) or None to skip
    row_filter: Callable[[ResearchOutcome], bool] = lambda o: True
    min_n: int = 5


def _spec_table() -> dict:
    g = lambda attr: (lambda o: getattr(o, attr) or None)
    era_g, stock_g, reg_g = g("era"), g("stock_group"), g("regime")
    S = QuestionSpec
    return {
        Q16.DURABLE_EXPERIMENTS: S(Q16.DURABLE_EXPERIMENTS, "which experiment types produce knowledge that is still true later",
                                   "durable", True, "exp_type", lambda o: o.durable,
                                   {"exp_type": g("exp_type"), "validation_method": g("validation_method"), "question_source": g("question_source")}),
        Q16.OVERFIT_TYPES: S(Q16.OVERFIT_TYPES, "which experiment types produce in-sample gains that vanish out of sample",
                             "overfit", False, "exp_type", lambda o: o.overfit,
                             {"exp_type": g("exp_type"), "family": g("family"), "complexity": lambda o: complexity_bucket(o.free_parameters),
                              "search_size": lambda o: search_bucket(o.n_tests)}),
        Q16.TRANSFERRING_FAMILIES: S(Q16.TRANSFERRING_FAMILIES, "which pattern families survive every new context they are tried in",
                                     "transferred", True, "family", transferred, {"family": g("family"), "exp_type": g("exp_type")}),
        Q16.FAILING_REPRESENTATIONS: S(Q16.FAILING_REPRESENTATIONS, "which representations repeatedly fail",
                                       "failed", False, "representation", lambda o: o.failed,
                                       {"representation": g("representation"), "family": g("family")}),
        Q16.FALSE_DISCOVERY_DATASETS: S(Q16.FALSE_DISCOVERY_DATASETS, "which datasets produce discoveries that later prove to be noise",
                                        "false_discovery", False, "dataset", lambda o: o.false_discovery,
                                        {"dataset": g("dataset"), "search_size": lambda o: search_bucket(o.n_tests)}),
        Q16.VALIDATION_METHODS: S(Q16.VALIDATION_METHODS, "which validation methods catch the errors that are really there",
                                  "error_caught", True, "validation_method", lambda o: o.validator_flagged,
                                  {"validation_method": g("validation_method")}, lambda o: o.error_present is True, 4),
        Q16.DECISION_QUESTIONS: S(Q16.DECISION_QUESTIONS, "which research questions end up changing a decision",
                                  "decision_changed", True, "question_source", lambda o: o.decision_changed,
                                  {"question_source": g("question_source"), "target": g("target")}),
        Q16.INFORMATIVE_FAILURES: S(Q16.INFORMATIVE_FAILURES, "which failed experiments still taught something",
                                    "informative", True, "exp_type", lambda o: informative(o),
                                    {"exp_type": g("exp_type"), "question_source": g("question_source")}, lambda o: o.failed),
        Q16.WASTED_COMPUTE: S(Q16.WASTED_COMPUTE, "which research paths repeatedly waste compute",
                              "wasted", False, "path", wasted, {"path": lambda o: o.path, "exp_type": g("exp_type")}, min_n=4),
        Q16.ERA_SURVIVAL: S(Q16.ERA_SURVIVAL, "which kinds of discovery survive a new era", "survives_new_era", True, "family",
                            lambda o: o.new_era_survived, {"family": g("family"), "era": era_g}),
        Q16.STOCK_SURVIVAL: S(Q16.STOCK_SURVIVAL, "which kinds of discovery survive new stocks", "survives_new_stock", True, "family",
                              lambda o: o.new_stock_survived, {"family": g("family"), "stock_group": stock_g}),
        Q16.REGIME_SURVIVAL: S(Q16.REGIME_SURVIVAL, "which kinds of discovery survive a new regime", "survives_new_regime", True, "family",
                               lambda o: o.new_regime_survived, {"family": g("family"), "regime": reg_g}),
    }


def complexity_bucket(free_parameters: int) -> str:
    """Overfitting is driven by freedom, so it is grouped by size class, not by the raw count."""
    return "p0_2" if free_parameters <= 2 else ("p3_6" if free_parameters <= 6 else ("p7_15" if free_parameters <= 15 else "p16_plus"))


def search_bucket(n_tests: int) -> str:
    """How wide a search the run performed: the multiple-testing burden that manufactures false discoveries."""
    return "tests_1" if n_tests <= 1 else ("tests_2_20" if n_tests <= 20 else ("tests_21_200" if n_tests <= 200 else "tests_201_plus"))


SPECS = _spec_table()
assert set(SPECS) == set(Q16), "every section-16 question needs a specification"


@dataclass(frozen=True)
class QuestionReport:
    question: Q16
    text: str
    tables: Mapping                                 # dimension name -> RateTable
    primary: str
    n_rows: int

    @property
    def main(self) -> RateTable:
        return self.tables[self.primary]

    def answer(self, k: int = 3) -> dict:
        t = self.main
        return {"best": t.best(k), "worst": t.worst(k), "pooled": round(t.pooled, 4), "unknown": list(t.unknown)}


def spec_rows(spec: QuestionSpec, records: Sequence[ResearchOutcome], dim: str | None = None) -> list:
    """(key, bool) pairs for one dimension; rows failing the filter, with no outcome known, or with no key are skipped."""
    fn = spec.groupers[dim or spec.primary]
    out = []
    for o in records:
        if not spec.row_filter(o):
            continue
        y = spec.outcome(o)
        k = fn(o)
        if y is None or k is None:
            continue
        out.append((k, bool(y)))
    return out


def analyse_question(question: Q16, records: Sequence[ResearchOutcome], fdr_q: float = 0.10, min_n: int | None = None) -> QuestionReport:
    spec = SPECS[question]
    mn = spec.min_n if min_n is None else min_n
    tables = {dim: rate_table(spec_rows(spec, records, dim), spec.outcome_name, spec.higher_is_better, mn, fdr_q) for dim in spec.groupers}
    used = sum(1 for o in records if spec.row_filter(o) and spec.outcome(o) is not None)
    return QuestionReport(question, spec.text, tables, spec.primary, used)


def analyse_all(records: Sequence[ResearchOutcome], fdr_q: float = 0.10) -> dict:
    return {q: analyse_question(q, records, fdr_q) for q in Q16}


# ------------------------------------------------------------------------------------------------ Q04: repeated failure

@dataclass(frozen=True)
class RepresentationVerdict:
    representation: str
    n: int
    failures: int
    current_streak: int                             # consecutive most-recent failures
    p_recovers: float                               # P(true failure rate <= tolerable) under the posterior: small = keeps failing
    verdict: str                                    # REPEATEDLY_FAILS / MIXED / WORKS / UNKNOWN


def representation_failures(records: Sequence[ResearchOutcome], tolerable_failure: float = 0.6, streak: int = 4,
                            min_n: int = 5, alpha: float = 0.10) -> list:
    """Repeatedly failing = a long unbroken recent streak AND a posterior that the failure rate is tolerable below alpha.
    One bad week is not a failing representation; the streak alone is not enough either (a coin can run cold)."""
    by: dict = defaultdict(list)
    for o in sorted(records, key=lambda o: (to_ts(o.resolved_at), o.run_id)):
        by[o.representation].append(o.failed)
    out = []
    for rep in sorted(by):
        flags = by[rep]
        n, f = len(flags), sum(flags)
        if n < min_n:
            out.append(RepresentationVerdict(rep, n, f, _streak(flags, True), float("nan"), "UNKNOWN"))
            continue
        p_ok = float(sps.beta.cdf(tolerable_failure, 1 + f, 1 + n - f))          # P(rate <= tolerable)
        st = _streak(flags, True)
        if st >= streak and p_ok < alpha:
            v = "REPEATEDLY_FAILS"
        elif p_ok > 1 - alpha:
            v = "WORKS"
        else:
            v = "MIXED"
        out.append(RepresentationVerdict(rep, n, f, st, p_ok, v))
    return out


# ------------------------------------------------------------------------------------------------ Q05: multiplicity and false discoveries

@dataclass(frozen=True)
class DatasetBurden:
    dataset: str
    runs: int
    total_tests: int
    expected_false: float                           # what pure chance would hand out at `alpha`
    observed_false: int
    excess: float                                   # observed - expected: >0 means more false findings than chance alone explains
    rate_per_run: float


def dataset_burden(records: Sequence[ResearchOutcome], alpha: float = 0.05) -> list:
    """A dataset can 'create' false discoveries simply by being searched more. Compare the false discoveries it produced with
    the number pure multiple testing would produce at the stated alpha across all the comparisons its runs made."""
    by: dict = defaultdict(lambda: [0, 0, 0])
    for o in records:
        if o.false_discovery is None:
            continue
        row = by[o.dataset]
        row[0] += 1
        row[1] += o.n_tests
        row[2] += int(o.false_discovery)
    out = []
    for ds in sorted(by):
        runs, tests, fd = by[ds]
        exp = tests * alpha
        out.append(DatasetBurden(ds, runs, tests, exp, fd, fd - exp, fd / runs))
    return out


def search_size_effect(records: Sequence[ResearchOutcome]) -> dict:
    """Spearman correlation between log(n_tests) and false-discovery indicator: does searching wider manufacture noise?"""
    rows = [(math.log(o.n_tests), 1.0 if o.false_discovery else 0.0) for o in records if o.false_discovery is not None]
    if len(rows) < 10:
        return {"rho": None, "n": len(rows), "verdict": "INSUFFICIENT"}
    rho = spearman([r[0] for r in rows], [r[1] for r in rows])
    if rho is None:
        return {"rho": None, "n": len(rows), "verdict": "CONSTANT"}
    t = rho * math.sqrt((len(rows) - 2) / max(1e-12, 1 - rho * rho))
    p = float(2 * sps.t.sf(abs(t), len(rows) - 2))
    return {"rho": float(rho), "n": len(rows), "p": p, "verdict": "WIDER_SEARCH_MEANS_MORE_NOISE" if rho > 0 and p < 0.05 else "NO_EFFECT_SEEN"}


# ------------------------------------------------------------------------------------------------ Q06: validation methods

@dataclass(frozen=True)
class MethodScore:
    method: str
    n_with_error: int
    caught: int
    recall: float
    recall_lo: float
    n_clean: int
    false_alarms: int
    false_alarm_rate: float
    youden: float                                   # recall - false alarm rate: 0 = a coin flip, whatever it costs
    precision: float | None
    minutes_per_run: float
    minutes_per_catch: float | None


def method_scores(records: Sequence[ResearchOutcome], min_n: int = 4) -> list:
    """Recall on runs that really contained a defect, false-alarm rate on clean runs, and cost per catch. A method with
    recall 0.9 and false-alarm 0.9 catches everything by raising the alarm on everything (Youden J shows it)."""
    by: dict = defaultdict(list)
    for o in records:
        if o.error_present is not None and o.validator_flagged is not None:
            by[o.validation_method].append(o)
    out = []
    for m in sorted(by):
        rows = by[m]
        bad = [o for o in rows if o.error_present]
        ok = [o for o in rows if not o.error_present]
        if len(bad) + len(ok) < min_n:
            continue
        caught = sum(1 for o in bad if o.validator_flagged)
        fa = sum(1 for o in ok if o.validator_flagged)
        rec = caught / len(bad) if bad else float("nan")
        far = fa / len(ok) if ok else float("nan")
        flagged = caught + fa
        cost = float(np.mean([o.cost_minutes for o in rows]))
        out.append(MethodScore(m, len(bad), caught, rec, wilson(caught, len(bad))[0] if bad else 0.0, len(ok), fa, far,
                               (rec if bad else 0.0) - (far if ok else 0.0), caught / flagged if flagged else None, cost,
                               (sum(o.cost_minutes for o in rows) / caught) if caught else None))
    return sorted(out, key=lambda s: (-s.youden, s.method))


def catch_matrix(records: Sequence[ResearchOutcome]) -> dict:
    """error_id -> {method: flagged}. The same defect seen by several methods, which is what lets us ask who catches what."""
    mat: dict = defaultdict(dict)
    for o in records:
        if o.error_id and o.error_present and o.validator_flagged is not None:
            mat[o.error_id][o.validation_method] = bool(o.validator_flagged) or mat[o.error_id].get(o.validation_method, False)
    return dict(mat)


def greedy_method_cover(records: Sequence[ResearchOutcome], target_share: float = 0.95) -> dict:
    """Cheapest ordered set of methods that together catch `target_share` of the known defects (greedy set cover by new
    catches per minute). Also lists the defects NO method caught: the honest blind spot of the validation stack."""
    mat = catch_matrix(records)
    if not mat:
        return {"order": [], "coverage": 0.0, "missed_by_all": [], "n_errors": 0}
    cost: dict = defaultdict(list)
    for o in records:
        cost[o.validation_method].append(o.cost_minutes)
    mean_cost = {m: max(1e-6, float(np.mean(v))) for m, v in cost.items()}
    caught_by: dict = defaultdict(set)
    for eid, per in mat.items():
        for m, hit in per.items():
            if hit:
                caught_by[m].add(eid)
    universe = set(mat)
    reachable = set().union(*caught_by.values()) if caught_by else set()
    covered: set = set()
    order = []
    while len(covered) / len(universe) < target_share:
        best = max(sorted(caught_by), key=lambda m: (len(caught_by[m] - covered) / mean_cost[m], m), default=None)
        if best is None or not (caught_by[best] - covered):
            break
        fresh = len(caught_by[best] - covered)
        covered |= caught_by[best]
        order.append({"method": best, "new_catches": fresh, "cum_coverage": len(covered) / len(universe)})
    return {"order": order, "coverage": len(covered) / len(universe), "missed_by_all": sorted(universe - reachable), "n_errors": len(universe)}


def method_agreement(records: Sequence[ResearchOutcome]) -> dict:
    """Pairwise Jaccard of the defect sets two methods catch. Two methods that catch the same defects are redundant; the
    scheduler should not pay for both."""
    mat = catch_matrix(records)
    sets: dict = defaultdict(set)
    for eid, per in mat.items():
        for m, hit in per.items():
            if hit:
                sets[m].add(eid)
    out = {}
    ms = sorted(sets)
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            u = sets[a] | sets[b]
            out[f"{a}~{b}"] = len(sets[a] & sets[b]) / len(u) if u else 0.0
    return out


# ------------------------------------------------------------------------------------------------ Q07 / Q08: decisions and informative failures

@dataclass(frozen=True)
class DecisionYield:
    source: str
    runs: int
    decisions_changed: int
    minutes: float
    changes_per_hour: float
    share_of_all_changes: float
    share_of_compute: float
    efficiency: float                               # share of changes / share of compute; >1 = pays for its compute


def decision_yield(records: Sequence[ResearchOutcome]) -> list:
    """Which question sources change decisions, per unit of compute (not just per run: a cheap question that moves a
    threshold beats an expensive one that moves it as often)."""
    by: dict = defaultdict(lambda: [0, 0, 0.0])
    for o in records:
        if o.decision_changed is None:
            continue
        row = by[o.question_source]
        row[0] += 1
        row[1] += int(o.decision_changed)
        row[2] += o.cost_minutes
    tot_ch = sum(v[1] for v in by.values())
    tot_min = sum(v[2] for v in by.values())
    out = []
    for s in sorted(by):
        runs, ch, mins = by[s]
        sc = ch / tot_ch if tot_ch else 0.0
        sm = mins / tot_min if tot_min else 0.0
        out.append(DecisionYield(s, runs, ch, mins, (ch / (mins / 60.0)) if mins > 0 else 0.0, sc, sm, (sc / sm) if sm > 0 else 0.0))
    return sorted(out, key=lambda d: (-d.efficiency, d.source))


@dataclass(frozen=True)
class FailureInformation:
    exp_type: str
    failures: int
    informative: int
    dead: int                                       # failed AND taught nothing AND spawned nothing
    informative_rate: float
    bits_total: float
    bits_per_minute: float
    dead_minutes: float


def failure_information(records: Sequence[ResearchOutcome], min_bits: float = 0.05) -> list:
    """Among FAILED runs, which were worth having failed? A failed run with no bits and no follow-up is 'dead': it cost
    compute and left nothing. A failed run that eliminated hypotheses is progress (contract section 34)."""
    by: dict = defaultdict(list)
    for o in records:
        if o.failed:
            by[o.exp_type].append(o)
    out = []
    for t in sorted(by):
        rows = by[t]
        inf = [o for o in rows if informative(o, min_bits)]
        dead = [o for o in rows if not informative(o, min_bits)]
        mins = sum(o.cost_minutes for o in rows)
        bits = sum(o.info_bits for o in rows)
        out.append(FailureInformation(t, len(rows), len(inf), len(dead), len(inf) / len(rows), bits, bits / mins if mins > 0 else 0.0,
                                      sum(o.cost_minutes for o in dead)))
    return out


# ------------------------------------------------------------------------------------------------ Q09: wasted compute and stopping

@dataclass(frozen=True)
class PathVerdict:
    path: str
    runs: int
    useful: int
    minutes: float
    wasted_minutes: float
    waste_share: float                              # share of the path's compute spent on wasted runs
    p_useful_ok: float                              # P(true useful rate >= floor) under the posterior
    recent_barren: int                              # consecutive most-recent wasted runs
    action: str                                     # STOP / THROTTLE / KEEP / UNKNOWN


def path_action(useful: int, runs: int, barren: int, useful_floor: float = 0.15, stop_p: float = 0.05, throttle_p: float = 0.25,
                min_runs: int = 4, barren_run: int = 4) -> str:
    """The single stop/throttle rule, shared by the verdicts and by the counterfactual replay so they cannot drift apart."""
    if runs < min_runs:
        return "UNKNOWN"
    prob = posterior_rate_above(useful, runs, useful_floor)
    if prob < stop_p and barren >= barren_run:
        return "STOP"
    return "THROTTLE" if prob < throttle_p else "KEEP"


def path_verdicts(records: Sequence[ResearchOutcome], useful_floor: float = 0.15, stop_p: float = 0.05, throttle_p: float = 0.25,
                  min_runs: int = 4, barren_run: int = 4) -> list:
    """Sequential stop rule per research path (exp_type|family|representation). STOP when the posterior probability that the
    path still delivers at least `useful_floor` useful runs has fallen below `stop_p` AND the last `barren_run` runs were all
    wasted; THROTTLE below `throttle_p`. Never STOP on too few runs: unknown is not wasteful."""
    by: dict = defaultdict(list)
    for o in sorted(records, key=lambda o: (to_ts(o.resolved_at), o.run_id)):
        w = wasted(o)
        if w is not None:
            by[o.path].append((o, w))
    out = []
    for p in sorted(by):
        rows = by[p]
        useful = sum(1 for _, w in rows if not w)
        mins = sum(o.cost_minutes for o, _ in rows)
        wmins = sum(o.cost_minutes for o, w in rows if w)
        prob = posterior_rate_above(useful, len(rows), useful_floor)
        barren = _streak([w for _, w in rows], True)
        act = path_action(useful, len(rows), barren, useful_floor, stop_p, throttle_p, min_runs, barren_run)
        out.append(PathVerdict(p, len(rows), useful, mins, wmins, wmins / mins if mins > 0 else 0.0, prob, barren, act))
    return sorted(out, key=lambda v: (-v.wasted_minutes, v.path))


def compute_waste_summary(verdicts: Sequence[PathVerdict]) -> dict:
    tot = sum(v.minutes for v in verdicts)
    wasted_m = sum(v.wasted_minutes for v in verdicts)
    stop = [v for v in verdicts if v.action == "STOP"]
    return {"total_minutes": tot, "wasted_minutes": wasted_m, "waste_share": wasted_m / tot if tot else 0.0,
            "stop_paths": [v.path for v in stop], "minutes_recoverable": sum(v.wasted_minutes for v in stop),
            "throttle_paths": [v.path for v in verdicts if v.action == "THROTTLE"]}


# ------------------------------------------------------------------------------------------------ Q10-Q12: survival and where it breaks

@dataclass(frozen=True)
class SurvivalBreakdown:
    dimension: str                                  # era / stock_group / regime
    kind: str                                       # the family (or other kind) being asked about
    label: str                                      # the era / stock group / regime label
    n: int
    survived: int
    rate: float
    lo: float


def survival_breakdown(records: Sequence[ResearchOutcome], dimension: str, min_n: int = 3) -> list:
    """For discoveries tested in a new era/stock-group/regime: survival by (family, that label). The Simpson guard for Q10-12:
    a family that 'fails to transfer' may fail in one regime only, and then the rule is a regime rule, not a family rule."""
    flag = {"era": "new_era_survived", "stock_group": "new_stock_survived", "regime": "new_regime_survived"}[dimension]
    by: dict = defaultdict(lambda: [0, 0])
    for o in records:
        y = getattr(o, flag)
        lab = getattr(o, dimension)
        if y is None or not lab:
            continue
        by[(o.family, lab)][0] += int(bool(y))
        by[(o.family, lab)][1] += 1
    out = []
    for (fam, lab), (s, n) in sorted(by.items()):
        if n >= min_n:
            out.append(SurvivalBreakdown(dimension, fam, lab, n, s, s / n, wilson(s, n)[0]))
    return out


def breaks_only_in(records: Sequence[ResearchOutcome], dimension: str, family: str, min_n: int = 3, gap: float = 0.3) -> list:
    """Labels in which `family` survives markedly worse than in its other labels: a conditional break, not a family failure."""
    rows = [b for b in survival_breakdown(records, dimension, min_n) if b.kind == family]
    if len(rows) < 2:
        return []
    out = []
    for b in rows:
        rest = [x for x in rows if x.label != b.label]
        ref = sum(x.survived for x in rest) / max(1, sum(x.n for x in rest))
        if ref - b.rate >= gap:
            out.append(b.label)
    return sorted(out)


def survival_gradient(records: Sequence[ResearchOutcome]) -> dict:
    """family -> (era, stock, regime) survival rates side by side, to see whether a family is robust on all three or on one."""
    out: dict = {}
    for fam in sorted({o.family for o in records}):
        rows = [o for o in records if o.family == fam]
        row = {}
        for name, attr in (("era", "new_era_survived"), ("stock", "new_stock_survived"), ("regime", "new_regime_survived")):
            ys = [getattr(o, attr) for o in rows if getattr(o, attr) is not None]
            row[name] = (sum(bool(y) for y in ys) / len(ys)) if ys else None
            row[f"n_{name}"] = len(ys)
        known = [v for k, v in row.items() if not k.startswith("n_") and v is not None]
        row["weakest"] = min(known) if known else None
        out[fam] = row
    return out


# ------------------------------------------------------------------------------------------------ interactions

def interaction_table(records: Sequence[ResearchOutcome], dim_a: str, dim_b: str, outcome: Callable[[ResearchOutcome], bool | None],
                      higher_is_better: bool, min_n: int = 6, fdr_q: float = 0.10) -> RateTable:
    """Two-way cross (e.g. exp_type x dataset) under the same BH-corrected leave-group-out test. The number of cells grows fast,
    which is exactly why the correction matters: uncorrected cross-tables find 'interactions' in any noise."""
    pairs = []
    for o in records:
        y = outcome(o)
        a, b = getattr(o, dim_a, None), getattr(o, dim_b, None)
        if y is None or not a or not b:
            continue
        pairs.append((f"{a}&{b}", bool(y)))
    return rate_table(pairs, f"{dim_a}x{dim_b}", higher_is_better, min_n, fdr_q)


def drift_in_process(records: Sequence[ResearchOutcome], outcome: Callable[[ResearchOutcome], bool | None], delta: float = 0.02,
                     lam: float = 4.0) -> list:
    """Has the research process itself changed? Page-Hinkley (engine.learning.meta_learning.SurvivalDrift) over the outcome
    stream in resolution order: [(resolved_at, RATE_UP | RATE_DOWN)]. A shift means advice fitted before it may be stale."""
    det = SurvivalDrift(delta, lam)
    alarms = []
    for o in sorted(records, key=lambda o: (to_ts(o.resolved_at), o.run_id)):
        y = outcome(o)
        if y is None:
            continue
        a = det.update(bool(y))
        if a:
            alarms.append((o.resolved_at, a))
    return alarms


# ------------------------------------------------------------------------------------------------ configuration

@dataclass(frozen=True)
class MetaResearchConfig:
    fdr_q: float = 0.10                             # Benjamini-Hochberg level for every group table
    min_group_n: int = 5
    min_train: int = 25
    min_test: int = 30
    folds: int = 4
    informative_bits: float = 0.05
    useful_floor: float = 0.15                      # a path must still deliver this useful-run rate or it is a stop candidate
    stop_p: float = 0.05
    throttle_p: float = 0.25
    barren_run: int = 4
    budget_share: float = 0.30                      # scheduler replay: fraction of a fold's compute the scheduler may spend
    n_draws: int = 200                              # random-baseline draws in the scheduler replay
    multiplier_lo: float = 0.05
    multiplier_hi: float = 2.0
    require_oos: bool = True                        # a question that does not beat the pooled rate out of sample is withheld
    certified_real: bool = False                    # only the real-data validation wave may set this

    def check(self) -> list:
        errs = []
        if not (0 < self.fdr_q < 1):
            errs.append("fdr_q must be in (0, 1)")
        if self.folds < 2 or self.min_train < 5 or self.min_test < 5 or self.min_group_n < 2:
            errs.append("folds>=2, min_train>=5, min_test>=5, min_group_n>=2 required")
        if not (0 < self.budget_share < 1) or not (0 < self.multiplier_lo < 1 < self.multiplier_hi):
            errs.append("budget_share in (0,1) and multiplier_lo < 1 < multiplier_hi required")
        return errs


# ------------------------------------------------------------------------------------------------ same-year rerun leak (rule 27)

def release_view(rows: Sequence[ResearchOutcome], replay_years: Iterable = ()) -> list:
    """Research filed under a real year must NEVER be released while that same year is being replayed in disguise. The trusted
    curator passes the years currently under replay; every run filed under one of them is withheld, so a disguised rerun of
    year Y can never learn from research about year Y."""
    banned = {str(y) for y in replay_years}
    return [r for r in rows if not (r.evidence_year and r.evidence_year in banned)]


def assert_identity_free(payload: Any, what: str = "advice") -> None:
    """Nothing handed on may carry a calendar date or a year (sections 29-31). Scans every string in a nested structure."""
    stack = [payload]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            leak = identity_leak(x)
            if leak:
                raise FirewallBreach(f"{what} {leak}: {x!r}")
        elif isinstance(x, Mapping):
            stack.extend(list(x.keys()) + list(x.values()))
        elif isinstance(x, (list, tuple, set, frozenset)):
            stack.extend(x)


# ------------------------------------------------------------------------------------------------ out-of-sample evaluation of each question

def question_rows(spec: QuestionSpec, records: Sequence[ResearchOutcome]) -> list:
    """(started_at, resolved_at, group key, outcome) rows in the shape walk_forward_groups expects."""
    fn = spec.groupers[spec.primary]
    rows = []
    for o in records:
        if not spec.row_filter(o):
            continue
        y, k = spec.outcome(o), fn(o)
        if y is None or k is None:
            continue
        rows.append((o.started_at, o.resolved_at, k, bool(y)))
    return rows


def evaluate_questions(records: Sequence[ResearchOutcome], now, seed: int, cfg: MetaResearchConfig) -> dict:
    """Walk-forward test of every question's group predictor against the pooled rate, on runs it never saw. This is the
    'meta-learning must itself be evaluated out of sample' requirement, one OosResult per question."""
    out = {}
    for i, q in enumerate(Q16):
        rows = question_rows(SPECS[q], records)
        out[q] = walk_forward_groups(rows, now, seed + i, q.value, cfg.min_train, cfg.min_test, cfg.folds, cfg.certified_real)
    return out


def question_usable(oos: OosResult, cfg: MetaResearchConfig) -> tuple:
    """(usable, reason). Usable = beat the pooled rate out of sample and was not failed by a leak audit."""
    if oos.label == ValidationLabel.FAILED_VALIDATION.value:
        return False, oos.reason
    if oos.label == ValidationLabel.INSUFFICIENT_EVIDENCE.value:
        return False, oos.reason
    if cfg.require_oos and not oos.beats_baseline:
        return False, "group predictor does not beat the pooled rate out of sample"
    return True, oos.reason


def to_discovery_records(records: Sequence[ResearchOutcome], extra: Callable[[ResearchOutcome], Mapping] | None = None) -> list:
    """Adapter so the run-level survival model of meta_learning can be reused on research runs. Features are only what was
    known when the run STARTED (planned cost, free parameters, search width): info_bits and follow-ups are results, so
    feeding them would be a label leak. `extra` exists to let tests plant such a leak and prove the audit sees it."""
    out = []
    for o in records:
        if o.durable is None:
            continue
        feats = {"log_cost": math.log1p(o.cost_minutes), "free_parameters": float(o.free_parameters), "log_n_tests": math.log(o.n_tests)}
        if extra is not None:
            feats.update({k: float(v) for k, v in extra(o).items()})
        out.append(DiscoveryRecord(o.run_id, o.exp_type, o.started_at, feats, bool(o.durable), o.resolved_at))
    return out


def run_level_oos(records: Sequence[ResearchOutcome], now, seed: int, cfg: MetaResearchConfig,
                  extra: Callable[[ResearchOutcome], Mapping] | None = None) -> dict:
    """Can the durability of a run be predicted from its start-time properties alone, out of sample? Plus the leak audit."""
    recs = to_discovery_records(records, extra)
    res = walk_forward_survival(recs, now, seed, cfg.folds, cfg.min_train, cfg.min_test, certified_real=cfg.certified_real)
    return {"oos": res, "leaks": tuple(leak_audit([r for r in recs if r.survived_oos is not None])), "n": len(recs)}


# ------------------------------------------------------------------------------------------------ the advice

@dataclass(frozen=True)
class SchedulerAdvice:
    """What research meta-learning tells the scheduler. Every mapping holds only groups whose question passed its
    out-of-sample test AND whose own group had enough data: absent means unknown, never neutral."""
    fitted_through: str
    n_runs: int
    exp_type_durable: Mapping = field(default_factory=dict)         # Q01
    exp_type_overfit: Mapping = field(default_factory=dict)         # Q02
    family_overfit: Mapping = field(default_factory=dict)           # Q02 (family view)
    family_transfer: Mapping = field(default_factory=dict)          # Q03
    representation_failure: Mapping = field(default_factory=dict)   # Q04
    blocked_representations: tuple = ()                             # Q04 REPEATEDLY_FAILS
    dataset_false_discovery: Mapping = field(default_factory=dict)  # Q05
    validation_recall: Mapping = field(default_factory=dict)        # Q06
    validation_order: tuple = ()                                    # Q06 cheapest-first cover
    source_decision: Mapping = field(default_factory=dict)          # Q07 by question source
    target_decision: Mapping = field(default_factory=dict)          # Q07 by ResearchTarget
    exp_type_informative_failure: Mapping = field(default_factory=dict)   # Q08
    stop_paths: tuple = ()                                          # Q09
    throttle_paths: tuple = ()                                      # Q09
    survival_era: Mapping = field(default_factory=dict)             # Q10
    survival_stock: Mapping = field(default_factory=dict)           # Q11
    survival_regime: Mapping = field(default_factory=dict)          # Q12
    pooled: Mapping = field(default_factory=dict)                   # question -> pooled base rate (the reference for ratios)
    withheld: Mapping = field(default_factory=dict)                 # question -> why it was NOT passed on
    oos_label: str = ValidationLabel.INSUFFICIENT_EVIDENCE.value
    usable_questions: tuple = ()

    @property
    def advice_id(self) -> str:
        return "SA" + stable_hash(canonical_json(self), 10)

    def trust(self) -> float:
        """0 with no data, saturating with n, scaled by the share of questions that survived their OOS test, halved unless VALIDATED."""
        base = self.n_runs / (self.n_runs + 60.0)
        share = len(self.usable_questions) / len(Q16)
        return base * share * (1.0 if self.oos_label == ValidationLabel.VALIDATED.value else 0.5)

    def check(self) -> list:
        errs = []
        for name in ("exp_type_durable", "exp_type_overfit", "family_overfit", "family_transfer", "representation_failure",
                     "dataset_false_discovery", "validation_recall", "source_decision", "target_decision",
                     "exp_type_informative_failure", "survival_era", "survival_stock", "survival_regime", "pooled"):
            for k, v in getattr(self, name).items():
                if not (isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0):
                    errs.append(f"{name}[{k}]={v!r} outside [0,1]")
        for p in self.stop_paths:
            if p in self.throttle_paths:
                errs.append(f"path {p} is both STOP and THROTTLE")
        try:
            assert_identity_free(self.payload(), "advice")
        except FirewallBreach as e:
            errs.append(str(e))
        return errs

    def to_dict(self) -> dict:
        """Full trusted-side form, including fitted_through (a real date)."""
        return json.loads(canonical_json(self))

    def payload(self) -> dict:
        """What may be handed on: the same content without the real fit date, which lives in the record's provenance."""
        d = self.to_dict()
        d.pop("fitted_through", None)
        return d

    def to_json(self) -> str:
        return canonical_json(self)

    def to_meta_advice(self) -> MetaAdvice:
        """The C62 MetaAdvice consumed by engine.learning.research_policy. Only the overlap is carried: overfit by
        exp_type/family, transfer by family, yield by ResearchTarget, and the context survival rates."""
        valid_targets = {t.value for t in ResearchTarget}
        overfit = {**{k: v for k, v in self.exp_type_overfit.items()}, **self.family_overfit}
        ctx = {f"era:{k}": v for k, v in self.survival_era.items()}
        ctx.update({f"stock:{k}": v for k, v in self.survival_stock.items()})
        ctx.update({f"regime:{k}": v for k, v in self.survival_regime.items()})
        return MetaAdvice(family_overfit=dict(overfit), family_survival=dict(self.family_transfer),
                          target_yield={k: v for k, v in self.target_decision.items() if k in valid_targets},
                          context_transfer=ctx, explanation_precision={}, fitted_through=self.fitted_through,
                          n_observations=self.n_runs, oos_label=self.oos_label)

    def matured_record(self, created_real: str, code_hash: str | None = None, parents: Sequence[str] = ()) -> MaturedRecord:
        """Wrap the advice as a MATURED_RESEARCH fact. It can only enter live state via .gate(now), which fails closed unless
        every run it was fitted on matured strictly before now."""
        errs = self.check()
        if errs:
            raise ValueError("; ".join(errs))
        prov = Provenance(created_real=created_real, learned_at=self.fitted_through, code_hash=code_hash or current_code_hash(),
                          outcomes_seen_through=self.fitted_through, parents=tuple(parents))
        return MaturedRecord(self.advice_id, self.fitted_through, self.payload(), prov, Namespace.MATURED_RESEARCH)


def advice_from_json(text: str) -> SchedulerAdvice:
    d = json.loads(text)
    tuples = ("blocked_representations", "validation_order", "stop_paths", "throttle_paths", "usable_questions")
    kw = {k: d[k] for k in SchedulerAdvice.__dataclass_fields__ if k in d}
    for k in tuples:
        if k in kw:
            kw[k] = tuple(kw[k])
    return SchedulerAdvice(**kw)


def _shrunk(table: RateTable) -> dict:
    return {k: round(g.shrunk, 4) for k, g in table.groups.items()}


def build_advice(records: Sequence[ResearchOutcome], reports: Mapping, oos: Mapping, cfg: MetaResearchConfig, fitted_through: str,
                 verdicts: Sequence[PathVerdict], reps: Sequence[RepresentationVerdict], cover: Mapping, run_oos: Mapping | None = None) -> SchedulerAdvice:
    """Assemble the advice, passing on only questions that were usable out of sample."""
    usable, withheld = [], {}
    for q in Q16:
        ok, why = question_usable(oos[q], cfg)
        if ok:
            usable.append(q)
        else:
            withheld[q.value] = why
    us = set(usable)
    R = lambda q, dim=None: reports[q].tables[dim or reports[q].primary]
    take = lambda q, dim=None: _shrunk(R(q, dim)) if q in us else {}
    stop = tuple(sorted(v.path for v in verdicts if v.action == "STOP")) if Q16.WASTED_COMPUTE in us else ()
    throttle = tuple(sorted(v.path for v in verdicts if v.action == "THROTTLE")) if Q16.WASTED_COMPUTE in us else ()
    blocked = tuple(sorted(v.representation for v in reps if v.verdict == "REPEATEDLY_FAILS")) if Q16.FAILING_REPRESENTATIONS in us else ()
    pooled = {q.value: round(reports[q].main.pooled, 4) for q in usable}
    if not usable:
        label = ValidationLabel.INSUFFICIENT_EVIDENCE.value
    elif cfg.certified_real and all(oos[q].label == ValidationLabel.VALIDATED.value for q in usable):
        label = ValidationLabel.VALIDATED.value
    else:
        label = ValidationLabel.NOT_VALIDATED.value
    order = tuple(s["method"] for s in cover.get("order", ())) if Q16.VALIDATION_METHODS in us else ()
    return SchedulerAdvice(
        fitted_through=fitted_through, n_runs=len(records),
        exp_type_durable=take(Q16.DURABLE_EXPERIMENTS), exp_type_overfit=take(Q16.OVERFIT_TYPES, "exp_type"),
        family_overfit=take(Q16.OVERFIT_TYPES, "family"), family_transfer=take(Q16.TRANSFERRING_FAMILIES),
        representation_failure=take(Q16.FAILING_REPRESENTATIONS), blocked_representations=blocked,
        dataset_false_discovery=take(Q16.FALSE_DISCOVERY_DATASETS), validation_recall=take(Q16.VALIDATION_METHODS),
        validation_order=order, source_decision=take(Q16.DECISION_QUESTIONS, "question_source"),
        target_decision=take(Q16.DECISION_QUESTIONS, "target"),
        exp_type_informative_failure=take(Q16.INFORMATIVE_FAILURES), stop_paths=stop, throttle_paths=throttle,
        survival_era=take(Q16.ERA_SURVIVAL, "era"), survival_stock=take(Q16.STOCK_SURVIVAL, "stock_group"),
        survival_regime=take(Q16.REGIME_SURVIVAL, "regime"), pooled=pooled, withheld=withheld, oos_label=label,
        usable_questions=tuple(q.value for q in usable))


# ------------------------------------------------------------------------------------------------ feeding the scheduler

def descriptor(o: ResearchOutcome) -> dict:
    """The start-time description of a run: everything a scheduler knows BEFORE choosing it, and nothing it learns after."""
    return {"exp_type": o.exp_type, "family": o.family, "representation": o.representation, "dataset": o.dataset,
            "validation_method": o.validation_method, "question_source": o.question_source, "target": o.target}


def descriptor_from_candidate(c: Candidate) -> dict:
    """Read a research_policy.Candidate's declared choices from its config; family/target come from the candidate itself."""
    cfg = dict(c.config)
    return {"exp_type": str(cfg.get("exp_type", "")), "family": str(cfg.get("family", c.family)), "representation": str(cfg.get("representation", "")),
            "dataset": str(cfg.get("dataset", "")), "validation_method": str(cfg.get("validation_method", "")),
            "question_source": str(cfg.get("question_source", "")), "target": c.target.value}


@dataclass(frozen=True)
class ScheduleDecision:
    cid: str
    action: str                                     # RUN / THROTTLE / BLOCK
    multiplier: float                               # priority multiplier in [lo, hi]; 0 when BLOCK
    reasons: tuple


def _ratio(rate: float | None, pooled: float | None, invert: bool = False, cap: float = 2.0) -> float | None:
    if rate is None or pooled is None or pooled <= 1e-9 or pooled >= 1 - 1e-9:
        return None
    num, den = ((1 - rate), (1 - pooled)) if invert else (rate, pooled)
    return min(cap, max(1.0 / cap, num / den))


def schedule_decision(desc: Mapping, adv: SchedulerAdvice, cid: str = "", cfg: MetaResearchConfig | None = None) -> ScheduleDecision:
    """Turn advice into a priority multiplier for one candidate. Every factor is a ratio against the pooled rate of its own
    question (so 1.0 = 'no better than average'), the product is blended toward 1 by advice trust, and a candidate whose path
    is on the STOP list, or whose representation repeatedly fails, is blocked outright. No advice means multiplier 1."""
    cfg = cfg or MetaResearchConfig()
    reasons, factors = [], []
    path = f"{desc.get('exp_type', '')}|{desc.get('family', '')}|{desc.get('representation', '')}"
    if path in adv.stop_paths:
        return ScheduleDecision(cid, "BLOCK", 0.0, (f"path {path} is on the STOP list (wasted compute, posterior below floor)",))
    if desc.get("representation") in adv.blocked_representations:
        return ScheduleDecision(cid, "BLOCK", 0.0, (f"representation {desc['representation']} repeatedly fails",))
    P = adv.pooled
    checks = (
        ("durable", adv.exp_type_durable.get(desc.get("exp_type")), P.get(Q16.DURABLE_EXPERIMENTS.value), False),
        ("overfit", adv.exp_type_overfit.get(desc.get("exp_type")), P.get(Q16.OVERFIT_TYPES.value), True),
        ("false-discovery dataset", adv.dataset_false_discovery.get(desc.get("dataset")), P.get(Q16.FALSE_DISCOVERY_DATASETS.value), True),
        ("decision source", adv.source_decision.get(desc.get("question_source")), P.get(Q16.DECISION_QUESTIONS.value), False),
        ("family transfer", adv.family_transfer.get(desc.get("family")), P.get(Q16.TRANSFERRING_FAMILIES.value), False),
        ("validation recall", adv.validation_recall.get(desc.get("validation_method")), P.get(Q16.VALIDATION_METHODS.value), False),
        ("informative failure", adv.exp_type_informative_failure.get(desc.get("exp_type")), P.get(Q16.INFORMATIVE_FAILURES.value), False),
        ("failing representation", adv.representation_failure.get(desc.get("representation")), P.get(Q16.FAILING_REPRESENTATIONS.value), True))
    for name, rate, pooled, invert in checks:
        r = _ratio(rate, pooled, invert)
        if r is not None:
            factors.append(r)
            reasons.append(f"{name} x{r:.2f}")
    if path in adv.throttle_paths:
        factors.append(0.4)
        reasons.append("path throttled (posterior of a useful run is low)")
    raw = float(np.prod(factors)) if factors else 1.0
    w = adv.trust()
    m = 1.0 + w * (raw - 1.0)
    m = min(cfg.multiplier_hi, max(cfg.multiplier_lo, m))
    if not factors:
        return ScheduleDecision(cid, "RUN", 1.0, ("no usable advice for this candidate: unknown, not neutral",))
    return ScheduleDecision(cid, "THROTTLE" if path in adv.throttle_paths else "RUN", m, tuple(reasons))


def rank_candidates(candidates: Sequence[Candidate], adv: SchedulerAdvice, base_scores: Mapping | None = None,
                    cfg: MetaResearchConfig | None = None) -> list:
    """Re-rank scheduler candidates by (base score x meta multiplier). Blocked candidates go last with score 0. Stable and
    deterministic: ties break on candidate id."""
    base_scores = base_scores or {}
    out = []
    for c in candidates:
        d = schedule_decision(descriptor_from_candidate(c), adv, c.cid, cfg)
        out.append((float(base_scores.get(c.cid, 1.0)) * d.multiplier, c.cid, d))
    out.sort(key=lambda t: (-t[0], t[1]))
    return [(cid, score, d) for score, cid, d in out]


def adjust_candidate(c: Candidate, adv: SchedulerAdvice) -> Candidate:
    """Feed the advice into a Candidate's own factors: the C62 meta adjustment plus the exp_type overfit rate as a floor on
    the candidate's overfit_hint (a caller may believe a search is safe; the record of that experiment type disagrees)."""
    c2 = meta_adjusted_factors(c, adv.to_meta_advice())
    d = descriptor_from_candidate(c2)
    rate = adv.exp_type_overfit.get(d["exp_type"])
    if rate is not None and adv.trust() > 0:
        hint = max(c2.overfit_hint, min(1.0, rate * min(1.0, adv.trust() * 2)))
        c2 = replace(c2, overfit_hint=hint)
    return c2


def policy_context_with(ctx, adv: SchedulerAdvice):
    """A research_policy.PolicyContext whose meta field is this advice. The original context is untouched."""
    return replace(ctx, meta=adv.to_meta_advice())


# ------------------------------------------------------------------------------------------------ does the advice actually help? (scheduler replay)

@dataclass(frozen=True)
class SchedulerEval:
    """Counterfactual replay: at each fold the advice is fitted on runs resolved before the fold, then asked to pick runs
    within a compute budget from the fold's runs. Compared with random selection of the same budget."""
    folds: int
    n_test: int
    excluded_advice_chosen: int                     # test runs the scheduler had itself selected: not an unbiased sample
    advice_durable: int
    random_durable_mean: float
    random_durable_p95: float
    lift: float                                     # advice - random mean, in durable runs found
    p_value: float                                  # share of random draws at least as good as the advice
    beats_random: bool
    label: str
    reason: str
    per_fold: tuple = ()


def _budgeted_pick(order: Sequence[int], costs: np.ndarray, budget: float) -> list:
    """Take runs in `order` while they fit in the budget (a run that does not fit is skipped, a cheaper later one may)."""
    used, picked = 0.0, []
    for i in order:
        if used + costs[i] <= budget:
            picked.append(i)
            used += costs[i]
    return picked


def evaluate_scheduler(rows: Sequence[ResearchOutcome], now, seed: int, cfg: MetaResearchConfig,
                       guard: SelfTrainingGuard | None = None,
                       advice_transform: Callable[[SchedulerAdvice], SchedulerAdvice] | None = None) -> SchedulerEval:
    """The out-of-sample test of the scheduler advice. Training rows for a fold are screened by the guard (no evaluation
    output, nothing resolved after the fold begins); test rows the scheduler itself chose are excluded."""
    guard = guard or SelfTrainingGuard()
    cut = to_ts(now)
    lab = sorted((r for r in rows if r.durable is not None and to_ts(r.resolved_at) < cut), key=lambda r: (to_ts(r.started_at), r.run_id))
    insufficient = lambda why: SchedulerEval(0, 0, 0, 0, float("nan"), float("nan"), float("nan"), 1.0, False,
                                             ValidationLabel.INSUFFICIENT_EVIDENCE.value, why)
    if len(lab) < cfg.min_train + cfg.folds:
        return insufficient(f"only {len(lab)} resolved runs")
    rng = np.random.default_rng(seed)
    edges = np.linspace(len(lab) // 2, len(lab), cfg.folds + 1).astype(int)
    total_adv, n_test, excluded, folds = 0, 0, 0, 0
    rand_totals = np.zeros(cfg.n_draws)
    per_fold = []
    for a, b in zip(edges[:-1], edges[1:]):
        chunk = lab[a:b]
        if not chunk:
            continue
        start = to_ts(chunk[0].started_at)
        train, _ = guard.screen([r for r in lab[:a] if to_ts(r.resolved_at) < start], start)
        test = [r for r in chunk if not r.chosen_by]
        excluded += len(chunk) - len(test)
        if len(train) < cfg.min_train or len(test) < 5:
            continue
        adv = fit_advice_quick(train, str(chunk[0].started_at), cfg, seed + folds)
        if advice_transform is not None:
            adv = advice_transform(adv)
        costs = np.array([max(1e-6, r.cost_minutes) for r in test])
        budget = cfg.budget_share * float(costs.sum())
        mult = [schedule_decision(descriptor(r), adv, r.run_id, cfg).multiplier for r in test]
        order = sorted(range(len(test)), key=lambda i: (-mult[i] / costs[i] ** 0.5, test[i].run_id))
        picked = _budgeted_pick(order, costs, budget)
        found = sum(1 for i in picked if test[i].durable)
        rand_found = np.zeros(cfg.n_draws)
        for d in range(cfg.n_draws):
            rp = _budgeted_pick(list(rng.permutation(len(test))), costs, budget)
            rand_found[d] = sum(1 for i in rp if test[i].durable)
        rand_totals += rand_found
        total_adv += found
        n_test += len(test)
        folds += 1
        per_fold.append({"fold": folds, "n_test": len(test), "advice_durable": found, "random_mean": float(rand_found.mean())})
    if not folds:
        return insufficient("no fold had enough clean training rows")
    p = float((1 + np.sum(rand_totals >= total_adv)) / (1 + cfg.n_draws))
    beats = bool(total_adv > rand_totals.mean() and p < 0.05)
    if n_test < cfg.min_test:
        label, why = ValidationLabel.INSUFFICIENT_EVIDENCE.value, f"only {n_test} test runs"
    elif not beats:
        label, why = ValidationLabel.FAILED_VALIDATION.value, "advice-ranked selection does not beat random selection of the same compute"
    elif not cfg.certified_real:
        label, why = ValidationLabel.NOT_VALIDATED.value, "beats random on records not certified as real: mechanism only"
    else:
        label, why = ValidationLabel.VALIDATED.value, "beats random out of sample on certified real records"
    return SchedulerEval(folds, n_test, excluded, int(total_adv), float(rand_totals.mean()), float(np.quantile(rand_totals, 0.95)),
                         float(total_adv - rand_totals.mean()), p, beats, label, why, tuple(per_fold))


def fit_advice_quick(train: Sequence[ResearchOutcome], now, cfg: MetaResearchConfig, seed: int) -> SchedulerAdvice:
    """Advice from the group tables alone (no per-question OOS pass): used INSIDE the replay, where the outer walk-forward
    is the out-of-sample test. Reports the questions whose tables are informative in-sample; the replay judges the result."""
    reports = analyse_all(train, cfg.fdr_q)
    fake = {q: OosResult(q.value, 0, 0, 0, 0.0, 0.0, None, {}, True, (), ValidationLabel.NOT_VALIDATED.value, "inner fit") for q in Q16}
    verdicts = path_verdicts(train, cfg.useful_floor, cfg.stop_p, cfg.throttle_p, 4, cfg.barren_run)
    reps = representation_failures(train, min_n=cfg.min_group_n)
    through = max((r.resolved_at for r in train), key=to_ts) if train else str(now)
    return build_advice(train, reports, fake, cfg, through, verdicts, reps, greedy_method_cover(train))


# ------------------------------------------------------------------------------------------------ the update: all twelve questions at once

@dataclass(frozen=True)
class ResearchMetaUpdate:
    now: str
    counts: Mapping
    guard: GuardReport
    reports: Mapping                                # Q16 -> QuestionReport
    oos: Mapping                                    # Q16 -> OosResult
    usable: tuple                                   # Q16 values that passed their OOS test
    run_level: Mapping                              # run-level durability predictor: oos, leaks
    scheduler: SchedulerEval | None
    paths: tuple                                    # PathVerdict
    representations: tuple                          # RepresentationVerdict
    methods: tuple                                  # MethodScore
    cover: Mapping
    decisions: tuple                                # DecisionYield
    failures: tuple                                 # FailureInformation
    burden: tuple                                   # DatasetBurden
    search_effect: Mapping
    breakdowns: Mapping                             # dimension -> list[SurvivalBreakdown]
    gradient: Mapping
    drift: Mapping                                  # name -> alarms
    advice: SchedulerAdvice
    store_hash: str
    code_hash: str
    label: str = LABEL

    def summary(self) -> dict:
        return {"now": self.now, "counts": dict(self.counts), "label": self.label, "store_hash": self.store_hash,
                "usable": list(self.usable), "withheld": dict(self.advice.withheld), "advice_id": self.advice.advice_id,
                "trust": round(self.advice.trust(), 4),
                "scheduler": None if self.scheduler is None else {"label": self.scheduler.label, "lift": self.scheduler.lift,
                                                                  "p": self.scheduler.p_value, "beats_random": self.scheduler.beats_random},
                "stop_paths": list(self.advice.stop_paths), "blocked_representations": list(self.advice.blocked_representations),
                "refused_rows": len(self.guard.refused), "leaks": [k for k, _ in self.run_level["leaks"]]}


class MetaResearchState:
    """Everything the research meta-learner owns: the outcome store, the sealed evaluations, the guard, the config, and its
    own history. One instance per research world. It lives entirely in MATURED_RESEARCH_STATE."""

    namespace = Namespace.MATURED_RESEARCH

    def __init__(self, store: OutcomeStore | None = None, cfg: MetaResearchConfig | None = None, vault: EvaluationVault | None = None,
                 strict_guard: bool = True):
        self.store = OutcomeStore() if store is None else store
        self.cfg = cfg or MetaResearchConfig()
        errs = self.cfg.check()
        if errs:
            raise ValueError("; ".join(errs))
        self.vault = EvaluationVault() if vault is None else vault
        self.guard = SelfTrainingGuard(self.vault, strict_guard)
        self.history: list = []
        self.last_advice: SchedulerAdvice | None = None

    def ingest(self, rows: Iterable[ResearchOutcome]) -> int:
        return self.store.extend(rows)


def fit_update(state: MetaResearchState, now, seed: int = 0, replay_years: Iterable = (), evaluate: bool = True) -> ResearchMetaUpdate:
    """Fit all twelve analyses on the runs visible at `now`, test each out of sample, and assemble the scheduler advice."""
    cfg = state.cfg
    visible = release_view(state.store.as_of(now), replay_years)
    train, grep = state.guard.screen(visible, now)
    reports = analyse_all(train, cfg.fdr_q)
    oos = evaluate_questions(train, now, seed, cfg)
    verdicts = path_verdicts(train, cfg.useful_floor, cfg.stop_p, cfg.throttle_p, 4, cfg.barren_run)
    reps = representation_failures(train, min_n=cfg.min_group_n)
    cover = greedy_method_cover(train)
    through = max((r.resolved_at for r in train), key=to_ts) if train else str(now)
    advice = build_advice(train, reports, oos, cfg, through, verdicts, reps, cover)
    run_level = run_level_oos(train, now, seed + 101, cfg)
    sched = evaluate_scheduler(train, now, seed + 202, cfg, state.guard) if evaluate else None
    if sched is not None and sched.label == ValidationLabel.FAILED_VALIDATION.value and advice.usable_questions:
        advice = replace(advice, oos_label=ValidationLabel.FAILED_VALIDATION.value)
    errs = advice.check()
    if errs:
        raise ValueError("; ".join(errs))
    drift = {"durable": drift_in_process(train, lambda o: o.durable), "overfit": drift_in_process(train, lambda o: o.overfit),
             "decision_changed": drift_in_process(train, lambda o: o.decision_changed)}
    return ResearchMetaUpdate(
        str(now), state.store.counts(now), grep, reports, oos, advice.usable_questions, run_level, sched, tuple(verdicts), tuple(reps),
        tuple(method_scores(train)), cover, tuple(decision_yield(train)), tuple(failure_information(train, cfg.informative_bits)),
        tuple(dataset_burden(train)), search_size_effect(train),
        {d: survival_breakdown(train, d) for d in ("era", "stock_group", "regime")}, survival_gradient(train), drift, advice,
        state.store.content_hash(), current_code_hash())


@dataclass(frozen=True)
class StepResult:
    update: ResearchMetaUpdate
    advice: SchedulerAdvice
    matured: MaturedRecord | None                   # None when nothing survived its out-of-sample test
    seal: EvalSeal | None                           # the scheduler evaluation, now quarantined from training
    verdict: GateVerdict


def step(state: MetaResearchState, now, seed: int = 0, replay_years: Iterable = (), evaluate: bool = True) -> StepResult:
    """PUBLIC ENTRY (wave-2 research loop calls this once per research cycle). Fit, test out of sample, quarantine the
    evaluation result, and emit advice as a MATURED_RESEARCH record. Nothing here reaches the blind trader: the record's
    .gate(now) is the only door, and it fails closed."""
    upd = fit_update(state, now, seed, replay_years, evaluate)
    seal = None
    if upd.scheduler is not None and math.isfinite(upd.scheduler.lift):
        seal = state.vault.seal("scheduler_replay", upd.advice.fitted_through, now, upd.scheduler.lift,
                                [r.run_id for r in state.store.as_of(now)][-50:])
    if not upd.advice.usable_questions:
        verdict = GateVerdict.NEEDS_MORE_EVIDENCE
        matured = None
    elif upd.advice.oos_label == ValidationLabel.FAILED_VALIDATION.value:
        verdict, matured = GateVerdict.FAILED, None
    else:
        verdict = GateVerdict.PROMOTE if upd.advice.oos_label == ValidationLabel.VALIDATED.value else GateVerdict.UNKNOWN
        matured = upd.advice.matured_record(str(now), parents=[seal.eval_id] if seal else ())
    state.history.append(upd.summary())
    state.last_advice = upd.advice
    return StepResult(upd, upd.advice, matured, seal, verdict)


def save_history(state: MetaResearchState, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(canonical_json(state.history[-1]) + "\n")


# ------------------------------------------------------------------------------------------------ learning about learning, over time

def research_learning_curve(store: OutcomeStore, checkpoints: Sequence[str], seed: int = 0, cfg: MetaResearchConfig | None = None) -> list:
    """Does meta-research improve as the record grows? At each checkpoint the advice is refitted using ONLY runs resolved
    before it and scored by the scheduler replay. A flat or falling curve means the meta-learner is not learning."""
    cfg = cfg or MetaResearchConfig()
    out = []
    for cp in checkpoints:
        st = MetaResearchState(OutcomeStore(store.as_of(cp)), cfg)
        oos = evaluate_questions(st.store.as_of(cp), cp, seed, cfg)
        usable = [q.value for q in Q16 if question_usable(oos[q], cfg)[0]]
        gains = [oos[q].baseline_brier - oos[q].model_brier for q in Q16 if oos[q].n_test > 0 and math.isfinite(oos[q].model_brier)]
        sch = evaluate_scheduler(st.store.as_of(cp), cp, seed + 1, cfg, st.guard)
        out.append({"checkpoint": str(cp), "n": len(st.store), "usable": len(usable),
                    "mean_brier_gain": float(np.mean(gains)) if gains else None, "scheduler_lift": sch.lift if math.isfinite(sch.lift) else None,
                    "scheduler_label": sch.label})
    return out


def advice_stability(old: SchedulerAdvice, new: SchedulerAdvice) -> dict:
    """Rank agreement (Spearman) of the exp-type durability ordering between two fits. Advice that reshuffles on every refit
    is noise, however well it scored once."""
    keys = sorted(set(old.exp_type_durable) & set(new.exp_type_durable))
    out = {"n_common": len(keys), "rho": None, "stop_added": sorted(set(new.stop_paths) - set(old.stop_paths)),
           "stop_removed": sorted(set(old.stop_paths) - set(new.stop_paths))}
    if len(keys) >= 3:
        out["rho"] = spearman([old.exp_type_durable[k] for k in keys], [new.exp_type_durable[k] for k in keys])
    return out


def stale_discount(adv: SchedulerAdvice, now, half_life_days: float = 365.0) -> SchedulerAdvice:
    """Advice ages: rates shrink toward the pooled rate of their question as the fit gets old (the research world changes).
    Structure (stop lists, blocks) is kept: a stale STOP is still a reason to re-test, not to keep spending."""
    age = max(0.0, (to_ts(now) - to_ts(adv.fitted_through)).total_seconds() / 86400.0)
    w = 0.5 ** (age / half_life_days)
    def pull(m: Mapping, pooled: float | None) -> dict:
        return dict(m) if pooled is None else {k: round(pooled + w * (v - pooled), 4) for k, v in m.items()}
    P = adv.pooled
    return replace(adv, exp_type_durable=pull(adv.exp_type_durable, P.get(Q16.DURABLE_EXPERIMENTS.value)),
                   exp_type_overfit=pull(adv.exp_type_overfit, P.get(Q16.OVERFIT_TYPES.value)),
                   family_transfer=pull(adv.family_transfer, P.get(Q16.TRANSFERRING_FAMILIES.value)),
                   dataset_false_discovery=pull(adv.dataset_false_discovery, P.get(Q16.FALSE_DISCOVERY_DATASETS.value)),
                   source_decision=pull(adv.source_decision, P.get(Q16.DECISION_QUESTIONS.value)))


# ------------------------------------------------------------------------------------------------ ingestion from the existing research ledgers

def sanitize_label(text: str) -> str:
    """Make a free-text label identity-free: dates and years are dropped, everything else becomes snake_case."""
    import re
    t = re.sub(r"(?<!\d)(19|20)\d{2}(-\d{2}(-\d{2})?)?s?(?!\d)", " ", str(text))
    t = re.sub(r"[^a-zA-Z0-9]+", "_", t).strip("_").lower()
    return t or "unspecified"


def outcomes_from_ledger(ledger, now, annotations: Mapping[str, Mapping], default_dataset: str = "unspecified") -> list:
    """ResearchOutcomes from engine.learning.experiment_memory.ExperimentLedger. The ledger knows what was run and what it
    said; whether the knowledge proved DURABLE is learned later, so it arrives through `annotations` (experiment_id ->
    fields incl. resolved_at). An experiment without an annotation is still unresolved and yields nothing."""
    out = []
    for eid, rec in sorted(ledger.view(now).items()):
        note = annotations.get(eid)
        if note is None or rec.result is None or rec.status not in ("ANSWERED", "INCONCLUSIVE"):
            continue
        cfg = dict(rec.experiment.config)
        bits = 0.0
        if rec.belief_update is not None and rec.belief_update.prior and rec.belief_update.posterior:
            from engine.learning.research_policy import realised_bits
            bits = realised_bits(rec.belief_update.prior, rec.belief_update.posterior)
        base = dict(run_id=eid, exp_type=sanitize_label(cfg.get("exp_type", rec.experiment.target or "unspecified")),
                    family=sanitize_label(cfg.get("family", "unspecified")), representation=sanitize_label(cfg.get("representation", "unspecified")),
                    dataset=sanitize_label(cfg.get("dataset", default_dataset)), validation_method=sanitize_label(cfg.get("validation_method", "unspecified")),
                    question_source=sanitize_label(cfg.get("question_source", "unspecified")), target=rec.experiment.target or "UNKNOWN_AREA",
                    started_at=rec.created_at, resolved_at=rec.result.observed_at, cost_minutes=float(rec.experiment.cost_minutes),
                    failed=rec.result.kind == "REFUTED", info_bits=bits, followups=1 if rec.next_action.strip() else 0)
        base.update(note)
        out.append(ResearchOutcome(**base))
    return out


def outcomes_from_failed_learners(registry, now, lag_days: int = 1, dataset: str = "learner_harness") -> list:
    """Every failed learner is a failed research run (contract section 17). The registry already holds cause, mode and
    generalisation; they map onto overfit / false_discovery / informative-failure fields. Cost is unrecorded there, so it
    is 0 rather than invented."""
    import datetime as dt
    from engine.learning.failed_learners import FailureMode, Generalization
    out = []
    for fl in registry.as_of(now):
        rec_at = to_ts(fl.recorded_at)
        known = fl.generalization in (Generalization.GENERALIZED, Generalization.REGIME_SPECIFIC)
        tags = sorted(fl.mechanism_tags)
        out.append(ResearchOutcome(
            run_id=f"FL-{fl.key}", exp_type="learner_evaluation", family=sanitize_label(fl.family or "learner"),
            representation=sanitize_label(tags[0]) if tags else "unspecified", dataset=dataset, validation_method="learner_harness",
            question_source="learner_proposal", target="NEW_REPRESENTATION", started_at=(rec_at - dt.timedelta(days=lag_days)).isoformat(),
            resolved_at=rec_at.isoformat(), cost_minutes=0.0, failed=True, info_bits=1.0 if (known and fl.reason.strip().upper() != "UNKNOWN") else 0.0,
            durable=False, overfit=fl.failure_mode in (FailureMode.OVERFIT, FailureMode.MEMORISATION),
            false_discovery=fl.failure_mode in (FailureMode.LEAKAGE, FailureMode.UNVALIDATED_CLAIM),
            followups=1 if known else 0, new_regime_survived=False if fl.regimes_failed else None))
    return out


def outcomes_from_realised_gains(gains: Iterable, lag_days: int = 1, dataset: str = "policy_history") -> list:
    """research_policy.RealisedGain history -> outcomes. Only what RealisedGain records is filled: durability and validation
    fields stay None (unknown), never guessed."""
    import datetime as dt
    out = []
    for g in gains:
        when = to_ts(g.when)
        out.append(ResearchOutcome(
            run_id=f"RG-{g.cid}", exp_type=sanitize_label(g.target.value), family=sanitize_label(g.family or "unspecified"),
            representation="unspecified", dataset=dataset, validation_method="belief_update", question_source="queue",
            target=g.target.value, started_at=(when - dt.timedelta(days=lag_days)).isoformat(), resolved_at=when.isoformat(),
            cost_minutes=float(g.cost_minutes), info_bits=float(g.realised_bits), decision_changed=bool(g.useful)))
    return out


# ------------------------------------------------------------------------------------------------ planted world (for C63 / tests)

WORLD_TRUTH = {
    "exp_durable": {"transfer_test": 0.60, "break_study": 0.55, "ablation": 0.40, "pattern_search": 0.25, "representation_probe": 0.12},
    "exp_overfit": {"transfer_test": 0.08, "break_study": 0.10, "ablation": 0.15, "pattern_search": 0.50, "representation_probe": 0.45},
    "rep_fail": {"raw_returns": 0.25, "rank_transform": 0.30, "zscore_context": 0.30, "fourier_bands": 0.92},
    "dataset_fd": {"panel_delisted": 0.06, "movers_only": 0.20, "panel_survivor": 0.38, "intraday_synth": 0.55},
    "source_decision": {"loss": 0.55, "surprise": 0.50, "break": 0.40, "missed_winner": 0.32, "curiosity": 0.05},
    "family_survive": {"vol_like": 0.72, "momentum_like": 0.55, "calendar_like": 0.15, "tiny_sample": 0.08},
    "method": {"walk_forward": (0.85, 0.08, 10.0), "permutation_null": (0.60, 0.05, 4.0), "holdout": (0.45, 0.10, 2.0), "eyeball": (0.08, 0.08, 0.5)},
    "regime_break": {"family": "vol_like", "regime": "crisis", "penalty": 0.5},
}


def synthetic_research_world(n: int, seed: int, spacing_days: int = 2, resolve_lag_days: int = 60, start: str = "2001-01-03",
                             n_defects: int = 80, dead_path: bool = True, truth: Mapping | None = None) -> list:
    """A planted research history with known truth (WORLD_TRUTH): per experiment type a hidden durable/overfit rate, one
    representation that almost always fails, datasets with different false-discovery rates, question sources with different
    decision yields, families with different survival, one regime in which one family breaks, and a validation stack
    whose methods differ in recall, false alarms and cost. `dead_path` plants a path that never produces anything."""
    import datetime as dt
    T = dict(truth or WORLD_TRUTH)
    rng = np.random.default_rng(seed)
    t0 = dt.date.fromisoformat(start)
    exps, reps, dss = sorted(T["exp_durable"]), sorted(T["rep_fail"]), sorted(T["dataset_fd"])
    srcs, fams = sorted(T["source_decision"]), sorted(T["family_survive"])
    methods = sorted(T["method"])
    targets = sorted(t.value for t in ResearchTarget)
    eras, groups, regimes = ["E1", "E2", "E3"], ["large", "mid", "small"], ["calm", "trend", "crisis"]
    out = []
    for i in range(n):
        et, rp, ds = exps[int(rng.integers(len(exps)))], reps[int(rng.integers(len(reps)))], dss[int(rng.integers(len(dss)))]
        src, fam = srcs[int(rng.integers(len(srcs)))], fams[int(rng.integers(len(fams)))]
        m = methods[int(rng.integers(len(methods)))]
        dead = dead_path and et == "representation_probe" and fam == "tiny_sample" and rp == "fourier_bands"
        failed = bool(rng.random() < T["rep_fail"][rp]) or dead
        overfit = bool(rng.random() < T["exp_overfit"][et]) and not dead
        fd = bool(rng.random() < T["dataset_fd"][ds]) and not dead
        durable = (not overfit) and (not fd) and (not failed) and bool(rng.random() < min(0.98, T["exp_durable"][et] / max(0.05, (1 - T["exp_overfit"][et]) * 0.8))) and not dead
        era, grp, reg = eras[int(rng.integers(3))], groups[int(rng.integers(3))], regimes[int(rng.integers(3))]
        p_surv = T["family_survive"][fam]
        pen = T["regime_break"]
        p_reg = p_surv - (pen["penalty"] if (fam == pen["family"] and reg == pen["regime"]) else 0.0)
        tested = rng.random() < 0.6
        d0 = t0 + dt.timedelta(days=i * spacing_days)
        bits = 0.0 if dead else float(rng.exponential(0.25 if failed else 0.15))
        informative_fail = failed and (not dead) and bool(rng.random() < (0.7 if et != "representation_probe" else 0.25))
        if informative_fail:
            bits = max(bits, 0.2)
        out.append(ResearchOutcome(
            f"r{i:05d}", et, fam, rp, ds, m, src, targets[int(rng.integers(len(targets)))], d0.isoformat(),
            (d0 + dt.timedelta(days=resolve_lag_days)).isoformat(), float(rng.lognormal(math.log(8.0), 0.5)), failed, bits,
            int(np.exp(rng.normal(3.0, 1.2))) + 1 if ds != "panel_delisted" else int(np.exp(rng.normal(2.0, 0.8))) + 1,
            int(rng.integers(0, 20)), durable, overfit, fd, (bool(rng.random() < T["source_decision"][src]) and not dead),
            1 if informative_fail else 0, None, None, "", (bool(rng.random() < p_surv) if tested else None) if durable or not failed else None,
            (bool(rng.random() < p_surv) if tested else None) if durable or not failed else None,
            (bool(rng.random() < p_reg) if tested else None) if durable or not failed else None, era, grp, reg))
    for j in range(n_defects):
        present = bool(rng.random() < 0.6)
        d0 = t0 + dt.timedelta(days=int(rng.integers(0, max(1, n * spacing_days))))
        for m in methods:
            rec, far, cost = T["method"][m]
            flagged = bool(rng.random() < (rec if present else far))
            out.append(ResearchOutcome(
                f"v{j:04d}_{m}", "validation_audit", "audit", "raw_returns", "panel_survivor", m, "audit", "DATA_QUALITY",
                d0.isoformat(), (d0 + dt.timedelta(days=resolve_lag_days)).isoformat(), cost * float(rng.uniform(0.8, 1.2)),
                error_present=present, validator_flagged=flagged, error_id=f"defect{j:04d}" if present else ""))
    return sorted(out, key=lambda o: (o.started_at, o.run_id))


def truth_recovery(update: ResearchMetaUpdate, truth: Mapping | None = None) -> dict:
    """Spearman agreement between what the meta-learner learned and the planted truth, per question. Only meaningful on
    planted worlds (no truth exists in real data); C63 uses it to prove the analyses can find a known effect."""
    T = dict(truth or WORLD_TRUTH)
    out = {}
    def rho(learned: Mapping, true: Mapping, sign: int = 1) -> float | None:
        ks = sorted(set(learned) & set(true))
        if len(ks) < 3:
            return None
        return spearman([learned[k] for k in ks], [sign * true[k] for k in ks])
    a = update.advice
    out["Q01"] = rho(a.exp_type_durable, T["exp_durable"])
    out["Q02"] = rho(a.exp_type_overfit, T["exp_overfit"])
    out["Q04"] = rho(a.representation_failure, T["rep_fail"])
    out["Q05"] = rho(a.dataset_false_discovery, T["dataset_fd"])
    out["Q07"] = rho(a.source_decision, T["source_decision"])
    out["Q03"] = rho(a.family_transfer, T["family_survive"])
    return out


# ------------------------------------------------------------------------------------------------ reports

def render(u: ResearchMetaUpdate) -> str:
    """Plain-text report: one section per section-16 question, then the scheduler test."""
    L = [f"Research meta-learning as of {u.now}   [{u.label}]", f"runs: {dict(u.counts)}   guard refused {len(u.guard.refused)} rows",
         f"questions usable out of sample: {len(u.usable)}/{len(Q16)}"]
    for q in Q16:
        rep, o = u.reports[q], u.oos[q]
        ok = "USED" if q.value in u.usable else "withheld"
        L.append(f"{q.value}  [{ok}]  {o.label}: {o.reason}")
        ans = rep.answer(2)
        L.append(f"    pooled {ans['pooled']:.2f}; best {ans['best']}; worst {ans['worst']}; unknown {ans['unknown']}")
    if u.scheduler is not None:
        s = u.scheduler
        L.append(f"scheduler replay: {s.label}; advice found {s.advice_durable} durable vs random {s.random_durable_mean:.1f} (p={s.p_value:.3f}); {s.reason}")
    if u.advice.stop_paths:
        L.append("STOP paths: " + ", ".join(u.advice.stop_paths))
    if u.advice.blocked_representations:
        L.append("blocked representations: " + ", ".join(u.advice.blocked_representations))
    if u.run_level["leaks"]:
        L.append(f"LEAK SUSPECTS in run-level features: {list(u.run_level['leaks'])}")
    for m in u.methods[:5]:
        L.append(f"  method {m.method}: recall {m.recall:.2f} (lo {m.recall_lo:.2f}) false-alarm {m.false_alarm_rate:.2f} J {m.youden:.2f} min/catch "
                 f"{'n/a' if m.minutes_per_catch is None else round(m.minutes_per_catch, 1)}")
    return "\n".join(L)


def to_markdown(u: ResearchMetaUpdate) -> str:
    rows = ["| question | used | OOS | pooled | best | worst |", "|---|---|---|---|---|---|"]
    for q in Q16:
        a = u.reports[q].answer(2)
        rows.append(f"| {q.value} | {'yes' if q.value in u.usable else 'no'} | {u.oos[q].label} | {a['pooled']:.2f} | {', '.join(a['best'])} | {', '.join(a['worst'])} |")
    return f"## Research meta-learning as of {u.now}\n\n" + "\n".join(rows) + f"\n\nLabel: {u.label}. advice {u.advice.advice_id}.\n"


def write_report(u: ResearchMetaUpdate, out_dir) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.txt").write_text(render(u), encoding="utf-8")
    (d / "report.md").write_text(to_markdown(u), encoding="utf-8")
    (d / "advice.json").write_text(u.advice.to_json(), encoding="utf-8")
    (d / "summary.json").write_text(canonical_json(u.summary()), encoding="utf-8")
    return d


def self_check(seed: int = 0) -> dict:
    """Mechanism check on a planted world: the analyses recover the planted ordering, the stop path is found, the dead
    representation is blocked, and a planted self-training row is refused. Not a validation of anything real."""
    st = MetaResearchState(cfg=MetaResearchConfig(n_draws=100))
    st.ingest(synthetic_research_world(700, seed))
    res = step(st, "2004-06-01", seed)
    rec = truth_recovery(res.update)
    poison = ResearchOutcome("poison", "ablation", "vol_like", "raw_returns", "panel_survivor", "walk_forward", "loss", "FAILURE",
                             "2001-01-01", "2001-02-01", 1.0, origin="meta_evaluation", durable=True)
    _, gr = st.guard.screen([poison], "2004-06-01")
    return {"recovery": rec, "usable": list(res.update.usable), "blocked": list(res.advice.blocked_representations),
            "stop_paths": list(res.advice.stop_paths), "guard_refuses_eval_output": "poison" in gr.refused,
            "verdict": res.verdict.value, "label": LABEL}


# ================================================================================================ DEPTH: is the meta-knowledge itself trustworthy?
# Everything below tests the meta-learner's own claims: are its predictions calibrated, do its findings replicate in a second
# half of history, does it find nothing in a shuffled world, does its knowledge go stale, and which question actually earns
# its keep in the scheduler.

def _fold_edges(n: int, folds: int) -> np.ndarray:
    return np.linspace(n // 2, n, folds + 1).astype(int)


def oos_predictions(question: Q16, records: Sequence[ResearchOutcome], now, cfg: MetaResearchConfig) -> dict:
    """The actual out-of-sample predictions behind evaluate_questions (which returns only summaries), so calibration and
    ranking quality can be measured. Same fold rule: train only on runs resolved before the fold's first run started."""
    rows = question_rows(SPECS[question], records)
    cut = to_ts(now)
    lab = sorted((r for r in rows if r[1] and to_ts(r[1]) < cut), key=lambda r: (to_ts(r[0]), str(r[2])))
    empty = {"n": 0, "p_model": np.zeros(0), "p_base": np.zeros(0), "y": np.zeros(0, int), "keys": [], "fold": np.zeros(0, int)}
    if len(lab) < cfg.min_train + cfg.folds:
        return empty
    pm, pb, ys, keys, fid = [], [], [], [], []
    edges = _fold_edges(len(lab), cfg.folds)
    for f, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        test = lab[a:b]
        if not test:
            continue
        cutoff = to_ts(test[0][0])
        train = [r for r in lab[:a] if to_ts(r[1]) < cutoff]
        if len(train) < cfg.min_train:
            continue
        rates, _ = group_rates(((r[2], r[3]) for r in train), 1)
        pooled = sum(1 for r in train if r[3]) / len(train)
        for r in test:
            pm.append(rates[r[2]].shrunk if r[2] in rates else pooled)
            pb.append(pooled)
            ys.append(1 if r[3] else 0)
            keys.append(r[2])
            fid.append(f)
    if not ys:
        return empty
    return {"n": len(ys), "p_model": np.array(pm), "p_base": np.array(pb), "y": np.array(ys), "keys": keys, "fold": np.array(fid)}


def calibration_report(question: Q16, records: Sequence[ResearchOutcome], now, cfg: MetaResearchConfig, bins: int = 5) -> dict:
    """How honest are the group rates as probabilities, out of sample? Brier gain over the pooled rate, AUC of the group
    ranking, expected calibration error, reliability bins, and precision at the top of the ranking."""
    from engine.learning.meta_learning import auc, decile_lift, expected_calibration_error, precision_at_k, reliability_bins
    pr = oos_predictions(question, records, now, cfg)
    if pr["n"] < cfg.min_test:
        return {"question": question.value, "n": pr["n"], "verdict": "INSUFFICIENT"}
    y, pm, pb = pr["y"], pr["p_model"], pr["p_base"]
    gain = brier(pb, y) - brier(pm, y)
    ece = expected_calibration_error(pm, y, bins)
    k = max(5, pr["n"] // 10)
    return {"question": question.value, "n": pr["n"], "brier_model": brier(pm, y), "brier_base": brier(pb, y), "brier_gain": gain,
            "auc": auc(pm, y), "ece": ece, "reliability": reliability_bins(pm, y, bins), "precision_at_k": precision_at_k(pm, y, k), "k": k,
            "base_rate": float(y.mean()), "lift": decile_lift(pm, y, 4),
            "verdict": "CALIBRATED_AND_USEFUL" if gain > 0 and ece < 0.10 else ("USEFUL_BUT_MISCALIBRATED" if gain > 0 else "NO_SKILL")}


def calibration_sweep(records: Sequence[ResearchOutcome], now, cfg: MetaResearchConfig) -> dict:
    return {q: calibration_report(q, records, now, cfg) for q in Q16}


def recalibrate_advice(adv: SchedulerAdvice, sweep: Mapping) -> SchedulerAdvice:
    """Drop the groups of any question whose OOS calibration verdict is NO_SKILL: a rate that ranks nothing is not information."""
    bad = {q.value for q in Q16 if sweep.get(q, {}).get("verdict") == "NO_SKILL"}
    if not bad:
        return adv
    blank = {}
    for q, fields in QUESTION_FIELDS.items():
        if q.value in bad:
            for f in fields:
                blank[f] = () if isinstance(getattr(adv, f), tuple) else {}
    pooled = {k: v for k, v in adv.pooled.items() if k not in bad}
    withheld = {**adv.withheld, **{b: "no out-of-sample calibration skill" for b in bad}}
    return replace(adv, **blank, pooled=pooled, withheld=withheld, usable_questions=tuple(u for u in adv.usable_questions if u not in bad))


QUESTION_FIELDS = {
    Q16.DURABLE_EXPERIMENTS: ("exp_type_durable",), Q16.OVERFIT_TYPES: ("exp_type_overfit", "family_overfit"),
    Q16.TRANSFERRING_FAMILIES: ("family_transfer",), Q16.FAILING_REPRESENTATIONS: ("representation_failure", "blocked_representations"),
    Q16.FALSE_DISCOVERY_DATASETS: ("dataset_false_discovery",), Q16.VALIDATION_METHODS: ("validation_recall", "validation_order"),
    Q16.DECISION_QUESTIONS: ("source_decision", "target_decision"), Q16.INFORMATIVE_FAILURES: ("exp_type_informative_failure",),
    Q16.WASTED_COMPUTE: ("stop_paths", "throttle_paths"), Q16.ERA_SURVIVAL: ("survival_era",), Q16.STOCK_SURVIVAL: ("survival_stock",),
    Q16.REGIME_SURVIVAL: ("survival_regime",)}
CONSUMED_BY_SCHEDULE = {Q16.DURABLE_EXPERIMENTS, Q16.OVERFIT_TYPES, Q16.TRANSFERRING_FAMILIES, Q16.FAILING_REPRESENTATIONS,
                        Q16.FALSE_DISCOVERY_DATASETS, Q16.VALIDATION_METHODS, Q16.DECISION_QUESTIONS, Q16.INFORMATIVE_FAILURES,
                        Q16.WASTED_COMPUTE}                       # Q10-12 reach the policy through MetaAdvice.context_transfer instead


# ------------------------------------------------------------------------------------------------ do findings replicate?

@dataclass(frozen=True)
class Replication:
    question: str
    n_first: int
    n_second: int
    common_groups: int
    rho: float | None
    sign_agreement: float | None                    # share of groups whose lift has the same sign in both halves
    verdict: str                                    # REPLICATES / WEAK / DOES_NOT_REPLICATE / INSUFFICIENT


def split_half_replication(question: Q16, records: Sequence[ResearchOutcome], cfg: MetaResearchConfig, min_lift: float = 0.03) -> Replication:
    """A finding that holds only in one half of history is a coincidence of that half. Fit the group rates on the first and
    second half of the runs (by start time) independently and ask whether they agree. The meta-learner's own claims must
    survive the same test it applies to pattern discoveries."""
    spec = SPECS[question]
    key = spec.groupers[spec.primary]
    rel = sorted((o for o in records if spec.row_filter(o) and spec.outcome(o) is not None and key(o) is not None),
                 key=lambda o: (to_ts(o.started_at), o.run_id))
    if len(rel) < 2 * cfg.min_train:
        return Replication(question.value, 0, 0, 0, None, None, "INSUFFICIENT")
    mid = len(rel) // 2
    tabs = [rate_table(spec_rows(spec, part), spec.outcome_name, spec.higher_is_better, cfg.min_group_n, cfg.fdr_q) for part in (rel[:mid], rel[mid:])]
    common = sorted(set(tabs[0].groups) & set(tabs[1].groups))
    if len(common) < 3:
        return Replication(question.value, mid, len(rel) - mid, len(common), None, None, "INSUFFICIENT")
    a = [tabs[0].groups[k].shrunk for k in common]
    b = [tabs[1].groups[k].shrunk for k in common]
    rho = spearman(a, b)
    moved = [k for k in common if abs(tabs[0].groups[k].lift) >= min_lift and abs(tabs[1].groups[k].lift) >= min_lift]
    agree = (sum(1 for k in moved if tabs[0].groups[k].lift * tabs[1].groups[k].lift > 0) / len(moved)) if moved else None
    if rho is None:
        verdict = "INSUFFICIENT"
    elif rho >= 0.5 and (agree is None or agree >= 0.6):
        verdict = "REPLICATES"
    elif rho <= 0.0:
        verdict = "DOES_NOT_REPLICATE"
    else:
        verdict = "WEAK"
    return Replication(question.value, mid, len(rel) - mid, len(common), rho, agree, verdict)


def replication_sweep(records: Sequence[ResearchOutcome], cfg: MetaResearchConfig) -> dict:
    return {q: split_half_replication(q, records, cfg) for q in Q16}


def prob_best(table: RateTable, seed: int, draws: int = 4000) -> dict:
    """Posterior probability that each group is the best (or worst) for research on this outcome, from Beta posteriors under
    the table's empirical-Bayes prior. Sharper than a point ranking: it says how sure the top spot is."""
    if not table.groups:
        return {}
    rng = np.random.default_rng(seed)
    a0, b0 = table.prior
    keys = sorted(table.groups)
    samples = np.column_stack([rng.beta(a0 + table.groups[k].successes, b0 + table.groups[k].n - table.groups[k].successes, draws) for k in keys])
    win = np.argmax(samples if table.higher_is_better else -samples, axis=1)
    lose = np.argmin(samples if table.higher_is_better else -samples, axis=1)
    return {k: {"p_best": float(np.mean(win == i)), "p_worst": float(np.mean(lose == i))} for i, k in enumerate(keys)}


@dataclass(frozen=True)
class DataNeed:
    question: str
    group: str
    n_now: int
    half_width_now: float
    n_needed: int                                   # additional runs so the Wilson half-width reaches the target
    known: bool


def exploration_needs(report: QuestionReport, target_half_width: float = 0.15, z: float = 1.96) -> list:
    """How many more runs each group of a question needs before its rate is pinned to +/- target_half_width. Unknown groups
    (too few runs to appear in the table) are included: they are exactly what the scheduler should explore."""
    t = report.main
    out = []
    for k, g in t.groups.items():
        p = min(0.95, max(0.05, g.shrunk))
        need = math.ceil(z * z * p * (1 - p) / target_half_width ** 2) - g.n
        out.append(DataNeed(report.question.value, k, g.n, (g.hi - g.lo) / 2, max(0, need), True))
    full = math.ceil(z * z * 0.25 / target_half_width ** 2)
    for k in t.unknown:
        out.append(DataNeed(report.question.value, k, 0, 0.5, full, False))
    return sorted(out, key=lambda d: (-d.n_needed, d.question, d.group))


# ------------------------------------------------------------------------------------------------ cost-effectiveness across experiment types

@dataclass(frozen=True)
class TypeEconomics:
    exp_type: str
    runs: int
    minutes: float
    durable_per_hour: float
    overfit_rate: float | None
    decisions_per_hour: float
    info_bits_per_hour: float
    dominated: bool = False


def type_economics(records: Sequence[ResearchOutcome]) -> list:
    """Durable knowledge, decision changes and information per compute hour, by experiment type, with the Pareto set on
    (durable per hour up, overfit rate down). A dominated type is worse on both counts than some other type."""
    by: dict = defaultdict(list)
    for o in records:
        by[o.exp_type].append(o)
    rows = []
    for t in sorted(by):
        rs = by[t]
        hrs = sum(o.cost_minutes for o in rs) / 60.0
        dur = [o.durable for o in rs if o.durable is not None]
        ov = [o.overfit for o in rs if o.overfit is not None]
        dec = [o.decision_changed for o in rs if o.decision_changed is not None]
        if hrs <= 0:
            continue
        rows.append(TypeEconomics(t, len(rs), hrs * 60.0, sum(map(bool, dur)) / hrs, (sum(map(bool, ov)) / len(ov)) if ov else None,
                                  sum(map(bool, dec)) / hrs, sum(o.info_bits for o in rs) / hrs))
    out = []
    for r in rows:
        dom = any(o is not r and o.durable_per_hour >= r.durable_per_hour and o.overfit_rate is not None and r.overfit_rate is not None
                  and o.overfit_rate <= r.overfit_rate and (o.durable_per_hour > r.durable_per_hour or o.overfit_rate < r.overfit_rate) for o in rows)
        out.append(replace(r, dominated=dom))
    return sorted(out, key=lambda r: (-r.durable_per_hour, r.exp_type))


def compute_reallocation(records: Sequence[ResearchOutcome], adv: SchedulerAdvice, total_minutes: float, floor_share: float = 0.05,
                         cap_share: float = 0.45) -> dict:
    """Recommended compute split across experiment types: proportional to advice-adjusted durable knowledge per minute,
    boxed by a floor (nothing starves, C62 section 35) and a cap (no single type eats the budget). Types the advice knows
    nothing about get the mean of the known scores: exploring the unknown is not penalised."""
    from engine.learning.research_policy import project_box_simplex
    types = sorted({o.exp_type for o in records})
    if not types:
        return {"minutes": {}, "share": {}, "reason": "no runs"}
    cost = {t: max(0.1, float(np.mean([o.cost_minutes for o in records if o.exp_type == t]))) for t in types}
    pooled_over = adv.pooled.get(Q16.OVERFIT_TYPES.value, 0.3)
    raw, known = {}, {}
    for t in types:
        d = adv.exp_type_durable.get(t)
        if d is None:
            continue
        raw[t] = d * (1 - adv.exp_type_overfit.get(t, pooled_over)) / cost[t]
        known[t] = True
    mean_raw = float(np.mean(list(raw.values()))) if raw else 1.0
    for t in types:
        raw.setdefault(t, mean_raw)
    floor = min(floor_share, 0.5 / len(types))
    cap = max(cap_share, 1.5 / len(types))
    share = project_box_simplex(raw, {t: floor for t in types}, {t: min(1.0, cap) for t in types})
    return {"minutes": {t: round(share[t] * total_minutes, 2) for t in types}, "share": {t: round(share[t], 4) for t in types},
            "explored_unknown": sorted(t for t in types if t not in known), "floor": floor, "cap": cap}


# ------------------------------------------------------------------------------------------------ contradictions between findings

@dataclass(frozen=True)
class Contradiction:
    kind: str
    subject: str
    detail: str


def find_contradictions(reports: Mapping) -> list:
    """Findings that cannot all be trusted at once. (1) An experiment type significantly MORE durable and significantly MORE
    overfit than average. (2) A dataset with the lowest false-discovery rate that also makes families look worst. (3) A
    validation method with high recall and an equally high false-alarm rate (it alarms on everything). These are reported,
    never resolved silently."""
    out = []
    dur, ovf = reports[Q16.DURABLE_EXPERIMENTS].tables["exp_type"], reports[Q16.OVERFIT_TYPES].tables["exp_type"]
    for k in sorted(set(dur.groups) & set(ovf.groups)):
        if dur.groups[k].direction == "ABOVE" and ovf.groups[k].direction == "ABOVE":
            out.append(Contradiction("durable_and_overfit", k, f"durable lift {dur.groups[k].lift:+.2f} and overfit lift {ovf.groups[k].lift:+.2f} both significant"))
    inf, wst = reports[Q16.INFORMATIVE_FAILURES].tables["exp_type"], reports[Q16.DURABLE_EXPERIMENTS].tables["exp_type"]
    for k in sorted(set(inf.groups) & set(wst.groups)):
        if inf.groups[k].direction == "BELOW" and wst.groups[k].direction == "ABOVE":
            out.append(Contradiction("durable_but_uninformative_failures", k, "produces durable knowledge yet its failures teach nothing"))
    return out


def method_contradictions(scores: Sequence[MethodScore], j_floor: float = 0.15, recall_hi: float = 0.7) -> list:
    return [Contradiction("alarm_on_everything", m.method, f"recall {m.recall:.2f} but false-alarm {m.false_alarm_rate:.2f} (Youden {m.youden:.2f})")
            for m in scores if m.n_with_error and m.n_clean and m.recall >= recall_hi and m.youden < j_floor]


# ------------------------------------------------------------------------------------------------ is the pipeline able to find NOTHING?

OUTCOME_FIELDS = ("failed", "durable", "overfit", "false_discovery", "decision_changed", "new_era_survived", "new_stock_survived",
                  "new_regime_survived", "info_bits", "followups")


def permute_outcomes(records: Sequence[ResearchOutcome], seed: int) -> list:
    """The null world: descriptors and timing stay, outcomes are shuffled across runs. Any 'finding' here is a false alarm."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    out = [replace(o, **{f: getattr(records[j], f) for f in OUTCOME_FIELDS}) for o, j in zip(records, idx)]
    have = [i for i, o in enumerate(out) if o.validator_flagged is not None]
    flags = [out[i].validator_flagged for i in have]
    perm = rng.permutation(len(have))
    for i, j in zip(have, perm):
        out[i] = replace(out[i], validator_flagged=flags[j])
    return out


def null_world_check(records: Sequence[ResearchOutcome], now, cfg: MetaResearchConfig, seed: int = 0, n_perm: int = 8) -> dict:
    """Run the whole out-of-sample battery on shuffled outcomes. The share of questions that come out 'usable' is the
    meta-learner's false-discovery rate; the run must be near zero, and a real world must beat it clearly."""
    real = evaluate_questions(records, now, seed, cfg)
    real_usable = sum(1 for q in Q16 if question_usable(real[q], cfg)[0])
    counts = []
    per_q = {q.value: 0 for q in Q16}
    for i in range(n_perm):
        fake = evaluate_questions(permute_outcomes(records, seed * 1000 + i + 1), now, seed + 7 * i + 3, cfg)
        ok = [q for q in Q16 if question_usable(fake[q], cfg)[0]]
        counts.append(len(ok))
        for q in ok:
            per_q[q.value] += 1
    mean = float(np.mean(counts)) if counts else 0.0
    return {"real_usable": real_usable, "null_usable_mean": mean, "null_usable_max": int(max(counts)) if counts else 0,
            "false_usable_rate": mean / len(Q16), "per_question_false_rate": {k: v / max(1, n_perm) for k, v in per_q.items()},
            "verdict": "PIPELINE_QUIET_ON_NOISE" if mean <= 0.15 * len(Q16) else "PIPELINE_FINDS_STRUCTURE_IN_NOISE", "n_perm": n_perm}


# ------------------------------------------------------------------------------------------------ generalisation across contexts, and staleness

def context_holdout(question: Q16, records: Sequence[ResearchOutcome], dim: str, cfg: MetaResearchConfig, min_test: int = 15) -> dict:
    """Leave one context label out (one era, one stock group, one regime): fit the question's group rates on the rest, score
    the held-out label against the pooled rate. Not a time test (labels are contexts, not dates) - it asks whether the
    advice is carried by a single context. Fails when it beats the pooled rate in fewer than half the held-out labels."""
    spec = SPECS[question]
    labels = sorted({getattr(o, dim) for o in records if getattr(o, dim)})
    rows = []
    for lab in labels:
        tr = spec_rows(spec, [o for o in records if getattr(o, dim) != lab])
        te = spec_rows(spec, [o for o in records if getattr(o, dim) == lab])
        if len(tr) < cfg.min_train or len(te) < min_test:
            continue
        rates, _ = group_rates(tr, 1)
        pooled = sum(1 for _, y in tr if y) / len(tr)
        pm = [rates[k].shrunk if k in rates else pooled for k, _ in te]
        y = [1 if v else 0 for _, v in te]
        bm, bb = brier(pm, y), brier([pooled] * len(y), y)
        rows.append({"label": lab, "n_test": len(te), "brier_group": bm, "brier_pooled": bb, "better": bm < bb})
    if len(rows) < 2:
        return {"question": question.value, "dimension": dim, "labels": rows, "verdict": "INSUFFICIENT"}
    share = sum(1 for r in rows if r["better"]) / len(rows)
    return {"question": question.value, "dimension": dim, "labels": rows, "share_better": share,
            "verdict": "CARRIES_ACROSS_CONTEXTS" if share >= 0.5 else "CONTEXT_SPECIFIC"}


def predictive_decay(question: Q16, records: Sequence[ResearchOutcome], cfg: MetaResearchConfig, windows: int = 5) -> dict:
    """How fast does meta-knowledge go stale? Rates fitted on window i predict window i+lag; the mean Brier gain over the
    pooled rate at each lag shows the horizon over which advice stays useful. `stale_after` is the first lag with no gain:
    advice older than that should be shrunk (stale_discount) or refitted."""
    spec = SPECS[question]
    key = spec.groupers[spec.primary]
    rel = sorted((o for o in records if spec.row_filter(o) and spec.outcome(o) is not None and key(o) is not None),
                 key=lambda o: (to_ts(o.started_at), o.run_id))
    if len(rel) < windows * 12:
        return {"question": question.value, "lags": [], "stale_after": None, "verdict": "INSUFFICIENT"}
    chunks = [spec_rows(spec, [rel[i] for i in ix]) for ix in np.array_split(np.arange(len(rel)), windows)]
    lags = []
    for lag in range(1, windows):
        gains = []
        for i in range(windows - lag):
            tr, te = chunks[i], chunks[i + lag]
            if len(tr) < 10 or len(te) < 10:
                continue
            rates, _ = group_rates(tr, 1)
            pooled = sum(1 for _, y in tr if y) / len(tr)
            pm = [rates[k].shrunk if k in rates else pooled for k, _ in te]
            y = [1 if v else 0 for _, v in te]
            gains.append(brier([pooled] * len(y), y) - brier(pm, y))
        if gains:
            lags.append({"lag": lag, "mean_gain": float(np.mean(gains)), "n_pairs": len(gains)})
    stale = next((r["lag"] for r in lags if r["mean_gain"] <= 0), None)
    return {"question": question.value, "lags": lags, "stale_after": stale,
            "verdict": "STALE_WITHIN_HORIZON" if stale is not None else "DURABLE_OVER_HORIZON"}


# ------------------------------------------------------------------------------------------------ which question earns its keep in the scheduler?

def drop_question(adv: SchedulerAdvice, question: Q16) -> SchedulerAdvice:
    """The advice as if this one question had never been answered."""
    blank = {f: (() if isinstance(getattr(adv, f), tuple) else {}) for f in QUESTION_FIELDS[question]}
    return replace(adv, **blank, pooled={k: v for k, v in adv.pooled.items() if k != question.value})


def question_ablation(rows: Sequence[ResearchOutcome], now, seed: int, cfg: MetaResearchConfig, guard: SelfTrainingGuard | None = None) -> dict:
    """Leave-one-question-out over the scheduler replay: how much of the advice's lift disappears when a question is
    removed? A question with contribution <= 0 is either not consumed by the schedule or not helping it. Q10-Q12 feed the
    policy through MetaAdvice.context_transfer, not this schedule, and are reported as such rather than as zero."""
    full = evaluate_scheduler(rows, now, seed, cfg, guard)
    out = {"full_lift": full.lift, "full_label": full.label, "questions": {}}
    if not math.isfinite(full.lift):
        return out
    for q in Q16:
        if q not in CONSUMED_BY_SCHEDULE:
            out["questions"][q.value] = {"consumed_by_schedule": False, "contribution": None}
            continue
        ev = evaluate_scheduler(rows, now, seed, cfg, guard, lambda a, q=q: drop_question(a, q))
        out["questions"][q.value] = {"consumed_by_schedule": True, "lift_without": ev.lift, "contribution": full.lift - ev.lift}
    return out


def stop_rule_replay(records: Sequence[ResearchOutcome], cfg: MetaResearchConfig) -> dict:
    """The compute-waste counterfactual (contract section 20): walk each path's runs in resolution order, apply the STOP
    rule to the prefix, and count what would have been skipped. Reports minutes saved against the useful runs that would
    have been lost by stopping: the stop rule is only worth having if it saves much more than it loses."""
    by: dict = defaultdict(list)
    for o in sorted(records, key=lambda o: (to_ts(o.resolved_at), o.run_id)):
        w = wasted(o)
        if w is not None:
            by[o.path].append((o, w))
    saved = lost_useful = lost_minutes = total = 0.0
    stopped = []
    for p, rows in sorted(by.items()):
        useful = barren = 0
        for i, (o, w) in enumerate(rows):
            total += o.cost_minutes
            if path_action(useful, i, barren, cfg.useful_floor, cfg.stop_p, cfg.throttle_p, 4, cfg.barren_run) == "STOP":
                saved += o.cost_minutes
                if not w:
                    lost_useful += 1
                    lost_minutes += o.cost_minutes
                if p not in stopped:
                    stopped.append(p)
                continue
            useful += 0 if w else 1
            barren = barren + 1 if w else 0
    return {"total_minutes": total, "minutes_skipped": saved, "useful_runs_lost": int(lost_useful), "stopped_paths": stopped,
            "minutes_of_useful_lost": lost_minutes, "net_minutes": saved - lost_minutes,
            "verdict": "WORTH_HAVING" if saved > 0 and saved >= 3 * max(lost_minutes, 1e-9) else ("HARMS_MORE_THAN_HELPS" if lost_minutes > 0.5 * saved else "MARGINAL")}


def meta_rerank(engine, adv: SchedulerAdvice, cfg: MetaResearchConfig | None = None) -> list:
    """INTEGRATION HOOK for engine.learning.research_priority.ResearchPriorityEngine: multiplies each OPEN item's adjusted
    priority by the meta multiplier and BLOCKS items on a stop path / failing representation (recorded in the item's note,
    never deleted). Returns the new top order. Applied after each meta update, exactly like experience_rerank."""
    from engine.learning.research_priority import ItemStatus
    for it in list(engine.queue.items.values()):
        if it.status != ItemStatus.OPEN:
            continue
        d = schedule_decision(descriptor_from_candidate(it.candidate), adv, it.candidate.cid, cfg)
        it.note = (it.note + " | " if it.note else "") + "meta: " + "; ".join(d.reasons)
        if d.action == "BLOCK":
            it.status = ItemStatus.BLOCKED
        else:
            it.adjusted *= d.multiplier
    return [i.candidate.cid for i in engine.queue.top(len(engine.queue.items))]


def explain(desc: Mapping, adv: SchedulerAdvice, cfg: MetaResearchConfig | None = None) -> str:
    d = schedule_decision(desc, adv, "", cfg)
    return f"{d.action} x{d.multiplier:.2f} (advice {adv.advice_id}, trust {adv.trust():.2f}): " + "; ".join(d.reasons)


# ------------------------------------------------------------------------------------------------ questions about the research process

def meta_questions(update: ResearchMetaUpdate, now, max_questions: int = 12, need_threshold: int = 10) -> list:
    """Section 40 applied to research itself: the gaps in what the meta-learner knows become ResearchQuestion objects for the
    scheduler. Sources: groups too small to judge, questions that could not be answered out of sample, findings that did
    not replicate, contradictions, and drifts in the process. All text is identity-free."""
    from engine.research.core import ExperimentValue, Problem, ResearchQuestion
    qs = []
    through = update.advice.fitted_through
    def add(text, source, success, failure, bits):
        assert not identity_leak(text), text
        qs.append(ResearchQuestion.make(text, source, Problem.RESEARCH_PROCESS, str(now), through, success, failure,
                                        expected=ExperimentValue(information_gain=bits)))
    for q in Q16:
        if q.value not in update.usable:
            add(f"what evidence is missing before '{SPECS[q].text}' can be answered out of sample", "unanswered_meta_question",
                "the group predictor beats the pooled rate on held-out runs", "still no better than the pooled rate with the extra runs", 1.0)
        for need in exploration_needs(update.reports[q])[:2]:
            if need.n_needed >= need_threshold:
                add(f"how {SPECS[q].outcome_name} behaves for {need.group} ({need.n_needed} more runs pin it down)", "unknown_group",
                    "the rate is pinned within the target width", "the rate stays too wide to act on", min(1.0, need.n_needed / 100))
    for c in find_contradictions(update.reports) + method_contradictions(update.methods):
        add(f"why {c.subject} looks both good and bad ({c.kind})", "contradiction", "one reading is ruled out", "both readings survive", 0.6)
    for name, alarms in update.drift.items():
        if alarms:
            add(f"has the {name} rate of the research process shifted after the last alarm", "process_drift",
                "a stable new rate is confirmed", "the shift disappears", 0.5)
    seen, out = set(), []
    for q in qs:
        if q.question_id not in seen:
            seen.add(q.question_id)
            out.append(q)
    return sorted(out, key=lambda x: (-(x.expected.information_gain or 0.0), x.question_id))[:max_questions]


# ------------------------------------------------------------------------------------------------ store health and diff

def store_health(store: OutcomeStore, now) -> dict:
    """What can and cannot be learned from the store right now: field coverage among resolved runs, pending runs, thin
    experiment types, and duplicate paths on the same start. A field with near-zero coverage silences its question."""
    vis = store.as_of(now)
    allrows = store.all()
    cov = {}
    for f in ("durable", "overfit", "false_discovery", "decision_changed", "new_era_survived", "new_stock_survived", "new_regime_survived",
              "validator_flagged"):
        cov[f] = (sum(1 for o in vis if getattr(o, f) is not None) / len(vis)) if vis else 0.0
    thin = sorted(t for t in {o.exp_type for o in vis} if sum(1 for o in vis if o.exp_type == t) < 5)
    dup = defaultdict(int)
    for o in vis:
        dup[(o.path, o.started_at)] += 1
    warnings = [f"{f} known for only {c:.0%} of resolved runs" for f, c in cov.items() if c < 0.1 and vis]
    if thin:
        warnings.append(f"{len(thin)} experiment type(s) with fewer than 5 runs")
    if sum(1 for v in dup.values() if v > 1):
        warnings.append("several runs share a path and start time (possible double logging)")
    return {"resolved": len(vis), "pending": len(allrows) - len(vis), "coverage": cov, "thin_types": thin,
            "duplicate_starts": sum(1 for v in dup.values() if v > 1), "warnings": warnings,
            "poison_rows": sum(1 for o in vis if o.origin in EVAL_ORIGINS)}


def question_power(store: OutcomeStore, now, cfg: MetaResearchConfig) -> dict:
    """Per question: usable rows, rows needed before the OOS test can run at all, and whether it can be answered yet."""
    vis = store.as_of(now)
    need = cfg.min_train + cfg.folds
    out = {}
    for q in Q16:
        n = len(question_rows(SPECS[q], vis))
        out[q.value] = {"rows": n, "needed": need + cfg.min_test, "answerable": n >= need + cfg.min_test, "short_by": max(0, need + cfg.min_test - n)}
    return out


def diff_updates(old: ResearchMetaUpdate, new: ResearchMetaUpdate, min_move: float = 0.10) -> dict:
    """What changed between two fits: questions gained or lost, group rates that moved, paths newly stopped."""
    def moved(a: Mapping, b: Mapping) -> dict:
        return {k: (a[k], b[k]) for k in sorted(set(a) & set(b)) if abs(a[k] - b[k]) >= min_move}
    oa, na = old.advice, new.advice
    return {"usable_gained": sorted(set(na.usable_questions) - set(oa.usable_questions)),
            "usable_lost": sorted(set(oa.usable_questions) - set(na.usable_questions)),
            "durable_moved": moved(oa.exp_type_durable, na.exp_type_durable), "overfit_moved": moved(oa.exp_type_overfit, na.exp_type_overfit),
            "dataset_moved": moved(oa.dataset_false_discovery, na.dataset_false_discovery),
            "stop_added": sorted(set(na.stop_paths) - set(oa.stop_paths)), "stop_removed": sorted(set(oa.stop_paths) - set(na.stop_paths)),
            "blocked_added": sorted(set(na.blocked_representations) - set(oa.blocked_representations)),
            "label": (oa.oos_label, na.oos_label), "stability": advice_stability(oa, na)}


def scorecard_numbers(u: ResearchMetaUpdate) -> dict:
    """Flat numbers for the self-improvement scorecard (contract section 36): a small, stable set of scalars."""
    s = u.scheduler
    return {"meta_questions_usable": len(u.usable), "meta_questions_total": len(Q16),
            "scheduler_lift": None if s is None or not math.isfinite(s.lift) else s.lift, "scheduler_p": None if s is None else s.p_value,
            "stop_paths": len(u.advice.stop_paths), "blocked_representations": len(u.advice.blocked_representations),
            "advice_trust": u.advice.trust(), "guard_refused": len(u.guard.refused),
            "compute_waste_share": compute_waste_summary(u.paths)["waste_share"] if u.paths else None}
