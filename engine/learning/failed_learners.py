"""Failed-learner memory (contract C62 section 38; checklist G09; Bible canon C58-C62).

A failed learning architecture is still evidence. For every one this registry stores the section-38 fields: learner,
hypothesis, implementation, failure mode, data regime, validation result, reason for failure, and whether the failure
generalised. The next learner must not repeat a known dead end without justification: `check_proposal` compares a proposed
learner with the failed ones (mechanism tags, hypothesis wording, family) and either clears it, blocks it, or clears it ONLY
against a structured Justification (a genuinely different mechanism tag, a different data regime, and the controls that
would have caught the earlier failure).

Seeded from the real history (`seed_history`): the memory bank (memorisation), the basis learner (no transfer), the episodic
memory (walk-forward skill -0.16), the missed-winner detector (uplift 0), lessons from other windows (-0.188%/week), the six
B24 learners of state/research/learners (band_cfg, band_pool, regime_map, lessons, mover_use, chain) and the direction model
on movers. `verify_evidence` reports which seeds' evidence files are actually on disk, so nothing is trusted on memory alone.

Append-only: a re-test writes a NEW record that supersedes the old one; failures are never deleted (section 13 spirit).
IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import enum
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .core import FirewallBreach, ValidationLabel, canonical_json, stable_hash
from .experiment_memory import question_similarity, to_ts

LABEL = ValidationLabel.NOT_VALIDATED.value


class FailureMode(str, enum.Enum):
    MEMORISATION = "MEMORISATION"                  # gains on the same year, none on unseen years
    NO_TRANSFER = "NO_TRANSFER"                    # transfer delta CI contains zero
    NO_SKILL = "NO_SKILL"                          # predictive skill at or below the base rate / control
    NO_EFFECT = "NO_EFFECT"                        # the learner adopted nothing / changed nothing measurable
    HARMFUL = "HARMFUL"                            # significantly worse than not learning
    TRADEOFF_HARM = "TRADEOFF_HARM"                # improves its design metric while a guard metric significantly worsens
    LEAKAGE = "LEAKAGE"                            # the apparent gain came from future information
    OVERFIT = "OVERFIT"                            # in-sample gain, out-of-sample loss
    INFEASIBLE = "INFEASIBLE"                      # cannot run on available data / compute
    UNVALIDATED_CLAIM = "UNVALIDATED_CLAIM"        # claimed a gain that no independent evaluation could reproduce

    def __str__(self):
        return self.value


class Generalization(str, enum.Enum):
    GENERALIZED = "GENERALIZED"                    # failed in two or more distinct regimes / independent replications
    REGIME_SPECIFIC = "REGIME_SPECIFIC"            # failed in some regimes and passed (or was untested) in others
    SINGLE_REGIME = "SINGLE_REGIME"                # only one regime tried: cannot say
    UNKNOWN = "UNKNOWN"

    def __str__(self):
        return self.value


class DeadEndStatus(str, enum.Enum):
    CLEAR = "CLEAR"                                # nothing similar has failed
    SIMILAR_TO_FAILED = "SIMILAR_TO_FAILED"        # a related learner failed, not similar enough to block; controls advised
    RETRY_ALLOWED_NEW_REGIME = "RETRY_ALLOWED_NEW_REGIME"
    JUSTIFIED = "JUSTIFIED"                        # would be blocked but a valid Justification was supplied
    BLOCKED_DEAD_END = "BLOCKED_DEAD_END"

    def __str__(self):
        return self.value


# What a repeat attempt MUST include to have any chance of telling a new result from the old failure.
REQUIRED_CONTROLS = {
    FailureMode.MEMORISATION: ("memoriser_control", "past_only_transfer", "disguised_rerun"),
    FailureMode.NO_TRANSFER: ("past_only_transfer", "noise_control"),
    FailureMode.NO_SKILL: ("walk_forward", "base_rate_baseline"),
    FailureMode.NO_EFFECT: ("planted_effect", "noise_control"),
    FailureMode.HARMFUL: ("guard_metrics", "no_learning_control"),
    FailureMode.TRADEOFF_HARM: ("guard_metrics", "tier_metrics"),
    FailureMode.LEAKAGE: ("leak_probe", "future_scramble"),
    FailureMode.OVERFIT: ("holdout_years", "shuffle_control"),
    FailureMode.INFEASIBLE: ("resource_estimate",),
    FailureMode.UNVALIDATED_CLAIM: ("independent_evaluation", "code_hash_pin")}


@dataclass(frozen=True)
class FailedLearner:
    learner: str
    hypothesis: str                                # what the learner was supposed to learn and why it should have worked
    implementation: str                            # 'module::Class' or script; the code that was tested
    failure_mode: FailureMode
    data_regime: tuple                             # regimes it was evaluated in, e.g. ('real:1990s-2026 past-only pairs',)
    validation_result: Mapping                     # numbers: estimates and intervals, exactly as reported
    reason: str                                    # WHY it failed, or 'UNKNOWN' (an honest answer)
    generalization: Generalization
    recorded_at: str
    mechanism_tags: tuple = ()
    family: str = ""
    regimes_failed: tuple = ()
    regimes_passed: tuple = ()
    evidence_paths: tuple = ()
    code_hash: str = ""
    supersedes: str = ""
    version: int = 1

    def validate(self) -> list:
        errs = []
        for f in ("learner", "hypothesis", "implementation", "reason"):
            if not str(getattr(self, f)).strip():
                errs.append(f"{f} empty")
        if not self.data_regime:
            errs.append("data_regime empty (a failure with no stated regime cannot be generalised)")
        if not self.validation_result:
            errs.append("validation_result empty (a failure needs its numbers)")
        for k, v in self.validation_result.items():
            if isinstance(v, float) and not math.isfinite(v):
                errs.append(f"validation_result[{k}] not finite")
        if not self.mechanism_tags:
            errs.append("mechanism_tags empty (needed to recognise the same idea again)")
        if self.generalization == Generalization.GENERALIZED and len(set(self.regimes_failed)) < 2:
            errs.append("GENERALIZED needs at least two distinct failed regimes")
        if set(self.regimes_failed) & set(self.regimes_passed):
            errs.append("a regime cannot be both failed and passed")
        return errs

    @property
    def key(self) -> str:
        return stable_hash([self.learner, self.version], 12)

    def to_json(self) -> str:
        return canonical_json(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "FailedLearner":
        return cls(learner=d["learner"], hypothesis=d["hypothesis"], implementation=d["implementation"],
                   failure_mode=FailureMode(d["failure_mode"]), data_regime=tuple(d["data_regime"]),
                   validation_result=dict(d["validation_result"]), reason=d["reason"],
                   generalization=Generalization(d["generalization"]), recorded_at=d["recorded_at"],
                   mechanism_tags=tuple(d.get("mechanism_tags", ())), family=d.get("family", ""),
                   regimes_failed=tuple(d.get("regimes_failed", ())), regimes_passed=tuple(d.get("regimes_passed", ())),
                   evidence_paths=tuple(d.get("evidence_paths", ())), code_hash=d.get("code_hash", ""),
                   supersedes=d.get("supersedes", ""), version=int(d.get("version", 1)))


def classify_generalization(regimes_failed: Sequence[str], regimes_passed: Sequence[str], independent_replications: int = 1) -> Generalization:
    """GENERALIZED when it failed in >= 2 distinct regimes or replicated failing >= 2 times independently and never passed;
    REGIME_SPECIFIC when some regime passed; SINGLE_REGIME otherwise; UNKNOWN with no regimes."""
    failed, passed = set(regimes_failed), set(regimes_passed)
    if not failed:
        return Generalization.UNKNOWN
    if passed:
        return Generalization.REGIME_SPECIFIC
    if len(failed) >= 2 or independent_replications >= 2:
        return Generalization.GENERALIZED
    return Generalization.SINGLE_REGIME


def classify_failure_mode(same_year_effect: float | None, same_year_ci: tuple | None, transfer_effect: float | None,
                          transfer_ci: tuple | None, guard_worse: bool = False, adopted_anything: bool = True,
                          leak_found: bool = False) -> FailureMode | None:
    """Failure mode from harness numbers, or None when the learner in fact PASSES (transfer CI above zero, guards fine).
    Order matters: a leak overrides everything; nothing adopted is NO_EFFECT before it can be anything else."""
    if leak_found:
        return FailureMode.LEAKAGE
    if not adopted_anything:
        return FailureMode.NO_EFFECT
    t_lo = transfer_ci[0] if transfer_ci else None
    t_hi = transfer_ci[1] if transfer_ci else None
    s_lo = same_year_ci[0] if same_year_ci else None
    if t_hi is not None and t_hi < 0:
        return FailureMode.HARMFUL
    if same_year_effect is not None and s_lo is not None and s_lo > 0 and (t_lo is None or t_lo <= 0):
        return FailureMode.MEMORISATION
    if t_lo is not None and t_lo > 0:
        return FailureMode.TRADEOFF_HARM if guard_worse else None
    return FailureMode.NO_TRANSFER


# ------------------------------------------------------------------------------------------------ registry

class FailedLearnerRegistry:
    """Append-only registry. Reads are always `as_of`: a failure recorded at/after `now` is invisible at `now`."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._rows: list = []
        self.unparseable = 0
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    self._rows.append(FailedLearner.from_dict(json.loads(line)))
                except (ValueError, KeyError, TypeError):
                    self.unparseable += 1

    def add(self, fl: FailedLearner) -> FailedLearner:
        errs = fl.validate()
        if errs:
            raise ValueError(f"failed learner {fl.learner!r} invalid: " + "; ".join(errs))
        prior = [r for r in self._rows if r.learner == fl.learner]
        if prior and fl.version <= max(r.version for r in prior):
            raise ValueError(f"{fl.learner}: version {fl.version} not newer than {max(r.version for r in prior)}; re-tests supersede")
        self._rows.append(fl)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(fl.to_json() + "\n")
        return fl

    def retest(self, learner: str, new: FailedLearner) -> FailedLearner:
        """Record a re-test of a known failure as a NEW version that supersedes the old one."""
        prior = [r for r in self._rows if r.learner == learner]
        if not prior:
            raise KeyError(f"unknown learner {learner!r}")
        last = max(prior, key=lambda r: r.version)
        return self.add(replace(new, learner=learner, version=last.version + 1, supersedes=last.key))

    def as_of(self, now) -> list:
        """Current (latest-version) record per learner, among those recorded strictly before `now`."""
        cut = to_ts(now)
        cur: dict = {}
        for r in self._rows:
            if to_ts(r.recorded_at) >= cut:
                continue
            if r.learner not in cur or r.version > cur[r.learner].version:
                cur[r.learner] = r
        return sorted(cur.values(), key=lambda r: r.learner)

    def get(self, learner: str, now) -> FailedLearner | None:
        return next((r for r in self.as_of(now) if r.learner == learner), None)

    def history(self, learner: str) -> list:
        return sorted((r for r in self._rows if r.learner == learner), key=lambda r: r.version)

    def __len__(self):
        return len({r.learner for r in self._rows})

    # -------------------------------------------------------------- knowledge extracted from the failures
    def mode_counts(self, now) -> dict:
        out: dict = {}
        for r in self.as_of(now):
            out[r.failure_mode.value] = out.get(r.failure_mode.value, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def dead_end_tags(self, now, min_failures: int = 2) -> dict:
        """Mechanism tags that appear in >= min_failures independent failed learners: ideas that keep failing."""
        tally: dict = {}
        for r in self.as_of(now):
            for t in set(r.mechanism_tags):
                e = tally.setdefault(t, {"learners": [], "generalized": 0})
                e["learners"].append(r.learner)
                e["generalized"] += int(r.generalization == Generalization.GENERALIZED)
        return {t: e for t, e in sorted(tally.items(), key=lambda kv: -len(kv[1]["learners"])) if len(e["learners"]) >= min_failures}

    def untested_regimes(self, now, all_regimes: Iterable[str]) -> dict:
        """Per learner, regimes it has never been evaluated in: where a retry would be new information, not a repeat."""
        out = {}
        universe = set(all_regimes)
        for r in self.as_of(now):
            tried = set(r.data_regime) | set(r.regimes_failed) | set(r.regimes_passed)
            gap = sorted(universe - tried)
            if gap:
                out[r.learner] = gap
        return out

    def verify_evidence(self, root, now) -> dict:
        """Which evidence files exist on disk. A record whose evidence is missing is marked UNVERIFIED, not deleted."""
        base = Path(root)
        present, missing = [], []
        for r in self.as_of(now):
            for p in r.evidence_paths:
                (present if (base / p).exists() else missing).append((r.learner, p))
        return {"present": present, "missing": missing, "verified_learners": sorted({l for l, _ in present} - {l for l, _ in missing}),
                "unverified_learners": sorted({l for l, _ in missing})}

    # -------------------------------------------------------------- the gate for the NEXT learner
    def check_proposal(self, proposal: "LearnerProposal", now, justification: "Justification | None" = None,
                       block_at: float = 0.55, warn_at: float = 0.30) -> "DeadEndVerdict":
        matches = []
        for r in self.as_of(now):
            sim = proposal_similarity(proposal, r)
            if sim >= warn_at:
                matches.append((r, sim))
        matches.sort(key=lambda m: -m[1])
        if not matches:
            return DeadEndVerdict(DeadEndStatus.CLEAR, "no failed learner resembles this proposal", (), ())
        top, sim = matches[0]
        needed = REQUIRED_CONTROLS.get(top.failure_mode, ())
        listing = tuple((r.learner, round(s, 3), r.failure_mode.value, r.generalization.value) for r, s in matches[:5])
        if sim < block_at:
            return DeadEndVerdict(DeadEndStatus.SIMILAR_TO_FAILED,
                                  f"resembles failed '{top.learner}' ({top.failure_mode}) at {sim:.2f}; include controls {list(needed)}",
                                  listing, needed)
        new_regime = [g for g in proposal.data_regime if g not in set(top.data_regime) | set(top.regimes_failed)]
        if top.generalization in (Generalization.SINGLE_REGIME, Generalization.REGIME_SPECIFIC) and new_regime:
            return DeadEndVerdict(DeadEndStatus.RETRY_ALLOWED_NEW_REGIME,
                                  f"'{top.learner}' failed only in {list(top.regimes_failed or top.data_regime)}; a retry in "
                                  f"{new_regime} is new information", listing, needed)
        ok, why = justification_ok(proposal, top, justification)
        if ok:
            return DeadEndVerdict(DeadEndStatus.JUSTIFIED, f"justified against '{top.learner}': {why}", listing, needed)
        return DeadEndVerdict(DeadEndStatus.BLOCKED_DEAD_END,
                              f"BLOCKED: repeats '{top.learner}' ({top.failure_mode}, {top.generalization}, similarity {sim:.2f}); "
                              f"unjustified: {why}", listing, needed)

    def require_clear(self, proposal: "LearnerProposal", now, justification: "Justification | None" = None) -> "DeadEndVerdict":
        """Raise instead of returning a blocked verdict, for callers that must not continue."""
        v = self.check_proposal(proposal, now, justification)
        if v.blocked:
            raise DeadEndRepeat(v)
        return v


@dataclass(frozen=True)
class LearnerProposal:
    name: str
    hypothesis: str
    mechanism_tags: tuple
    family: str = ""
    data_regime: tuple = ()


@dataclass(frozen=True)
class Justification:
    """Why a proposal is NOT the earlier failure. Must name at least one mechanism that is new, and the controls that would
    have exposed the earlier failure mode. 'It is better this time' is not a justification."""
    new_mechanism_tags: tuple
    explanation: str
    controls_planned: tuple
    new_evidence: tuple = ()
    regime_changed: bool = False


@dataclass(frozen=True)
class DeadEndVerdict:
    status: DeadEndStatus
    message: str
    similar: tuple
    required_controls: tuple

    @property
    def blocked(self) -> bool:
        return self.status == DeadEndStatus.BLOCKED_DEAD_END


class DeadEndRepeat(Exception):
    def __init__(self, verdict: DeadEndVerdict):
        super().__init__(verdict.message)
        self.verdict = verdict


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def proposal_similarity(p: LearnerProposal, f: FailedLearner) -> float:
    """0.5 x tag Jaccard + 0.3 x hypothesis wording + 0.2 x same family. Tags dominate: renaming a learner does not hide it."""
    fam = 1.0 if p.family and p.family == f.family else 0.0
    return 0.5 * jaccard(p.mechanism_tags, f.mechanism_tags) + 0.3 * question_similarity(p.hypothesis, f.hypothesis) + 0.2 * fam


def justification_ok(p: LearnerProposal, failed: FailedLearner, j: Justification | None) -> tuple:
    if j is None:
        return False, "no justification supplied"
    if not j.explanation.strip():
        return False, "explanation empty"
    truly_new = set(j.new_mechanism_tags) - set(failed.mechanism_tags)
    if not truly_new:
        return False, f"no mechanism tag that the failed learner lacks (it had {list(failed.mechanism_tags)})"
    if not set(j.new_mechanism_tags) <= set(p.mechanism_tags):
        return False, "justification cites tags the proposal does not carry"
    need = set(REQUIRED_CONTROLS.get(failed.failure_mode, ()))
    lacking = need - set(j.controls_planned)
    if lacking:
        return False, f"controls that would have exposed {failed.failure_mode} are missing: {sorted(lacking)}"
    return True, f"new mechanism {sorted(truly_new)} with controls {sorted(need)}"


# ------------------------------------------------------------------------------------------------ real history seeds

_R1 = "real archive windows, past-only pairs (ls1/ls2: 12 pairs)"
_R3 = "real archive windows, past-only pairs (ls3_s11: 16 pairs)"
_RA = "real archive windows 2001+ and pre-1997 (ld1/ld2: 12 pairs)"
_RW = "real walk-forward 2017-2026"


def seed_history(recorded_at: str = "2026-09-29T00:00:00") -> list:
    """The failed learners the repository already proved, with the numbers exactly as their reports state them. Reasons say
    'UNKNOWN' where the report established that it failed but not why."""
    def fl(learner, hypo, impl, mode, regime, val, reason, gen, tags, fam, failed=(), passed=(), ev=()):
        return FailedLearner(learner=learner, hypothesis=hypo, implementation=impl, failure_mode=mode, data_regime=tuple(regime),
                             validation_result=val, reason=reason, generalization=gen, recorded_at=recorded_at, mechanism_tags=tuple(tags),
                             family=fam, regimes_failed=tuple(failed), regimes_passed=tuple(passed), evidence_paths=tuple(ev))
    return [
        fl("memory_bank", "Storing window-level numeric memories lets the system recall what worked and improve when the same or similar market recurs",
           "engine/learning_delta.py::MemoryBankLearner", FailureMode.MEMORISATION, (_RA,),
           {"same_year_effect_mean_week": 0.002269, "same_year_ci_lo": 0.0000676, "same_year_ci_hi": 0.00544,
            "transfer_mean_week": -0.0003211, "gap_ci_lo": 0.000124, "gap_ci_hi": 0.00585, "in_band_effect": -0.0143, "pairs": 12},
           "A disguised rerun of the same real year is the same numbers with new labels; a stored numeric memory recognises it, so the gain is recall not learning; nothing transfers to unseen years",
           Generalization.GENERALIZED, ("stored_numeric_memory", "memory_bank", "exact_path_recall"), "memory", (_RA, _R1),
           ev=("state/research/learning_delta/ld1/report.md", "state/research/learning_delta/ld1/summary.json")),
        fl("basis_learner", "A learned basis over window features (S0) generalises what the memory bank memorises",
           "engine/learning_delta.py::BasisLearner", FailureMode.NO_TRANSFER, (_RA,),
           {"transfer_mean_week": -0.00002, "transfer_ci_lo": -0.00034, "transfer_ci_hi": 0.00030, "p_signflip": 1.0,
            "same_year_effect": 0.00093, "same_year_ci_lo": 0.000205, "same_year_ci_hi": 0.00177, "pairs": 12},
           "Transfer CI contains zero; the same-year effect far exceeds transfer, so the residual is memorisation (harness verdict MEMORISATION)",
           Generalization.SINGLE_REGIME, ("basis_expansion", "memory_bank", "stored_numeric_memory"), "memory", (_RA,),
           ev=("state/research/learning_delta/ld2_basis/report.md", "state/research/learning_delta/ld2_basis/summary.json")),
        fl("episodic_memory_retrieval", "Retrieving similar past situations gives forward-looking skill for the next decision",
           "engine/memory.py::Memory", FailureMode.NO_SKILL, (_RW,),
           {"walk_forward_skill": -0.16, "skill_ci_lo": -0.20, "skill_ci_hi": -0.12, "ic": -0.029},
           "UNKNOWN: retrieval reproduces past outcomes but its walk-forward skill is significantly negative; the mechanism of the negative sign was not isolated",
           Generalization.SINGLE_REGIME, ("episodic_retrieval", "analog_memory", "similarity_lookup"), "memory", (_RW,),
           ev=("state/research/memory_adapter/results.json",)),
        fl("missed_winner_detector", "Analysing which winners we missed identifies a rule that would have captured them",
           "engine/missed_winners.py::MissedWinnerDetector", FailureMode.NO_EFFECT, ("real 2017-2026",),
           {"detector_uplift": 0.0, "p_vs_control": 0.64, "verdict": "false"},
           "Uplift over a control detector is zero (p 0.64): the winners were not predictable from what the detector looks at",
           Generalization.SINGLE_REGIME, ("missed_winner_rules", "post_hoc_capture"), "failure_learning", ("real 2017-2026",),
           ev=("state/research/memory_adapter/results.json",)),
        fl("lessons_from_other_windows", "Veto lessons distilled from other windows protect a new window",
           "engine/antimemo.py::archive_experiment; scripts/lessons_archive.py", FailureMode.HARMFUL, ("real archive windows",),
           {"weekly_effect": -0.00188, "ci90_lo": -0.00337, "ci90_hi": -0.00019},
           "Lessons learned from other windows cost about 0.19 percent per week: they veto names that would have done fine (the lessons do not transfer)",
           Generalization.SINGLE_REGIME, ("lessons_from_other_windows", "veto_rules", "stored_numeric_memory"), "lessons",
           ("real archive windows",), ev=("state/research/lessons/archive_v2.md",)),
        fl("band_cfg", "Configuration knobs chosen across years move the weekly result toward the 5-10 percent band",
           "engine/learners.py::BandCfgLearner", FailureMode.TRADEOFF_HARM, (_R1, _R3),
           {"ls1_in_band_transfer": 0.01107, "ls1_ci_lo": 0.00160, "ls1_ci_hi": 0.02204, "ls3_in_band_transfer": 0.02120,
            "ls3_ci_lo": 0.00118, "ls3_ci_hi": 0.04833, "ls3_tiers_worse": "worst5,max_dd"},
           "Lifts the share of weeks in band but significantly worsens worst-5-percent week and max drawdown in the replication (C39: no tier may be worse)",
           Generalization.REGIME_SPECIFIC, ("cfg_knob_search", "band_targeting"), "learners", (_R3,), (_R1,),
           ev=("state/research/learners/ls1/report.md", "state/research/learners/ls3_s11/report.md")),
        fl("band_pool", "Ranking windows on volatility, max20, p_move and keeping the pool where the portfolio lands in band transfers",
           "engine/learners.py::BandPoolLearner", FailureMode.NO_EFFECT, (_R1, _R3),
           {"ls1_in_band_transfer": 0.0, "ls3_in_band_transfer": 0.0, "adopted": 0},
           "The adoption gate never opened (no candidate held in enough years with a consistent sign): the learner adopted nothing",
           Generalization.GENERALIZED, ("pool_rank_rules", "band_targeting"), "learners", (_R1, _R3),
           ev=("state/research/learners/ls1/report.md", "state/research/learners/ls3_s11/report.md")),
        fl("regime_map", "A market-regime bucket to pool rule map, learned across years with shrinkage, transfers",
           "engine/learners.py::RegimeMapLearner", FailureMode.NO_EFFECT, (_R1, _R3),
           {"ls1_in_band_transfer": 0.0, "ls3_in_band_transfer": 0.0, "adopted": 0},
           "UNKNOWN: no regime bucket rule cleared the cross-year gate; whether none exists or the data cannot show one is not determined",
           Generalization.GENERALIZED, ("regime_bucket_rules", "pool_rank_rules"), "learners", (_R1, _R3),
           ev=("state/research/learners/ls1/report.md", "state/research/learners/ls3_s11/report.md")),
        fl("lesson_learner", "Veto lessons of the form 'names in the top decile of X lose' that held in many distinct years transfer",
           "engine/learners.py::LessonLearner", FailureMode.NO_EFFECT, (_R1, _R3),
           {"ls1_mean_week_transfer": 0.0, "ls3_mean_week_transfer": 0.0, "adopted": 0},
           "No lesson held in enough distinct years; consistent with the harmful archive lessons (see lessons_from_other_windows)",
           Generalization.GENERALIZED, ("veto_rules", "lessons_from_other_windows"), "learners", (_R1, _R3),
           ev=("state/research/learners/ls1/report.md", "state/research/learners/ls3_s11/report.md")),
        fl("mover_use", "A movement score fitted on realised absolute move across years selects better movers",
           "engine/learners.py::MoverUseLearner", FailureMode.NO_EFFECT, (_R1, _R3),
           {"ls1_in_band_transfer": 0.0, "ls3_in_band_transfer": 0.0, "adopted": 0},
           "The learned movement score never beat the incumbent p_move under the sign-consistency gate",
           Generalization.GENERALIZED, ("movement_score", "mover_selection"), "learners", (_R1, _R3),
           ev=("state/research/learners/ls1/report.md", "state/research/learners/ls3_s11/report.md")),
        fl("chain_learner", "Chaining the five single learners compounds their transfer",
           "engine/learners.py::chain", FailureMode.NO_EFFECT, (_R1, _R3),
           {"ls1_in_band_transfer": 0.0, "ls3_in_band_transfer": 0.0, "adopted": 0},
           "Chaining components that each adopt nothing (or fail a guard) cannot create transfer",
           Generalization.GENERALIZED, ("learner_chain", "pool_rank_rules", "veto_rules"), "learners", (_R1, _R3),
           ev=("state/research/learners/ls1/report.md", "state/research/learners/ls3_s11/report.md")),
        fl("cross_year_learners_all", "Any of the six cross-year learners passes past-only transfer on unseen years",
           "engine/learners.py (6 learners)", FailureMode.NO_TRANSFER, (_R1, _R3),
           {"learners_tested": 6, "transfer_passing_learners": 0, "independent_replications": 3, "headline": "NO_TRANSFER"},
           "No learner passes the past-only transfer gate in three independent pair sets; the headline verdict on weekly mean is NO_TRANSFER every time",
           Generalization.GENERALIZED, ("cross_year_learner", "past_only_transfer_gate"), "learners", (_R1, _R3),
           ev=("state/research/learners/ls1/report.md", "state/research/learners/ls2_replicate/report.md", "state/research/learners/ls3_s11/report.md")),
        fl("direction_on_movers", "The direction of a mover's next week can be predicted above the base rate",
           "engine/direction.py", FailureMode.NO_SKILL, ("real movers panel",),
           {"accuracy": 0.518, "brier_vs_base_rate": 0.0, "gate_80pct_opened": False},
           "Accuracy is 51.8 percent and the Brier score equals the base rate: no direction signal exists in the features tried",
           Generalization.SINGLE_REGIME, ("direction_classifier", "mover_features"), "direction", ("real movers panel",),
           ev=("state/research/direction",)),
    ]


def seed_registry(reg: FailedLearnerRegistry, recorded_at: str = "2026-09-29T00:00:00") -> dict:
    """Load the historical failures into a registry (skips learners already present). Returns counts."""
    have = {r.learner for r in reg._rows}
    added = 0
    for fl_ in seed_history(recorded_at):
        if fl_.learner not in have:
            reg.add(fl_)
            added += 1
    return {"added": added, "skipped": len(seed_history(recorded_at)) - added, "label": LABEL}


def render_report(reg: FailedLearnerRegistry, now) -> str:
    rows = reg.as_of(now)
    lines = [f"Failed-learner memory as of {now}   [{LABEL}]", f"{len(rows)} failed learners; modes: {reg.mode_counts(now)}"]
    for r in rows:
        lines.append(f"- {r.learner}: {r.failure_mode} / {r.generalization} :: {r.reason[:110]}")
    dead = reg.dead_end_tags(now)
    lines.append("mechanism tags that keep failing:")
    lines += [f"  {t}: {len(e['learners'])} learners ({e['generalized']} generalised)" for t, e in list(dead.items())[:8]]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ from harness output

def from_harness_summary(learner: str, summary: Mapping, recorded_at: str, implementation: str, hypothesis: str,
                         mechanism_tags: Sequence[str], family: str = "", regime: str = "", evidence: Sequence[str] = (),
                         guard_worse: bool = False) -> FailedLearner | None:
    """Turn one learner's harness aggregate (the shape engine.learning_delta / learners write: per-metric 'same', 'transfer'
    and 'gap' blocks with mean/lo/hi) into a FailedLearner, or None when it actually PASSES. The failure mode comes from
    `classify_failure_mode`, not from the caller's opinion, and the numbers are copied, never rounded away."""
    metrics = summary.get("metrics") or summary.get("aggregate", {}).get("metrics") or {}
    design = summary.get("design_metric") or ("mean_week" if "mean_week" in metrics else next(iter(metrics), None))
    if design is None:
        raise ValueError("harness summary has no metrics")
    block = metrics[design]
    same, tr = block.get("same") or {}, block.get("transfer") or {}
    adopted = bool(summary.get("adopted", 1)) and not (tr.get("mean") == 0.0 and tr.get("lo") == 0.0 and tr.get("hi") == 0.0)
    mode = classify_failure_mode(same.get("mean"), (same.get("lo"), same.get("hi")) if same else None, tr.get("mean"),
                                 (tr.get("lo"), tr.get("hi")) if tr else None, guard_worse, adopted, bool(summary.get("leak_found")))
    if mode is None:
        return None
    val = {"design_metric": design, "transfer_mean": tr.get("mean"), "transfer_ci_lo": tr.get("lo"), "transfer_ci_hi": tr.get("hi"),
           "same_year_mean": same.get("mean"), "same_year_ci_lo": same.get("lo"), "same_year_ci_hi": same.get("hi"),
           "pairs": tr.get("n", same.get("n")), "guard_worse": guard_worse}
    val = {k: v for k, v in val.items() if v is not None}
    reason = {FailureMode.MEMORISATION: "same-year gain with no transfer to unseen years",
              FailureMode.NO_TRANSFER: "transfer interval contains zero",
              FailureMode.NO_EFFECT: "the learner adopted nothing measurable",
              FailureMode.HARMFUL: "transfer interval is entirely below zero",
              FailureMode.TRADEOFF_HARM: "design metric improved while a guard metric worsened",
              FailureMode.LEAKAGE: "a future-information leak explains the gain"}.get(mode, "UNKNOWN")
    regs = (regime,) if regime else ("unspecified regime",)
    return FailedLearner(learner=learner, hypothesis=hypothesis, implementation=implementation, failure_mode=mode, data_regime=regs,
                         validation_result=val, reason=reason, generalization=classify_generalization(regs, (), 1),
                         recorded_at=recorded_at, mechanism_tags=tuple(mechanism_tags), family=family, regimes_failed=regs,
                         evidence_paths=tuple(evidence))


def advice_for_next_learner(reg: FailedLearnerRegistry, proposal: LearnerProposal, now, all_regimes: Iterable[str] = ()) -> dict:
    """Everything the failures teach a would-be builder BEFORE they write code: the verdict, the controls that must be
    present, the tags that already failed repeatedly, and the regimes nobody has tried."""
    verdict = reg.check_proposal(proposal, now)
    dead = reg.dead_end_tags(now)
    touching = {t: dead[t] for t in proposal.mechanism_tags if t in dead}
    return {"verdict": verdict.status.value, "message": verdict.message, "required_controls": list(verdict.required_controls),
            "similar": [list(s) for s in verdict.similar], "tags_that_keep_failing": {t: e["learners"] for t, e in touching.items()},
            "untested_regimes": reg.untested_regimes(now, all_regimes), "label": LABEL}


def audit_registry(reg: FailedLearnerRegistry, now, root=None) -> dict:
    """Integrity of the memory itself: every current record validates, GENERALIZED claims are backed by regimes, versions are
    contiguous, and (when a repository root is given) evidence files exist. A registry that cannot pass its own audit is not memory."""
    problems = []
    for r in reg.as_of(now):
        problems += [f"{r.learner}: {e}" for e in r.validate()]
        hist = reg.history(r.learner)
        if [h.version for h in hist] != list(range(1, len(hist) + 1)):
            problems.append(f"{r.learner}: version gap {[h.version for h in hist]}")
        if r.failure_mode in (FailureMode.MEMORISATION, FailureMode.NO_TRANSFER) and not any(
                k.startswith(("transfer", "same_year", "ls")) or "transfer" in k for k in r.validation_result):
            problems.append(f"{r.learner}: {r.failure_mode} without a transfer/same-year number")
    ev = reg.verify_evidence(root, now) if root is not None else {"unverified_learners": []}
    return {"ok": not problems, "problems": problems, "unverified_learners": ev["unverified_learners"], "n": len(reg.as_of(now)),
            "unparseable_lines": reg.unparseable}
