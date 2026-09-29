"""Learning scorecard and the claim gate (contract C62 sections 47, 59, 25 controls; checklist L06-L12).

STATUS: IMPLEMENTED - NOT VALIDATED.

Every learner version produces a LearningScorecard with every section-47 field (baseline, post-learning, same-year / cross-year /
cross-regime / cross-stock gain, transfer ratio, risk and drawdown change, 5-10% band share, movement / direction / mover
performance, calibration, memorisation gap, identity gap, future-leak status, stability, compute cost). A field that was not
measured is UNTESTED, which is different from zero and is never rendered as a number.

"Never report 'learning improved' without the corresponding controls": gate_improvement_claim() refuses the claim unless the five
controls of section 25 were run (A no learning, B the learner, C identity memoriser, D random learner, E leaky learner) and behave
as they must: A is flat, D sets a luck floor the learner beats, C is flagged as a memoriser (so the instrument can see one) while
the learner is not, E is CAUGHT by the leak audit (so the audit can see leaks) and the learner's own leak status is clean, and
the gain survives on unseen years/contexts. Text is checked too: assert_claim_ok() rejects a sentence that says 'improved',
'validated', 'production ready', 'done' ... when the gate refused, and even a passing gate never permits 'validated'.

Scorecards are immutable, hashed, and appended to a hash-chained store that refuses to overwrite a version (section 49)."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .core import FirewallBreach, ValidationLabel, _StrEnum, as_date, canonical_json, stable_hash
from . import transfer_score as TS

SECTION_47_FIELDS = ("baseline_performance", "post_learning_performance", "same_year_gain", "cross_year_gain", "cross_regime_gain",
                     "cross_stock_gain", "transfer_ratio", "risk_change", "drawdown_change", "band_share", "movement_performance",
                     "direction_performance", "mover_performance", "calibration", "memorization_gap", "identity_gap",
                     "future_leak_status", "stability", "compute_cost")
CONTROL_NAMES = ("A_no_learning", "B_learner", "C_identity_memoriser", "D_random_learner", "E_leaky_learner")
REQUIRED_OTHER_CONTROLS = ("A_no_learning", "C_identity_memoriser", "D_random_learner", "E_leaky_learner")
FORBIDDEN_ALWAYS = ("validated", "production ready", "done", "finished", "complete", "learning works")
FORBIDDEN_WITHOUT_GATE = ("improved", "improvement", "learning improved")


class UnsupportedClaim(Exception):
    """Raised when text asserts a result the scorecard's controls do not support."""


class LeakStatus(_StrEnum):
    CLEAN = "CLEAN"
    LEAK_DETECTED = "LEAK_DETECTED"
    UNAUDITED = "UNAUDITED"


class MStatus(_StrEnum):
    MEASURED = "MEASURED"
    UNTESTED = "UNTESTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"     # measured, but the quantity is undefined (e.g. a ratio with no same-context gain)


@dataclass(frozen=True)
class Measured:
    """One scorecard number with its uncertainty. value/lo/hi are NaN unless status is MEASURED."""
    value: float = float("nan")
    lo: float = float("nan")
    hi: float = float("nan")
    n: int = 0
    status: MStatus = MStatus.UNTESTED
    note: str = ""

    @classmethod
    def untested(cls, note: str = "not measured") -> "Measured":
        return cls(note=note)

    @classmethod
    def undefined(cls, note: str) -> "Measured":
        return cls(status=MStatus.NOT_APPLICABLE, note=note)

    @classmethod
    def from_boot(cls, b: TS.BootMean, note: str = "") -> "Measured":
        if b.n == 0 or math.isnan(b.mean):
            return cls.untested(note or "no units")
        return cls(b.mean, b.lo, b.hi, b.n, MStatus.MEASURED, note)

    @classmethod
    def from_gap(cls, g: TS.GapResult | None, note: str = "") -> "Measured":
        if g is None or math.isnan(g.gap):
            return cls.untested(note or "not measured")
        return cls(g.gap, g.lo, g.hi, min(g.n_a, g.n_b), MStatus.MEASURED, note)

    @classmethod
    def point(cls, value: float, n: int = 0, note: str = "") -> "Measured":
        return cls(float(value), float("nan"), float("nan"), n, MStatus.MEASURED, note) if math.isfinite(value) else cls.untested(note)

    @property
    def measured(self) -> bool:
        return self.status == MStatus.MEASURED

    @property
    def significantly_positive(self) -> bool:
        return self.measured and math.isfinite(self.lo) and self.lo > 0

    @property
    def significantly_negative(self) -> bool:
        return self.measured and math.isfinite(self.hi) and self.hi < 0

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class ComputeCost:
    cpu_seconds: float = float("nan")
    wall_seconds: float = float("nan")
    peak_ram_mb: float = float("nan")
    n_fits: int = 0
    n_evaluations: int = 0

    def check(self) -> list[str]:
        errs = []
        for f in ("cpu_seconds", "wall_seconds", "peak_ram_mb"):
            v = getattr(self, f)
            if not (math.isnan(v) or v >= 0):
                errs.append(f"{f}={v} is negative")
        return errs

    @property
    def measured(self) -> bool:
        return not math.isnan(self.cpu_seconds) or not math.isnan(self.wall_seconds)


class ComputeMeter:
    """Context manager: `with ComputeMeter() as m: ...; m.cost(n_fits=3)`. CPU and wall seconds from the process; peak RAM from
    psutil when installed (NaN otherwise, never a guess)."""

    def __enter__(self):
        self._c0, self._w0 = time.process_time(), time.perf_counter()
        self._peak = float("nan")
        return self

    def sample(self):
        try:
            import psutil
            rss = psutil.Process().memory_info().rss / 1e6
            self._peak = rss if math.isnan(self._peak) else max(self._peak, rss)
        except Exception:
            pass

    def __exit__(self, *exc):
        self.sample()
        self._cpu, self._wall = time.process_time() - self._c0, time.perf_counter() - self._w0
        return False

    def cost(self, n_fits: int = 0, n_evaluations: int = 0) -> ComputeCost:
        return ComputeCost(self._cpu, self._wall, self._peak, n_fits, n_evaluations)


@dataclass(frozen=True)
class ControlResult:
    """One of the five section-25 controls, run through the same harness as the learner."""
    name: str
    gain: Measured                                   # primary-metric gain over the no-learning baseline
    detected: bool | None = None                     # C: flagged as a memoriser?  E: leak caught?  else None
    gains: tuple = field(default=(), repr=False, compare=False)     # per-unit gains (for the resemblance test)
    note: str = ""

    def check(self) -> list[str]:
        errs = []
        if self.name not in CONTROL_NAMES:
            errs.append(f"unknown control {self.name!r}")
        if self.name in ("C_identity_memoriser", "E_leaky_learner") and self.detected is None:
            errs.append(f"{self.name}: 'detected' must be recorded (was the planted defect seen?)")
        return errs


def summarise_control(name: str, gains: Sequence[float], clusters=None, *, detected: bool | None = None, seed: int = 0, note: str = "") -> ControlResult:
    """Build a ControlResult from per-unit gains (cluster bootstrap)."""
    g = np.asarray(gains, float)
    return ControlResult(name, Measured.from_boot(TS.cluster_bootstrap_mean(g, clusters, seed=seed)), detected, tuple(float(x) for x in g[np.isfinite(g)]), note)


def leak_status_from(findings: Sequence | None, *, audited: bool = True, breach: Exception | None = None) -> LeakStatus:
    """CLEAN / LEAK_DETECTED / UNAUDITED from audit findings (objects with .severity == 'fail' as in blind_gates.Finding, or dicts
    with 'severity'/'status'). Not audited, or no findings list at all, is UNAUDITED - never CLEAN by default."""
    if isinstance(breach, FirewallBreach):
        return LeakStatus.LEAK_DETECTED
    if not audited or findings is None:
        return LeakStatus.UNAUDITED
    for f in findings:
        sev = f.get("severity", f.get("status")) if isinstance(f, Mapping) else getattr(f, "severity", getattr(f, "status", ""))
        if str(sev).lower() in ("fail", "breach", "leak"):
            return LeakStatus.LEAK_DETECTED
    return LeakStatus.CLEAN


# ---------------------------------------------------------------------------------------------------------------
# calibration (Measured wrapper over the existing pattern_reliability.brier / ece)
# ---------------------------------------------------------------------------------------------------------------
def calibration_measure(p, y, *, n_boot: int = 300, seed: int = 0, bins: int = 10) -> Measured:
    """Expected calibration error of probabilities `p` against 0/1 outcomes `y` (lower is better), bootstrap interval over
    samples, Brier in the note. Uses engine.pattern_reliability.brier/ece (equal-mass bins); fewer than 3*bins samples is UNTESTED."""
    from .. import pattern_reliability as PR
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) != len(y) or ((p < 0) | (p > 1)).any() or not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("p must be in [0,1] and y in {0,1}")
    e = PR.ece(p, y, bins)
    if not math.isfinite(e):
        return Measured.untested(f"only {len(p)} samples (< {3 * bins})")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(p), len(p))
        v = PR.ece(p[i], y[i], bins)
        if math.isfinite(v):
            draws.append(v)
    lo, hi = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))) if len(draws) >= 20 else (float("nan"), float("nan"))
    return Measured(e, lo, hi, len(p), MStatus.MEASURED, f"ECE (lower is better); Brier {PR.brier(p, y):.4f}; base-rate Brier {float(y.mean() * (1 - y.mean())):.4f}")


# ---------------------------------------------------------------------------------------------------------------
# the scorecard
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LearningScorecard:
    learner_version: str
    now: dt.date
    code_hash: str
    seed: int | None
    baseline_performance: Measured = field(default_factory=Measured)
    post_learning_performance: Measured = field(default_factory=Measured)
    learning_gain: Measured = field(default_factory=Measured)       # post - baseline, paired: THE primary gain the gate tests
    same_year_gain: Measured = field(default_factory=Measured)
    cross_year_gain: Measured = field(default_factory=Measured)
    cross_regime_gain: Measured = field(default_factory=Measured)
    cross_stock_gain: Measured = field(default_factory=Measured)
    transfer_ratio: Measured = field(default_factory=Measured)
    risk_change: Measured = field(default_factory=Measured)          # positive = less risk
    drawdown_change: Measured = field(default_factory=Measured)      # positive = shallower drawdown
    band_share: Measured = field(default_factory=Measured)           # share of weeks in the 5-10% band (post-learning)
    movement_performance: Measured = field(default_factory=Measured)
    direction_performance: Measured = field(default_factory=Measured)
    mover_performance: Measured = field(default_factory=Measured)
    calibration: Measured = field(default_factory=Measured)
    memorization_gap: Measured = field(default_factory=Measured)
    identity_gap: Measured = field(default_factory=Measured)
    future_leak_status: LeakStatus = LeakStatus.UNAUDITED
    stability: Measured = field(default_factory=Measured)
    compute_cost: ComputeCost = field(default_factory=ComputeCost)
    controls: Mapping[str, ControlResult] = field(default_factory=dict)
    data_hash: str = ""
    config_hash: str = ""
    controls_hash: str = ""                                          # hash of the source of the five control learners (frozen across versions)
    transfer_verdict: str = "UNTESTED"
    curve_verdict: str = "UNTESTED"
    label: ValidationLabel = ValidationLabel.NOT_VALIDATED

    def check(self) -> list[str]:
        errs = []
        if not self.learner_version:
            errs.append("learner_version missing")
        if not self.code_hash:
            errs.append("code_hash missing: a scorecard must say which code produced it")
        if self.seed is None:
            errs.append("seed missing")
        for f in SECTION_47_FIELDS:
            if not hasattr(self, f):
                errs.append(f"section-47 field {f} missing")
        errs += self.compute_cost.check()
        for c in self.controls.values():
            errs += c.check()
        if self.label == ValidationLabel.VALIDATED:
            errs.append("a scorecard may never carry the VALIDATED label; validation needs a sealed window")
        return errs

    def untested_fields(self) -> list[str]:
        out = []
        for f in SECTION_47_FIELDS:
            v = getattr(self, f)
            if isinstance(v, Measured) and not v.measured:
                out.append(f)
            elif isinstance(v, LeakStatus) and v == LeakStatus.UNAUDITED:
                out.append(f)
            elif isinstance(v, ComputeCost) and not v.measured:
                out.append(f)
        return out

    def as_record(self) -> dict:
        rec = {"learner_version": self.learner_version, "now": self.now.isoformat(), "code_hash": self.code_hash, "data_hash": self.data_hash,
               "config_hash": self.config_hash, "controls_hash": self.controls_hash, "seed": self.seed, "label": self.label.value, "transfer_verdict": self.transfer_verdict,
               "curve_verdict": self.curve_verdict}
        for f in SECTION_47_FIELDS + ("learning_gain",):
            v = getattr(self, f)
            rec[f] = v.as_dict() if isinstance(v, Measured) else (v.value if isinstance(v, LeakStatus) else dataclasses.asdict(v))
        rec["controls"] = {n: {"gain": c.gain.as_dict(), "detected": c.detected, "note": c.note, "gains_hash": stable_hash(list(c.gains))}
                           for n, c in sorted(self.controls.items())}
        rec["card_id"] = stable_hash(rec)
        return rec

    @property
    def card_id(self) -> str:
        return self.as_record()["card_id"]


# ---------------------------------------------------------------------------------------------------------------
# the claim gate
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GateCheck:
    name: str
    ok: bool
    blocking: bool
    message: str


@dataclass(frozen=True)
class ClaimDecision:
    allowed: bool                  # may the words 'improved'/'improvement' be used about this learner version?
    checks: tuple
    label: ValidationLabel

    @property
    def blockers(self) -> tuple:
        return tuple(c.message for c in self.checks if c.blocking and not c.ok)

    @property
    def warnings(self) -> tuple:
        return tuple(c.message for c in self.checks if not c.blocking and not c.ok)

    def statement(self, card: LearningScorecard) -> str:
        if self.allowed:
            g = card.learning_gain
            return (f"{card.learner_version}: gain over its own no-learning baseline {g.value:+.4g} (CI {g.lo:+.3g}..{g.hi:+.3g}) survives controls A-E "
                    f"and unseen-year transfer. Label stays {ValidationLabel.NOT_VALIDATED.value}: a sealed blind window has not confirmed it.")
        return (f"{card.learner_version}: no claim of improvement. {self.label.value}. Blocked by: " + "; ".join(self.blockers[:6]) +
                (f" (+{len(self.blockers) - 6} more)" if len(self.blockers) > 6 else "") + ".")


def gate_improvement_claim(card: LearningScorecard, *, resemblance_max: float = 0.8, ratio_floor: float = TS.RATIO_FLOOR, stability_min: float = 0.3,
                           gap_tol: float = 0.0) -> ClaimDecision:
    """Decide whether 'learning improved' may be said about `card`. Blocking checks (all must pass):
      scorecard_valid       the card passes its own check()
      controls_present      A, C, D, E were run and B (the learner) has a gain
      learning_gain         the learner's paired gain over its own baseline has a CI above zero
      no_learning_flat      control A's gain interval contains zero (else the harness itself moves: the run is void)
      beats_luck_floor      the learner's gain exceeds the upper bound of the random learner's gain (D)
      memoriser_seen        control C was FLAGGED as a memoriser: the instrument can see memorisation
      leak_seen             control E's leak was CAUGHT: the audit can see a leak
      learner_leak_clean    the learner's own future-leak status is CLEAN (not UNAUDITED)
      not_memoriser         memorisation gap and identity gap are measured and not significantly positive; the learner's per-unit
                            gains do not correlate with the memoriser's above resemblance_max
      cross_year_transfer   the unseen-year gain is measured with a CI above zero and the transfer ratio is not below ratio_floor
      another_context       at least one of regime/stock gain is measured and none is significantly negative
      risk_not_worse        risk and drawdown changes are not significantly negative
      stability             stability measured and at least stability_min
    Non-blocking warnings: calibration untested, compute cost unmeasured, fields untested."""
    cs = []
    add = lambda name, ok, msg, blocking=True: cs.append(GateCheck(name, bool(ok), blocking, msg if not ok else "ok"))
    errs = card.check()
    add("scorecard_valid", not errs, "scorecard invalid: " + "; ".join(errs))
    missing = [n for n in REQUIRED_OTHER_CONTROLS if n not in card.controls or not card.controls[n].gain.measured]
    add("controls_present", not missing and card.learning_gain.measured,
        f"controls not run/measured: {', '.join(missing) or 'none'}; learner gain {'measured' if card.learning_gain.measured else 'UNTESTED'}")
    add("learning_gain", card.learning_gain.significantly_positive, "the learner's gain over its own baseline is not significantly above zero")
    A = card.controls.get("A_no_learning")
    add("no_learning_flat", A is not None and A.gain.measured and A.gain.lo <= 0 <= A.gain.hi,
        "the no-learning control is not flat: the harness moves without learning (void run)")
    D = card.controls.get("D_random_learner")
    add("beats_luck_floor", D is not None and D.gain.measured and card.learning_gain.measured and card.learning_gain.value > max(D.gain.hi, 0.0),
        "the learner's gain does not exceed the random learner's upper bound (luck floor)")
    C, E = card.controls.get("C_identity_memoriser"), card.controls.get("E_leaky_learner")
    add("memoriser_seen", C is not None and C.detected is True, "the identity-memoriser control was not flagged: the memorisation instrument cannot see a memoriser")
    add("leak_seen", E is not None and E.detected is True, "the leaky-learner control was not caught: the future-leak audit cannot see a leak")
    add("learner_leak_clean", card.future_leak_status == LeakStatus.CLEAN, f"future-leak status is {card.future_leak_status.value}, not CLEAN")
    mg, ig = card.memorization_gap, card.identity_gap
    resembles = False
    B = card.controls.get("B_learner")
    if B is not None and C is not None and len(B.gains) == len(C.gains) >= 8:
        a, b = np.asarray(B.gains), np.asarray(C.gains)
        if a.std() > 0 and b.std() > 0:
            resembles = float(np.corrcoef(a, b)[0, 1]) > resemblance_max
    add("not_memoriser", mg.measured and ig.measured and not (mg.lo > gap_tol) and not (ig.lo > gap_tol) and not resembles,
        "memorisation/identity gap untested" if not (mg.measured and ig.measured) else
        ("the learner's per-unit gains resemble the identity memoriser's" if resembles else "memorisation or identity gap is significantly positive"))
    ratio_ok = card.transfer_ratio.measured and card.transfer_ratio.value >= ratio_floor
    add("cross_year_transfer", card.cross_year_gain.significantly_positive and ratio_ok,
        "unseen-year gain is not significantly positive" if not card.cross_year_gain.significantly_positive
        else f"transfer ratio {card.transfer_ratio.status.value if not card.transfer_ratio.measured else format(card.transfer_ratio.value, '.2f')} is below {ratio_floor:.2f} (over-specialised)")
    others = [card.cross_regime_gain, card.cross_stock_gain]
    add("another_context", any(o.measured for o in others) and not any(o.significantly_negative for o in others),
        "no other context (regime/stock) was tested" if not any(o.measured for o in others) else "the gain is significantly negative in another context")
    add("tier_measured", card.band_share.measured and card.risk_change.measured and card.drawdown_change.measured,
        "band share, risk change or drawdown change untested: the tiered objective cannot be judged")
    add("risk_not_worse", not card.risk_change.significantly_negative and not card.drawdown_change.significantly_negative,
        "risk or drawdown got significantly worse")
    add("stability", card.stability.measured and card.stability.value >= stability_min, f"transfer stability {'untested' if not card.stability.measured else format(card.stability.value, '.2f')} < {stability_min}")
    add("calibration_measured", card.calibration.measured, "calibration untested", blocking=False)
    add("compute_measured", card.compute_cost.measured, "compute cost unmeasured", blocking=False)
    allowed = all(c.ok for c in cs if c.blocking)
    # FAILED_VALIDATION needs measured evidence AGAINST the learner; missing controls/measurements are INSUFFICIENT_EVIDENCE
    against = (card.learning_gain.significantly_negative or card.cross_year_gain.significantly_negative or mg.lo > gap_tol or ig.lo > gap_tol
               or resembles or card.risk_change.significantly_negative or card.drawdown_change.significantly_negative
               or (card.transfer_ratio.measured and card.cross_year_gain.measured and card.transfer_ratio.value < ratio_floor)
               or (D is not None and D.gain.measured and card.learning_gain.measured and card.learning_gain.value <= max(D.gain.hi, 0.0)
                   and card.learning_gain.hi < D.gain.hi))
    lab = ValidationLabel.NOT_VALIDATED if allowed else (ValidationLabel.FAILED_VALIDATION if against else ValidationLabel.INSUFFICIENT_EVIDENCE)
    return ClaimDecision(allowed, tuple(cs), lab)


# ---------------------------------------------------------------------------------------------------------------
# text guard
# ---------------------------------------------------------------------------------------------------------------
def _word_re(words: Sequence[str]):
    body = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(r"(?<!not )(?<!no )(?<!never )(?<!without )(?<!cannot be )(?<!un)(?<!in)\b(" + body + r")\b", re.IGNORECASE)


def claim_violations(text: str, decision: ClaimDecision) -> list[str]:
    """Words in `text` that assert more than the decision supports. 'validated', 'production ready', 'done', 'finished',
    'complete', 'learning works' are never allowed (a scorecard cannot validate); 'improved', 'improvement' only when the gate
    passed. Negated uses ('not validated', 'no improvement') are fine."""
    bad = [m.group(1).lower() for m in _word_re(FORBIDDEN_ALWAYS).finditer(text)]
    if not decision.allowed:
        bad += [m.group(1).lower() for m in _word_re(FORBIDDEN_WITHOUT_GATE).finditer(text)]
    return sorted(set(bad))


def assert_claim_ok(text: str, decision: ClaimDecision) -> None:
    v = claim_violations(text, decision)
    if v:
        raise UnsupportedClaim(f"text claims {v} but the scorecard controls do not support it: {'; '.join(decision.blockers[:3]) or 'validation needs a sealed window'}")


def render_scorecard(card: LearningScorecard, decision: ClaimDecision | None = None) -> str:
    """Human-readable card. The rendering passes its own claim check: it cannot contain a claim the gate refused."""
    d = decision or gate_improvement_claim(card)
    fmt = lambda m: (f"{m.value:+.5f} [{m.lo:+.4f},{m.hi:+.4f}] n={m.n}" if m.measured and math.isfinite(m.lo) else
                     f"{m.value:+.5f} n={m.n}" if m.measured else f"{m.status.value}: {m.note}")
    L = [f"LEARNING SCORECARD {card.learner_version}  as of {card.now}  code {card.code_hash}  seed {card.seed}  [{d.label.value}]"]
    for f in SECTION_47_FIELDS:
        v = getattr(card, f)
        L.append(f"  {f:<26} " + (v.value if isinstance(v, LeakStatus) else (f"cpu {v.cpu_seconds:.1f}s wall {v.wall_seconds:.1f}s ram {v.peak_ram_mb:.0f}MB fits {v.n_fits}" if isinstance(v, ComputeCost) and v.measured else "UNTESTED" if isinstance(v, ComputeCost) else fmt(v))))
    L.append(f"  {'learning_gain':<26} {fmt(card.learning_gain)}")
    for n in CONTROL_NAMES:
        c = card.controls.get(n)
        L.append(f"  control {n:<24} " + ("NOT RUN" if c is None else f"{fmt(c.gain)}" + (f" detected={c.detected}" if c.detected is not None else "")))
    L.append("  " + d.statement(card))
    for w in d.warnings:
        L.append(f"  warning: {w}")
    text = "\n".join(L)
    assert_claim_ok(text.replace(d.statement(card), ""), d)          # numbers and labels only; the statement is built to comply
    return text


# ---------------------------------------------------------------------------------------------------------------
# building a card from the module reports
# ---------------------------------------------------------------------------------------------------------------
def build_scorecard(*, learner_version: str, now, code_hash: str, seed: int | None, baseline: Measured, post: Measured, learning_gain: Measured,
                    transfer=None, delta=None, value=None, calibration: Measured | None = None, leak: LeakStatus = LeakStatus.UNAUDITED,
                    compute: ComputeCost | None = None, controls: Sequence[ControlResult] = (), data_hash: str = "", config_hash: str = "",
                    curve_verdict: str = "UNTESTED", band_share: Measured | None = None) -> LearningScorecard:
    """Assemble a card from a transfer.TransferReport, a learning_curve.LearningDelta and a portfolio_value.ValueDecomposition
    (all optional: what is missing stays UNTESTED). Nothing is invented: a missing report leaves its fields untested."""
    from .transfer import Axis
    kw: dict = {}
    if transfer is not None:
        ax = transfer.axes
        get = lambda a: ax.get(a)
        yr = get(Axis.YEAR)
        kw["same_year_gain"] = Measured.from_boot(yr.same, "familiar year, dated after learning") if yr else Measured.untested("no year axis")
        kw["cross_year_gain"] = Measured.from_boot(yr.cross, "unseen later years") if yr and yr.tested else Measured.untested("no unseen-year units")
        for name, a in (("cross_regime_gain", Axis.REGIME), ("cross_stock_gain", Axis.STOCK)):
            r = get(a)
            kw[name] = Measured.from_boot(r.cross, f"unseen {a.value.lower()}") if r and r.tested else Measured.untested(f"no unseen-{a.value.lower()} units")
        if yr is not None:
            rr = yr.ratio
            kw["transfer_ratio"] = (Measured(rr.value, yr.ratio_ci.lo if yr.ratio_ci and yr.ratio_ci.lo is not None else float("nan"),
                                             yr.ratio_ci.hi if yr.ratio_ci and yr.ratio_ci.hi is not None else float("nan"), yr.cross.n,
                                             MStatus.MEASURED, rr.reason) if rr.value is not None else
                                    Measured.undefined(f"{rr.status.value}: {rr.reason}"))
            kw["memorization_gap"] = Measured.from_gap(yr.memorization, "replay minus forward on familiar years")
        kw["identity_gap"] = Measured.from_gap(transfer.identity, "identity kept minus scrambled")
        stabs = [r.stability.score for r in ax.values() if r.stability.n_groups >= 3]
        kw["stability"] = Measured.point(min(stabs), len(stabs), "worst axis stability score") if stabs else Measured.untested("fewer than 3 held-out groups on every axis")
        kw["transfer_verdict"] = transfer.overall.value
    if delta is not None:
        def dv(name):
            v = delta.values[name]
            return Measured(v.delta, v.lo, v.hi, v.n, MStatus.MEASURED, f"{v.state.value}; source {v.source}") if v.state.value != "UNTESTED" else Measured.untested(f"no {v.source}")
        kw["risk_change"], kw["drawdown_change"] = dv("risk_delta"), dv("drawdown_delta")
        if band_share is None:
            band_share = dv("band_share_delta")
    if value is not None:
        def cv(name, attr="new"):
            c = value.components[name]
            return Measured.untested(c.note) if c.status != "MEASURED" else Measured(getattr(c, attr), c.lo if attr == "delta" else float("nan"),
                                                                                       c.hi if attr == "delta" else float("nan"), c.n, MStatus.MEASURED,
                                                                                       f"{name}: delta {c.delta:+.4f} [{c.lo:+.4f},{c.hi:+.4f}]")
        kw["movement_performance"] = cv("movement_prediction")
        kw["direction_performance"] = cv("direction_value")
        kw["mover_performance"] = cv("selection_value")
    return LearningScorecard(learner_version=learner_version, now=as_date(now), code_hash=code_hash, seed=seed, baseline_performance=baseline,
                             post_learning_performance=post, learning_gain=learning_gain, band_share=band_share or Measured.untested("no band share"),
                             calibration=calibration or Measured.untested("no calibration data"), future_leak_status=leak,
                             compute_cost=compute or ComputeCost(), controls={c.name: c for c in controls}, data_hash=data_hash, config_hash=config_hash,
                             curve_verdict=curve_verdict, **kw)


# ---------------------------------------------------------------------------------------------------------------
# versions: comparison and the append-only store
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Comparison:
    old: str
    new: str
    regressions: tuple             # fields where the newer version is significantly worse
    gains: tuple                   # fields where it is significantly better
    untested_in_new: tuple
    newer_is_better: bool

    def statement(self) -> str:
        if self.newer_is_better:
            return f"{self.new} beats {self.old} on {', '.join(self.gains)} with no regression, and its own claim gate passes."
        return (f"{self.new} is NOT shown to be better than {self.old}: regressions [{', '.join(self.regressions) or 'none'}]; gains "
                f"[{', '.join(self.gains) or 'none'}]; untested [{', '.join(self.untested_in_new) or 'none'}].")


_HIGHER_BETTER = ("learning_gain", "same_year_gain", "cross_year_gain", "cross_regime_gain", "cross_stock_gain", "risk_change", "drawdown_change",
                  "band_share", "movement_performance", "direction_performance", "mover_performance", "stability")
_LOWER_BETTER = ("calibration", "memorization_gap", "identity_gap")


def compare_scorecards(old: LearningScorecard, new: LearningScorecard) -> Comparison:
    """Field-by-field comparison. Intervals are compared, not point values: a difference counts only when the two intervals do not
    overlap. `newer_is_better` needs at least one separated gain, no separated regression, and a passing claim gate on `new`."""
    reg, gain, unt = [], [], []
    for f in _HIGHER_BETTER + _LOWER_BETTER:
        a, b = getattr(old, f), getattr(new, f)
        if not b.measured:
            unt.append(f)
            continue
        if not a.measured or not (math.isfinite(a.lo) and math.isfinite(b.lo)):
            continue
        if f in _HIGHER_BETTER:
            better, worse = b.lo > a.hi, b.hi < a.lo
        else:
            better, worse = b.hi < a.lo, b.lo > a.hi
        if better:
            gain.append(f)
        elif worse:
            reg.append(f)
    ok = bool(gain) and not reg and gate_improvement_claim(new).allowed
    return Comparison(old.learner_version, new.learner_version, tuple(reg), tuple(gain), tuple(unt), ok)


class ScorecardStore:
    """Append-only, hash-chained JSONL of scorecard records. A learner_version can be written once; changing behaviour means a new
    version and a new line (section 49). verify() re-derives every hash and the chain."""

    def __init__(self, path):
        self.path = Path(path)

    def _lines(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def append(self, card: LearningScorecard, *, allow_new_controls: bool = False) -> str:
        errs = card.check()
        if errs:
            raise ValueError("refusing to store an invalid scorecard: " + "; ".join(errs))
        rows = self._lines()
        first = next((r["record"].get("controls_hash", "") for r in rows if r["record"].get("controls_hash", "")), "")
        if first and card.controls_hash and card.controls_hash != first and not allow_new_controls:
            raise ValueError("the control learners changed since the first stored scorecard: gains are no longer comparable across versions "
                             "(pass allow_new_controls=True to start a new control epoch, knowingly)")
        if any(r["record"]["learner_version"] == card.learner_version for r in rows):
            raise FileExistsError(f"scorecard for {card.learner_version} already stored: history is never overwritten; use a new version")
        prev = rows[-1]["chain"] if rows else "GENESIS"
        rec = card.as_record()
        chain = stable_hash({"prev": prev, "record": rec}, 32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(canonical_json({"prev": prev, "chain": chain, "record": rec}) + "\n")
        return rec["card_id"]

    def load(self) -> list[dict]:
        return [r["record"] for r in self._lines()]

    def verify(self) -> list[str]:
        """[] when intact; otherwise one message per broken link or altered record."""
        problems, prev = [], "GENESIS"
        for i, r in enumerate(self._lines()):
            rec = r["record"]
            body = {k: v for k, v in rec.items() if k != "card_id"}
            if stable_hash(body) != rec.get("card_id"):
                problems.append(f"line {i}: record content does not match its card_id (edited)")
            if r["prev"] != prev:
                problems.append(f"line {i}: prev hash {r['prev']} != {prev} (line removed or reordered)")
            if stable_hash({"prev": r["prev"], "record": rec}, 32) != r["chain"]:
                problems.append(f"line {i}: chain hash does not match its content")
            prev = r["chain"]
        return problems

    # ---- reading the history ----------------------------------------------------------------------------------
    def latest(self) -> dict | None:
        rows = self.load()
        return rows[-1] if rows else None

    def trajectory(self, field_name: str) -> list[tuple]:
        """[(learner_version, value, lo, hi)] of one Measured field across stored versions, in storage order."""
        out = []
        for r in self.load():
            m = r.get(field_name)
            if isinstance(m, dict) and m.get("status") == "MEASURED":
                num = lambda v: float(v) if v not in (None, "nan") else float("nan")
                out.append((r["learner_version"], num(m["value"]), num(m["lo"]), num(m["hi"])))
        return out

    def regression_alerts(self, fields: Sequence[str] = ("cross_year_gain", "cross_regime_gain", "cross_stock_gain", "stability", "learning_gain")) -> list[str]:
        """Consecutive stored versions where a later one is significantly worse than the earlier on a higher-is-better field
        (intervals separate downward). An alert is a prompt to look, not a verdict."""
        out = []
        for f in fields:
            tr = self.trajectory(f)
            for (v0, _, lo0, hi0), (v1, _, lo1, hi1) in zip(tr, tr[1:]):
                if all(math.isfinite(x) for x in (lo0, hi0, lo1, hi1)) and hi1 < lo0:
                    out.append(f"{v1} is significantly worse than {v0} on {f}")
        return out


# ---------------------------------------------------------------------------------------------------------------
# the five controls, run through the same harness as the learner (section 25)
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeResult:
    changed_share: float           # share of rows whose weight changed when the outcome columns were shuffled
    leaked: bool
    n: int


def outcome_dependence_probe(predict, frame, outcome_cols: Sequence[str] = ("alt", "base", "fwd", "learned"), *, seed: int = 0, tol: float = 1e-12) -> ProbeResult:
    """Mutation test for future-information use. A legitimate predictor's weights depend on the FEATURES of the rows it is asked
    about; if the realised outcomes in those rows are jointly shuffled and any weight changes, the predictor read the answer.
    Deterministic given `seed`. Rows whose weight is unchanged prove nothing about other rows, so the share is reported."""
    import pandas as pd
    cols = [c for c in outcome_cols if c in frame.columns]
    if not cols or len(frame) < 4:
        return ProbeResult(float("nan"), False, len(frame))
    w0 = np.asarray(predict(frame), float)
    perm = np.random.default_rng(seed).permutation(len(frame))
    shuffled = frame.copy()
    for c in cols:
        shuffled[c] = frame[c].to_numpy()[perm]
    w1 = np.asarray(predict(shuffled), float)
    ch = float((np.abs(w0 - w1) > tol).mean())
    return ProbeResult(ch, ch > 0.0, int(len(frame)))


def no_learning_fit(train):
    """CONTROL A: never takes the altered action. Its gain is identically zero; it verifies the harness adds nothing of its own."""
    return lambda t: np.zeros(len(t))


def leaky_learner_fit(train):
    """CONTROL E: reads the realised outcome of the very rows it is asked about (alt > base) - future information. Its held-out gain
    is spectacular and worthless; the leak audit must catch it."""
    return lambda t: (t["alt"].to_numpy(float) > t["base"].to_numpy(float)).astype(float)


@dataclass(frozen=True)
class ControlSuite:
    controls: tuple
    learning_gain: Measured
    memorization_gap: Measured
    identity_gap: Measured
    leak: LeakStatus
    fold_axis: object                # transfer.AxisResult of the learner
    n_fits: int


def run_controls(units, learner_fit, now, *, seed: int = 0, n_boot: int = 300, min_units: int = 30) -> ControlSuite:
    """Run A (no learning), B (`learner_fit`), C (identity memoriser), D (random) and E (leaky) through the SAME forward-year folds of
    the unit-table harness and return ControlResults plus the learner's gain, memorisation gap (replay minus held-out), identity gap
    (identity kept minus scrambled, on the situations it trained on) and leak status (outcome-dependence probe on held-out rows).
    C is `detected` when the harness flags it over-specialised or identity dependent; E when the probe sees it read the answer."""
    from . import transfer as T
    d = T.prepare_units(units.assign(learned=units["base"]) if "learned" not in units else units, now)
    folds = T.make_folds(d, T.Axis.YEAR)
    fits = {"A_no_learning": no_learning_fit, "B_learner": learner_fit, "C_identity_memoriser": T.identity_memoriser_fit,
            "D_random_learner": T.random_learner_fit(seed), "E_leaky_learner": leaky_learner_fit}
    res, out = {}, []
    for name, fit in fits.items():
        fr = T.run_folds(units, fit, folds, now)
        ax = T.axis_result_from_folds(T.Axis.YEAR, fr, n_boot=n_boot, seed=seed, min_units=min_units)
        gains = np.concatenate([r.gain_test for r in fr]) if fr else np.array([])
        cl = np.concatenate([r.cluster_test for r in fr]) if fr else np.array([])
        res[name] = (fr, ax, gains, cl)
        detected = None
        if name == "C_identity_memoriser":
            detected = bool(ax.specialisation.over_specialised or ax.verdict.label in (TS.TransferVerdictLabel.OVER_SPECIALISED, TS.TransferVerdictLabel.IDENTITY_DEPENDENT))
        if name == "E_leaky_learner" and folds:
            f = folds[-1]
            detected = outcome_dependence_probe(fit(d.iloc[f.train_idx].copy()), d.iloc[f.test_idx], seed=seed).leaked
        out.append(ControlResult(name, Measured.from_boot(TS.cluster_bootstrap_mean(gains, cl, n_boot=n_boot, seed=seed)), detected,
                                 tuple(float(x) for x in gains[np.isfinite(gains)]), "held-out forward years"))
    fr_b, ax_b, g_b, c_b = res["B_learner"]
    replay = np.concatenate([r.gain_replay for r in fr_b]) if fr_b else np.array([])
    rc = np.concatenate([r.cluster_replay for r in fr_b]) if fr_b else np.array([])
    mem = Measured.from_gap(TS.memorization_gap(replay, g_b, rc, c_b, n_boot=n_boot, seed=seed), "replay minus held-out") if len(replay) and len(g_b) else Measured.untested("no folds")
    ident, leak = Measured.untested("no folds"), LeakStatus.UNAUDITED
    if folds:
        te = T.identity_probe_units(units, learner_fit, folds[-1], now, seed=seed, on="train")
        ident = Measured.from_gap(TS.identity_gap(te["learned"] - te["base"], te["learned_disguised"] - te["base"], te["cluster"], n_boot=n_boot, seed=seed),
                                  "identity kept minus scrambled, on trained situations")
        leak = LeakStatus.LEAK_DETECTED if outcome_dependence_probe(learner_fit(d.iloc[folds[-1].train_idx].copy()), d.iloc[folds[-1].test_idx],
                                                                     seed=seed).leaked else LeakStatus.CLEAN
    return ControlSuite(tuple(out), Measured.from_boot(TS.cluster_bootstrap_mean(g_b, c_b, n_boot=n_boot, seed=seed), "held-out forward years"), mem, ident,
                        leak, ax_b, len(fits) * len(folds))


def control_code_hashes() -> dict:
    """Hash of the source of each control learner, so 'the controls did not change' is checkable, not assumed (WP4: frozen)."""
    import inspect
    from . import transfer as T
    fns = {"A_no_learning": no_learning_fit, "C_identity_memoriser": T.identity_memoriser_fit, "D_random_learner": T.random_learner_fit,
           "E_leaky_learner": leaky_learner_fit, "probe": outcome_dependence_probe}
    return {n: stable_hash(inspect.getsource(f)) for n, f in sorted(fns.items())}


def scorecard_for_learner(units, learner_fit, *, learner_version: str, now, code_hash: str, seed: int = 0, n_boot: int = 300, min_units: int = 30) -> LearningScorecard:
    """A scorecard for a re-trainable learner on a unit table: controls A-E, forward-year gain, memorisation and identity gaps,
    leak status, compute cost. Risk, drawdown, band share and calibration need weekly portfolios and stay UNTESTED here, so the
    claim gate will (correctly) refuse until a portfolio-level card supplies them."""
    from . import transfer as T
    with ComputeMeter() as meter:
        suite = run_controls(units, learner_fit, now, seed=seed, n_boot=n_boot, min_units=min_units)
        d = T.prepare_units(units.assign(learned=units["base"]) if "learned" not in units else units, now)
        extra = {}
        for ax, name in ((T.Axis.STOCK, "cross_stock_gain"), (T.Axis.REGIME, "cross_regime_gain")):
            folds = T.make_folds(d, ax, seed=seed)
            if folds:
                r = T.axis_result_from_folds(ax, T.run_folds(units, learner_fit, folds, now), n_boot=n_boot, seed=seed, min_units=min_units)
                extra[name] = Measured.from_boot(r.cross, f"leave-one-{ax.value.lower()}-out")
        ax_b = suite.fold_axis
        base_m = Measured.from_boot(TS.cluster_bootstrap_mean(d["base"], d["cluster"], n_boot=n_boot, seed=seed), "mean baseline outcome per unit")
        post = (Measured.point(float(base_m.value + suite.learning_gain.value), suite.learning_gain.n, "baseline + held-out gain")
                if suite.learning_gain.measured and base_m.measured else Measured.untested("no held-out gain"))
        ratio = ax_b.ratio
        card_kw = dict(same_year_gain=Measured.from_boot(ax_b.same, "trained model replayed on its own years"), cross_year_gain=suite.learning_gain,
                       transfer_ratio=(Measured(ratio.value, ax_b.ratio_ci.lo if ax_b.ratio_ci and ax_b.ratio_ci.lo is not None else float("nan"),
                                                ax_b.ratio_ci.hi if ax_b.ratio_ci and ax_b.ratio_ci.hi is not None else float("nan"), ax_b.cross.n,
                                                MStatus.MEASURED, ratio.reason) if ratio.value is not None else Measured.undefined(f"{ratio.status.value}: {ratio.reason}")),
                       stability=Measured.point(ax_b.stability.score, ax_b.stability.n_groups, "share positive x mean/(mean+sd) over held-out years")
                       if ax_b.stability.n_groups >= 3 else Measured.untested("fewer than 3 held-out years"))
    card = LearningScorecard(learner_version=learner_version, now=as_date(now), code_hash=code_hash, seed=seed, baseline_performance=base_m,
                             post_learning_performance=post, learning_gain=suite.learning_gain, memorization_gap=suite.memorization_gap,
                             identity_gap=suite.identity_gap, future_leak_status=suite.leak,
                             compute_cost=meter.cost(n_fits=suite.n_fits, n_evaluations=len(suite.controls)), controls={c.name: c for c in suite.controls},
                             controls_hash=stable_hash(control_code_hashes()), transfer_verdict=ax_b.verdict.label.value, **{**card_kw, **extra})
    return card


def scorecard_markdown(card: LearningScorecard, decision: ClaimDecision | None = None) -> str:
    """Markdown table of the section-47 fields, the controls, and the gate's verdict. Passes its own claim check."""
    d = decision or gate_improvement_claim(card)
    def cell(v):
        if isinstance(v, LeakStatus):
            return v.value
        if isinstance(v, ComputeCost):
            return f"cpu {v.cpu_seconds:.1f}s / wall {v.wall_seconds:.1f}s / {v.peak_ram_mb:.0f} MB" if v.measured else "UNTESTED"
        if not v.measured:
            return f"{v.status.value}"
        return f"{v.value:+.5f}" + (f" [{v.lo:+.4f}, {v.hi:+.4f}]" if math.isfinite(v.lo) and math.isfinite(v.hi) else "")
    L = [f"### Learning scorecard: {card.learner_version}", f"as of {card.now}, code `{card.code_hash}`, seed {card.seed}, **{d.label.value}**", "",
         "| field | value |", "|---|---|"]
    for f in SECTION_47_FIELDS + ("learning_gain",):
        L.append(f"| {f} | {cell(getattr(card, f))} |")
    L += ["", "| control | gain | detected |", "|---|---|---|"]
    for n in CONTROL_NAMES:
        c = card.controls.get(n)
        L.append(f"| {n} | {'NOT RUN' if c is None else cell(c.gain)} | {'' if c is None or c.detected is None else c.detected} |")
    L += ["", d.statement(card)] + [f"- blocker: {b}" for b in d.blockers]
    text = "\n".join(L)
    assert_claim_ok(text.replace(d.statement(card), ""), d)
    return text


def scorecard_diff(old: LearningScorecard, new: LearningScorecard):
    """Field-by-field table of two cards: both values and whether the intervals separate, and in which direction. Untested on either
    side is shown as such. This is the table behind compare_scorecards, for reports."""
    import pandas as pd
    rows = []
    for f in _HIGHER_BETTER + _LOWER_BETTER:
        a, b = getattr(old, f), getattr(new, f)
        state = "untested"
        if a.measured and b.measured and all(math.isfinite(x) for x in (a.lo, a.hi, b.lo, b.hi)):
            up = (b.lo > a.hi) if f in _HIGHER_BETTER else (b.hi < a.lo)
            down = (b.hi < a.lo) if f in _HIGHER_BETTER else (b.lo > a.hi)
            state = "better" if up else "worse" if down else "same"
        rows.append({"field": f, old.learner_version: a.value, new.learner_version: b.value, "state": state})
    return pd.DataFrame(rows)


def missing_for_claim(card: LearningScorecard) -> list[str]:
    """What still has to be run or measured before the gate could pass: the blocking checks that failed for lack of evidence rather than
    for evidence against. A to-do list, not a verdict."""
    d = gate_improvement_claim(card)
    need = {"controls_present", "memoriser_seen", "leak_seen", "learner_leak_clean", "tier_measured", "another_context", "stability", "scorecard_valid"}
    return [c.message for c in d.checks if c.blocking and not c.ok and c.name in need]


# ---------------------------------------------------------------------------------------------------------------
# section 47 completion: each of the 19 fields is tied to the control that makes it reportable
# ---------------------------------------------------------------------------------------------------------------
# field -> (controls whose evidence it needs, extra condition name). A field is only 'claimable' as an improvement when it is measured
# AND every control it depends on was run and behaves as it must. 'own' means the field is its own evidence (a gap or a status).
FIELD_CONTROLS = {
    "baseline_performance": ("A_no_learning",), "post_learning_performance": ("A_no_learning", "D_random_learner"),
    "same_year_gain": ("A_no_learning", "D_random_learner", "C_identity_memoriser"),
    "cross_year_gain": ("A_no_learning", "D_random_learner", "C_identity_memoriser", "E_leaky_learner"),
    "cross_regime_gain": ("A_no_learning", "D_random_learner", "C_identity_memoriser"),
    "cross_stock_gain": ("A_no_learning", "D_random_learner", "C_identity_memoriser"),
    "transfer_ratio": ("C_identity_memoriser",), "risk_change": ("A_no_learning",), "drawdown_change": ("A_no_learning",),
    "band_share": ("A_no_learning", "D_random_learner"), "movement_performance": ("D_random_learner",),
    "direction_performance": ("D_random_learner",), "mover_performance": ("D_random_learner",),
    "calibration": (), "memorization_gap": ("C_identity_memoriser",), "identity_gap": ("C_identity_memoriser",),
    "future_leak_status": ("E_leaky_learner",), "stability": ("A_no_learning",), "compute_cost": (),
}


@dataclass(frozen=True)
class FieldClaim:
    field: str
    measured: bool
    controls_needed: tuple
    controls_missing: tuple        # needed but not run / not behaving
    may_claim_improved: bool
    reason: str


def _control_ok(card: LearningScorecard, name: str) -> bool:
    c = card.controls.get(name)
    if c is None or not c.gain.measured:
        return False
    if name == "A_no_learning":
        return c.gain.lo <= 0 <= c.gain.hi                      # must be flat
    if name in ("C_identity_memoriser", "E_leaky_learner"):
        return c.detected is True                               # the planted defect must have been seen
    return True


def field_claims(card: LearningScorecard) -> dict:
    """For each of the 19 section-47 fields: is it measured, which controls does it depend on, which of them are missing or misbehaving,
    and therefore may the field be described as an improvement. Fields with no control dependency (calibration, compute cost) need a
    reference instead: calibration must beat the base-rate Brier noted in its own text, compute cost is a fact, never an improvement."""
    out = {}
    for f in SECTION_47_FIELDS:
        v = getattr(card, f)
        if isinstance(v, LeakStatus):
            measured = v != LeakStatus.UNAUDITED
        elif isinstance(v, ComputeCost):
            measured = v.measured
        else:
            measured = v.measured
        need = FIELD_CONTROLS[f]
        missing = tuple(n for n in need if not _control_ok(card, n))
        if f == "compute_cost":
            ok, why = False, "a cost is a fact, never an improvement"
        elif f == "future_leak_status":
            ok, why = card.future_leak_status == LeakStatus.CLEAN and not missing, "clean status counts only if the leak control was caught"
        elif not measured:
            ok, why = False, "not measured"
        elif missing:
            ok, why = False, "controls missing or misbehaving: " + ", ".join(missing)
        elif isinstance(v, Measured):
            ok, why = (v.significantly_positive if f not in ("calibration", "memorization_gap", "identity_gap") else (v.hi < 0 if f == "calibration" else not (v.lo > 0))), "interval-based"
        else:
            ok, why = False, "no claim form"
        out[f] = FieldClaim(f, measured, need, missing, bool(ok), why)
    return out


def refuse_unsupported_field_claims(card: LearningScorecard, text: str) -> list[str]:
    """Scan free text for a field name paired with 'improved' / 'improvement' / 'better' (same sentence). Returns the fields the text
    claims an improvement in although the card does not allow it; an empty list means the text is within the evidence."""
    claims = field_claims(card)
    bad = []
    for sentence in re.split(r"[.;\n]", text):
        s = sentence.lower()
        if not re.search(r"(?<!not )(?<!no )\b(improved|improvement|better|gain)\b", s):
            continue
        for f, c in claims.items():
            if (f.replace("_", " ") in s or f in s) and not c.may_claim_improved:
                bad.append(f)
    return sorted(set(bad))


def calibration_control(p, y) -> Measured:
    """The control for the calibration field: Brier skill against always predicting the base rate. Positive skill with an interval is what
    a calibration 'improvement' has to show; the value is skill (higher is better), the note carries both Brier scores."""
    from .. import pattern_reliability as PR
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) < 30 or len(p) != len(y):
        return Measured.untested("fewer than 30 samples")
    ref = np.full(len(y), y.mean())
    rng = np.random.default_rng(0)
    skills = []
    for _ in range(300):
        i = rng.integers(0, len(p), len(p))
        b0 = PR.brier(np.full(len(i), y[i].mean()), y[i])
        skills.append(1 - PR.brier(p[i], y[i]) / b0 if b0 > 0 else np.nan)
    skills = np.array([s for s in skills if np.isfinite(s)])
    sk = 1 - PR.brier(p, y) / PR.brier(ref, y) if PR.brier(ref, y) > 0 else float("nan")
    if len(skills) < 20 or not np.isfinite(sk):
        return Measured.untested("degenerate outcomes")
    return Measured(float(sk), float(np.quantile(skills, 0.025)), float(np.quantile(skills, 0.975)), len(p), MStatus.MEASURED,
                    f"Brier skill vs base rate; Brier {PR.brier(p, y):.4f} vs base-rate {PR.brier(ref, y):.4f}")


def claim_matrix(card: LearningScorecard):
    """The 19 fields as a table: measured, controls needed, controls missing, and whether an improvement may be claimed for each."""
    import pandas as pd
    rows = [{"field": f, "measured": c.measured, "controls_needed": ",".join(c.controls_needed), "controls_missing": ",".join(c.controls_missing),
             "may_claim_improved": c.may_claim_improved, "reason": c.reason} for f, c in field_claims(card).items()]
    return pd.DataFrame(rows)


def claimable_fields(card: LearningScorecard) -> list[str]:
    """Names of the fields for which the card supports the word 'improved'. Usually a strict subset of what was measured."""
    return [f for f, c in field_claims(card).items() if c.may_claim_improved]
