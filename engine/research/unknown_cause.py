"""Unknown-cause system (contract C66 section 33; canons C62 section 42, C66). RESEARCH / AUDIT WORLD ONLY.

"We do not know" is a successful research result. UNKNOWN_CAUSE is a first-class verdict for an unexplained extreme move, and an
explanation is granted only when the evidence for that CAUSE beats a random-explanation null across the whole population of moves
(cause labels shuffled among moves with the same prevalence, Holm-corrected over every cause tried, replicated in both halves of
history) AND the individual move carries enough evidence of it. The module tracks the unknown rate by regime, volatility, sector and
model confidence, and answers the section's decisive question: when the unknown rate falls, is that genuine discovery of
transferable causes, or the classifier simply forcing labels? The answer is an end-to-end placebo (decouple evidence from moves and
re-run the whole pipeline: a genuine explainer loses its explanations, a forcing one keeps them), a transfer test (causes supported
earlier must explain later moves) and a check for loosened thresholds.

Builds on: engine.research.knowability (assessments, ledger, wilson), engine.learning.unknowns (UnknownLedger, one vocabulary for
unknowns), engine.pattern_reliability (holm, unknown_cause_share for pattern breaks), engine.learning.core (FailureCause).
Public entry: step(state, now, assessments) -> (state, UnknownCauseReport).
Status: IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.learning.core import FailureCause, FirewallBreach, Unknown, as_date, stable_hash
from engine.learning.unknowns import Availability as ActionAvailability
from engine.learning.unknowns import UnknownLedger, UnknownReason, unknown_state
from engine.pattern_reliability import holm, unknown_cause_share
from engine.research.core import Availability, Knowability
from engine.research.knowability import KnowabilityAssessment, KnowabilityError, wilson

UNKNOWN_CAUSE = "UNKNOWN_CAUSE"
CAUSES = ("EARNINGS", "FILING_EVENT", "INSIDER", "MACRO_EVENT", "CORPORATE_ACTION", "NEWS", "MARKET_WIDE", "MOMENTUM",
          "VOLUME_SURGE", "TECHNICAL_PATTERN", "REGIME", "PATTERN", "MEMORY", "SECTOR", "MODEL_FEATURE")
CHANNEL_CAUSE = {"EVENT": "FILING_EVENT", "MACRO": "MACRO_EVENT", "PRICE": "MOMENTUM", "VOLUME": "VOLUME_SURGE",
                 "TECHNICAL": "TECHNICAL_PATTERN", "MARKET_STATE": "REGIME", "PATTERNS": "PATTERN", "MEMORY": "MEMORY",
                 "CROSS_SECTION": "SECTOR", "FEATURES": "MODEL_FEATURE"}
KIND_CAUSE = {"FILING": "FILING_EVENT", "EVENT": "EARNINGS", "INSIDER": "INSIDER", "MACRO": "MACRO_EVENT",
              "CORPORATE_ACTION": "CORPORATE_ACTION", "NEWS_EXTERNAL": "NEWS"}
_AVAIL = {a.value: a for a in Availability}


class UnknownCauseError(ValueError):
    """Malformed cause record or configuration. Never swallowed."""


# ==================================================================================================================
# records: per-move cause evidence (identity-free tags only: no ticker, no date beyond the maturity stamp)
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class CauseEvidence:
    """Evidence that `cause` is present for a move, in [0,1], with WHEN the evidence was public. Strength is evidence of the
    cause's presence, never of its having produced the move: that is what the population test decides."""
    cause: str
    strength: float
    availability: Availability
    source_ids: tuple[str, ...] = ()

    def check(self) -> list[str]:
        errs = []
        if self.cause not in CAUSES:
            errs.append(f"unknown cause {self.cause!r}")
        if not 0.0 <= self.strength <= 1.0 or math.isnan(self.strength):
            errs.append(f"{self.cause}: strength outside [0,1]")
        return errs


@dataclasses.dataclass(frozen=True)
class MoveCauseRecord:
    move_id: str
    matured_at: str
    regime: str
    vol_bucket: str
    sector: str
    confidence_bucket: str
    abs_z: float
    evidence: tuple[CauseEvidence, ...] = ()
    decision_year: int = 0                         # kept for the same-year release guard; never handed to the trader

    def check(self) -> list[str]:
        errs = [e for ev in self.evidence for e in ev.check()]
        if not math.isfinite(self.abs_z) or self.abs_z < 0:
            errs.append(f"{self.move_id}: abs_z must be finite and >= 0")
        causes = [e.cause for e in self.evidence]
        if len(causes) != len(set(causes)):
            errs.append(f"{self.move_id}: duplicate cause evidence")
        return errs

    def strength(self, cause: str) -> float:
        return max((e.strength for e in self.evidence if e.cause == cause), default=0.0)

    def get(self, cause: str) -> CauseEvidence | None:
        return next((e for e in self.evidence if e.cause == cause), None)


def confidence_bucket(c: float | None) -> str:
    return "NONE" if c is None else "LOW" if c < 0.34 else "MID" if c < 0.67 else "HIGH"


def record_from_assessment(a: KnowabilityAssessment, min_strength: float = 0.05) -> MoveCauseRecord | None:
    """Knowability assessment -> cause evidence. Channels contribute KNOWN_BEFORE evidence of their cause; discovered
    explanations contribute after-the-fact evidence with their own availability; a large systematic share is MARKET_WIDE
    evidence (KNOWN_ONLY_AFTER: the market move was itself the outcome). DATA_FAILURE moves and moves with no measurable
    magnitude yield None: neither is an unexplained market move."""
    if a.classification == Knowability.DATA_FAILURE or a.abs_move_z is None:
        return None
    ev: dict[str, CauseEvidence] = {}

    def put(cause, strength, avail, ids):
        cur = ev.get(cause)
        if cur is None or strength > cur.strength:
            ev[cause] = CauseEvidence(cause, float(min(1.0, max(0.0, strength))), avail, tuple(ids))
    for ch, st in a.knowledge_state_at_decision.items():
        if st.get("anticipation") is not None and st["anticipation"] >= min_strength and ch in CHANNEL_CAUSE:
            put(CHANNEL_CAUSE[ch], st["anticipation"], Availability.KNOWN_BEFORE_EVENT, (ch,))
    for iid, av, strength, _prec in a.explanations:
        kind = iid.split("-")[0]
        cause = "EARNINGS" if kind == "ER" else "FILING_EVENT" if kind in ("EDGAR", "8K", "10KA") else "INSIDER" if kind == "INS" \
            else "CORPORATE_ACTION" if kind == "SPLIT" else "MACRO_EVENT" if kind == "MACRO" else "NEWS"
        put(cause, strength, _AVAIL.get(av, Availability.UNCERTAIN), (iid,))
    if a.systematic_share is not None and a.systematic_share >= min_strength:
        put("MARKET_WIDE", a.systematic_share, Availability.KNOWN_ONLY_AFTER_EVENT, ("systematic",))
    return MoveCauseRecord(a.move_id, a.matured_at, a.regime or "UNSPECIFIED", a.vol_bucket, a.sector or "UNSPECIFIED",
                           confidence_bucket(a.model_confidence), float(a.abs_move_z), tuple(sorted(ev.values(), key=lambda e: e.cause)),
                           as_date(a.decision_date).year)


# ==================================================================================================================
# configuration: every rule that can make an explanation easier is in one hashed object
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class AssignerConfig:
    """`min_strength` (per-move evidence bar), `require_supported` (population evidence bar), alpha/min_lift/min_present/n_perm
    (how hard the population test is) and `allow_after_fact` (may an explanation rest on information that arrived later) are the
    knobs a forcing classifier turns. They are hashed so a rate change can be traced to a threshold change."""
    min_strength: float = 0.35
    require_supported: bool = True
    allow_after_fact: bool = True
    alpha: float = 0.05
    min_lift: float = 1.3
    min_present: int = 8
    big_z: float = 3.0
    n_perm: int = 400
    seed: int = 0
    require_replication: bool = True

    def check(self) -> list[str]:
        errs = []
        if not 0.0 <= self.min_strength <= 1.0:
            errs.append("min_strength outside [0,1]")
        if not 0.0 < self.alpha < 1.0:
            errs.append("alpha outside (0,1)")
        if self.min_present < 2 or self.n_perm < 50 or self.min_lift < 1.0:
            errs.append("min_present >= 2, n_perm >= 50, min_lift >= 1 required")
        return errs

    def hash(self) -> str:
        return stable_hash(self, 12)


def loosened(before: AssignerConfig, after: AssignerConfig) -> list[str]:
    """Knobs that changed in the direction that makes explanations easier to obtain. Tightening is not listed."""
    out = []
    if after.min_strength < before.min_strength:
        out.append(f"min_strength {before.min_strength} -> {after.min_strength}")
    if before.require_supported and not after.require_supported:
        out.append("population support no longer required")
    if not before.allow_after_fact and after.allow_after_fact:
        out.append("explanations may now rest on after-the-fact information")
    if after.alpha > before.alpha:
        out.append(f"alpha {before.alpha} -> {after.alpha}")
    if after.min_lift < before.min_lift:
        out.append(f"min_lift {before.min_lift} -> {after.min_lift}")
    if after.min_present < before.min_present:
        out.append(f"min_present {before.min_present} -> {after.min_present}")
    if before.require_replication and not after.require_replication:
        out.append("replication no longer required")
    return out


# ==================================================================================================================
# the random-explanation null: does a cause explain moves better than a random label with the same prevalence?
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class CauseTest:
    cause: str
    mode: str                                     # "magnitude" (present vs absent among movers) or "prevalence" (movers vs ordinary days)
    n_present: int
    n_other: int
    effect: float
    lift: float
    p_perm: float
    p_holm: float
    replicated: bool | None
    supported: bool
    why: str


def _flags(records: Sequence[MoveCauseRecord], cause: str, bar: float, allow_after: bool) -> np.ndarray:
    out = np.zeros(len(records), dtype=bool)
    for i, r in enumerate(records):
        e = r.get(cause)
        if e is None or e.strength < bar:
            continue
        if not allow_after and e.availability != Availability.KNOWN_BEFORE_EVENT:
            continue
        out[i] = True
    return out


def _seed_for(cfg: AssignerConfig, cause: str, salt: str = "") -> int:
    return int(stable_hash({"s": cfg.seed, "c": cause, "salt": salt}, 8), 16) % (2 ** 31)


def _perm_p(stat_fn: Callable[[np.ndarray], float], flags: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs = stat_fn(flags)
    ge = sum(stat_fn(rng.permutation(flags)) >= obs for _ in range(n_perm))
    return obs, (1 + ge) / (n_perm + 1)


def _magnitude_stat(z: np.ndarray) -> Callable[[np.ndarray], float]:
    def stat(f: np.ndarray) -> float:
        return float(z[f].mean() - z[~f].mean()) if f.any() and (~f).any() else 0.0
    return stat


def _lift(z: np.ndarray, f: np.ndarray, big: float) -> float:
    if not f.any() or f.all():
        return float("nan")
    p1 = (float((z[f] >= big).sum()) + 0.5) / (f.sum() + 1.0)
    p0 = (float((z[~f] >= big).sum()) + 0.5) / ((~f).sum() + 1.0)
    return p1 / p0


def test_causes(records: Sequence[MoveCauseRecord], cfg: AssignerConfig = AssignerConfig(),
                reference: Sequence[MoveCauseRecord] | None = None, causes: Sequence[str] = CAUSES) -> dict[str, CauseTest]:
    """For every cause: is its presence associated with bigger moves (or, given ordinary-day `reference` records, with being a
    mover at all) by more than a random label of equal prevalence would be? One-sided permutation p, Holm across the causes
    tried, a lift floor, and replication (same sign and lift >= 1 in both chronological halves). `supported` needs all of it."""
    errs = cfg.check()
    if errs:
        raise UnknownCauseError("; ".join(errs))
    recs = sorted(records, key=lambda r: (r.matured_at, r.move_id))
    z = np.array([r.abs_z for r in recs], dtype=float)
    big = float(np.quantile(z, 0.75)) if len(z) else cfg.big_z          # "big" is relative to this population, not a fixed sigma
    out: dict[str, CauseTest] = {}
    raw_p, tested = {}, []
    for c in causes:
        f = _flags(recs, c, cfg.min_strength, cfg.allow_after_fact)
        if reference is not None:
            fr = _flags(list(reference), c, cfg.min_strength, cfg.allow_after_fact)
            pooled = np.concatenate([f, fr])
            n_m = len(f)
            if f.sum() + fr.sum() < cfg.min_present or n_m == 0 or len(fr) == 0:
                out[c] = CauseTest(c, "prevalence", int(f.sum()), int(fr.sum()), 0.0, float("nan"), 1.0, 1.0, None, False, "too few occurrences")
                continue
            lab = np.zeros(len(pooled), dtype=bool)
            lab[:n_m] = True
            rng = np.random.default_rng(_seed_for(cfg, c, "prev"))
            obs = f.mean() - fr.mean()
            ge = 0
            for _ in range(cfg.n_perm):
                pl = rng.permutation(lab)
                ge += (pooled[pl].mean() - pooled[~pl].mean()) >= obs
            p = (1 + ge) / (cfg.n_perm + 1)
            lift = (f.mean() + 0.5 / (len(f) + 1)) / (fr.mean() + 0.5 / (len(fr) + 1))
            out[c] = CauseTest(c, "prevalence", int(f.sum()), int(fr.sum()), float(obs), float(lift), p, p, None, False, "")
            raw_p[c] = p
            tested.append(c)
            continue
        if f.sum() < cfg.min_present or (~f).sum() < cfg.min_present:
            out[c] = CauseTest(c, "magnitude", int(f.sum()), int((~f).sum()), 0.0, float("nan"), 1.0, 1.0, None, False,
                               f"needs >= {cfg.min_present} present and absent, has {int(f.sum())}/{int((~f).sum())}")
            continue
        eff, p = _perm_p(_magnitude_stat(z), f, cfg.n_perm, _seed_for(cfg, c, "mag"))
        out[c] = CauseTest(c, "magnitude", int(f.sum()), int((~f).sum()), eff, _lift(z, f, big), p, p, None, False, "")
        raw_p[c] = p
        tested.append(c)
    if tested:
        adj = holm([raw_p[c] for c in tested])
        half = len(recs) // 2
        for c, ph in zip(tested, adj):
            t = out[c]
            rep = None
            if reference is None and half >= 2 * cfg.min_present:
                effs = []
                for sl in (slice(0, half), slice(half, None)):
                    ff = _flags(recs[sl], c, cfg.min_strength, cfg.allow_after_fact)
                    zz = z[sl]
                    effs.append((_magnitude_stat(zz)(ff), _lift(zz, ff, big)) if ff.any() and (~ff).any() else (0.0, float("nan")))
                rep = all(e > 0 and (l == l and l >= 1.0) for e, l in effs)
            ok = ph < cfg.alpha and t.effect > 0 and t.lift == t.lift and t.lift >= cfg.min_lift
            if cfg.require_replication and rep is not None:
                ok = ok and rep
            why = "supported" if ok else f"p_holm={ph:.3f}, effect={t.effect:+.2f}, lift={t.lift:.2f}" + ("" if rep is None else f", replicated={rep}")
            out[c] = dataclasses.replace(t, p_holm=float(ph), replicated=rep, supported=bool(ok), why=why)
    return out


def supported_causes(tests: Mapping[str, CauseTest]) -> frozenset[str]:
    return frozenset(c for c, t in tests.items() if t.supported)


# ==================================================================================================================
# verdicts: UNKNOWN_CAUSE unless the cause is supported by the population AND evidenced in this move
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class CauseVerdict:
    move_id: str
    cause: str                                     # a cause, or UNKNOWN_CAUSE
    strength: float
    tier: str                                      # EXPLAINED_BEFORE / EXPLAINED_AFTER / UNKNOWN
    rejected: tuple[str, ...] = ()                 # candidate causes with evidence in this move that the population did not support

    @property
    def unknown(self) -> bool:
        return self.cause == UNKNOWN_CAUSE


def assign(records: Sequence[MoveCauseRecord], tests: Mapping[str, CauseTest], cfg: AssignerConfig = AssignerConfig()) -> list[CauseVerdict]:
    """One verdict per record. The strongest SUPPORTED cause with enough evidence in the move wins; otherwise UNKNOWN_CAUSE, with
    the unsupported candidates listed so a human can see what was declined (declining is not the same as not having looked)."""
    sup = supported_causes(tests)
    out = []
    for r in records:
        cands = [e for e in r.evidence if e.strength >= cfg.min_strength and (cfg.allow_after_fact or e.availability == Availability.KNOWN_BEFORE_EVENT)]
        ok = [e for e in cands if (e.cause in sup) or not cfg.require_supported]
        rej = tuple(sorted(e.cause for e in cands if e not in ok))
        if not ok:
            out.append(CauseVerdict(r.move_id, UNKNOWN_CAUSE, 0.0, "UNKNOWN", rej))
            continue
        best = max(ok, key=lambda e: (e.strength, e.cause))
        tier = "EXPLAINED_BEFORE" if best.availability == Availability.KNOWN_BEFORE_EVENT else "EXPLAINED_AFTER"
        out.append(CauseVerdict(r.move_id, best.cause, best.strength, tier, rej))
    return out


def force_assign(records: Sequence[MoveCauseRecord], fallback: str = "MOMENTUM") -> list[CauseVerdict]:
    """The FOIL: always names a cause, the strongest available or a default. This is exactly the behaviour section 33 calls
    failure. It exists so the forced-label detector can be shown to catch it."""
    out = []
    for r in records:
        best = max(r.evidence, key=lambda e: (e.strength, e.cause), default=None)
        cause = best.cause if best is not None else fallback
        out.append(CauseVerdict(r.move_id, cause, best.strength if best else 0.0,
                                "EXPLAINED_BEFORE" if best and best.availability == Availability.KNOWN_BEFORE_EVENT else "EXPLAINED_AFTER"))
    return out


def unknown_counts(verdicts: Iterable[CauseVerdict]) -> tuple[int, int]:
    v = list(verdicts)
    return sum(1 for x in v if x.unknown), len(v)


# ==================================================================================================================
# unknown-cause rates by regime, volatility, sector and model confidence
# ==================================================================================================================
DIMENSIONS = ("regime", "vol_bucket", "sector", "confidence_bucket")


def _dim(r: MoveCauseRecord, by: str) -> str:
    if by not in DIMENSIONS:
        raise UnknownCauseError(f"unknown dimension {by!r}; choose from {DIMENSIONS}")
    return str(getattr(r, by)) or "UNSPECIFIED"


def rate_table(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], by: str, min_n: int = 1) -> pd.DataFrame:
    """Unknown rate per group with a Wilson interval. Groups smaller than min_n are omitted (their rate is not knowledge)."""
    v = {x.move_id: x for x in verdicts}
    groups: dict[str, list[bool]] = {}
    for r in records:
        if r.move_id in v:
            groups.setdefault(_dim(r, by), []).append(v[r.move_id].unknown)
    rows = []
    for g, flags in sorted(groups.items()):
        n, k = len(flags), int(sum(flags))
        if n < min_n:
            continue
        lo, hi = wilson(k, n)
        rows.append({"group": g, "n": n, "unknown": k, "rate": k / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows, columns=["group", "n", "unknown", "rate", "lo", "hi"])


def rate_tables(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], min_n: int = 1) -> dict[str, pd.DataFrame]:
    return {d: rate_table(records, verdicts, d, min_n) for d in DIMENSIONS}


def rate_heterogeneity(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], by: str, n_perm: int = 500,
                       seed: int = 0, min_group: int = 5) -> dict[str, float]:
    """Is the unknown rate different across groups beyond chance? Permutation test on the spread of group rates (group labels
    shuffled), so a regime that merely has few moves cannot look special."""
    v = {x.move_id: x.unknown for x in verdicts}
    pairs = [(_dim(r, by), v[r.move_id]) for r in records if r.move_id in v]
    keep = {g for g in {p[0] for p in pairs} if sum(1 for x in pairs if x[0] == g) >= min_group}
    pairs = [p for p in pairs if p[0] in keep]
    if len(keep) < 2:
        return {"spread": float("nan"), "p": float("nan"), "groups": len(keep), "n": len(pairs)}
    g = np.array([p[0] for p in pairs])
    u = np.array([p[1] for p in pairs], dtype=float)

    def spread(gg):
        rates = [u[gg == k].mean() for k in keep]
        return float(max(rates) - min(rates))
    obs = spread(g)
    rng = np.random.default_rng(seed)
    ge = sum(spread(rng.permutation(g)) >= obs for _ in range(n_perm))
    return {"spread": obs, "p": (1 + ge) / (n_perm + 1), "groups": len(keep), "n": len(pairs)}


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided z-test p-value for a difference of rates (1.0 when either sample is empty or the pooled rate is 0/1)."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return float(math.erfc(abs(z) / math.sqrt(2)))


# ==================================================================================================================
# forced labels versus genuine discovery
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class RateChange:
    before_rate: float
    after_rate: float
    drop: float
    p_drop: float
    n_before: int
    n_after: int
    transferable_share: float | None              # of newly explained moves, share explained by a cause supported on EARLIER data
    weak_share: float | None                      # of newly explained moves, share resting on a cause the population did not support
    placebo_ratio: float | None                   # explained share when evidence is decoupled from moves / real explained share
    loosened: tuple[str, ...]
    verdict: str
    reasons: tuple[str, ...]


VERDICTS = ("GENUINE_DISCOVERY", "FORCED_LABELS", "UNCONFIRMED", "NO_SIGNIFICANT_CHANGE", "UNKNOWN_RATE_ROSE", "INSUFFICIENT_EVIDENCE")


def _decouple(records: Sequence[MoveCauseRecord], rng: np.random.Generator) -> list[MoveCauseRecord]:
    """Same records with the magnitudes shuffled among them: evidence stays, but it no longer belongs to the move it explains."""
    z = rng.permutation([r.abs_z for r in records])
    return [dataclasses.replace(r, abs_z=float(zz)) for r, zz in zip(records, z)]


def placebo_explained_share(records: Sequence[MoveCauseRecord], cfg: AssignerConfig, n: int = 5, seed: int = 0,
                            assigner: Callable[[Sequence[MoveCauseRecord]], list[CauseVerdict]] | None = None) -> float:
    """Explained share after the full pipeline (population test, then per-move assignment) is run on decoupled data. A real
    explainer must collapse toward zero here; if it keeps explaining, its explanations never depended on the moves. `assigner`
    replaces the pipeline for classifiers that are not built from test_causes/assign."""
    rng = np.random.default_rng(seed)
    shares = []
    for _ in range(n):
        d = _decouple(records, rng)
        vs = assigner(d) if assigner is not None else assign(d, test_causes(d, cfg), cfg)
        k, m = unknown_counts(vs)
        shares.append(1.0 - k / m if m else 0.0)
    return float(np.mean(shares))


def diagnose_rate_change(before: Sequence[MoveCauseRecord], after: Sequence[MoveCauseRecord], cfg_before: AssignerConfig = AssignerConfig(),
                         cfg_after: AssignerConfig | None = None, min_n: int = 30, seed: int = 0,
                         assigner_after: Callable[[Sequence[MoveCauseRecord]], list[CauseVerdict]] | None = None) -> RateChange:
    """Did the unknown rate fall because causes were genuinely discovered, or because labels were forced?
    1. Significance: two-proportion test of the drop.
    2. Transfer: of the explained moves in the later half of `after`, how many rest on a cause supported on `before` or the earlier half of `after`?
    3. Support: how many rest on a cause the population test did NOT support (forced by a loosened bar)?
    4. Placebo: run the whole pipeline on decoupled `after` data; the ratio of its explained share to the real one.
    5. Drift: which thresholds were loosened between the two periods.
    `assigner_after` lets a black-box classifier be examined: it is used for the real verdicts and the placebo runs."""
    cfg_after = cfg_after or cfg_before
    tests_b = test_causes(before, cfg_before)
    v_b = assign(before, tests_b, cfg_before)
    if assigner_after is not None:
        v_a = assigner_after(after)
        tests_a = test_causes(after, cfg_after)
    else:
        tests_a = test_causes(after, cfg_after)
        v_a = assign(after, tests_a, cfg_after)
    kb, nb = unknown_counts(v_b)
    ka, na = unknown_counts(v_a)
    rb, ra = (kb / nb if nb else float("nan")), (ka / na if na else float("nan"))
    drift = tuple(loosened(cfg_before, cfg_after))
    reasons = []
    if nb < min_n or na < min_n:
        return RateChange(rb, ra, rb - ra, 1.0, nb, na, None, None, None, drift, "INSUFFICIENT_EVIDENCE",
                          (f"need >= {min_n} moves in each period, have {nb} and {na}",))
    p = two_proportion_p(kb, nb, ka, na)
    drop = rb - ra
    sup_b, sup_a = supported_causes(tests_b), supported_causes(tests_a)
    explained = [x for x in v_a if not x.unknown]
    ordered = sorted(after, key=lambda r: (r.matured_at, r.move_id))
    half = len(ordered) // 2
    sup_early = sup_b | supported_causes(test_causes(ordered[:half], dataclasses.replace(cfg_after, require_replication=False)))
    late_ids = {r.move_id for r in ordered[half:]}
    late_expl = [x for x in explained if x.move_id in late_ids]
    transfer = (sum(1 for x in late_expl if x.cause in sup_early) / len(late_expl)) if late_expl else None
    weak = (sum(1 for x in explained if x.cause not in sup_a) / len(explained)) if explained else None
    real_share = 1.0 - ra
    plc = placebo_explained_share(after, cfg_after, seed=seed, assigner=assigner_after)
    ratio = plc / real_share if real_share > 0 else None
    if p >= 0.10 or abs(drop) < 0.03:
        verdict = "NO_SIGNIFICANT_CHANGE"
        reasons.append(f"rate moved {rb:.2f} -> {ra:.2f} (p={p:.2f})")
    elif drop < 0:
        verdict = "UNKNOWN_RATE_ROSE"
        reasons.append(f"unknown rate rose {rb:.2f} -> {ra:.2f} (p={p:.3f}); more honesty, or a harder period")
    else:
        forced = bool(drift) or (ratio is not None and ratio > 0.6) or (weak is not None and weak > 0.4)
        if drift:
            reasons.append("thresholds loosened: " + "; ".join(drift))
        if ratio is not None and ratio > 0.6:
            reasons.append(f"placebo keeps {ratio:.0%} of its explanations with evidence decoupled from moves")
        if weak is not None and weak > 0.4:
            reasons.append(f"{weak:.0%} of new explanations rest on causes the population did not support")
        if forced:
            verdict = "FORCED_LABELS"
        elif transfer is not None and transfer >= 0.6 and (ratio is None or ratio <= 0.4):
            verdict = "GENUINE_DISCOVERY"
            reasons.append(f"{transfer:.0%} of new explanations use causes supported on earlier data; placebo collapses to {ratio:.0%}")
        else:
            verdict = "UNCONFIRMED"
            reasons.append(f"drop is significant but transfer={transfer if transfer is None else round(transfer, 2)} and placebo="
                           f"{ratio if ratio is None else round(ratio, 2)} do not confirm discovery")
    return RateChange(rb, ra, drop, p, nb, na, transfer, weak, ratio, drift, verdict, tuple(reasons))


def transfer_of_causes(early: Sequence[MoveCauseRecord], late: Sequence[MoveCauseRecord], cfg: AssignerConfig = AssignerConfig()) -> dict[str, Any]:
    """Freeze the causes supported on `early`; on `late`, do moves carrying their evidence have larger magnitudes than moves
    without it, by more than a random cause set of the same size would show (the null for 'any cause set would look this good')?
    Presence alone is not the test: common decoy causes are present on many moves and would win a coverage contest."""
    sup = sorted(supported_causes(test_causes(early, cfg)))
    nan = float("nan")
    if len(late) < 2 * cfg.min_present:
        return {"supported": sup, "effect": nan, "random_effect": nan, "p_random_ge": nan}
    z = np.array([r.abs_z for r in late], dtype=float)

    def effect(cs: Sequence[str]) -> float:
        f = np.array([any(r.strength(c) >= cfg.min_strength for c in cs) for r in late])
        return float(z[f].mean() - z[~f].mean()) if f.any() and (~f).any() else 0.0
    real = effect(sup) if sup else 0.0
    rng = np.random.default_rng(cfg.seed)
    present = sorted({e.cause for r in late for e in r.evidence})
    rand = [effect(list(rng.choice(present, size=len(sup), replace=False))) for _ in range(200)] if sup and len(present) >= len(sup) else []
    return {"supported": sup, "effect": real, "random_effect": float(np.mean(rand)) if rand else nan,
            "p_random_ge": (1 + sum(x >= real for x in rand)) / (len(rand) + 1) if rand else nan}


# ==================================================================================================================
# a book of windows: the unknown rate over time, and which causes were discovered when
# ==================================================================================================================
@dataclasses.dataclass(frozen=True)
class WindowRecord:
    window_id: str
    matured_through: str
    n_moves: int
    n_unknown: int
    config_hash: str
    supported: tuple[str, ...]
    verdict: str = ""

    @property
    def rate(self) -> float:
        return self.n_unknown / self.n_moves if self.n_moves else float("nan")


class UnknownRateBook:
    """Append-only history of windows. Answers: what is the unknown-rate trend, when was each cause first supported, and did
    every rate fall carry a GENUINE_DISCOVERY verdict?"""

    def __init__(self):
        self._w: list[WindowRecord] = []

    def __len__(self) -> int:
        return len(self._w)

    def windows(self) -> list[WindowRecord]:
        return list(self._w)

    def add(self, w: WindowRecord) -> None:
        if self._w and as_date(w.matured_through) < as_date(self._w[-1].matured_through):
            raise UnknownCauseError("windows must arrive in maturity order")
        if any(x.window_id == w.window_id for x in self._w):
            raise UnknownCauseError(f"window {w.window_id} already recorded")
        self._w.append(w)

    def trend(self) -> pd.DataFrame:
        return pd.DataFrame([{"window": w.window_id, "through": w.matured_through, "n": w.n_moves, "rate": w.rate,
                              "config": w.config_hash, "verdict": w.verdict} for w in self._w],
                            columns=["window", "through", "n", "rate", "config", "verdict"])

    def first_supported(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for w in self._w:
            for c in w.supported:
                out.setdefault(c, w.window_id)
        return out

    def persistence(self) -> dict[str, float]:
        """Share of windows after first support in which the cause stayed supported. A discovered cause that vanishes was a
        period effect, not a cause."""
        first = self.first_supported()
        ids = [w.window_id for w in self._w]
        out = {}
        for c, w0 in first.items():
            after = self._w[ids.index(w0):]
            out[c] = sum(1 for w in after if c in w.supported) / len(after)
        return out

    def unearned_drops(self, tol: float = 0.03) -> list[str]:
        """Windows where the unknown rate fell by more than tol without a GENUINE_DISCOVERY verdict: the failure mode of section 33."""
        bad = []
        for a, b in zip(self._w, self._w[1:]):
            if a.rate == a.rate and b.rate == b.rate and a.rate - b.rate > tol and b.verdict != "GENUINE_DISCOVERY":
                bad.append(b.window_id)
        return bad


# ==================================================================================================================
# one vocabulary with engine.learning.unknowns / failure / pattern_reliability
# ==================================================================================================================
def failure_cause(v: CauseVerdict) -> FailureCause:
    """UNKNOWN_CAUSE is the same statement as FailureCause.UNKNOWN: a correct answer, not a gap."""
    return FailureCause.UNKNOWN if v.unknown else FailureCause.INSUFFICIENT_EVIDENCE if v.strength < 0.5 else FailureCause.SELECTION_ERROR


def open_unknowns(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], ledger: UnknownLedger, now,
                  min_cluster: int = 5) -> list[str]:
    """One open unknown per (regime, volatility) cluster holding at least min_cluster UNKNOWN_CAUSE moves, in the shared
    UnknownLedger, so the rest of the system sees 'we do not know why these move' as an unknown with an action (collect data).
    Subjects are identity-free. Returns the subjects opened or kept."""
    v = {x.move_id: x for x in verdicts}
    clusters: dict[tuple[str, str], int] = {}
    for r in records:
        if r.move_id in v and v[r.move_id].unknown:
            clusters[(r.regime, r.vol_bucket)] = clusters.get((r.regime, r.vol_bucket), 0) + 1
    opened = []
    for (reg, vb), n in sorted(clusters.items()):
        if n < min_cluster:
            continue
        rec = unknown_state(f"unknown_cause:{reg}:{vb}", now, UnknownReason.UNEXPLAINED, ActionAvailability(can_collect_data=True))
        ledger.open(dataclasses.replace(rec, action_detail=f"{n} extreme moves unexplained; {rec.action_detail}"))
        opened.append(rec.subject)
    return opened


def resolve_on_discovery(ledger: UnknownLedger, change: RateChange, evidence_through, now) -> list[str]:
    """Close the open unknown-cause records ONLY when the rate change was confirmed as GENUINE_DISCOVERY. A forced or unconfirmed
    drop never resolves an unknown."""
    if change.verdict != "GENUINE_DISCOVERY":
        return []
    done = []
    for r in list(ledger.open_rows()):
        if r.subject.startswith("unknown_cause:"):
            ledger.resolve(r.subject, now, evidence_through, "; ".join(change.reasons) or "genuine discovery")
            done.append(r.subject)
    return done


def one_vocabulary_summary(verdicts: Sequence[CauseVerdict], break_investigations: pd.DataFrame | None = None) -> dict[str, Any]:
    """The unknown share of extreme MOVES and of pattern BREAKS (engine.pattern_reliability.unknown_cause_share) side by side,
    both from the same definition: unexplained after evidence was required."""
    k, n = unknown_counts(verdicts)
    lo, hi = wilson(k, n)
    out = {"moves": {"n": n, "unknown": k, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}}
    if break_investigations is not None and len(break_investigations):
        out["breaks"] = unknown_cause_share(break_investigations)
    return out


# ==================================================================================================================
# planted worlds: moves whose true cause structure is known by construction
# ==================================================================================================================
def planted_records(n: int = 400, seed: int = 0, real_causes: Sequence[str] = ("EARNINGS", "MARKET_WIDE"),
                    decoy_causes: Sequence[str] = ("MOMENTUM", "TECHNICAL_PATTERN"), p_real: float = 0.25, p_decoy: float = 0.4,
                    p_unexplained: float = 0.3, start: str = "2015-01-05", days_span: int = 1500) -> list[MoveCauseRecord]:
    """A world where REAL causes truly enlarge moves and DECOY causes are present about as often but change nothing.
    A fraction of moves have a big magnitude and no cause evidence at all: the honest UNKNOWN_CAUSE moves. Decoy evidence is
    the temptation section 33 warns about ("momentum", "technical pattern" exist for almost any move)."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    out = []
    for i in range(n):
        day = t0 + pd.Timedelta(days=int(i * days_span / n))
        ev, z = {}, float(abs(rng.normal(3.2, 0.5)))
        if rng.uniform() < p_unexplained:
            z += float(abs(rng.normal(0.8, 0.3)))
        else:
            for c in real_causes:
                if rng.uniform() < p_real:
                    ev[c] = CauseEvidence(c, float(rng.uniform(0.6, 0.95)), Availability.KNOWN_ONLY_AFTER_EVENT, (c,))
                    z += 0.9
        for c in decoy_causes:
            if rng.uniform() < p_decoy:
                ev[c] = CauseEvidence(c, float(rng.uniform(0.4, 0.9)), Availability.KNOWN_BEFORE_EVENT, (c,))
        out.append(MoveCauseRecord(f"PM{seed}-{i:05d}", str((day + pd.Timedelta(days=6)).date()), str(rng.choice(["R1", "R2", "R3"])),
                                   str(rng.choice(["LOW", "MID", "HIGH"])), str(rng.choice(["S1", "S2", "S3", "S4"])),
                                   str(rng.choice(["LOW", "MID", "HIGH"])), z, tuple(sorted(ev.values(), key=lambda e: e.cause)), day.year))
    return out


def null_records(n: int = 400, seed: int = 0) -> list[MoveCauseRecord]:
    """A world where EVERY cause is a decoy: evidence is independent of magnitude. Nothing may be supported here."""
    return planted_records(n, seed, real_causes=(), decoy_causes=("EARNINGS", "MARKET_WIDE", "MOMENTUM", "TECHNICAL_PATTERN", "VOLUME_SURGE"),
                           p_decoy=0.35, p_unexplained=0.0)


def truth_check(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], real: Sequence[str] = ("EARNINGS", "MARKET_WIDE")) -> dict[str, float]:
    """Score verdicts against planted truth: a move with real-cause evidence should be explained by a real cause; a move with
    no real cause should be UNKNOWN_CAUSE (or at worst explained by nothing real). Returns the four rates that matter."""
    v = {x.move_id: x for x in verdicts}
    tp = fn = fp = tn = decoy_named = 0
    for r in records:
        x = v[r.move_id]
        has_real = any(r.strength(c) >= 0.35 for c in real)
        if has_real:
            tp += int((not x.unknown) and x.cause in real)
            fn += int(x.unknown or x.cause not in real)
        else:
            tn += int(x.unknown)
            fp += int(not x.unknown)
            decoy_named += int((not x.unknown) and x.cause not in real)
    return {"recall_real": tp / max(1, tp + fn), "unknown_when_no_real": tn / max(1, tn + fp),
            "forced_label_rate": fp / max(1, tn + fp), "decoy_named_rate": decoy_named / max(1, tn + fp)}


# ==================================================================================================================
# sensitivity of the unknown rate to every rule, and per-cause profiles
# ==================================================================================================================
def strength_sweep(records: Sequence[MoveCauseRecord], cfg: AssignerConfig = AssignerConfig(),
                   bars: Sequence[float] = (0.15, 0.25, 0.35, 0.5, 0.65, 0.8)) -> pd.DataFrame:
    """Unknown rate as the per-move evidence bar moves, with the population test held to its own rules at each bar. A rate that
    swings wildly with the bar is a property of the bar. A flat rate is a property of the data."""
    rows = []
    for b in bars:
        c = dataclasses.replace(cfg, min_strength=float(b))
        tests = test_causes(records, c)
        k, n = unknown_counts(assign(records, tests, c))
        rows.append({"bar": b, "n": n, "unknown_rate": k / n if n else float("nan"), "n_supported": len(supported_causes(tests))})
    return pd.DataFrame(rows, columns=["bar", "n", "unknown_rate", "n_supported"])


def forcing_curve(records: Sequence[MoveCauseRecord], cfg: AssignerConfig = AssignerConfig()) -> dict[str, float]:
    """Unknown rate under the honest pipeline, with population support switched off, and with the FOIL. The gap between the first
    two is exactly the amount of 'explanation' that only exists because the null was skipped."""
    honest = unknown_counts(assign(records, test_causes(records, cfg), cfg))
    lax = unknown_counts(assign(records, {}, dataclasses.replace(cfg, require_supported=False)))
    forced = unknown_counts(force_assign(records))
    rate = lambda t: t[0] / t[1] if t[1] else float("nan")
    return {"honest": rate(honest), "no_population_test": rate(lax), "forced_foil": rate(forced),
            "explanations_owed_to_skipping_the_null": rate(honest) - rate(lax)}


def cause_profile(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], tests: Mapping[str, CauseTest]) -> pd.DataFrame:
    """Per cause: how often it was the named cause, how often it was present but declined, and the population test result."""
    named: dict[str, int] = {}
    declined: dict[str, int] = {}
    for x in verdicts:
        if not x.unknown:
            named[x.cause] = named.get(x.cause, 0) + 1
        for c in x.rejected:
            declined[c] = declined.get(c, 0) + 1
    rows = []
    for c in CAUSES:
        t = tests.get(c)
        rows.append({"cause": c, "named": named.get(c, 0), "declined": declined.get(c, 0), "supported": bool(t and t.supported),
                     "p_holm": t.p_holm if t else float("nan"), "lift": t.lift if t else float("nan"), "why": t.why if t else "not tested"})
    return pd.DataFrame(rows, columns=["cause", "named", "declined", "supported", "p_holm", "lift", "why"])


def after_fact_share(verdicts: Sequence[CauseVerdict]) -> dict[str, float]:
    """Of explained moves, how many rest on hindsight-only evidence. High values are legitimate research findings but say nothing
    about predictive knowledge (section 7: do not convert hindsight into predictive knowledge)."""
    ex = [x for x in verdicts if not x.unknown]
    if not ex:
        return {"explained": 0, "before": float("nan"), "after": float("nan")}
    b = sum(1 for x in ex if x.tier == "EXPLAINED_BEFORE")
    return {"explained": len(ex), "before": b / len(ex), "after": 1 - b / len(ex)}


def verdict_stability(records: Sequence[MoveCauseRecord], cfg: AssignerConfig = AssignerConfig(), n: int = 8, frac: float = 0.8,
                      seed: int = 0) -> dict[str, float]:
    """Re-run the whole pipeline on random 80% subsamples. For each move, the share of runs that agree with the full-data verdict
    on unknown/explained. Low agreement on individual moves means the explanation was a coin flip of which other moves were present."""
    full = {x.move_id: x.unknown for x in assign(records, test_causes(records, cfg), cfg)}
    rng = np.random.default_rng(seed)
    agree, seen = {k: 0 for k in full}, {k: 0 for k in full}
    for _ in range(n):
        idx = rng.choice(len(records), size=max(2, int(frac * len(records))), replace=False)
        sub = [records[i] for i in sorted(idx)]
        vs = {x.move_id: x.unknown for x in assign(sub, test_causes(sub, cfg), cfg)}
        for k, u in vs.items():
            seen[k] += 1
            agree[k] += int(u == full[k])
    rates = [agree[k] / seen[k] for k in full if seen[k]]
    return {"mean_agreement": float(np.mean(rates)) if rates else float("nan"),
            "share_unstable": float(np.mean([r < 0.75 for r in rates])) if rates else float("nan"), "n_runs": n}


def supported_set_stability(records: Sequence[MoveCauseRecord], cfg: AssignerConfig = AssignerConfig(), n: int = 8, frac: float = 0.8,
                            seed: int = 0) -> dict[str, float]:
    """Per cause: share of subsample runs in which it is supported. A cause supported in 60% of subsamples is a fragile finding."""
    rng = np.random.default_rng(seed)
    tally = {c: 0 for c in CAUSES}
    for _ in range(n):
        idx = rng.choice(len(records), size=max(2, int(frac * len(records))), replace=False)
        for c in supported_causes(test_causes([records[i] for i in sorted(idx)], cfg)):
            tally[c] += 1
    return {c: v / n for c, v in tally.items()}


def per_year_support(records: Sequence[MoveCauseRecord], cfg: AssignerConfig = AssignerConfig(), min_year_n: int = 60) -> dict[str, Any]:
    """Consistent = holds in every year (C58-C61). A cause is CONSISTENT only if the population test supports it in every year with
    enough moves (year-level tests are underpowered by design; a cause that passes anyway is strong)."""
    by_year: dict[int, list[MoveCauseRecord]] = {}
    for r in records:
        by_year.setdefault(int(r.matured_at[:4]), []).append(r)
    years = {y: g for y, g in sorted(by_year.items()) if len(g) >= min_year_n}
    per = {y: supported_causes(test_causes(g, dataclasses.replace(cfg, require_replication=False))) for y, g in years.items()}
    consistent = sorted(set.intersection(*[set(s) for s in per.values()])) if per else []
    return {"years": sorted(years), "by_year": {y: sorted(s) for y, s in per.items()}, "consistent": consistent}


# ==================================================================================================================
# per-move confidence in an UNKNOWN verdict (unknown because nothing is there, or because we could not look?)
# ==================================================================================================================
def unknown_honesty(r: MoveCauseRecord, v: CauseVerdict, blind_causes: Iterable[str] = ()) -> dict[str, Any]:
    """Splits UNKNOWN_CAUSE into TRULY_UNEXPLAINED (channels that could have shown a cause were present and silent) and
    NOT_LOOKED (relevant causes had no evidence at all because the data was missing). The second is a coverage problem; only the
    first is a statement about the world."""
    if not v.unknown:
        return {"kind": "EXPLAINED"}
    blind = set(blind_causes)
    looked = {e.cause for e in r.evidence}
    unlooked = sorted(blind - looked)
    if v.rejected:
        return {"kind": "DECLINED_UNSUPPORTED", "candidates": list(v.rejected)}
    if unlooked and len(unlooked) >= max(1, len(CAUSES) // 3):
        return {"kind": "NOT_LOOKED", "blind": unlooked}
    return {"kind": "TRULY_UNEXPLAINED", "checked": sorted(looked)}


def honesty_mix(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], blind_causes: Iterable[str] = ()) -> dict[str, int]:
    v = {x.move_id: x for x in verdicts}
    mix: dict[str, int] = {}
    for r in records:
        k = unknown_honesty(r, v[r.move_id], blind_causes)["kind"]
        mix[k] = mix.get(k, 0) + 1
    return mix


# ==================================================================================================================
# state, report, firewall, persistence
# ==================================================================================================================
@dataclasses.dataclass
class UnknownCauseState:
    cfg: AssignerConfig = AssignerConfig()
    seen: dict[str, MoveCauseRecord] = dataclasses.field(default_factory=dict)
    book: UnknownRateBook = dataclasses.field(default_factory=UnknownRateBook)
    unknowns: UnknownLedger = dataclasses.field(default_factory=UnknownLedger)
    last_window: tuple[MoveCauseRecord, ...] = ()
    last_verdicts: tuple[CauseVerdict, ...] = ()


@dataclasses.dataclass(frozen=True)
class UnknownCauseReport:
    now: str
    n_records: int
    n_new: int
    unknown_rate: float | None
    rate_lo: float | None
    rate_hi: float | None
    supported: tuple[str, ...]
    tables: Mapping[str, Any]
    change: RateChange | None
    opened: tuple[str, ...]
    resolved: tuple[str, ...]
    unearned_drops: tuple[str, ...]
    honesty: Mapping[str, int]
    notes: tuple[str, ...]


def step(state: UnknownCauseState, now, assessments: Iterable[KnowabilityAssessment], window_id: str | None = None,
         blind_causes: Iterable[str] = (), min_n: int = 30) -> tuple[UnknownCauseState, UnknownCauseReport]:
    """Fold newly matured knowability assessments into the unknown-cause books. Only assessments that matured strictly before real
    `now` are used (an outcome maturing today is not yet a fact). Then: population test over ALL matured moves, verdicts, rate
    tables, a forced-versus-genuine diagnosis against the previous window, and unknown-ledger bookkeeping."""
    from engine.learning.core import require_past
    notes, new = [], []
    for a in assessments:
        try:
            require_past(a.matured_at, now, f"assessment {a.move_id}")
        except FirewallBreach:
            notes.append(f"{a.move_id}: not yet matured, skipped")
            continue
        if a.move_id in state.seen:
            continue
        rec = record_from_assessment(a)
        if rec is None:
            notes.append(f"{a.move_id}: {a.classification.value}, not an unexplained-market-move candidate")
            continue
        errs = rec.check()
        if errs:
            raise UnknownCauseError("; ".join(errs))
        state.seen[rec.move_id] = rec
        new.append(rec)
    recs = sorted(state.seen.values(), key=lambda r: (r.matured_at, r.move_id))
    if not recs:
        return state, UnknownCauseReport(str(as_date(now)), 0, 0, None, None, None, (), {}, None, (), (), tuple(state.book.unearned_drops()),
                                         {}, tuple(notes) + ("no matured moves yet",))
    tests = test_causes(recs, state.cfg)
    verdicts = assign(recs, tests, state.cfg)
    k, n = unknown_counts(verdicts)
    lo, hi = wilson(k, n)
    change = None
    if state.last_window and state.last_verdicts:
        prev = [r for r in state.last_window]
        cur_only = [r for r in recs if r.move_id not in {p.move_id for p in prev}]
        if len(cur_only) >= min_n and len(prev) >= min_n:
            change = diagnose_rate_change(prev, cur_only, state.cfg, state.cfg, min_n=min_n)
    wid = window_id or f"W{len(state.book) + 1:04d}"
    state.book.add(WindowRecord(wid, recs[-1].matured_at, n, k, state.cfg.hash(), tuple(sorted(supported_causes(tests))),
                                change.verdict if change else ""))
    opened = open_unknowns(recs, verdicts, state.unknowns, now)
    resolved = resolve_on_discovery(state.unknowns, change, recs[-1].matured_at, now) if change else []
    state.last_window, state.last_verdicts = tuple(recs), tuple(verdicts)
    return state, UnknownCauseReport(str(as_date(now)), n, len(new), k / n, lo, hi, tuple(sorted(supported_causes(tests))),
                                     rate_tables(recs, verdicts), change, tuple(opened), tuple(resolved),
                                     tuple(state.book.unearned_drops()), honesty_mix(recs, verdicts, blind_causes), tuple(notes))


def render_report(rep: UnknownCauseReport) -> str:
    """Plain text with counts and shares only (no ids, tickers or dates other than the run date)."""
    if rep.n_records == 0:
        return "unknown-cause: no matured moves yet"
    lines = [f"unknown-cause: {rep.n_records} moves ({rep.n_new} new), unknown rate {rep.unknown_rate:.1%} [{rep.rate_lo:.1%}, {rep.rate_hi:.1%}]",
             f"  supported causes: {', '.join(rep.supported) or 'none (every explanation was declined)'}"]
    for d, t in rep.tables.items():
        if len(t):
            worst = t.sort_values("rate", ascending=False).iloc[0]
            lines.append(f"  {d}: highest unknown rate {worst['group']} {worst['rate']:.0%} (n={int(worst['n'])})")
    if rep.change is not None:
        lines.append(f"  rate change: {rep.change.verdict}; " + "; ".join(rep.change.reasons))
    if rep.unearned_drops:
        lines.append(f"  WARNING unearned drops in windows: {', '.join(rep.unearned_drops)}")
    if rep.honesty:
        lines.append("  unknown mix: " + ", ".join(f"{k}={v}" for k, v in sorted(rep.honesty.items())))
    return "\n".join(lines)


def refuse_unknown_cause_features(columns: Iterable[str]) -> None:
    """Fail closed if a training/feature column looks like a cause label or unknown-cause verdict (hindsight labels)."""
    marks = ("unknown_cause", "cause_verdict", "explained_after", "explained_before", "cause_label", "is_unexplained")
    bad = sorted(str(c) for c in columns if any(m in str(c).lower() for m in marks))
    if bad:
        raise FirewallBreach(f"cause/unknown-cause labels offered as features: {bad}")


def release_prior(state: UnknownCauseState, now, replaying_years: Iterable[int] = (), min_n: int = 30) -> dict[str, Any]:
    """The only road to the trader: aggregate unknown rate per regime (quarter-rounded, groups under min_n withheld), over moves
    that matured strictly before real `now` and whose decision year is not being replayed in disguise. No ids, tickers or dates."""
    from engine.learning.core import require_past
    replay = {int(y) for y in replaying_years}
    recs = [r for r in state.seen.values() if r.decision_year not in replay]
    for r in recs:
        try:
            require_past(r.matured_at, now, "release")
        except FirewallBreach:
            recs = [x for x in recs if x.matured_at < str(as_date(now))]
            break
    if not recs:
        return {"prior": {}, "withheld_groups": 0}
    v = {x.move_id: x for x in assign(recs, test_causes(recs, state.cfg), state.cfg)}
    groups: dict[str, list[bool]] = {}
    for r in recs:
        groups.setdefault(r.regime, []).append(v[r.move_id].unknown)
    return {"prior": {g: {"unknown_rate": round(float(np.mean(f)) * 4) / 4} for g, f in sorted(groups.items()) if len(f) >= min_n},
            "withheld_groups": sum(1 for f in groups.values() if len(f) < min_n)}


def save_state(state: UnknownCauseState, path) -> str:
    """Persist the moves seen and the window book (the rest is recomputable). Atomic. Returns a content hash."""
    import json, os
    body = {"cfg": dataclasses.asdict(state.cfg), "cfg_hash": state.cfg.hash(),
            "records": [{**{k: v for k, v in dataclasses.asdict(r).items() if k != "evidence"},
                         "evidence": [{"cause": e.cause, "strength": e.strength, "availability": e.availability.value,
                                       "source_ids": list(e.source_ids)} for e in r.evidence]}
                        for r in sorted(state.seen.values(), key=lambda r: r.move_id)],
            "windows": [dataclasses.asdict(w) for w in state.book.windows()]}
    h = stable_hash(body, 20)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"hash": h, "body": body}, f, sort_keys=True)
    os.replace(tmp, path)
    return h


def load_state(path) -> UnknownCauseState:
    import json, os
    if not os.path.exists(path):
        return UnknownCauseState()
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if stable_hash(d["body"], 20) != d["hash"]:
        raise UnknownCauseError("unknown-cause state file failed its integrity hash")
    b = d["body"]
    st = UnknownCauseState(AssignerConfig(**b["cfg"]))
    for r in b["records"]:
        ev = tuple(CauseEvidence(e["cause"], e["strength"], Availability(e["availability"]), tuple(e["source_ids"])) for e in r["evidence"])
        rec = MoveCauseRecord(**{**{k: v for k, v in r.items() if k != "evidence"}, "evidence": ev})
        st.seen[rec.move_id] = rec
    for w in b["windows"]:
        st.book.add(WindowRecord(**{**w, "supported": tuple(w["supported"])}))
    return st


# ==================================================================================================================
# per-move evidence against the random-explanation null
# ==================================================================================================================
def evidence_null_p(records: Sequence[MoveCauseRecord], r: MoveCauseRecord, cause: str) -> float:
    """How surprising is THIS move's evidence strength for `cause`, given how strongly that cause's evidence shows up across
    all the other moves? Fraction of other moves whose strength for the cause is at least as high (add-one smoothed). A cause
    whose evidence is strong on most moves is common, so strong evidence on one move explains little."""
    s = r.strength(cause)
    others = [x.strength(cause) for x in records if x.move_id != r.move_id]
    if not others:
        return 1.0
    return (1 + sum(o >= s for o in others)) / (len(others) + 1)


def commonness(records: Sequence[MoveCauseRecord], cutoff: float = 0.35) -> dict[str, float]:
    """Share of moves that carry each cause's evidence at all. Near 1.0 = the cause 'exists' for nearly every move and cannot
    distinguish any of them (momentum, technical patterns)."""
    n = len(records)
    return {c: (sum(1 for r in records if r.strength(c) >= cutoff) / n if n else float("nan")) for c in CAUSES}


def overexplained(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], common_at: float = 0.5) -> list[str]:
    """Verdicts that name a cause present on more than `common_at` of all moves: an explanation that would fit almost anything.
    Returned as move ids for review; the assigner itself stays governed by the population test."""
    com = commonness(records)
    return sorted(x.move_id for x in verdicts if not x.unknown and com.get(x.cause, 0.0) > common_at)


def cause_dimension_matrix(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], by: str) -> pd.DataFrame:
    """Named-cause counts per group (rows) and cause (columns), UNKNOWN_CAUSE as its own column."""
    v = {x.move_id: x for x in verdicts}
    rows = [{"group": _dim(r, by), "cause": v[r.move_id].cause} for r in records if r.move_id in v]
    if not rows:
        return pd.DataFrame()
    return pd.crosstab(pd.DataFrame(rows)["group"], pd.DataFrame(rows)["cause"])


def shrunk_rates(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], by: str, strength: float = 10.0) -> pd.DataFrame:
    """Empirical-Bayes rates: each group's unknown rate pulled toward the overall rate with weight `strength` pseudo-moves, so a
    group of 4 moves cannot report 0% or 100% with a straight face."""
    t = rate_table(records, verdicts, by)
    if t.empty:
        return t
    overall = float(t["unknown"].sum() / t["n"].sum())
    t = t.copy()
    t["shrunk"] = (t["unknown"] + strength * overall) / (t["n"] + strength)
    return t


def interaction_rates(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], a: str = "regime", b: str = "vol_bucket",
                      min_n: int = 10) -> pd.DataFrame:
    """Unknown rate for pairs of dimensions (regime x volatility). Cells below min_n are dropped: a 2-way table fragments quickly."""
    v = {x.move_id: x.unknown for x in verdicts}
    cells: dict[tuple[str, str], list[bool]] = {}
    for r in records:
        if r.move_id in v:
            cells.setdefault((_dim(r, a), _dim(r, b)), []).append(v[r.move_id])
    rows = []
    for (ga, gb), f in sorted(cells.items()):
        if len(f) >= min_n:
            lo, hi = wilson(sum(f), len(f))
            rows.append({a: ga, b: gb, "n": len(f), "rate": sum(f) / len(f), "lo": lo, "hi": hi})
    return pd.DataFrame(rows, columns=[a, b, "n", "rate", "lo", "hi"])


# ==================================================================================================================
# forced-label detection on a series of windows
# ==================================================================================================================
def rolling_diagnosis(windows: Sequence[Sequence[MoveCauseRecord]], cfgs: Sequence[AssignerConfig] | AssignerConfig = AssignerConfig(),
                      min_n: int = 30, seed: int = 0) -> list[RateChange]:
    """diagnose_rate_change between each consecutive pair of windows. `cfgs` is one config for all windows or one per window (so
    a config change is attributed to the window where it happened)."""
    cf = list(cfgs) if not isinstance(cfgs, AssignerConfig) else [cfgs] * len(windows)
    if len(cf) != len(windows):
        raise UnknownCauseError("need one config per window")
    return [diagnose_rate_change(windows[i], windows[i + 1], cf[i], cf[i + 1], min_n, seed + i) for i in range(len(windows) - 1)]


def forced_detector_score(changes: Sequence[RateChange]) -> dict[str, Any]:
    """Summary over windows: how many falls were genuine, forced, unconfirmed. A healthy research process shows few forced
    falls; ANY forced fall is a defect to investigate."""
    tally = {v: 0 for v in VERDICTS}
    for c in changes:
        tally[c.verdict] += 1
    falls = tally["GENUINE_DISCOVERY"] + tally["FORCED_LABELS"] + tally["UNCONFIRMED"]
    return {"tally": tally, "falls": falls, "forced_share_of_falls": tally["FORCED_LABELS"] / falls if falls else float("nan"),
            "genuine_share_of_falls": tally["GENUINE_DISCOVERY"] / falls if falls else float("nan")}


def make_forcing_assigner(strength_floor: float = 0.0, share: float = 1.0, seed: int = 0) -> Callable[[Sequence[MoveCauseRecord]], list[CauseVerdict]]:
    """A classifier that skips the population test and names the strongest cause above `strength_floor` for a `share` of moves.
    Test double for the black-box path of diagnose_rate_change; it is what a drifting classifier looks like from the outside."""
    def run(records: Sequence[MoveCauseRecord]) -> list[CauseVerdict]:
        rng = np.random.default_rng(seed)
        out = []
        for r in records:
            best = max((e for e in r.evidence if e.strength >= strength_floor), key=lambda e: (e.strength, e.cause), default=None)
            if best is None or rng.uniform() > share:
                out.append(CauseVerdict(r.move_id, UNKNOWN_CAUSE, 0.0, "UNKNOWN"))
            else:
                out.append(CauseVerdict(r.move_id, best.cause, best.strength,
                                        "EXPLAINED_BEFORE" if best.availability == Availability.KNOWN_BEFORE_EVENT else "EXPLAINED_AFTER"))
        return out
    return run


def explanation_novelty(before: Sequence[MoveCauseRecord], after: Sequence[MoveCauseRecord], cfg: AssignerConfig = AssignerConfig()) -> dict[str, Any]:
    """Causes newly supported / no longer supported between two periods. A newly supported cause is the mechanism of genuine
    discovery; a cause that stops being supported means earlier explanations built on it should be re-examined."""
    sb, sa = supported_causes(test_causes(before, cfg)), supported_causes(test_causes(after, cfg))
    return {"new": sorted(sa - sb), "lost": sorted(sb - sa), "kept": sorted(sa & sb)}


def reexamine_lost(records: Sequence[MoveCauseRecord], lost: Iterable[str], verdicts: Sequence[CauseVerdict]) -> list[str]:
    """Moves whose explanation used a cause that is no longer supported (candidates to revert to UNKNOWN_CAUSE)."""
    gone = set(lost)
    return sorted(x.move_id for x in verdicts if x.cause in gone)


# ==================================================================================================================
# integrity checks on the verdict set itself
# ==================================================================================================================
def audit_verdicts(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], tests: Mapping[str, CauseTest],
                   cfg: AssignerConfig = AssignerConfig()) -> list[str]:
    """Invariants of an honest assignment. Any line returned is a defect: a verdict that names an unsupported cause, names a
    cause with no evidence, rests on hindsight when hindsight was forbidden, or a missing/duplicated verdict."""
    bad = []
    ids = [x.move_id for x in verdicts]
    if len(ids) != len(set(ids)):
        bad.append("duplicate verdicts")
    have = {r.move_id: r for r in records}
    if set(ids) != set(have):
        bad.append(f"verdicts cover {len(set(ids) & set(have))} of {len(have)} moves")
    sup = supported_causes(tests)
    for x in verdicts:
        r = have.get(x.move_id)
        if r is None or x.unknown:
            continue
        e = r.get(x.cause)
        if e is None or e.strength < cfg.min_strength:
            bad.append(f"{x.move_id}: names {x.cause} without enough evidence")
        if cfg.require_supported and x.cause not in sup:
            bad.append(f"{x.move_id}: names unsupported cause {x.cause}")
        if not cfg.allow_after_fact and e is not None and e.availability != Availability.KNOWN_BEFORE_EVENT:
            bad.append(f"{x.move_id}: rests on after-the-fact information")
    return bad


def duplicate_evidence_ids(records: Sequence[MoveCauseRecord]) -> dict[str, list[str]]:
    """Source ids used as evidence for more than one cause in the same move (double counting one fact as two causes)."""
    out: dict[str, list[str]] = {}
    for r in records:
        seen: dict[str, str] = {}
        for e in r.evidence:
            for sid in e.source_ids:
                if sid in seen and seen[sid] != e.cause:
                    out.setdefault(r.move_id, []).append(f"{sid}: {seen[sid]}+{e.cause}")
                seen[sid] = e.cause
    return out


def future_evidence(records: Sequence[MoveCauseRecord], now) -> list[str]:
    """Records matured at or after `now` (they must not be in any window judged at `now`)."""
    return sorted(r.move_id for r in records if as_date(r.matured_at) >= as_date(now))


def config_lineage(book: UnknownRateBook) -> list[tuple[str, str]]:
    """(window id, config hash) each time the assigner configuration changed. Every unexplained-rate shift must be read
    against this list."""
    out, prev = [], None
    for w in book.windows():
        if w.config_hash != prev:
            out.append((w.window_id, w.config_hash))
            prev = w.config_hash
    return out


def rate_fall_needs_evidence(book: UnknownRateBook, tol: float = 0.03) -> dict[str, Any]:
    """The section-33 acceptance test on a book: every fall in the unknown rate must carry GENUINE_DISCOVERY and must not
    coincide with a config change that loosened a rule."""
    ws = book.windows()
    problems, falls = [], 0
    for a, b in zip(ws, ws[1:]):
        if a.rate == a.rate and b.rate == b.rate and a.rate - b.rate > tol:
            falls += 1
            if b.verdict != "GENUINE_DISCOVERY":
                problems.append(f"{b.window_id}: fell {a.rate:.2f}->{b.rate:.2f} with verdict {b.verdict or 'none'}")
            if a.config_hash != b.config_hash:
                problems.append(f"{b.window_id}: fall coincides with a configuration change")
    return {"falls": falls, "problems": problems, "passes": not problems}


# ==================================================================================================================
# pattern breaks share the vocabulary: engine.pattern_reliability.investigate() output -> the same unknown-cause rates
# ==================================================================================================================
def break_unknown_table(inv: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Unknown-cause share of RESOLVED pattern breaks (pattern_reliability's own definition: verdict EXPLAINED_AND_GATED or
    DISCARDED_UNPREDICTABLE, cause == 'unknown'), optionally per column `by`, with Wilson intervals. Uses the source module's
    definition so the move-level and break-level numbers are comparable."""
    cols = ["group", "resolved", "unknown", "share", "lo", "hi"]
    if inv is None or len(inv) == 0:
        return pd.DataFrame(columns=cols)
    res = inv[inv["verdict"].isin(["EXPLAINED_AND_GATED", "DISCARDED_UNPREDICTABLE"])]
    if len(res) == 0:
        return pd.DataFrame(columns=cols)
    groups = [("all", res)] if by is None else [(str(k), g) for k, g in res.groupby(by)]
    rows = []
    for k, g in groups:
        n, u = len(g), int((g["cause"] == "unknown").sum())
        lo, hi = wilson(u, n)
        rows.append({"group": k, "resolved": n, "unknown": u, "share": u / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows, columns=cols)


def break_forcing_check(inv_before: pd.DataFrame, inv_after: pd.DataFrame, min_n: int = 20) -> dict[str, Any]:
    """Did the unknown share of pattern breaks fall between two investigation runs, and is that fall explained by a lower
    evidence bar? Only the fall and its significance can be judged from the frames; the reasons are the pattern_reliability
    `detail` strings, of which we report how many now cite no driver at all (a resolution without reasoning is forcing)."""
    def share(df):
        s = unknown_cause_share(df) if len(df) else {"resolved": 0, "unknown": 0, "share": float("nan")}
        return s
    a, b = share(inv_before), share(inv_after)
    if a["resolved"] < min_n or b["resolved"] < min_n:
        return {"verdict": "INSUFFICIENT_EVIDENCE", "before": a, "after": b}
    p = two_proportion_p(a["unknown"], a["resolved"], b["unknown"], b["resolved"])
    empty_detail = int((inv_after["detail"].astype(str).str.strip() == "").sum()) if "detail" in inv_after else 0
    fell = a["share"] - b["share"] > 0.03 and p < 0.10
    return {"verdict": "FELL_CHECK_REASONING" if fell else "NO_SIGNIFICANT_FALL", "before": a, "after": b, "p": p,
            "resolutions_without_reasoning": empty_detail}


# ==================================================================================================================
# streaming by year (market-wide research cannot hold every name-day; only exception rows and one snapshot per day survive)
# ==================================================================================================================
def stream_by_year(year_batches: Iterable[tuple[int, Sequence[KnowabilityAssessment]]], state: UnknownCauseState, now,
                   drop_after: bool = True, min_n: int = 30) -> list[UnknownCauseReport]:
    """Feed one year's assessments at a time. After each year the raw assessments may be dropped by the caller (`drop_after`
    documents that only MoveCauseRecords, a few hundred bytes each, are retained in the state). Years must arrive in order."""
    reports, last_year = [], None
    for year, batch in year_batches:
        if last_year is not None and year <= last_year:
            raise UnknownCauseError(f"years must arrive in increasing order, got {year} after {last_year}")
        last_year = year
        _, rep = step(state, now, batch, window_id=f"Y{year}", min_n=min_n)
        reports.append(rep)
    return reports


def memory_footprint(state: UnknownCauseState) -> dict[str, float]:
    """Rough bytes retained: records only, never bars. Lets the loop refuse to keep growing without bound."""
    import sys
    n = len(state.seen)
    ev = sum(len(r.evidence) for r in state.seen.values())
    return {"records": n, "evidence_items": ev, "approx_mb": (n * 400 + ev * 150 + sys.getsizeof(state.seen)) / 1e6}


def blind_causes_of(records: Sequence[MoveCauseRecord], floor: float = 0.05) -> list[str]:
    """Causes for which fewer than `floor` of moves carry any evidence at all: the data behind them is missing, not silent.
    Pass the result as `blind_causes` so UNKNOWN moves are split into TRULY_UNEXPLAINED and NOT_LOOKED."""
    n = len(records)
    if n == 0:
        return list(CAUSES)
    return sorted(c for c in CAUSES if sum(1 for r in records if r.get(c) is not None) / n < floor)


def cause_trend(book: UnknownRateBook) -> pd.DataFrame:
    """Window by window: which causes are supported (1/0), one column per cause. A cause that flips on and off is not a
    discovery yet; the persistence() number summarises the same table."""
    rows = []
    for w in book.windows():
        row = {"window": w.window_id, "rate": w.rate}
        row.update({c: int(c in w.supported) for c in CAUSES})
        rows.append(row)
    return pd.DataFrame(rows, columns=["window", "rate"] + list(CAUSES))


# ==================================================================================================================
# does a supported cause hold everywhere it is claimed? (a cause that works in one regime only is regime-bound, not general)
# ==================================================================================================================
def cause_by_group(records: Sequence[MoveCauseRecord], cause: str, by: str, cfg: AssignerConfig = AssignerConfig(),
                   min_group: int = 40) -> pd.DataFrame:
    """Magnitude effect of `cause` (mean |z| with minus without its evidence) inside each group of `by`, with a permutation p per
    group. Groups too small to test are reported as untestable rather than dropped."""
    cols = ["group", "n", "n_present", "effect", "p", "testable"]
    groups: dict[str, list[MoveCauseRecord]] = {}
    for r in records:
        groups.setdefault(_dim(r, by), []).append(r)
    rows = []
    for g, rs in sorted(groups.items()):
        f = _flags(rs, cause, cfg.min_strength, cfg.allow_after_fact)
        if len(rs) < min_group or f.sum() < cfg.min_present or (~f).sum() < cfg.min_present:
            rows.append({"group": g, "n": len(rs), "n_present": int(f.sum()), "effect": float("nan"), "p": float("nan"), "testable": False})
            continue
        z = np.array([r.abs_z for r in rs], dtype=float)
        eff, p = _perm_p(_magnitude_stat(z), f, cfg.n_perm, _seed_for(cfg, cause, f"grp:{by}:{g}"))
        rows.append({"group": g, "n": len(rs), "n_present": int(f.sum()), "effect": eff, "p": p, "testable": True})
    return pd.DataFrame(rows, columns=cols)


def regime_bound_causes(records: Sequence[MoveCauseRecord], cfg: AssignerConfig = AssignerConfig(), by: str = "regime",
                        min_group: int = 40) -> dict[str, str]:
    """Label each supported cause GENERAL (positive effect in every testable group), REGIME_BOUND (significant in some groups,
    absent or reversed in others) or UNTESTED_BY_GROUP (fewer than two testable groups). Only GENERAL causes should ever be trusted to
    explain a move in a group they were not tested in."""
    out = {}
    for c in sorted(supported_causes(test_causes(records, cfg))):
        t = cause_by_group(records, c, by, cfg, min_group)
        t = t[t["testable"]]
        if len(t) < 2:
            out[c] = "UNTESTED_BY_GROUP"
        elif (t["effect"] > 0).all():
            out[c] = "GENERAL"
        else:
            out[c] = "REGIME_BOUND"
    return out


def hindsight_fraction_by_cause(records: Sequence[MoveCauseRecord], cutoff: float = 0.35) -> dict[str, float]:
    """For each cause, the share of its strong evidence that only existed after the move. 1.0 means the cause can be studied
    but never anticipated; 0.0 means it was there to be used. Feeds the decision of which causes are worth a predictive feature."""
    out = {}
    for c in CAUSES:
        ev = [e for r in records if r.strength(c) >= cutoff and (e := r.get(c)) is not None]
        out[c] = (sum(1 for e in ev if e.availability != Availability.KNOWN_BEFORE_EVENT) / len(ev)) if ev else float("nan")
    return out


def anticipable_causes(records: Sequence[MoveCauseRecord], cfg: AssignerConfig = AssignerConfig(), max_hindsight: float = 0.5) -> list[str]:
    """Supported causes whose evidence is mostly available before the move: the ones worth turning into decision-time features."""
    frac = hindsight_fraction_by_cause(records, cfg.min_strength)
    return sorted(c for c in supported_causes(test_causes(records, cfg)) if frac[c] == frac[c] and frac[c] <= max_hindsight)


# ==================================================================================================================
# cause redundancy and residual magnitude (is one cause just another one in disguise? how big are the unexplained moves?)
# ==================================================================================================================
def cooccurrence(records: Sequence[MoveCauseRecord], cutoff: float = 0.35) -> pd.DataFrame:
    """Jaccard overlap of the moves carrying each pair of causes' evidence. Two supported causes with Jaccard near 1 are one cause
    counted twice, and crediting both inflates the explained share."""
    flags = {c: np.array([r.strength(c) >= cutoff for r in records]) for c in CAUSES}
    rows = []
    for i, a in enumerate(CAUSES):
        for b in CAUSES[i + 1:]:
            u = int((flags[a] | flags[b]).sum())
            if u:
                rows.append({"a": a, "b": b, "jaccard": float((flags[a] & flags[b]).sum() / u)})
    return pd.DataFrame(rows, columns=["a", "b", "jaccard"])


def redundant_supported(records: Sequence[MoveCauseRecord], cfg: AssignerConfig = AssignerConfig(), bar: float = 0.7) -> list[tuple[str, str]]:
    """Pairs of SUPPORTED causes that overlap above `bar`."""
    sup = supported_causes(test_causes(records, cfg))
    co = cooccurrence(records, cfg.min_strength)
    return [(r.a, r.b) for r in co.itertuples() if r.a in sup and r.b in sup and r.jaccard >= bar]


def residual_magnitude(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict]) -> dict[str, float]:
    """Mean |z| of UNKNOWN_CAUSE moves against explained ones. Unknown moves that are as large as explained ones mean the
    unexplained region is not just noise; much smaller ones mean 'unknown' is mostly the tail of ordinary volatility."""
    v = {x.move_id: x.unknown for x in verdicts}
    unk = [r.abs_z for r in records if v.get(r.move_id)]
    exp = [r.abs_z for r in records if v.get(r.move_id) is False]
    return {"unknown_mean_z": float(np.mean(unk)) if unk else float("nan"), "explained_mean_z": float(np.mean(exp)) if exp else float("nan"),
            "n_unknown": len(unk), "n_explained": len(exp)}


def verdict_frame(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict]) -> pd.DataFrame:
    """Flat research-side table: one row per move with its group tags, verdict, tier and the causes declined."""
    v = {x.move_id: x for x in verdicts}
    rows = []
    for r in records:
        x = v.get(r.move_id)
        if x is None:
            continue
        rows.append({"move_id": r.move_id, "regime": r.regime, "vol_bucket": r.vol_bucket, "sector": r.sector,
                     "confidence_bucket": r.confidence_bucket, "abs_z": r.abs_z, "cause": x.cause, "tier": x.tier,
                     "strength": x.strength, "declined": ",".join(x.rejected), "unknown": x.unknown})
    return pd.DataFrame(rows, columns=["move_id", "regime", "vol_bucket", "sector", "confidence_bucket", "abs_z", "cause", "tier",
                                       "strength", "declined", "unknown"])


def declined_summary(verdicts: Sequence[CauseVerdict]) -> dict[str, int]:
    """How many UNKNOWN_CAUSE verdicts declined each candidate: the causes the system was tempted by and refused."""
    out: dict[str, int] = {}
    for x in verdicts:
        if x.unknown:
            for c in x.rejected:
                out[c] = out.get(c, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def unknown_share_of_extremes(records: Sequence[MoveCauseRecord], verdicts: Sequence[CauseVerdict], top_frac: float = 0.1) -> dict[str, float]:
    """Unknown rate among the largest `top_frac` of moves versus the rest: section 33 is about unexplained EXTREME moves, so the
    rate that matters is the one at the top of the magnitude distribution."""
    if not records:
        return {"extreme": float("nan"), "rest": float("nan"), "n_extreme": 0}
    v = {x.move_id: x.unknown for x in verdicts}
    cut = float(np.quantile([r.abs_z for r in records], 1.0 - top_frac))
    ext = [v[r.move_id] for r in records if r.abs_z >= cut and r.move_id in v]
    rest = [v[r.move_id] for r in records if r.abs_z < cut and r.move_id in v]
    return {"extreme": float(np.mean(ext)) if ext else float("nan"), "rest": float(np.mean(rest)) if rest else float("nan"), "n_extreme": len(ext)}
