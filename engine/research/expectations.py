"""Immutable pre-prediction expectations (C68 checklist A; extends C62 archive hash chains, C66 two-stage decisions).

Before a position exists the system writes down EVERYTHING it expects: return and its distribution, direction, confidence,
volatility, time to peak, the exit window, best/worst excursion, market and sector regime, patterns with strengths and
interactions, the holding period, a probability distribution, why it chose this name and why it rejected the competitors, its
uncertainty, the alternative hypotheses it still entertains, the model version, the feature and knowledge state, the
timestamp and the information set it was allowed to see. The record is frozen the moment it is written:

  * it lives on a hash-chained lane (`SealedLane`, a typed lane on engine.learning.archive.ChainFile, i.e. on the same
    append-only chain PatternMemory writes), so any edit, deletion or reordering breaks every later hash;
  * the prediction id is derived from (subject, decided_at, model), so a second, different expectation for the same
    prediction is refused (`ExpectationRewrite`) - only a byte-identical re-record is accepted (idempotent);
  * `record` refuses an expectation that claims the future, one written after its entry session began (the outcome path
    exists from then on), and one whose information set or feature names contain anything dated or labelled after the decision;
  * `verify` re-reads the chain (from disk when persistent), recomputes every content hash and checks caller-held anchors,
    so a consistent whole-chain rewrite is caught by a head hash the caller stored earlier.

Volatility convention used across checklists A-C: `predicted_volatility` is the DAILY standard deviation of position returns
over the holding period. Excursions are signed position returns (mfe >= 0 >= mae). LABEL = IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from engine.learning.archive import ChainCorrupt, ChainFile
from engine.learning.core import (FirewallBreach, Provenance, as_date, canonical_json, current_code_hash, require_past,
                                  stable_hash)
from engine.research.two_stage import FLAT, FORBIDDEN_PREFIXES, OUTCOME_COLUMNS, Reason

LABEL = "IMPLEMENTED - NOT VALIDATED"
LANE_EXPECTATION = "exp68"
MIN_QUANTILES = 3
PROB_TOL = 1e-6


class ExpectationInvalid(ValueError):
    """The expectation is incomplete or inconsistent; nothing was written."""


class ExpectationRewrite(FirewallBreach):
    """An attempt to change, replace or delete a frozen expectation. Never caught-and-continued."""


class LateExpectation(FirewallBreach):
    """Recorded after the position's entry session began: the outcome may already have existed."""


class LedgerTampered(FirewallBreach):
    """`verify` found the lane edited, truncated, reordered or disagreeing with a stored anchor."""


def _finite(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(float(x))


def _frozen_map(m: Mapping | None) -> Mapping:
    return MappingProxyType(dict(sorted((str(k), v) for k, v in (m or {}).items())))


def _forbidden_name(name: str) -> bool:
    return str(name) in OUTCOME_COLUMNS or str(name).startswith(FORBIDDEN_PREFIXES)


# ------------------------------------------------------------------------------------------------ small value types
@dataclasses.dataclass(frozen=True)
class ReturnDistribution:
    """Predicted return distribution as ((q, value), ...) quantiles. Everything derived (cdf, PIT, pinball loss, band
    probability) is computed from the quantiles alone, so the stored object is the whole truth."""
    quantiles: tuple

    def __post_init__(self):
        object.__setattr__(self, "quantiles", tuple(sorted((float(q), float(v)) for q, v in self.quantiles)))

    def validate(self) -> list[str]:
        errs = []
        if len(self.quantiles) < MIN_QUANTILES:
            return [f"distribution needs >= {MIN_QUANTILES} quantiles"]
        qs = [q for q, _ in self.quantiles]
        vs = [v for _, v in self.quantiles]
        if not all(math.isfinite(q) and 0.0 < q < 1.0 for q in qs):
            errs.append("quantile levels must lie strictly in (0,1)")
        if len(set(qs)) != len(qs):
            errs.append("duplicate quantile levels")
        if not all(math.isfinite(v) for v in vs):
            errs.append("non-finite quantile value")
        elif any(b < a for a, b in zip(vs, vs[1:])):
            errs.append("quantile values must be non-decreasing in q (crossing quantiles)")
        return errs

    def value_at(self, q: float) -> float:
        qs, vs = zip(*self.quantiles)
        return float(np.interp(q, qs, vs))

    @property
    def median(self) -> float:
        return self.value_at(0.5)

    def cdf(self, x: float) -> float:
        """P(R <= x). Linear between quantile points; beyond the outer quantiles the end-segment slope is extended and the
        result clipped to [0.0005, 0.9995] so a PIT is always finite and a 'never predicted' outcome stays visible."""
        qs = np.array([q for q, _ in self.quantiles])
        vs = np.array([v for _, v in self.quantiles])
        if x <= vs[0]:
            w = vs[1] - vs[0]
            c = qs[0] - ((qs[1] - qs[0]) / w) * (vs[0] - x) if w > 0 else (0.0005 if x < vs[0] else qs[0])
        elif x >= vs[-1]:
            w = vs[-1] - vs[-2]
            c = qs[-1] + ((qs[-1] - qs[-2]) / w) * (x - vs[-1]) if w > 0 else (0.9995 if x > vs[-1] else qs[-1])
        else:
            c = float(np.interp(x, vs, qs))
        return float(min(max(c, 0.0005), 0.9995))

    def prob_within(self, lo: float, hi: float) -> float:
        return max(0.0, self.cdf(hi) - self.cdf(lo))

    def pinball(self, actual: float) -> float:
        """Mean pinball (quantile) loss of the stated quantiles against the realised value: a proper score for the whole
        distribution, so a wrong spread is penalised even when the median is right."""
        return float(np.mean([max(q * (actual - v), (q - 1.0) * (actual - v)) for q, v in self.quantiles]))


@dataclasses.dataclass(frozen=True)
class Alternative:
    """A hypothesis the system still entertains at decision time, with the return it would predict."""
    hypothesis: str
    probability: float
    predicted_return: float


@dataclasses.dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    p: float


# ------------------------------------------------------------------------------------------------ the expectation
@dataclasses.dataclass(frozen=True, kw_only=True)
class Expectation:
    """Every checklist-A field, frozen. Build it, `validate()` it, hand it to `ExpectationLedger.record`."""
    subject: str                               # the instrument (research side only; never shown to the blind trader)
    decided_at: str                            # session whose close the decision was taken at
    entry_at: str                              # the session the fill happens (next open); the outcome path starts here
    timestamp: str                             # full ISO time the expectation was formed
    model_version: str
    predicted_return: float                    # expected realisable position return at the learned exit
    distribution: ReturnDistribution
    direction: int                             # +1 long / -1 short
    confidence: float                          # P(direction correct), calibrated
    predicted_volatility: float                # daily std of position returns over the hold
    time_to_peak: float                        # sessions after entry until the best close
    exit_window: tuple                         # (earliest, latest) sessions after entry of the best exit
    holding_period: float                      # sessions the learned exit policy expects to hold
    mfe: float                                 # expected maximum favourable excursion (>= 0)
    mae: float                                 # expected maximum adverse excursion (<= 0)
    market_regime: str
    sector_regime: str
    patterns: tuple                            # relevant pattern keys (identity-free)
    pattern_strengths: Mapping                 # key -> expected strength
    interactions: Mapping                      # "a x b" -> expected strength; each side must be in `patterns`
    prob_distribution: tuple                   # Bin(lo, hi, p) partition of the return line, p sums to 1
    reasons_selected: tuple
    reasons_rejected: Mapping                  # competitor -> why it lost (may be empty only if there were no competitors)
    uncertainty: Mapping                       # {"aleatoric": .., "epistemic": ..} (>= 0)
    alternatives: tuple                        # Alternative(...) ; probabilities sum to <= 1
    feature_state: Mapping                     # feature name -> value at decision time (None = missing)
    knowledge_state: Mapping                   # {"digest": .., "n_items": ..} of what the model knew
    information_set: Mapping                   # source -> as_of date; every as_of <= decided_at
    n_competitors: int = 0
    notes: str = ""

    def __post_init__(self):
        for f in ("decided_at", "entry_at"):
            object.__setattr__(self, f, as_date(getattr(self, f)).isoformat())
        object.__setattr__(self, "exit_window", tuple(float(v) for v in self.exit_window))
        object.__setattr__(self, "patterns", tuple(str(p) for p in self.patterns))
        object.__setattr__(self, "reasons_selected", tuple(str(r) for r in self.reasons_selected))
        bins = tuple(b if isinstance(b, Bin) else Bin(*b) for b in self.prob_distribution)
        object.__setattr__(self, "prob_distribution", tuple(sorted(bins, key=lambda b: (b.lo, b.hi))))
        alts = tuple(a if isinstance(a, Alternative) else Alternative(**a) for a in self.alternatives)
        object.__setattr__(self, "alternatives", alts)
        if not isinstance(self.distribution, ReturnDistribution):
            object.__setattr__(self, "distribution", ReturnDistribution(tuple(map(tuple, self.distribution))))
        for f in ("pattern_strengths", "interactions", "reasons_rejected", "uncertainty", "feature_state", "knowledge_state"):
            object.__setattr__(self, f, _frozen_map(getattr(self, f)))
        object.__setattr__(self, "information_set", _frozen_map({k: as_date(v).isoformat() for k, v in self.information_set.items()}))

    # ---------------------------------------------------------------------------------------- identity
    @property
    def prediction_id(self) -> str:
        """Same subject + decision session + model = same prediction. Content may not differ under one id."""
        return "X" + stable_hash([self.subject, self.decided_at, self.model_version], 14)

    def content(self) -> dict:
        return dict(json.loads(canonical_json(self)))

    @property
    def content_hash(self) -> str:
        return stable_hash(self.content(), 32)

    # ---------------------------------------------------------------------------------------- derived (ex-ante only)
    def prob_in_band(self, lo: float = 0.05, hi: float = 0.10) -> float:
        """Predicted probability that the realisable return lies in [lo, hi], from the stated bins (falls back to the quantile
        distribution when the bins do not resolve the band). Uses only what was recorded before the outcome."""
        cover = 0.0
        for b in self.prob_distribution:
            a, z = max(b.lo, lo), min(b.hi, hi)
            if z > a and b.hi > b.lo:
                cover += b.p * (z - a) / (b.hi - b.lo)
        return float(cover) if self.prob_distribution else self.distribution.prob_within(lo, hi)

    # ---------------------------------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        e: list[str] = []
        for name in ("subject", "model_version", "market_regime", "sector_regime", "timestamp"):
            if not str(getattr(self, name)).strip():
                e.append(f"{name} is empty")
        d, en = as_date(self.decided_at), as_date(self.entry_at)
        if en <= d:
            e.append("entry_at must be after decided_at (a close decision fills at the NEXT session's open)")
        try:
            if as_date(self.timestamp) != d:
                e.append("timestamp is not on decided_at")
        except ValueError:
            e.append("timestamp is not ISO")
        for name in ("predicted_return", "predicted_volatility", "time_to_peak", "holding_period", "mfe", "mae", "confidence"):
            if not _finite(getattr(self, name)):
                e.append(f"{name} is not a finite number")
        if e:
            return e                                # numeric checks below need finite values
        e += self.distribution.validate()
        if self.direction not in (-1, 1):
            e.append("direction must be +1 or -1 (an abstention is not an expectation)")
        if not 0.5 <= self.confidence <= 1.0:
            e.append("confidence is P(direction correct) and must be in [0.5, 1]")
        if self.predicted_volatility <= 0:
            e.append("predicted_volatility must be > 0")
        if self.holding_period <= 0 or self.time_to_peak < 0:
            e.append("holding_period must be > 0 and time_to_peak >= 0")
        if len(self.exit_window) != 2 or not (0 <= self.exit_window[0] <= self.exit_window[1]):
            e.append("exit_window must be (earliest, latest) with 0 <= earliest <= latest")
        elif not self.exit_window[0] <= self.holding_period <= self.exit_window[1]:
            e.append("holding_period lies outside the predicted exit window")
        if not (self.mae <= self.predicted_return <= self.mfe and self.mae <= 0 <= self.mfe):
            e.append("excursions must bracket the return: mae <= 0 <= mfe and mae <= predicted_return <= mfe")
        e += self._validate_structure()
        e += self._validate_information()
        return e

    def _validate_structure(self) -> list[str]:
        e = []
        if not self.patterns:
            e.append("no relevant patterns listed (an expectation with no pattern basis must say so via a pattern key)")
        if set(self.pattern_strengths) != set(self.patterns):
            e.append("pattern_strengths must have exactly the keys of patterns")
        if not all(_finite(v) for v in self.pattern_strengths.values()):
            e.append("non-finite pattern strength")
        for k, v in self.interactions.items():
            parts = [p.strip() for p in str(k).split(" x ")]
            if len(parts) < 2 or any(p not in self.patterns for p in parts):
                e.append(f"interaction {k!r} names a pattern that is not in patterns")
            if not _finite(v):
                e.append(f"interaction {k!r} strength is not finite")
        if not self.reasons_selected:
            e.append("reasons_selected empty")
        if self.n_competitors > 0 and not self.reasons_rejected:
            e.append("competitors existed but no rejection reasons recorded")
        if any(not str(r).strip() for r in self.reasons_rejected.values()):
            e.append("blank rejection reason")
        if not self.prob_distribution:
            e.append("prob_distribution missing")
        else:
            if any(b.hi <= b.lo or not 0 <= b.p <= 1 for b in self.prob_distribution):
                e.append("prob_distribution has an empty bin or p outside [0,1]")
            if any(nxt.lo < cur.hi - 1e-12 for cur, nxt in zip(self.prob_distribution, self.prob_distribution[1:])):
                e.append("prob_distribution bins overlap")
            if abs(sum(b.p for b in self.prob_distribution) - 1.0) > PROB_TOL:
                e.append("prob_distribution does not sum to 1")
        for k in ("aleatoric", "epistemic"):
            if not (_finite(self.uncertainty.get(k)) and self.uncertainty[k] >= 0):
                e.append(f"uncertainty.{k} missing or negative")
        if not self.alternatives:
            e.append("no alternative hypotheses recorded")
        elif sum(a.probability for a in self.alternatives) > 1 + PROB_TOL or any(a.probability < 0 for a in self.alternatives):
            e.append("alternative probabilities invalid")
        if "digest" not in self.knowledge_state:
            e.append("knowledge_state.digest missing")
        if not self.feature_state:
            e.append("feature_state empty")
        return e

    def _validate_information(self) -> list[str]:
        e = []
        d = as_date(self.decided_at)
        if not self.information_set:
            e.append("information_set empty: say what was legally available")
        for src, asof in self.information_set.items():
            if as_date(asof) > d:
                e.append(f"information source {src!r} is dated {asof}, after the decision session {self.decided_at}")
        for name in self.feature_state:
            if _forbidden_name(name):
                e.append(f"feature {name!r} is an outcome/future column")
        for name in self.pattern_strengths:
            if _forbidden_name(name):
                e.append(f"pattern {name!r} is an outcome/future column")
        return e

    def require_valid(self) -> "Expectation":
        errs = self.validate()
        if errs:
            raise ExpectationInvalid("; ".join(errs))
        return self

    # ---------------------------------------------------------------------------------------- serialisation
    @classmethod
    def from_record(cls, d: Mapping) -> "Expectation":
        d = dict(d)
        d["distribution"] = ReturnDistribution(tuple(map(tuple, d["distribution"]["quantiles"])))
        d["prob_distribution"] = tuple(Bin(**b) for b in d["prob_distribution"])
        d["alternatives"] = tuple(Alternative(**a) for a in d["alternatives"])
        return cls(**d)


# ------------------------------------------------------------------------------------------------ sealed lane
class SealedLane:
    """An append-only lane of bodies on the shared hash chain. Adds to ChainFile what every immutable ledger here needs:
    a memory of the lines read, a from-scratch `verify` that compares the chain on disk with what this process saw, and
    anchor checking (a head hash stored elsewhere must still be part of the chain)."""

    def __init__(self, root=None, kind: str = LANE_EXPECTATION):
        self.kind, self.root = kind, root
        self.chain = ChainFile(root, kind)
        self.chain.sync()
        self._lines: list[dict] = self.chain.take_new()

    def __len__(self) -> int:
        return len(self._lines)

    @property
    def head(self) -> str:
        return self.chain.head

    def lines(self) -> tuple:
        return tuple(self._lines)

    def append(self, body: Mapping) -> dict:
        self.chain.append_many([body])
        self.chain.sync()
        self._lines.extend(self.chain.take_new())
        return self._lines[-1]

    def _fresh_view(self) -> tuple[list[dict], list[str]]:
        """(lane lines, all chain hashes) read again from the medium, independently of the cached copies."""
        if self.root is None:
            recs = self.chain._all()
            return [r for r in recs if r["kind"] == self.kind], [r["hash"] for r in recs]
        fresh = ChainFile(self.root, self.kind)
        fresh.sync()
        return fresh.take_new(), [r["hash"] for r in fresh._all()]

    def verify(self, anchors: Iterable[str] = (), body_check: Callable[[dict], str | None] | None = None) -> dict:
        problems: list[str] = []
        try:
            rep = self.chain.verify()
            if not rep.get("ok"):
                problems.append(f"chain broken at seq {rep.get('first_bad_seq')}")
            lane, hashes = self._fresh_view()
        except ChainCorrupt as ex:
            return {"ok": False, "n": len(self._lines), "problems": [f"chain corrupt: {ex}"], "head": self.head}
        cached = [ln["hash"] for ln in self._lines]
        fresh_h = [ln.get("hash") for ln in lane]
        if fresh_h[:len(cached)] != cached:
            problems.append("lane on the medium differs from what this process recorded (edited or reordered history)")
        if len(fresh_h) < len(cached):
            problems.append("lane shorter than recorded (truncation)")
        known = set(hashes)
        for a in anchors:
            if a not in known:
                problems.append(f"anchor {str(a)[:12]} is no longer part of the chain (history rewritten)")
        if body_check is not None:
            for ln in lane:
                msg = body_check(ln["body"])
                if msg:
                    problems.append(msg)
        return {"ok": not problems, "n": len(lane), "problems": problems, "head": self.head}


# ------------------------------------------------------------------------------------------------ the ledger
class ExpectationLedger:
    """Append-only, hash-chained store of expectations. The only write is `record`; there is no update or delete."""

    def __init__(self, root=None, code_hash: str | None = None):
        self.lane = SealedLane(root, LANE_EXPECTATION)
        self.code_hash = code_hash or current_code_hash()
        self._by_id: dict[str, tuple[Expectation, dict]] = {}
        for ln in self.lane.lines():
            self._ingest(ln)

    def _ingest(self, ln: dict) -> None:
        b = ln["body"]
        exp = Expectation.from_record(b["expectation"])
        self._by_id[b["prediction_id"]] = (exp, {"recorded_at": b["recorded_at"], "content_hash": b["content_hash"],
                                                 "hash": ln["hash"], "seq": ln["seq"]})

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, pid: str) -> bool:
        return pid in self._by_id

    @property
    def head(self) -> str:
        return self.lane.head

    def anchor(self) -> str:
        """Head hash to store OUTSIDE the ledger; `verify(anchors=[...])` later proves nothing before it was rewritten."""
        return self.lane.head

    # ---------------------------------------------------------------------------------------- write
    def record(self, exp: Expectation, now) -> str:
        """Freeze `exp`. `now` is the time of writing: it may not precede the decision and may not be after the fill session
        began (the outcome path exists from entry_at). Re-recording identical content is a no-op; different content under the
        same prediction id is a rewrite and is refused."""
        exp.require_valid()
        if as_date(now) < as_date(exp.decided_at):
            raise FirewallBreach(f"expectation for {exp.decided_at} written at {now}: it claims the future")
        if as_date(now) > as_date(exp.entry_at):
            raise LateExpectation(f"expectation {exp.prediction_id} written {now}, after entry session {exp.entry_at} began")
        pid = exp.prediction_id
        if pid in self._by_id:
            old, _ = self._by_id[pid]
            if old.content_hash != exp.content_hash:
                raise ExpectationRewrite(f"prediction {pid} already has a frozen expectation with different content")
            return pid
        body = {"prediction_id": pid, "content_hash": exp.content_hash, "recorded_at": as_date(now).isoformat(),
                "code_hash": self.code_hash, "expectation": exp.content()}
        self._ingest(self.lane.append(body))
        return pid

    def update(self, *_a, **_k):
        raise ExpectationRewrite("expectations are immutable: no update")

    def delete(self, *_a, **_k):
        raise ExpectationRewrite("expectations are immutable: no delete")

    amend = supersede = update

    # ---------------------------------------------------------------------------------------- read
    def get(self, pid: str, now=None) -> Expectation:
        """The frozen expectation; with `now`, only if it had been recorded strictly before `now`."""
        exp, meta = self._by_id[pid]
        if now is not None:
            require_past(meta["recorded_at"], now, f"expectation {pid} recorded_at")
        return exp

    def meta(self, pid: str) -> dict:
        return dict(self._by_id[pid][1])

    def ids(self, now=None) -> list[str]:
        if now is None:
            return sorted(self._by_id)
        return sorted(p for p, (_, m) in self._by_id.items() if as_date(m["recorded_at"]) < as_date(now))

    def frozen_before(self, pid: str, outcome_start) -> bool:
        """True iff the expectation was recorded no later than the day the outcome path began."""
        return as_date(self._by_id[pid][1]["recorded_at"]) <= as_date(outcome_start)

    def provenance(self, pid: str, learned_at) -> Provenance:
        m = self._by_id[pid][1]
        return Provenance(created_real=m["recorded_at"], learned_at=as_date(learned_at).isoformat(), code_hash=self.code_hash,
                          parents=(pid,))

    # ---------------------------------------------------------------------------------------- audit
    def verify(self, anchors: Iterable[str] = ()) -> dict:
        """Chain integrity + every stored content hash recomputed from the stored content + prediction ids re-derived."""
        def check(body: dict) -> str | None:
            try:
                exp = Expectation.from_record(body["expectation"])
            except (KeyError, TypeError, ValueError) as ex:
                return f"unreadable expectation body: {ex}"
            if exp.content_hash != body.get("content_hash"):
                return f"content of {body.get('prediction_id')} does not match its recorded hash"
            if exp.prediction_id != body.get("prediction_id"):
                return f"prediction id of {body.get('prediction_id')} does not match its content"
            return None
        rep = self.lane.verify(anchors, check)
        return {**rep, "n_expectations": len(self._by_id)}

    def assert_intact(self, anchors: Iterable[str] = ()) -> dict:
        rep = self.verify(anchors)
        if not rep["ok"]:
            raise LedgerTampered("; ".join(rep["problems"]))
        return rep

    def coverage(self, decided_days: Sequence[Any]) -> dict:
        """Which decision sessions have expectations: a session with positions but no expectation is a gap the audit reports."""
        days = {as_date(d).isoformat() for d in decided_days}
        have = {e.decided_at for e, _ in self._by_id.values()}
        return {"days": len(days), "with_expectations": len(days & have), "missing": sorted(days - have)}


# ------------------------------------------------------------------------------------------------ from the two-stage funnel
TRAJECTORY_FIELDS = ("predicted_return", "distribution", "time_to_peak", "exit_window", "holding_period", "mfe", "mae",
                     "predicted_volatility", "prob_distribution", "uncertainty", "alternatives")


def expectations_from_day(day, path_model: Callable[[Any, Any], Mapping], context: Mapping, now, *, model_version: str,
                          information_set: Mapping, feature_columns: Sequence[str] | None = None,
                          today: Any = None, max_rejected: int = 5) -> list[Expectation]:
    """One expectation per POSITION of a two-stage `DayDecision`. The funnel supplies direction, confidence, the reason each
    competitor was rejected and the knowledge digest; the trajectory (return distribution, excursions, exit window, ...) comes
    from `path_model(row, subject)`, which must return every name in TRAJECTORY_FIELDS - nothing is invented here, and a
    missing field raises ExpectationInvalid. `context` supplies market_regime, sector_regime, patterns, pattern_strengths,
    interactions, entry_at, timestamp and per-subject reasons if any."""
    table = day.table
    pos = table[table["side"] != FLAT]
    rejected_pool = table[(table["side"] == FLAT) & table["mover"].astype(bool)].sort_values("p_move", ascending=False)
    out: list[Expectation] = []
    for idx, row in pos.iterrows():
        subject = str(idx[-1] if isinstance(idx, tuple) else idx)
        traj = dict(path_model(row, subject))
        miss = [f for f in TRAJECTORY_FIELDS if f not in traj]
        if miss:
            raise ExpectationInvalid(f"path_model for {subject} did not supply {miss}")
        side = int(row["side"])
        conf = float(row["p_up"] if side > 0 else row["p_down"])
        rej = {str(i[-1] if isinstance(i, tuple) else i): f"{r['reason']} (p_move={float(r['p_move']):.3f})"
               for i, r in rejected_pool.head(max_rejected).iterrows() if i != idx}
        feats = {}
        if today is not None and idx in today.index:
            cols = feature_columns or [c for c in today.columns if not _forbidden_name(c)]
            feats = {c: (None if not _finite(today.loc[idx, c]) else float(today.loc[idx, c])) for c in cols}
        reasons = (f"{Reason.POSITION}: p_move={float(row['p_move']):.3f}, direction confidence={conf:.3f}",
                   f"funnel: {day.funnel.as_dict().get('positions', {}).get('note', '')}")
        fields = {**dict(context), **traj, "subject": subject, "decided_at": day.decided_at, "model_version": model_version,
                  "direction": side, "confidence": conf, "reasons_selected": reasons, "reasons_rejected": rej,
                  "n_competitors": len(rej), "feature_state": feats, "information_set": information_set,
                  "knowledge_state": {"digest": day.knowledge_digest, "gate": day.gate.get("verdict")}}
        exp = Expectation(**fields)
        out.append(exp.require_valid())
    return out
