"""Surprise engine (contract C62 section 52; checklist C07 surprise detection, J10 surprise monitoring). Bible phase serving:
the learning loop's MEASURE SURPRISE stage (section 4) and RESEARCH PRIORITY input (section 51).

Per resolved decision: expected outcome, actual outcome, surprise magnitude (standardised |z| and information in bits),
direction (over/under-performance versus expectation) and - across a situation cell - persistence. High surprises open an
investigation; REPEATED surprises in similar situations raise research priority. Cells are opaque strings supplied by the
caller (a context signature), never tickers or dates, so the engine cannot memorise identities.

Scale of "normal" error comes from the tracker's own PAST errors only (matured strictly before the decision was taken), or an
explicit scale from the caller, or a stated default - and the record says which. Nothing dated at/after `now` is ever used."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as sps

from .core import FirewallBreach, Health, as_date, canonical_json, require_past, stable_hash

NOMINAL_BIG_RATE = 2 * (1 - sps.norm.cdf(2.0))         # share of |z| >= 2 under pure expected noise (~4.6%)


@dataclasses.dataclass(frozen=True)
class SurpriseRecord:
    """One resolved expectation. `z` is signed: positive = outcome better than expected."""
    record_id: str
    cell: str
    decided_at: str
    matured_at: str
    expected: float
    actual: float
    scale: float
    scale_source: str                      # "given" | "history" | "default" | "binomial"
    z: float
    bits: float                            # -log2 P(|Z| >= |z|): how improbable an error this large was
    kind: str = "continuous"
    knowledge_ids: tuple[str, ...] = ()

    @property
    def magnitude(self) -> float:
        return abs(self.z)

    @property
    def direction(self) -> int:
        return 0 if self.z == 0 else (1 if self.z > 0 else -1)

    def validate(self) -> list[str]:
        errs = []
        if not all(math.isfinite(v) for v in (self.expected, self.actual, self.scale, self.z, self.bits)):
            errs.append("non-finite field")
        if self.scale <= 0:
            errs.append("scale <= 0")
        if as_date(self.decided_at) > as_date(self.matured_at):
            errs.append("decided_at after matured_at")
        if self.kind not in ("continuous", "binary"):
            errs.append(f"unknown kind {self.kind}")
        return errs


def surprise_bits(z: float) -> float:
    """Information in a standardised error: -log2 of the two-sided tail probability, floored so extreme z stays finite."""
    p = float(2.0 * sps.norm.sf(abs(z)))
    return -math.log2(max(p, 1e-300))


def continuous_z(expected: float, actual: float, scale: float) -> float:
    if not (scale > 0 and math.isfinite(scale)):
        raise ValueError(f"scale must be positive and finite, got {scale}")
    if not (math.isfinite(expected) and math.isfinite(actual)):
        raise ValueError("expected/actual must be finite")
    return (actual - expected) / scale


def binary_z(p_expected: float, outcome: int) -> float:
    """Standardised surprise of a yes/no outcome against a stated probability: (o - p) / sqrt(p(1-p)), p clipped so that a
    confident wrong call is very surprising but finite."""
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")
    p = min(max(float(p_expected), 1e-3), 1 - 1e-3)
    return (outcome - p) / math.sqrt(p * (1 - p))


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.10) -> list[bool]:
    """FDR control across cells: many cells are tested at once, so a raw p<0.05 would flag noise cells by the dozen."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals, kind="stable")
    thresh = -1
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / m:
            thresh = rank
    keep = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= thresh:
            keep[idx] = True
    return keep


@dataclasses.dataclass(frozen=True)
class SurpriseConfig:
    z_high: float = 2.5                    # single-record investigation trigger
    z_big: float = 2.0                     # counted as a 'big' surprise for repetition / rate
    min_history: int = 12                  # past errors needed before the tracker trusts its own scale
    default_scale: float = 1.0
    min_cell_n: int = 6                    # cell statistics are reported but never acted on below this
    fdr_q: float = 0.10
    recency_half_life_days: float = 180.0
    repeat_saturation: float = 3.0         # number of big surprises at which repetition counts ~63%
    neighbour_weight: float = 0.5
    window_days: int = 90                  # monitoring window (J10)
    rate_alarm_ratio: float = 2.0          # window big-surprise rate vs nominal that raises an alarm

    def validate(self) -> list[str]:
        errs = []
        if self.z_high < self.z_big:
            errs.append("z_high below z_big")
        if self.default_scale <= 0:
            errs.append("default_scale <= 0")
        if not 0 < self.fdr_q < 1:
            errs.append("fdr_q outside (0,1)")
        if self.min_cell_n < 3:
            errs.append("min_cell_n < 3")
        return errs


@dataclasses.dataclass(frozen=True)
class CellStats:
    cell: str
    n: int
    mean_z: float
    mean_abs_z: float
    big_n: int
    big_rate: float
    bias_p: float                           # two-sided p that mean z differs from 0 (persistent over/under-performance)
    rate_p: float                           # one-sided binomial p that big surprises exceed the nominal rate
    sign_persistence: float                 # P(next big surprise has the same sign as the previous one); 0.5 = no memory
    autocorr: float                         # lag-1 autocorrelation of z within the cell
    last_big_at: str | None
    under_n: int
    over_n: int


@dataclasses.dataclass(frozen=True)
class Investigation:
    cell: str
    reason: str
    priority: float
    record_ids: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ResearchPriority:
    cell: str
    score: float
    components: Mapping[str, float]
    reason: str


class SurpriseTracker:
    """Append-only store of surprise records with per-cell statistics, investigations, priorities and monitoring."""

    def __init__(self, cfg: SurpriseConfig | None = None):
        self.cfg = cfg or SurpriseConfig()
        errs = self.cfg.validate()
        if errs:
            raise ValueError("invalid SurpriseConfig: " + "; ".join(errs))
        self._recs: list[SurpriseRecord] = []
        self._ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._recs)

    def records(self, now=None, cell: str | None = None) -> list[SurpriseRecord]:
        """Records known at `now` (matured strictly before it), oldest first."""
        cut = as_date(now) if now is not None else None
        out = [r for r in self._recs if (cut is None or as_date(r.matured_at) < cut) and (cell is None or r.cell == cell)]
        return sorted(out, key=lambda r: (r.matured_at, r.record_id))

    # ---- scale from the past only
    def scale_at(self, decided_at, cell: str | None = None) -> tuple[float, str]:
        """Robust error scale from errors that had matured strictly before `decided_at` (what the decider could have known)."""
        cut = as_date(decided_at)
        pool = [r for r in self._recs if as_date(r.matured_at) < cut and r.kind == "continuous"]
        cellp = [r for r in pool if cell is not None and r.cell == cell]
        for src, rs in (("history", cellp if len(cellp) >= self.cfg.min_history else pool),):
            if len(rs) >= self.cfg.min_history:
                errs = np.array([(r.actual - r.expected) for r in rs])
                mad = float(np.median(np.abs(errs - np.median(errs))))
                sc = 1.4826 * mad
                if sc > 0:
                    return sc, src
        return self.cfg.default_scale, "default"

    # ---- observe
    def observe(self, cell: str, expected: float, actual: float, decided_at, matured_at, now, scale: float | None = None,
                knowledge_ids: Iterable[str] = ()) -> SurpriseRecord:
        """C07. Record a continuous expectation vs outcome. The outcome must have matured strictly before `now`."""
        require_past(matured_at, now, "surprise outcome matured_at")
        if not cell:
            raise ValueError("cell must be a non-empty situation key")
        if scale is None:
            sc, src = self.scale_at(decided_at, cell)
        else:
            sc, src = float(scale), "given"
        z = continuous_z(expected, actual, sc)
        return self._store(cell, "continuous", decided_at, matured_at, expected, actual, sc, src, z, knowledge_ids)

    def observe_binary(self, cell: str, p_expected: float, outcome: int, decided_at, matured_at, now,
                       knowledge_ids: Iterable[str] = ()) -> SurpriseRecord:
        require_past(matured_at, now, "surprise outcome matured_at")
        z = binary_z(p_expected, outcome)
        return self._store(cell, "binary", decided_at, matured_at, float(p_expected), float(outcome), 1.0, "binomial", z,
                           knowledge_ids)

    def _store(self, cell, kind, decided_at, matured_at, expected, actual, scale, src, z, kids) -> SurpriseRecord:
        rid = stable_hash([cell, kind, str(as_date(decided_at)), str(as_date(matured_at)), round(expected, 10), round(actual, 10),
                           sorted(kids)], 16)
        if rid in self._ids:
            raise ValueError(f"duplicate surprise record {rid}: the same expectation was already recorded")
        rec = SurpriseRecord(rid, cell, as_date(decided_at).isoformat(), as_date(matured_at).isoformat(), float(expected),
                             float(actual), float(scale), src, float(z), surprise_bits(z), kind, tuple(sorted(kids)))
        errs = rec.validate()
        if errs:
            raise ValueError("invalid surprise record: " + "; ".join(errs))
        self._recs.append(rec)
        self._ids.add(rid)
        return rec

    # ---- statistics
    def cell_stats(self, cell: str, now) -> CellStats | None:
        rs = self.records(now, cell)
        if not rs:
            return None
        z = np.array([r.z for r in rs])
        n = len(z)
        big = np.abs(z) >= self.cfg.z_big
        bn = int(big.sum())
        bias_p = float(2 * sps.norm.sf(abs(z.mean()) * math.sqrt(n))) if n >= 2 else 1.0
        rate_p = float(sps.binomtest(bn, n, NOMINAL_BIG_RATE, alternative="greater").pvalue)
        signs = np.sign(z[big])
        sp = float(np.mean(signs[1:] == signs[:-1])) if len(signs) >= 2 else 0.5
        ac = float(np.corrcoef(z[:-1], z[1:])[0, 1]) if n >= 4 and z.std() > 0 else 0.0
        big_recs = [r for r, b in zip(rs, big) if b]
        return CellStats(cell, n, float(z.mean()), float(np.abs(z).mean()), bn, bn / n, bias_p, rate_p, sp,
                         0.0 if not np.isfinite(ac) else ac, big_recs[-1].matured_at if big_recs else None,
                         int((z < 0).sum()), int((z > 0).sum()))

    def cells(self, now) -> list[str]:
        return sorted({r.cell for r in self.records(now)})

    def all_stats(self, now) -> list[CellStats]:
        return [s for c in self.cells(now) if (s := self.cell_stats(c, now)) is not None]

    # ---- investigations (high surprise triggers)
    def investigations(self, now, since=None) -> list[Investigation]:
        """Single high-surprise records since `since`, plus cells whose persistent bias or excess big-surprise rate survives
        FDR control. Sorted by priority, ties broken by a content hash (not by the cell name)."""
        out: list[Investigation] = []
        lo = as_date(since) if since is not None else None
        for r in self.records(now):
            if r.magnitude >= self.cfg.z_high and (lo is None or as_date(r.matured_at) > lo):
                out.append(Investigation(r.cell, f"single surprise z={r.z:+.2f} ({r.bits:.1f} bits): "
                                         f"expected {r.expected:.4g}, got {r.actual:.4g}", r.magnitude, (r.record_id,)))
        stats = [s for s in self.all_stats(now) if s.n >= self.cfg.min_cell_n]
        for label, attr in (("persistent bias", "bias_p"), ("excess surprise rate", "rate_p")):
            keep = benjamini_hochberg([getattr(s, attr) for s in stats], self.cfg.fdr_q)
            for s, k in zip(stats, keep):
                if k:
                    pr = -math.log10(max(getattr(s, attr), 1e-12))
                    side = "under" if s.mean_z < 0 else "over"
                    extra = f", mostly {side}-performing (mean z {s.mean_z:+.2f})" if attr == "bias_p" else f", {s.big_n}/{s.n} big"
                    out.append(Investigation(s.cell, f"{label} in cell over {s.n} outcomes{extra}", pr))
        return sorted(out, key=lambda i: (-i.priority, stable_hash([i.cell, i.reason], 8)))

    # ---- research priority
    def research_priority(self, now, similar: Mapping[str, Sequence[tuple[str, float]]] | None = None) -> list[ResearchPriority]:
        """Repeated surprise in a cell raises its research priority: repetition x magnitude x directional persistence x
        recency, plus a share of the repetition score of similar cells (neighbours vote: a surprise pattern that recurs across
        similar situations is a stronger hint than one isolated cell)."""
        cfg, cut = self.cfg, as_date(now)
        base: dict[str, dict[str, float]] = {}
        for s in self.all_stats(now):
            if s.n < cfg.min_cell_n or s.big_n == 0:
                continue
            rep = 1.0 - math.exp(-s.big_n / cfg.repeat_saturation)
            bigs = [abs(r.z) for r in self.records(now, s.cell) if abs(r.z) >= cfg.z_big]
            mag = min(3.0, float(np.mean(bigs)) / cfg.z_big)
            pers = 0.5 + 0.5 * max(0.0, 2.0 * s.sign_persistence - 1.0)
            age = (cut - as_date(s.last_big_at)).days if s.last_big_at else 10 ** 6
            rec = 0.5 ** (age / cfg.recency_half_life_days)
            base[s.cell] = {"repetition": rep, "magnitude": mag, "persistence": pers, "recency": rec,
                            "score": rep * mag * pers * rec}
        out = []
        for cell, comp in base.items():
            bonus = 0.0
            for nb, sim in (similar or {}).get(cell, ()):
                if nb in base and 0.0 <= sim <= 1.0:
                    bonus += cfg.neighbour_weight * sim * base[nb]["score"]
            comp = dict(comp, neighbours=bonus)
            total = comp["score"] + bonus
            out.append(ResearchPriority(cell, total, comp,
                                        f"{cell}: repetition {comp['repetition']:.2f} x magnitude {comp['magnitude']:.2f} x "
                                        f"persistence {comp['persistence']:.2f} x recency {comp['recency']:.2f}"
                                        + (f" + neighbours {bonus:.2f}" if bonus else "")))
        return sorted(out, key=lambda p: (-p.score, stable_hash(p.cell, 8)))

    # ---- monitoring (J10)
    def monitor(self, now) -> dict[str, Any]:
        """Is the system being surprised more than its own error model says it should be, and is that getting worse?"""
        cut, w = as_date(now), self.cfg.window_days
        rs = self.records(now)
        recent = [r for r in rs if (cut - as_date(r.matured_at)).days <= w]
        prior = [r for r in rs if w < (cut - as_date(r.matured_at)).days <= 2 * w]
        out: dict[str, Any] = {"n_total": len(rs), "n_window": len(recent), "n_prior": len(prior)}
        if len(recent) < self.cfg.min_cell_n:
            out.update(health=Health.INSUFFICIENT_EVIDENCE.value, reason=f"only {len(recent)} outcomes in the last {w} days")
            return out
        zr = np.array([r.z for r in recent])
        big = int((np.abs(zr) >= self.cfg.z_big).sum())
        rate = big / len(zr)
        p = float(sps.binomtest(big, len(zr), NOMINAL_BIG_RATE, alternative="greater").pvalue)
        out.update(big_rate=rate, nominal=NOMINAL_BIG_RATE, rate_p=p, mean_z=float(zr.mean()),
                   bias_p=float(2 * sps.norm.sf(abs(zr.mean()) * math.sqrt(len(zr)))),
                   under_share=float((zr < 0).mean()))
        if len(prior) >= self.cfg.min_cell_n:
            zp = np.array([r.z for r in prior])
            out["prior_big_rate"] = float((np.abs(zp) >= self.cfg.z_big).mean())
            out["rate_trend"] = rate - out["prior_big_rate"]
        alarm = rate >= self.cfg.rate_alarm_ratio * NOMINAL_BIG_RATE and p < 0.05
        biased = out["bias_p"] < 0.01
        if alarm and out.get("rate_trend", 0.0) > 0:
            out.update(health=Health.DEGRADING.value, reason="surprise rate is above nominal and rising")
        elif alarm or biased:
            out.update(health=Health.UNSTABLE.value, reason="surprise rate above nominal" if alarm else "systematic bias in errors")
        else:
            out.update(health=Health.HEALTHY.value, reason="surprise within the expected range")
        return out

    def by_knowledge(self, now) -> dict[str, dict[str, float]]:
        """Which knowledge items were behind the biggest average surprises (feeds credit/blame later)."""
        acc: dict[str, list[float]] = {}
        for r in self.records(now):
            for k in r.knowledge_ids:
                acc.setdefault(k, []).append(r.z)
        return {k: {"n": len(v), "mean_z": float(np.mean(v)), "mean_abs_z": float(np.mean(np.abs(v)))}
                for k, v in sorted(acc.items())}

    def dump(self, path) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(canonical_json(dataclasses.asdict(r)) for r in self._recs) + ("\n" if self._recs else ""),
                     encoding="utf-8")
        return len(self._recs)

    @classmethod
    def load(cls, path, cfg: SurpriseConfig | None = None) -> "SurpriseTracker":
        t = cls(cfg)
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    d["knowledge_ids"] = tuple(d.get("knowledge_ids", ()))
                    rec = SurpriseRecord(**d)
                    if rec.validate():
                        raise FirewallBreach(f"stored surprise record {rec.record_id} is invalid: {rec.validate()}")
                    t._recs.append(rec)
                    t._ids.add(rec.record_id)
        return t

    def report(self, now, top: int = 5) -> str:
        m = self.monitor(now)
        lines = [f"SURPRISE REPORT as of {as_date(now)} - IMPLEMENTED - NOT VALIDATED",
                 f"health: {m['health']} ({m['reason']}); records known: {m['n_total']}"]
        for i in self.investigations(now)[:top]:
            lines.append(f"  investigate {i.cell}: {i.reason}")
        for p in self.research_priority(now)[:top]:
            lines.append(f"  priority {p.score:.2f}  {p.reason}")
        return "\n".join(lines)
