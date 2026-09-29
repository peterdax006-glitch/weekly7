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

from engine import pattern_stats

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
    """FDR control across cells (many cells are tested at once, so raw p<0.05 would flag noise cells by the dozen).
    Delegates to engine.pattern_stats.bh_reject - the audited step-up rule the miner also uses; NaN p counts as 1."""
    if len(pvals) == 0:
        return []
    clean = [1.0 if not math.isfinite(float(v)) else float(v) for v in pvals]
    return [bool(x) for x in pattern_stats.bh_reject(clean, q)]


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
        rs = cellp if len(cellp) >= self.cfg.min_history else pool
        if len(rs) >= self.cfg.min_history:
            errs = np.array([(r.actual - r.expected) for r in rs])
            sc = 1.4826 * float(np.median(np.abs(errs - np.median(errs))))
            if sc > 0:
                return sc, "history"
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
        rid = stable_hash([cell, kind, str(as_date(decided_at)), str(as_date(matured_at)), round(float(expected), 10), round(float(actual), 10),
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


# ------------------------------------------------------------------------------------------------- similar situations

def parse_cell(cell: str) -> frozenset[str]:
    """A cell key is a '|'-joined context signature such as 'vol=high|trend=up|size=small'. Tokens are compared as sets, so
    two cells are similar by the context they share, never by any identity."""
    return frozenset(t for t in str(cell).split("|") if t)


def cell_similarity(a: str, b: str) -> float:
    """Jaccard similarity of context tokens, in [0, 1]. 1.0 only for identical signatures."""
    ta, tb = parse_cell(a), parse_cell(b)
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def auto_similar(tracker: SurpriseTracker, now, min_sim: float = 0.5, max_neighbours: int = 8) -> dict[str, list[tuple[str, float]]]:
    """Neighbour map for `research_priority` built from the cells' own signatures, so repeated surprises across SIMILAR situations
    raise priority without the caller hand-listing neighbours. Deterministic: ties on similarity break by content hash."""
    cells = tracker.cells(now)
    out: dict[str, list[tuple[str, float]]] = {}
    for c in cells:
        nb = [(o, cell_similarity(c, o)) for o in cells if o != c]
        nb = [(o, s) for o, s in nb if s >= min_sim]
        nb.sort(key=lambda x: (-x[1], stable_hash(x[0], 8)))
        if nb:
            out[c] = nb[:max_neighbours]
    return out


def rollup(tracker: SurpriseTracker, now, drop: str) -> dict[str, CellStats]:
    """Aggregate to parent cells by dropping one context dimension (e.g. drop='size'): a surprise that only shows up once the
    cells are pooled is a surprise about the dimension that was dropped being irrelevant - or about the pooled context."""
    groups: dict[str, list[SurpriseRecord]] = {}
    for r in tracker.records(now):
        toks = sorted(t for t in parse_cell(r.cell) if not t.startswith(drop + "="))
        groups.setdefault("|".join(toks), []).append(r)
    out = {}
    for parent, rs in sorted(groups.items()):
        if not parent:
            continue
        sub = SurpriseTracker(tracker.cfg)
        for r in rs:
            sub._recs.append(dataclasses.replace(r, cell=parent))
        st = sub.cell_stats(parent, now)
        if st:
            out[parent] = st
    return out


def persistence_profile(tracker: SurpriseTracker, cell: str, now) -> dict[str, Any]:
    """Run structure of surprises in one cell: how long do same-direction big surprises last, and is the longest run longer
    than independence would produce? A run of k same-sign big surprises among m big ones has probability about 2 * 0.5**k
    under no memory, so a long run is evidence that the surprise is one persistent thing rather than several accidents."""
    rs = tracker.records(now, cell)
    big = [r for r in rs if r.magnitude >= tracker.cfg.z_big]
    signs = [r.direction for r in big]
    runs: list[int] = []
    prev: Any = None
    for s in signs:
        if runs and s == prev:
            runs[-1] += 1
        else:
            runs.append(1)
        prev = s
    longest = max(runs) if runs else 0
    chance = min(1.0, len(big) * 0.5 ** longest) if longest else 1.0
    return {"cell": cell, "n": len(rs), "big": len(big), "runs": runs, "longest_run": longest,
            "longest_run_chance": chance, "persistent": bool(longest >= 4 and chance < 0.05),
            "direction": (1 if sum(signs) > 0 else -1) if signs and sum(signs) != 0 else 0}


def summary_table(tracker: SurpriseTracker, now) -> list[dict[str, Any]]:
    """One row per cell, everything a reviewer needs: counts, bias, big-surprise rate, persistence, last big surprise."""
    rows = []
    for st in tracker.all_stats(now):
        per = persistence_profile(tracker, st.cell, now)
        rows.append({"cell": st.cell, "n": st.n, "mean_z": st.mean_z, "big_rate": st.big_rate, "bias_p": st.bias_p,
                     "rate_p": st.rate_p, "longest_run": per["longest_run"], "persistent": per["persistent"],
                     "last_big_at": st.last_big_at})
    return sorted(rows, key=lambda r: (r["bias_p"], stable_hash(r["cell"], 8)))


# ------------------------------------------------------------------------------------------------- charts and skill

def ewma_chart(tracker: SurpriseTracker, cell: str, now, lam: float = 0.2, width: float = 3.0) -> dict[str, Any]:
    """EWMA control chart of |z| for one cell: a slow rise in how surprised we are is caught earlier than by single records.
    Control limit: expected |z| under noise (sqrt(2/pi)) plus `width` steady-state standard deviations of the EWMA."""
    zs = [abs(r.z) for r in tracker.records(now, cell)]
    if not 0 < lam <= 1:
        raise ValueError("lam must be in (0, 1]")
    if len(zs) < tracker.cfg.min_cell_n:
        return {"n": len(zs), "alarm_index": None, "ewma": [], "limit": None}
    mu0 = math.sqrt(2 / math.pi)
    sd0 = math.sqrt(1 - 2 / math.pi)
    limit = mu0 + width * sd0 * math.sqrt(lam / (2 - lam))
    e, series, alarm = mu0, [], None
    for i, z in enumerate(zs):
        e = lam * z + (1 - lam) * e
        series.append(e)
        if alarm is None and e > limit:
            alarm = i
    return {"n": len(zs), "alarm_index": alarm, "ewma": series, "limit": limit}


def expectation_skill(tracker: SurpriseTracker, now, kind: str = "continuous") -> dict[str, float]:
    """Are the EXPECTATIONS themselves informative? Regress actual on expected across all resolved records. A slope near 1 and
    intercept near 0 mean expected outcomes track reality; a slope near 0 means the expectations carry no information, and
    every 'surprise' is then just the outcome itself."""
    rs = [r for r in tracker.records(now) if r.kind == kind]
    if len(rs) < 10:
        return {"n": len(rs), "slope": float("nan"), "intercept": float("nan"), "corr": float("nan"), "p": float("nan")}
    e = np.array([r.expected for r in rs])
    a = np.array([r.actual for r in rs])
    if float(np.ptp(e)) < 1e-12:
        return {"n": len(rs), "slope": float("nan"), "intercept": float("nan"), "corr": float("nan"), "p": float("nan")}
    fit = sps.linregress(e, a)
    return {"n": len(rs), "slope": float(fit.slope), "intercept": float(fit.intercept), "corr": float(fit.rvalue),
            "p": float(fit.pvalue)}


def direction_table(tracker: SurpriseTracker, now) -> list[dict[str, Any]]:
    """Per cell: does the system tend to be pleasantly or unpleasantly surprised? Asymmetry matters: repeated UNDER-performance
    in a context is a lesson about the item, repeated OVER-performance is a missed opportunity (section 22)."""
    rows = []
    for st in tracker.all_stats(now):
        rows.append({"cell": st.cell, "n": st.n, "mean_z": st.mean_z, "under": st.under_n, "over": st.over_n,
                     "tilt": (st.over_n - st.under_n) / st.n, "bias_p": st.bias_p})
    return sorted(rows, key=lambda r: (r["bias_p"], stable_hash(r["cell"], 8)))


# ------------------------------------------------------------------------------------------------- investigation log

INVESTIGATION_STATES = ("OPEN", "EXPLAINED", "NO_CAUSE_FOUND", "SUPERSEDED")


@dataclasses.dataclass(frozen=True)
class InvestigationEntry:
    seq: int
    cell: str
    state: str
    at: str
    note: str


class InvestigationLog:
    """Append-only record of what was done about each flagged cell, so the same surprise is not re-raised every day and a
    cell that keeps surprising after being 'explained' is visible as such. Nothing is edited; a new entry supersedes."""

    def __init__(self):
        self._e: list[InvestigationEntry] = []

    def record(self, cell: str, state: str, at, note: str = "") -> InvestigationEntry:
        if state not in INVESTIGATION_STATES:
            raise ValueError(f"unknown investigation state {state}")
        if self._e and as_date(at) < as_date(self._e[-1].at):
            raise FirewallBreach("investigation entries must be recorded in date order")
        e = InvestigationEntry(len(self._e), cell, state, as_date(at).isoformat(), note)
        self._e.append(e)
        return e

    def current(self, cell: str, as_of) -> InvestigationEntry | None:
        rel = [e for e in self._e if e.cell == cell and as_date(e.at) < as_date(as_of)]
        return rel[-1] if rel else None

    def last_reviewed(self, cell: str, as_of) -> str | None:
        e = self.current(cell, as_of)
        return e.at if e else None

    def pending(self, tracker: SurpriseTracker, now) -> list[Investigation]:
        """Investigations worth acting on: never reviewed, or flagged again by records that matured after the last review."""
        out = []
        for inv in tracker.investigations(now):
            last = self.last_reviewed(inv.cell, now)
            if last is None:
                out.append(inv)
                continue
            newer = [r for r in tracker.records(now, inv.cell) if as_date(r.matured_at) > as_date(last)]
            if len(newer) >= tracker.cfg.min_cell_n and sum(1 for r in newer if r.magnitude >= tracker.cfg.z_big) >= 2:
                out.append(dataclasses.replace(inv, reason=inv.reason + f" (re-flagged after review on {last})"))
        return out

    def repeat_offenders(self, min_entries: int = 3) -> list[str]:
        """Cells investigated and closed at least `min_entries` times without the surprises stopping."""
        counts: dict[str, int] = {}
        for e in self._e:
            if e.state in ("EXPLAINED", "NO_CAUSE_FOUND"):
                counts[e.cell] = counts.get(e.cell, 0) + 1
        return sorted(c for c, n in counts.items() if n >= min_entries)

    def __len__(self) -> int:
        return len(self._e)


def discriminating_tokens(tracker: SurpriseTracker, now, min_cells: int = 2) -> list[dict[str, Any]]:
    """Which context tokens (e.g. 'vol=high') travel with the surprises? For each token, compare the pooled mean |z| of cells
    that contain it with cells that do not (Welch t on cell-level mean |z|). Ranking the tokens tells the research queue WHICH
    condition to split on first - it turns 'this area surprises us' into 'test a condition on this dimension'."""
    stats = [s for s in tracker.all_stats(now) if s.n >= tracker.cfg.min_cell_n]
    tokens = sorted({t for s in stats for t in parse_cell(s.cell)})
    rows = []
    for tok in tokens:
        a = np.array([s.mean_abs_z for s in stats if tok in parse_cell(s.cell)])
        b = np.array([s.mean_abs_z for s in stats if tok not in parse_cell(s.cell)])
        if len(a) < min_cells or len(b) < min_cells:
            continue
        t, p = sps.ttest_ind(a, b, equal_var=False)
        rows.append({"token": tok, "cells_with": len(a), "cells_without": len(b), "mean_abs_z_with": float(a.mean()),
                     "mean_abs_z_without": float(b.mean()), "t": float(t) if np.isfinite(t) else 0.0,
                     "p": float(p) if np.isfinite(p) else 1.0})
    return sorted(rows, key=lambda r: (r["p"], -abs(r["t"]), r["token"]))


def research_questions(tracker: SurpriseTracker, now, top: int = 5) -> list[dict[str, Any]]:
    """Turn the priority list into concrete questions for the research policy (section 51): the cell, why it ranks, and the
    condition token most associated with surprise across cells (the suggested split)."""
    suggested = discriminating_tokens(tracker, now)
    lead = suggested[0]["token"] if suggested and suggested[0]["p"] < 0.10 else None
    out = []
    for pr in tracker.research_priority(now, auto_similar(tracker, now))[:top]:
        out.append({"cell": pr.cell, "priority": pr.score, "why": pr.reason,
                    "question": f"why is {pr.cell} repeatedly surprising?" + (f" test a split on '{lead}'" if lead else ""),
                    "suggested_split": lead if lead and lead in parse_cell(pr.cell) else None})
    return out


# ------------------------------------------------------------------------------------------------- clusters, half-life, hand-off

def cluster_cells(tracker: SurpriseTracker, now, min_sim: float = 0.5) -> list[list[str]]:
    """Group similar situations (connected components of the token-similarity graph). Each cluster is one 'kind of situation'
    whose surprises are pooled, so a surprise that recurs across neighbouring cells is treated as one persistent thing."""
    cells = tracker.cells(now)
    parent = {c: c for c in cells}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    for c, nbs in auto_similar(tracker, now, min_sim, max_neighbours=len(cells)).items():
        for o, _ in nbs:
            parent[find(o)] = find(c)
    groups: dict[str, list[str]] = {}
    for c in cells:
        groups.setdefault(find(c), []).append(c)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), stable_hash(g, 8)))


def surprise_half_life(zs: Sequence[float], days: Sequence[float], max_lag: float = 365.0) -> dict[str, float | None]:
    """How long does a surprise persist? Fit rho(lag) = rho0 * exp(-lag / tau) to the average product of standardised z over
    pairs of records at each time lag (grid over tau). half_life = tau * ln 2. None when there is no positive persistence -
    surprises that do not carry over in time have no half-life, and inventing one would be false precision."""
    z, d = np.asarray(zs, float), np.asarray(days, float)
    if len(z) < 8 or float(z.std()) == 0:
        return {"tau": None, "half_life": None, "rho0": None, "n_pairs": 0}
    z = (z - z.mean()) / z.std()
    i, j = np.triu_indices(len(z), 1)
    lag, prod = np.abs(d[j] - d[i]), z[i] * z[j]
    keep = lag <= max_lag
    lag, prod = lag[keep], prod[keep]
    if len(lag) < 10:
        return {"tau": None, "half_life": None, "rho0": None, "n_pairs": int(len(lag))}
    edges = np.quantile(lag, np.linspace(0, 1, 7))
    idx = np.clip(np.searchsorted(edges, lag, side="right") - 1, 0, 5)
    xs = np.array([lag[idx == b].mean() for b in range(6) if (idx == b).any()])
    ys = np.array([prod[idx == b].mean() for b in range(6) if (idx == b).any()])
    best: tuple[float, Any, Any] = (float("inf"), None, None)
    for tau in np.exp(np.linspace(math.log(3.0), math.log(2000.0), 60)):
        basis = np.exp(-xs / tau)
        rho0 = float((basis * ys).sum() / (basis * basis).sum())
        sse = float(((ys - rho0 * basis) ** 2).sum())
        if rho0 > 0 and sse < best[0]:
            best = (sse, float(tau), rho0)
    if best[1] is None or best[2] < 0.05:
        return {"tau": None, "half_life": None, "rho0": best[2], "n_pairs": int(len(lag))}
    return {"tau": best[1], "half_life": best[1] * math.log(2), "rho0": best[2], "n_pairs": int(len(lag))}


def cluster_persistence(tracker: SurpriseTracker, cluster: Sequence[str], now) -> dict[str, Any]:
    """Pooled statistics and surprise half-life of one cluster of similar cells (records in matured order)."""
    rs = sorted((r for c in cluster for r in tracker.records(now, c)), key=lambda r: (r.matured_at, r.record_id))
    z = [r.z for r in rs]
    days = [float(as_date(r.matured_at).toordinal()) for r in rs]
    hl = surprise_half_life(z, days)
    return {"cells": list(cluster), "n": len(rs), "mean_z": float(np.mean(z)) if z else 0.0,
            "big_rate": float(np.mean([abs(v) >= tracker.cfg.z_big for v in z])) if z else 0.0, **hl}


@dataclasses.dataclass(frozen=True)
class FailureHandoff:
    """Surprise -> failure-learning hand-off (section 9): a hypothesis about the cause, never a verdict. `cause` is UNKNOWN
    unless the surprise structure discriminates."""
    cluster: tuple[str, ...]
    cause: str
    subsystem: str
    direction: int
    confidence: str
    evidence: Mapping[str, Any]
    record_ids: tuple[str, ...] = ()


def failure_handoffs(tracker: SurpriseTracker, now, min_sim: float = 0.5) -> list[FailureHandoff]:
    """Turn surprise clusters into FailureCause hypotheses for engine.learning failure modules. Rules: too little data ->
    INSUFFICIENT_EVIDENCE; a persistent adverse cluster with a long half-life (>= 90 days) -> REGIME_CHANGE; adverse but
    short-lived -> TEMPORARY_INACTIVITY; big surprises in both directions with no bias -> WRONG_CONTEXT (the context splits
    the outcomes); a persistent favourable cluster -> SELECTION_ERROR (opportunity missed); otherwise UNKNOWN."""
    from .core import FailureCause, Subsystem
    out = []
    for cl in cluster_cells(tracker, now, min_sim):
        info = cluster_persistence(tracker, cl, now)
        rec_ids = tuple(r.record_id for c in cl for r in tracker.records(now, c) if abs(r.z) >= tracker.cfg.z_big)[:20]
        z = np.array([r.z for c in cl for r in tracker.records(now, c)])
        if len(z) < tracker.cfg.min_cell_n:
            cause, conf = FailureCause.INSUFFICIENT_EVIDENCE, "none"
        else:
            bias_p = float(2 * sps.norm.sf(abs(z.mean()) * math.sqrt(len(z))))
            two_sided = (z >= tracker.cfg.z_big).any() and (z <= -tracker.cfg.z_big).any() and info["big_rate"] > 0.15
            hl = info["half_life"]
            if bias_p < 0.01 and z.mean() < 0:
                cause, conf = (FailureCause.REGIME_CHANGE, "medium") if hl and hl >= 90 else \
                    (FailureCause.TEMPORARY_INACTIVITY, "low")
            elif bias_p < 0.01 and z.mean() > 0:
                cause, conf = FailureCause.SELECTION_ERROR, "low"
            elif two_sided:
                cause, conf = FailureCause.WRONG_CONTEXT, "low"
            else:
                cause, conf = FailureCause.UNKNOWN, "none"
        out.append(FailureHandoff(tuple(cl), cause.value, Subsystem.SELECTION.value,
                                  0 if not len(z) else (1 if z.mean() > 0 else -1), conf, info, rec_ids))
    return out
