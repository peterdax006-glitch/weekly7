"""The real-vs-noise pattern benchmark (canon C70, C71, C72, C73, C74; C54-C58, C63, C64; C66 sections 30-31; C69 sections 12-14,
27, 28, 31). IMPLEMENTED - NOT VALIDATED: this module MEASURES the current discovery -> gate path; it validates nothing and changes no
research threshold.

C70  hundreds of simulated worlds, each with hard-to-notice REAL patterns and convincing NOISE; the system must surface every potential
     pattern, sort real from noise on past data only, and be scored against the exact truth.
C71  it must hold whatever the era: every world is built from calm / volatile / crisis / trending / choppy regimes (different base rates,
     market volatility and cross-sectional spreads, regime shifts inside a world), and the world never reveals its year (C64: the
     calendar start is drawn independently of the era, and no era label reaches the frame).
C72  the answer key (every planted pattern labelled REAL / NOISE-kind, its strength and its detectability) is written to a SEALED file
     BEFORE the system under test runs; its sha256 and write time go to an append-only manifest; the system never reads it; the key
     is opened only by `open_key`, only after the system's answers are saved, and the score fails closed when the key is missing,
     altered, or newer than the answers.
C73  besides right / wrong: HOW CLOSE the system was - confidence vs truth (Brier, calibration table), estimated vs planted effect,
     the rank of each real pattern among all candidates, partial credit for near misses (a promoted proxy, the right feature in the
     wrong form or sign, a late promotion vs the earliest detectable look), and how close each noise pattern came to promotion.
C74  every world plants >= 200 noise candidates (many of them multiple-testing winners by construction) and >= 24 real patterns in
     four strength bands: the gate's family-wise control is what is being tested; false positives are a count per world (target 0),
     reported next to the count the gate's own alpha would allow.

What runs as 'the current system' (`run_system`): the research loop's candidate generation and gate at DEFAULT settings, on past data
only - st_feature_screen (volatility_lab.oriented_scan over every derived feature, loop defaults screen_t / screen_top) raises
candidates, and every raised candidate is judged by evidence.assemble + evidence.gate under loop.REGATE_PLAN (numbered looks, the
persistent replication ledger, re-gated every loop.REGATE_NEW_DATES dates, retired on futility / repeated FAILED looks (F26,
SequentialPlan.retire) / quarantine / the evidence horizon), on the rolling frame the loop's feed serves (feeds.FeedConfig.frame_weeks).
F26: the gate is told the screen's full search size (FindingSpec.n_scanned = every feature scored), and BenchConfig.evidence_ablation
turns individual F26 evidence fixes off for attribution only (never the world). Deviations, all stated in the report: (1) the
screen runs once per re-gate interval with its per-call cap multiplied by the interval (one call instead of 13 weekly calls); (2) the
ladder between the screen and the gate (cheap screen -> stronger tests -> cross-year -> fresh holdout) is not run - a raised candidate
goes to the gate directly, which is the gate-only path the task names; (3) the benchmark's planted columns enter the candidate
universe through the same runtime registry of derived features the interaction module uses (vol_hypotheses.DERIVED), removed again
afterwards; (4) the firewall's planted-corpus self-test (a pure function of no input) is computed once per process instead of once per
bundle. Nothing in evidence / quality_gate / replication / volatility_lab is edited or re-implemented.

Public entry: `run_world(seed, out_dir, cfg)` (generate -> seal key -> system -> save answers -> open key -> score); `aggregate(rows)`
and `report_markdown(...)` for the tables; `development_seeds` / `heldout_seeds` / `tuning_seeds` for the fixed held-out split."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from engine.research.core import stable_hash

LABEL = "IMPLEMENTED - NOT VALIDATED"
PREFIX = "bm_"
REAL, NOISE, PART = "REAL", "NOISE", "PART"
# C75 3A genuine kinds: obvious .. extremely subtle are the strength BANDS; the kinds are the shapes a genuine pattern can take
REAL_KINDS = ("linear", "rare", "threshold", "conditional", "interactive", "xor", "delayed", "regime", "changing", "lifecycle")
LIFECYCLES = ("appear", "strengthen", "weaken", "die", "reverse", "return")
# C75 3A noise kinds (every listed family) plus the benchmark's own: proxy, leak, fluke, early_decay, context, base_null
NOISE_KINDS = ("null", "identity_null", "mt_winner", "autocorr_trap", "regime_corr", "vol_corr", "sample_size", "threshold_illusion",
               "near_pattern", "coincidence", "early_decay", "reversal", "delayed_coincidence", "interaction_decoy", "xor_trap",
               "selection_bias", "survivor_bias", "strong_nontransferable", "adversarial_near", "proxy", "fluke", "leak", "context",
               "base_null")
PAIRED_NOISE = ("interaction_decoy", "xor_trap")    # one process, two candidate columns
# zero per-date effect in every evaluation window AND no per-name persistence (the gate's alpha bounds how often these may pass).
# Excluded: per-name persistent columns (identity_null, context, vol_corr, base_null - the name's volatility level - and near_pattern
# built on a context): over 48 names they correlate with the names' base rates by chance, for ever (an identity trap, not a null).
PURE_NULL_KINDS = ("null", "mt_winner", "sample_size", "threshold_illusion", "regime_corr", "selection_bias")
ADVERSARIAL_KINDS = ("mt_winner", "threshold_illusion", "near_pattern", "adversarial_near", "autocorr_trap", "strong_nontransferable",
                     "leak", "survivor_bias")
# C75 3F difficulty tiers: strength multiplier on every genuine pattern, multiplier on the adversarial noise counts, conditional share;
# tier 0 = NULL world (no genuine pattern; the right answer is 'no reliable signal')
TIERS: dict[int, dict[str, float]] = {0: {"strength": 0.0, "adv": 1.0, "ctx": 0.5}, 1: {"strength": 1.6, "adv": 0.5, "ctx": 0.5},
                                      2: {"strength": 1.25, "adv": 0.75, "ctx": 0.5}, 3: {"strength": 1.0, "adv": 1.0, "ctx": 0.5},
                                      4: {"strength": 1.0, "adv": 2.0, "ctx": 0.5}, 5: {"strength": 0.85, "adv": 1.5, "ctx": 0.25},
                                      6: {"strength": 0.7, "adv": 2.0, "ctx": 0.25}}
ERAS: dict[str, dict[str, float]] = {
    # base: logit of a +-10% touch; vol: market / name volatility multiplier; spread: cross-sectional (per-name) dispersion of the
    # base rate; drift: weekly market drift (trending) ; alt: alternating weekly market sign (choppy)
    "calm": {"base": -2.5, "vol": 0.7, "spread": 0.8, "drift": 0.002, "alt": 0.0},
    "volatile": {"base": -1.6, "vol": 1.5, "spread": 1.2, "drift": 0.0, "alt": 0.0},
    "crisis": {"base": -0.9, "vol": 2.6, "spread": 1.5, "drift": -0.012, "alt": 0.0},
    "trending": {"base": -2.2, "vol": 0.9, "spread": 1.0, "drift": 0.010, "alt": 0.0},
    "choppy": {"base": -1.9, "vol": 1.2, "spread": 1.1, "drift": 0.0, "alt": 0.02},
}
REGIME_ERAS = ("volatile", "crisis")            # where a regime-limited real pattern lives (in every occurrence of those eras)
# F28: noise kinds grouped by the mechanism that makes them convincing (the report's noise-family table)
NOISE_FAMILIES: dict[str, tuple[str, ...]] = {
    "pure null (per-date effect zero everywhere)": PURE_NULL_KINDS,
    "per-name persistent (identity-shaped)": ("identity_null", "context", "vol_corr", "base_null", "autocorr_trap"),
    "correlated with a real pattern (proxy-shaped)": ("proxy", "adversarial_near", "near_pattern"),
    "future-derived (leak-shaped)": ("leak", "survivor_bias"),
    "transient or local effect": ("coincidence", "early_decay", "fluke", "reversal", "delayed_coincidence", "interaction_decoy", "xor_trap",
                                  "strong_nontransferable"),
    "interaction component (part of a real pattern)": ("interaction_component",),
}
UNDETECTABLE = "UNDETECTABLE_IN_PRINCIPLE"
NOT_REPRESENTABLE = "NOT_REPRESENTABLE"         # an oracle with the true form detects it; the single feature the system can test does not
DETECTABLE = "DETECTABLE"


class SealError(RuntimeError):
    """The answer key is missing, altered, or not older than the system's answers: the world cannot be scored (fail closed)."""


class HeldOutAccess(PermissionError):
    """Tuning code asked for a held-out seed."""


# ================================================================================================================ configuration
@dataclasses.dataclass(frozen=True)
class BenchConfig:
    n_names: int = 48
    n_dates: int = 208                     # weekly decision dates (four years)
    n_sectors: int = 6
    frame_weeks: int = 156                 # the loop feed's rolling research frame (feeds.FeedConfig.frame_weeks)
    first_look: int = 156                  # decision-date index of the first screen + gate (the loop's rolling frame is full here)
    look_every: int = 13                   # = loop.REGATE_NEW_DATES
    bands: tuple = (("obvious", 0.60), ("moderate", 0.35), ("subtle", 0.20), ("faint", 0.12), ("extremely_subtle", 0.06))
    real_kinds: tuple = REAL_KINDS
    # ~500 noise processes (C75): the null count absorbs whatever the tier's adversarial multiplier adds or removes
    noise_counts: tuple = (("null", 130), ("identity_null", 30), ("mt_winner", 60), ("autocorr_trap", 25), ("regime_corr", 20),
                           ("vol_corr", 20), ("sample_size", 20), ("threshold_illusion", 20), ("near_pattern", 15), ("coincidence", 15),
                           ("early_decay", 10), ("reversal", 15), ("delayed_coincidence", 15), ("interaction_decoy", 10), ("xor_trap", 10),
                           ("selection_bias", 10), ("survivor_bias", 10), ("strong_nontransferable", 10), ("adversarial_near", 15),
                           ("proxy", 15), ("fluke", 10), ("leak", 10))
    null_world_share: float = 0.10         # C75 3E: worlds with zero genuine patterns
    tiers: tuple = (1, 2, 3, 4, 5, 6)      # C75 3F: drawn uniformly per world (tier 0 = the null worlds)
    slow_sd: float = 0.4                   # stationary sd of the slow per-name base-rate process (what autocorrelation traps align with)
    mt_pool: int = 40                      # each multiple-testing winner is the best of this many null columns ...
    mt_window: float = 0.35                # ... on this earliest share of the dates (it looks great in-sample, is null afterwards)
    ticker_sd: float = 0.5                 # persistent per-name base-rate effect (what identity nulls can latch onto)
    oracle_draws: int = 24                 # outcome resamples from the TRUE probabilities for detectability
    power_floor: float = 0.5               # oracle power below this = UNDETECTABLE_IN_PRINCIPLE (more likely missed than found)
    orient_frac: float = 0.30              # evidence.EvidenceConfig.orient_frac (the oracle uses the gate's train/test split)
    single_era_share: float = 0.5
    min_segment: int = 26
    year_range: tuple = (1978, 2046)       # calendar start drawn uniformly: the year carries no information (C64)
    screen_top: int = 6                    # loop.LoopConfig defaults (read from the loop at run time; kept here for the key)
    screen_t: float = 2.0
    gate: bool = True                      # False = screen only (tests)
    screen_calls: int | None = None        # weekly screen calls one benchmark screen stands for (None = look_every)
    seed_salt: str = "F19-heldout-v1"
    # F26 attribution only (never changes the world): EvidenceConfig fix switches to turn OFF, and whether the gate is told the
    # screen's full search size (False = the pre-F26 count of candidates raised)
    evidence_ablation: tuple = ()
    honest_multiplicity: bool = True

    def validate(self) -> list[str]:
        errs = []
        from engine.research import evidence as EV
        flags = {f.name for f in dataclasses.fields(EV.EvidenceConfig) if f.name.startswith(("f26_", "f28_"))}
        if set(self.evidence_ablation) - flags:
            errs.append(f"unknown evidence ablation {sorted(set(self.evidence_ablation) - flags)}; known: {sorted(flags)}")
        if self.n_names < 16 or self.n_dates < 60:
            errs.append("n_names >= 16 and n_dates >= 60 required")
        if not 8 <= self.first_look <= self.n_dates or self.look_every < 1:
            errs.append("first_look in [8, n_dates] and look_every >= 1 required")
        if self.frame_weeks < 52:
            errs.append("frame_weeks >= 52 required")
        if not self.bands or any(b <= 0 for _, b in self.bands):
            errs.append("bands need positive strengths")
        if [b for _, b in self.bands] != sorted([b for _, b in self.bands], reverse=True):
            errs.append("bands must be listed strongest first")
        if set(self.real_kinds) - set(REAL_KINDS) or {k for k, _ in self.noise_counts} - set(NOISE_KINDS):
            errs.append("unknown pattern kind")
        if not 0 < self.power_floor < 1 or self.oracle_draws < 8 or self.mt_pool < 2 or not 0.1 <= self.mt_window <= 0.6:
            errs.append("power_floor in (0,1), oracle_draws >= 8, mt_pool >= 2, mt_window in [0.1, 0.6] required")
        if not 0 <= self.null_world_share < 1 or not self.tiers or set(self.tiers) - set(TIERS) - {0}:
            errs.append("null_world_share in [0,1) and tiers from TIERS required")
        return errs

    def counts_for(self, tier: int) -> dict[str, int]:
        """Noise process counts for a tier: adversarial families scaled, nulls absorb the difference (total stays constant)."""
        base = dict(self.noise_counts)
        total = sum(base.values())
        adv = TIERS[int(tier)]["adv"]
        out = {k: (int(round(n * adv)) if k in ADVERSARIAL_KINDS else n) for k, n in base.items()}
        if "null" in out:
            out["null"] = max(min(base["null"], 10), total - sum(v for k, v in out.items() if k != "null"))
        return out

    def looks(self) -> list[int]:
        return list(range(self.first_look, self.n_dates + 1, self.look_every))


def full_scale_ok(cfg: BenchConfig) -> list[str]:
    """C74 floor for a benchmark run (tests may use smaller worlds; the runner refuses them)."""
    n_noise = sum(n for _, n in cfg.noise_counts)
    n_real = len(cfg.real_kinds) * len(cfg.bands)
    errs = []
    if n_noise < 200:
        errs.append(f"C74: {n_noise} planted noise candidates < 200")
    if n_real < 24:
        errs.append(f"C74: {n_real} real patterns < 24")
    return errs


# ================================================================================================================ seeds (fixed held-out)
def is_heldout(seed: int, salt: str = BenchConfig.seed_salt) -> bool:
    """One third of all seeds, chosen by a salted hash fixed in code (not by any result)."""
    return int(stable_hash([salt, int(seed)], 8), 16) % 3 == 0


def heldout_seeds(upto: int, salt: str = BenchConfig.seed_salt) -> list[int]:
    return [s for s in range(int(upto)) if is_heldout(s, salt)]


def development_seeds(upto: int, salt: str = BenchConfig.seed_salt) -> list[int]:
    return [s for s in range(int(upto)) if not is_heldout(s, salt)]


def tuning_seeds(seeds: Iterable[int], salt: str = BenchConfig.seed_salt) -> list[int]:
    """The ONLY way tuning code may obtain worlds: any held-out seed is refused (the held-out set is never used for tuning)."""
    out = [int(s) for s in seeds]
    bad = [s for s in out if is_heldout(s, salt)]
    if bad:
        raise HeldOutAccess(f"held-out seeds requested by tuning code: {bad[:10]}")
    return out


def split_of(seed: int, salt: str = BenchConfig.seed_salt) -> str:
    return "heldout" if is_heldout(seed, salt) else "development"


def write_heldout_record(path: Path, upto: int, salt: str = BenchConfig.seed_salt) -> dict:
    """Record the held-out list once; a later call with a different list is refused (the set is fixed now)."""
    seeds = heldout_seeds(upto, salt)
    body = {"salt": salt, "upto": int(upto), "heldout": seeds, "sha256": hashlib.sha256(json.dumps(seeds).encode()).hexdigest()}
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("salt") != salt or old["heldout"] != [s for s in seeds if s < old["upto"]]:
            raise HeldOutAccess(f"held-out record {path} disagrees with the fixed rule: refusing to rewrite it")
        if old["upto"] >= upto:
            return old
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(body, indent=1).encode())
    return body


# ================================================================================================================ fast per-date AUC
def date_aucs(score: np.ndarray, y: np.ndarray, min_names: int = 8) -> np.ndarray:
    """Per-date rank AUC of `score` (T x N) for labels `y` (... x T x N, bool); NaN where a date has one class. Average ranks, so a
    binary score is scored exactly like the gate's rank AUC (ties count one half)."""
    score = np.asarray(score, float)
    fin = np.isfinite(score)
    k = (~fin).sum(-1, keepdims=True)                       # missing scores rank lowest; shifting by k ranks the finite ones alone
    r = rankdata(np.where(fin, score, -np.inf), axis=-1) - k
    y = np.asarray(y, bool) & fin
    n1 = y.sum(-1).astype(float)
    n0 = fin.sum(-1) - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        a = ((r * y).sum(-1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    a[(n1 < 1) | (n0 < 1) | (np.broadcast_to(fin.sum(-1), n1.shape) < min_names)] = np.nan
    return a


def _t(e: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.isfinite(e).sum(axis)
    m = np.nanmean(e, axis)
    s = np.nanstd(e, axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= 3, m / (s / np.sqrt(n)), np.nan)


def oracle_crit() -> float:
    """The statistical bar an oracle that KNOWS which feature is real faces: the gate's own first-look bar (sequential plan alpha_1,
    never below the promotion policy's OOS t floor) with no multiplicity."""
    from engine.learning import promotion as PR
    from engine.research import loop as LP
    a = LP.REGATE_PLAN.alpha_at(1)
    return float(max(PR.PromotionPolicy().min_oos_t, NormalDist().inv_cdf(1 - a)))


def windows(cfg: BenchConfig) -> list[tuple[int, int, int]]:
    """(look index, first test date, end) per look: matured decision dates are [lo, L-1) (an outcome ends 8 days after its date), the
    first `orient_frac` of them train (evidence.split_dates without the year-end move), the rest are the test dates."""
    out = []
    for L in cfg.looks():
        lo, hi = max(0, L - cfg.frame_weeks), max(0, L - 1)
        n = hi - lo
        k = max(2, int(n * cfg.orient_frac))
        out.append((L, lo + k, hi))
    return out


def power_curve(score: np.ndarray, y_draws: np.ndarray, date_mask: np.ndarray, wins: Sequence[tuple[int, int, int]],
                crit: float) -> list[tuple[float, float]]:
    """(power, mean effect) per look window of a one-sided per-date AUC t-test of `score` over that window's test dates where the
    effect exists. The per-date AUCs are computed once for all draws and dates, then sliced per window."""
    A = date_aucs(score, y_draws) - 0.5                  # draws x T
    out = []
    for _, a, b in wins:
        m = np.zeros(A.shape[-1], bool)
        m[a:b] = True
        m &= date_mask
        if m.sum() < 3:
            out.append((0.0, 0.0))
            continue
        E = A[:, m]
        t = _t(E)
        out.append((float(np.mean(np.nan_to_num(t, nan=-np.inf) > crit)), float(np.nanmean(E)) if np.isfinite(E).any() else 0.0))
    return out


def oracle_power(score: np.ndarray, y_draws: np.ndarray, date_mask: np.ndarray, test: tuple[int, int], crit: float) -> tuple[float, float]:
    """(power, mean effect) on one test window [a, b)."""
    return power_curve(score, y_draws, date_mask, [(0, test[0], test[1])], crit)[0]


# ================================================================================================================ the world
@dataclasses.dataclass
class World:
    seed: int
    frame: pd.DataFrame                    # the LEARNER world: observations only (base columns, planted columns, outcomes)
    key: dict                              # the TRUTH world: patterns, generator configuration, ids (sealed before the learner runs)


def _era_schedule(rng: np.random.Generator, cfg: BenchConfig) -> list[tuple[int, str]]:
    names = list(ERAS)
    if rng.random() < cfg.single_era_share:
        return [(0, str(rng.choice(names)))]
    k = int(rng.integers(2, 4))
    for _ in range(50):
        cuts = sorted(int(c) for c in rng.integers(cfg.min_segment, cfg.n_dates - cfg.min_segment, k - 1))
        if all(b - a >= cfg.min_segment for a, b in zip([0] + cuts, cuts + [cfg.n_dates])):
            break
    else:
        cuts = [cfg.n_dates // 2]
    eras, last = [], None
    for c in [0] + cuts:
        e = str(rng.choice([n for n in names if n != last]))
        eras.append((c, e))
        last = e
    return eras


def era_per_date(schedule: Sequence[tuple[int, str]], T: int) -> np.ndarray:
    out = np.empty(T, dtype=object)
    for i, (start, e) in enumerate(schedule):
        end = schedule[i + 1][0] if i + 1 < len(schedule) else T
        out[start:end] = e
    return out


def _z(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def _ar_names(rng: np.random.Generator, T: int, N: int, phi: float, sd: float) -> np.ndarray:
    """Per-name AR(1) over weeks with stationary sd `sd`."""
    x = np.zeros((T, N))
    x[0] = rng.normal(0, sd, N)
    e = sd * math.sqrt(1 - phi * phi)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + rng.normal(0, e, N)
    return x


def lifecycle_weights(kind: str, T: int, rng: np.random.Generator) -> np.ndarray:
    """C75 3G: the pattern's signed strength over time (1 = its full effect)."""
    t = np.arange(T) / T
    a, b = sorted(rng.uniform(0.2, 0.8, 2))
    if kind == "appear":
        return (t >= a).astype(float)
    if kind == "strengthen":
        return 0.3 + 1.4 * t
    if kind == "weaken":
        return 1.7 - 1.4 * t
    if kind == "die":
        return (t < a).astype(float)
    if kind == "reverse":
        return np.where(t < a, 1.0, -1.0)
    return np.where((t >= a) & (t < b), 0.0, 1.0)            # return: on, gone for a while, back


def scan_universe(columns: Iterable[str]) -> list[str]:
    """Every feature the loop's screen scores on a frame with these columns (st_feature_screen's own filter), planted columns included."""
    from engine.research import two_stage as TS
    from engine.research import vol_hypotheses as VH
    cols = set(columns)
    base = [f for f in VH.DERIVED if not f.startswith(("ix__", "ixnull")) and f not in TS.HISTORY_DEPENDENT
            and not VH.missing_columns((f,), cols) and not f.startswith(PREFIX)]
    return base + sorted(c for c in cols if c.startswith(PREFIX))


def world_design(seed: int, cfg: BenchConfig) -> dict:
    """The seed's world-level draws (tier, eras, calendar start): part of the TRUTH world, never shown to the learner."""
    rng = np.random.default_rng(np.random.SeedSequence([20260930, int(seed), 1]))
    tier = 0 if rng.random() < cfg.null_world_share else int(rng.choice(list(cfg.tiers)))
    return {"tier": tier, "rng": rng}


def make_world(seed: int, cfg: BenchConfig = BenchConfig()) -> World:
    """SEED -> WORLD -> TRUTH (C75 Firewall 9). Base columns come from volatility_lab.planted_frame (the research-frame schema the gate
    reads), re-scaled per era; the touch outcome is re-drawn from a logit that holds the era base rate, a persistent and a slowly
    moving per-name effect, and every planted process's contribution; planted columns get opaque, shuffled names (the learner cannot
    know counts, ids, labels or parameters from them)."""
    from engine.research import volatility_lab as VL
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid benchmark config: " + "; ".join(errs))
    wd = world_design(seed, cfg)
    tier, rng = wd["tier"], wd["rng"]
    tp = TIERS[tier]
    T, N = cfg.n_dates, cfg.n_names
    year = int(rng.integers(*cfg.year_range))
    first = str((pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(rng.integers(0, 330)))).date())
    F = VL.planted_frame("null", n_dates=T, n_tickers=N, seed=int(rng.integers(0, 2**31)), first=first, n_sectors=cfg.n_sectors)
    sched = _era_schedule(rng, cfg)
    era = era_per_date(sched, T)
    par = {k: np.array([ERAS[e][k] for e in era]) for k in ("base", "vol", "spread", "drift", "alt")}
    vf = np.repeat(par["vol"], N)
    for c in ("vol20", "atr", "range20", "max5", "absr1", "absr5", "m_vol"):
        F[c] = F[c].to_numpy(float) * vf
    alt = par["alt"] * np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    F["m_r5"] = F["m_r5"].to_numpy(float) + np.repeat(par["drift"] + alt, N)
    F["m_r20"] = F["m_r20"].to_numpy(float) + np.repeat(4 * par["drift"], N)

    shape = (T, N)
    tt = np.arange(T)
    u_name = rng.normal(0, cfg.ticker_sd, N)
    slow = _ar_names(rng, T, N, 0.98, cfg.slow_sd)          # slowly moving per-name base rate: what autocorrelation traps align with
    logit = par["base"][:, None] + par["spread"][:, None] * u_name[None, :] + slow
    cols: dict[str, np.ndarray] = {}
    pats: list[dict] = []
    forms: dict[str, tuple[np.ndarray, np.ndarray]] = {}    # pid -> (oriented oracle score, date mask where the effect exists)

    def new_col(v: np.ndarray) -> str:
        name = f"_c{len(cols):04d}"
        cols[name] = np.asarray(v, float).astype(np.float32)
        return name

    def add(label, kind, band, beta, sign, colnames, parent=None, **extra) -> dict:
        p = {"pid": f"P{len(pats):03d}", "label": label, "kind": kind, "band": band, "beta": float(beta), "sign": float(sign),
             "columns": list(colnames), "parent": parent, **extra}
        pats.append(p)
        return p

    regime_dates = np.isin(era, REGIME_ERAS)
    ctx_share = tp["ctx"]
    # ---------------------------------------------------------------- GENUINE patterns: every kind in every strength band (none in a NULL world)
    for kind in (cfg.real_kinds if tier else ()):
        for band, b0 in cfg.bands:
            beta = b0 * tp["strength"]
            s = float(rng.choice([-1.0, 1.0]))
            x = rng.normal(0, 1, shape)
            mask = np.ones(T, bool)
            extra: dict[str, Any] = {}
            comps: list[str] = []
            if kind == "linear":
                contrib, form = beta * s * x, s * x
            elif kind == "rare":
                ind = (x > 1.5).astype(float)                     # ~7% of name-weeks
                contrib, form = 3.0 * beta * s * ind, s * ind
            elif kind == "threshold":
                cut = float(rng.uniform(0.3, 0.9))
                ind = (x > cut).astype(float)
                contrib, form = 1.6 * beta * s * ind, s * ind
                extra["threshold"] = cut
            elif kind == "conditional":
                ctx = np.repeat((rng.random(N) < ctx_share)[None, :], T, 0).astype(float)
                ctx = np.where(rng.random(shape) < 0.05, 1 - ctx, ctx)
                contrib, form = (1.0 / ctx_share) * beta * s * x * ctx, s * x * ctx
                extra["context_share"] = ctx_share
            elif kind in ("interactive", "xor"):
                b2 = rng.normal(0, 1, shape)
                prod = x * b2 if kind == "interactive" else np.sign(x) * np.sign(b2)
                contrib, form = 1.5 * beta * s * prod, s * prod
            elif kind == "delayed":
                lag = int(rng.integers(1, 4))
                x = _ar_names(rng, T, N, 0.5, 1.0)                # the learner sees x_t; the outcome answers to x_(t-lag)
                xl = np.vstack([np.zeros((lag, N)), x[:-lag]])
                contrib, form = beta * s * xl, s * xl
                mask[:lag] = False
                extra["lag"] = lag
            elif kind == "regime":
                contrib, form = 2.2 * beta * s * x * regime_dates[:, None], s * x
                mask = regime_dates.copy()
                extra["regime"] = list(REGIME_ERAS)
            elif kind == "changing":
                phase = float(rng.uniform(0, 2 * math.pi))
                amp = 1.0 + 0.7 * np.sin(2 * math.pi * tt / (1.4 * T) + phase)
                contrib, form = beta * s * amp[:, None] * x, s * x
            else:                                                 # lifecycle: appear / strengthen / weaken / die / reverse / return
                lc = str(LIFECYCLES[len([p for p in pats if p["kind"] == "lifecycle"]) % len(LIFECYCLES)])
                w = lifecycle_weights(lc, T, rng)
                contrib, form = beta * s * w[:, None] * x, s * np.sign(w)[:, None] * x
                mask = w != 0
                extra["lifecycle"] = lc
                extra["final_sign"] = float(s * np.sign(w[-1])) if w[-1] != 0 else 0.0
            logit = logit + contrib
            xc = new_col(x)
            if kind in ("interactive", "xor"):
                comps = [xc, new_col(b2)]
                p = add(REAL, kind, band, beta, s, [], columns_marginal=comps, **extra)
                for c in comps:
                    add(PART, "interaction_component", band, 0.0, s, [c], parent=p["pid"])
            else:
                p = add(REAL, kind, band, beta, s, [xc], columns_marginal=[xc], **extra)
            forms[p["pid"]] = (form, mask)
            p["_x"] = xc
            if kind == "conditional":
                add(NOISE, "context", None, 0.0, 1.0, [new_col(ctx)], parent=p["pid"])
    reals = [p for p in pats if p["label"] == REAL]
    counts = cfg.counts_for(tier)

    def parent_col() -> tuple[str, dict | None]:
        """A real pattern's column to imitate (a null column in a NULL world: the imitation is then pure noise)."""
        cand = [p for p in reals if p["kind"] not in ("interactive", "xor")]
        if cand:
            src = cand[int(rng.integers(0, len(cand)))]
            return src["_x"], src
        return new_col(rng.normal(0, 1, shape)), None
    # ---------------------------------------------------------------- NOISE that moves the outcome for a while, or only for a few names
    k_in = max(3, int(cfg.mt_window * T))
    for kind in ("coincidence", "early_decay", "fluke", "reversal", "delayed_coincidence", "interaction_decoy", "xor_trap",
                 "strong_nontransferable"):
        for _ in range(counts.get(kind, 0)):
            s = float(rng.choice([-1.0, 1.0]))
            x = rng.normal(0, 1, shape)
            names = np.ones(N, bool)
            if kind == "coincidence":
                w = np.where(tt < 0.25 * T, 0.35, 0.0)
            elif kind == "early_decay":
                w = 0.35 * np.clip(1 - tt / (0.5 * T), 0, 1)
            elif kind == "fluke":
                a = int(rng.integers(0, T - 10))
                w = np.where((tt >= a) & (tt < a + 10), 0.6, 0.0)
            elif kind == "reversal":
                w = np.where(tt < T // 2, 0.35, -0.35)
            elif kind == "delayed_coincidence":
                a = int(rng.integers(int(0.6 * T), T - 13))
                w = np.where((tt >= a) & (tt < a + 13), 0.5, 0.0)
            elif kind in ("interaction_decoy", "xor_trap"):
                a = int(rng.integers(0, max(1, int(0.3 * T))))
                w = np.where((tt >= a) & (tt < a + 26), 0.6, 0.0)
            else:
                w = np.full(T, 0.9)
                names = np.zeros(N, bool)
                names[rng.choice(N, 3, replace=False)] = True
            act = np.flatnonzero(w)
            info: dict[str, Any] = {"active": [int(act[0]), int(act[-1]) + 1] if len(act) else None}
            if kind in ("interaction_decoy", "xor_trap"):
                b2 = rng.normal(0, 1, shape)
                prod = x * b2 if kind == "interaction_decoy" else np.sign(x) * np.sign(b2)
                logit = logit + s * w[:, None] * prod
                add(NOISE, kind, None, float(np.abs(w).max()), s, [new_col(x), new_col(b2)], **info)
                continue
            logit = logit + s * w[:, None] * x * names[None, :]
            if kind == "strong_nontransferable":
                info["names"] = int(names.sum())
            add(NOISE, kind, None, float(np.abs(w).max()), s, [new_col(x)], **info)
    p_true = 1.0 / (1.0 + np.exp(-logit))
    touch = rng.random(shape) < p_true
    # ---------------------------------------------------------------- the outcome columns (planted_frame's recipe)
    vol20 = F["vol20"].to_numpy(float).reshape(shape)
    quiet = np.minimum(0.095, vol20 * np.sqrt(5) * np.abs(rng.normal(0, 1, shape)) * 0.8)
    absmove = np.where(touch, 0.10 + rng.exponential(0.05, shape), quiet)
    sgn = np.where(rng.random(shape) < 0.5, 1.0, -1.0)
    F["touch"] = touch.reshape(-1).astype(float)
    F["absmove"] = absmove.reshape(-1)
    F["tday"] = np.where(touch, rng.integers(1, 6, shape), 0).reshape(-1).astype(float)
    F["close"] = (sgn * absmove * rng.uniform(0.6, 1.0, shape)).reshape(-1)
    F["up"] = (F["close"] > 0).astype(float)
    F["end"] = pd.to_datetime(F.index.get_level_values(0)) + pd.Timedelta(days=8)
    # ---------------------------------------------------------------- NOISE that never moves the outcome
    for _ in range(counts.get("null", 0)):
        d = int(rng.integers(0, 4))
        x = (rng.normal(0, 1, shape) if d == 0 else rng.standard_t(3, shape) if d == 1 else rng.exponential(1.0, shape) if d == 2
             else rng.integers(0, 5, shape).astype(float))
        add(NOISE, "null", None, 0.0, 1.0, [new_col(x)])
    for _ in range(counts.get("identity_null", 0)):
        add(NOISE, "identity_null", None, 0.0, 1.0, [new_col(rng.normal(0, 1, N)[None, :] + 0.35 * rng.normal(0, 1, shape))])
    for _ in range(counts.get("autocorr_trap", 0)):
        add(NOISE, "autocorr_trap", None, 0.0, 1.0, [new_col(_ar_names(rng, T, N, 0.98, 1.0))])
    for _ in range(counts.get("regime_corr", 0)):                 # tracks the era's base rate: pooled AUC sees it, per-date AUC cannot
        add(NOISE, "regime_corr", None, 0.0, 1.0, [new_col(rng.normal(0, 1, shape) + 1.5 * _z(par["base"])[:, None])])
    lv = _z(np.log(vol20))
    for _ in range(counts.get("vol_corr", 0)):
        add(NOISE, "vol_corr", None, 0.0, 1.0, [new_col(lv + 0.6 * rng.normal(0, 1, shape))])
    for _ in range(counts.get("sample_size", 0)):
        x = np.where(rng.random(shape) < 0.02, rng.normal(0, 1, shape), np.nan)
        add(NOISE, "sample_size", None, 0.0, 1.0, [new_col(x)])
    y_in = np.broadcast_to(touch[:k_in], (cfg.mt_pool, k_in, N))
    for _ in range(counts.get("mt_winner", 0)):
        pool = rng.normal(0, 1, (cfg.mt_pool, k_in, N))
        eff = np.nanmean(date_aucs(pool, y_in) - 0.5, axis=-1)
        j = int(np.argmax(np.abs(eff)))
        x = rng.normal(0, 1, shape)
        x[:k_in] = pool[j]
        add(NOISE, "mt_winner", None, 0.0, float(np.sign(eff[j]) or 1.0), [new_col(x)], in_sample_effect=float(eff[j]), window=[0, k_in])
    qs = np.linspace(0.05, 0.95, 19)
    for _ in range(counts.get("threshold_illusion", 0)):
        x = rng.normal(0, 1, shape)
        cuts = np.quantile(x[:k_in], qs)
        inds = (x[None, :k_in] > cuts[:, None, None]).astype(float)
        eff = np.nanmean(date_aucs(inds, np.broadcast_to(touch[:k_in], inds.shape)) - 0.5, axis=-1)
        j = int(np.argmax(np.abs(eff)))
        add(NOISE, "threshold_illusion", None, 0.0, float(np.sign(eff[j]) or 1.0), [new_col((x > cuts[j]).astype(float))],
            threshold=float(cuts[j]), in_sample_effect=float(eff[j]))
    near_src = [p for p in pats if p["kind"] in ("interaction_component", "context")]
    for _ in range(counts.get("near_pattern", 0)):               # near a real pattern (its component or context) but null itself
        src = near_src[int(rng.integers(0, len(near_src)))]["columns"][0] if near_src else new_col(rng.normal(0, 1, shape))
        v = cols[src].astype(float)
        add(NOISE, "near_pattern", None, 0.0, 1.0, [new_col(0.5 * _z(v) + math.sqrt(0.75) * rng.normal(0, 1, shape))])
    for _ in range(counts.get("adversarial_near", 0)):           # identical to a real pattern's column in-sample, unrelated afterwards
        c, src = parent_col()
        x = rng.normal(0, 1, shape)
        x[:k_in] = cols[c][:k_in]
        add(NOISE, "adversarial_near", src["band"] if src else None, 0.0, src["sign"] if src else 1.0, [new_col(x)],
            parent=src["pid"] if src else None, window=[0, k_in])
    for _ in range(counts.get("proxy", 0)):
        c, src = parent_col()
        rho = float(rng.uniform(0.6, 0.9))
        x = rho * cols[c].astype(float) + math.sqrt(1 - rho * rho) * rng.normal(0, 1, shape)
        add(NOISE, "proxy", src["band"] if src else None, 0.0, src["sign"] if src else 1.0, [new_col(x)],
            parent=src["pid"] if src else None, rho=rho)
    prev_touch = np.vstack([np.zeros((1, N), bool), touch[:-1]])
    for _ in range(counts.get("selection_bias", 0)):             # observed only after a quiet week (a past-selected sample), null values
        add(NOISE, "selection_bias", None, 0.0, 1.0, [new_col(np.where(prev_touch, np.nan, rng.normal(0, 1, shape)))])
    fut = np.zeros(shape)
    for k in range(1, 9):                                       # 'survives' = no +-10% week in the next eight weeks (future-derived)
        fut += np.vstack([touch[k:], np.zeros((k, N), bool)])
    survivor = (fut == 0).astype(float)
    for _ in range(counts.get("survivor_bias", 0)):
        add(NOISE, "survivor_bias", None, 0.8, -1.0, [new_col(rng.normal(0, 1, shape) + 0.8 * survivor)])
    gs = (0.3, 0.6, 1.0)
    for i in range(counts.get("leak", 0)):
        g = gs[i % len(gs)]
        add(NOISE, "leak", None, g, 1.0, [new_col(rng.normal(0, 1, shape) + g * _z(absmove))])   # overlaps the outcome window itself
    # ---------------------------------------------------------------- opaque names, frame, detectability, key
    order = rng.permutation(len(cols))
    rename = {old: f"{PREFIX}{int(order[i]):03d}" for i, old in enumerate(cols)}
    F = pd.concat([F, pd.DataFrame({rename[o]: v.reshape(-1) for o, v in cols.items()}, index=F.index)], axis=1)
    for p in pats:
        p["columns"] = [rename[c] for c in p["columns"]]
        if "columns_marginal" in p:
            p["columns_marginal"] = [rename[c] for c in p["columns_marginal"]]
        if "_x" in p:
            p["_x"] = rename[p["_x"]]
    for f in [f for f in scan_universe(F.columns) if not f.startswith(PREFIX)]:
        add(NOISE, "base_null", None, 0.0, 1.0, [f])
    F.attrs.clear()
    _detectability(pats, forms, F, p_true, rng, cfg)
    wins = windows(cfg)
    _, a_last, b_last = wins[-1]
    test_era = pd.Series(era[a_last:b_last]).value_counts()
    dates = pd.DatetimeIndex(F.index.get_level_values(0).unique())
    key = {"world_id": f"W{int(seed):05d}", "seed": int(seed), "split": split_of(seed, cfg.seed_salt), "tier": tier, "start": first,
           "year": year, "eras": [[int(a), e] for a, e in sched], "eval_era": str(test_era.index[0]), "shift": len(sched) > 1,
           "eval_era_share": float(test_era.iloc[0] / test_era.sum()), "base_rate": float(touch.mean()),
           "looks": [str(dates[L].date()) if L < T else str((dates[-1] + pd.Timedelta(days=7)).date()) for L in cfg.looks()],
           "config": stable_hash(dataclasses.asdict(cfg), 16), "noise_counts": counts, "oracle_crit": oracle_crit(),
           "power_floor": cfg.power_floor, "patterns": [{k: v for k, v in p.items() if not k.startswith("_")} for p in pats],
           "columns": {c: p["pid"] for p in pats for c in (p["columns"] or p.get("columns_marginal", []))}}
    return World(int(seed), F, key)


def _detectability(pats: list[dict], forms: Mapping[str, tuple], F: pd.DataFrame, p_true: np.ndarray, rng: np.random.Generator,
                   cfg: BenchConfig) -> None:
    """Per planted pattern: the power an ORACLE that knows the true form (and where the effect exists) would have on this sample at
    every look (outcomes resampled from the true probabilities, common random numbers across patterns), and the same for the single
    feature the system can test (the marginal). Status: UNDETECTABLE_IN_PRINCIPLE below the floor at the final look; NOT_REPRESENTABLE
    when only the true form clears it. The floor is the stated `power_floor`; the test bar is the gate's own first-look bar."""
    T, N = p_true.shape
    crit = oracle_crit()
    y = rng.random((cfg.oracle_draws, T, N)) < p_true[None]
    wins = windows(cfg)
    allm = np.ones(T, bool)
    for p in pats:
        cols = p.get("columns_marginal") or p["columns"]
        if not cols or cols[0] not in F:
            continue
        x = F[cols[0]].to_numpy(float).reshape(T, N)
        sign = p.get("final_sign", p["sign"]) or p["sign"]
        marg = power_curve(sign * x, y, allm, wins, crit)
        p["marginal_power"], p["planted_effect"] = marg[-1]
        p["marginal_power_by_look"] = [round(m[0], 3) for m in marg]
        if p["pid"] in forms:
            form, mask = forms[p["pid"]]
            orc = power_curve(form, y, mask, wins, crit)
            p["oracle_power"], p["oracle_effect"] = orc[-1]
            p["oracle_power_by_look"] = [round(o[0], 3) for o in orc]
            p["earliest_look"] = next((i for i, o in enumerate(orc) if o[0] >= cfg.power_floor), None)
            if p["oracle_power"] < cfg.power_floor:
                p["status"] = UNDETECTABLE
            elif p["marginal_power"] < cfg.power_floor:
                p["status"] = NOT_REPRESENTABLE
            else:
                p["status"] = DETECTABLE


# ================================================================================================================ C72: the sealed key
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class Manifest:
    """Append-only JSON-lines run manifest: one 'key' row per world (sha256, write time) BEFORE the system starts, one 'answers' row
    after it finishes. Each row carries the hash of the previous row, so an edited row breaks the chain."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, row: dict) -> dict:
        rows = self.rows()
        prev = rows[-1]["row_sha"] if rows else ""
        body = {**row, "prev": prev}
        body["row_sha"] = _sha(json.dumps(body, sort_keys=True).encode())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as fh:
            fh.write((json.dumps(body, sort_keys=True) + "\n").encode())
        return body

    def verify_chain(self) -> list[str]:
        errs, prev = [], ""
        for i, r in enumerate(self.rows()):
            body = {k: v for k, v in r.items() if k != "row_sha"}
            if r.get("prev") != prev or _sha(json.dumps(body, sort_keys=True).encode()) != r.get("row_sha"):
                errs.append(f"manifest row {i} ({r.get('world_id')}, {r.get('kind')}) breaks the hash chain")
            prev = r.get("row_sha", "")
        return errs

    def last(self, world_id: str, kind: str) -> dict | None:
        rs = [r for r in self.rows() if r.get("world_id") == world_id and r.get("kind") == kind]
        return rs[-1] if rs else None


def key_path(out: Path, world_id: str) -> Path:
    return Path(out) / "keys" / f"{world_id}.key.json"


def answers_path(out: Path, world_id: str) -> Path:
    return Path(out) / "answers" / f"{world_id}.answers.json"


def seal_key(key: dict, out: Path, manifest: Manifest) -> dict:
    """Write the answer key BEFORE the system runs; record its sha256 and write time. Refuses to overwrite an existing key."""
    p = key_path(out, key["world_id"])
    if p.exists():
        raise SealError(f"answer key {p.name} already exists: a key is written once, before its system run")
    data = json.dumps(key, sort_keys=True, default=float).encode()
    _write_new(p, data)
    row = manifest.append({"kind": "key", "world_id": key["world_id"], "sha256": _sha(data), "written_ns": time.time_ns(),
                           "mtime_ns": p.stat().st_mtime_ns})
    while time.time_ns() <= max(row["written_ns"], row["mtime_ns"]):   # the clock is coarse on Windows: anything later is strictly later
        time.sleep(0.001)
    return row


def save_answers(answers: dict, out: Path, manifest: Manifest, started_ns: int) -> dict:
    data = json.dumps(answers, sort_keys=True, default=float).encode()
    p = answers_path(out, answers["world_id"])
    _write_new(p, data)
    return manifest.append({"kind": "answers", "world_id": answers["world_id"], "sha256": _sha(data), "started_ns": int(started_ns),
                            "written_ns": time.time_ns(), "mtime_ns": p.stat().st_mtime_ns})


def open_key(out: Path, world_id: str, manifest: Manifest) -> tuple[dict, dict]:
    """THE ONLY READER of a sealed key (scoring only). Fails closed unless: the manifest chain is intact, the key row predates the
    system's start, the answers were saved after it, and both files still hash to their manifest rows."""
    errs = manifest.verify_chain()
    krow, arow = manifest.last(world_id, "key"), manifest.last(world_id, "answers")
    kp, ap = key_path(out, world_id), answers_path(out, world_id)
    if krow is None or not kp.exists():
        errs.append(f"{world_id}: answer key missing")
    if arow is None or not ap.exists():
        errs.append(f"{world_id}: system answers missing (the key is opened only after the answers are saved)")
    if errs or krow is None or arow is None:
        raise SealError("; ".join(errs))
    kb, ab = kp.read_bytes(), ap.read_bytes()
    if _sha(kb) != krow["sha256"]:
        errs.append(f"{world_id}: answer key altered (sha256 differs from the manifest)")
    if _sha(ab) != arow["sha256"]:
        errs.append(f"{world_id}: answers altered after they were saved")
    if not krow["written_ns"] < arow["started_ns"] <= arow["written_ns"]:
        errs.append(f"{world_id}: the key was not written before the system started")
    if kp.stat().st_mtime_ns > arow["written_ns"] or kp.stat().st_mtime_ns != krow["mtime_ns"]:
        errs.append(f"{world_id}: the key file was rewritten after it was sealed")
    if errs:
        raise SealError("; ".join(errs))
    return json.loads(kb), json.loads(ab)


# ================================================================================================================ the system under test
def _column_getter(c: str):
    def get(F: pd.DataFrame) -> pd.Series:
        return F[c]
    return get


@contextlib.contextmanager
def registered(features: Sequence[str]):
    """Planted columns enter the candidate universe through the runtime derived-feature registry (as interactions do), then leave."""
    from engine.research import vol_hypotheses as VH
    added = []
    try:
        for c in features:
            if c not in VH.DERIVED:
                VH.DERIVED[c] = ((c,), _column_getter(c))
                added.append(c)
        yield added
    finally:
        for c in added:
            VH.DERIVED.pop(c, None)


_CORPUS: dict = {}


@contextlib.contextmanager
def corpus_once():
    """firewalls.run_corpus() takes no input and returns the same verdict every call; evidence.leakage calls it once per bundle (~1 s).
    Computed once here and restored afterwards - the verdict each bundle sees is the one it would have computed."""
    from engine.learning import firewalls as FW
    orig = FW.run_corpus

    def cached(gate=None):
        if gate is not None:
            return orig(gate)
        if "v" not in _CORPUS:
            _CORPUS["v"] = orig()
        return _CORPUS["v"]
    FW.run_corpus = cached
    try:
        yield
    finally:
        FW.run_corpus = orig


def loop_screen_defaults() -> tuple[int, float]:
    from engine.research import loop as LP
    f = LP.LoopConfig.__dataclass_fields__
    top: Any = f["screen_top"].default
    t_min: Any = f["screen_t"].default
    return int(top), float(t_min)


# what evidence.assemble reads besides the feature (F26: + absmove, the realised magnitude its future-dependence leak screen tests)
EVIDENCE_COLUMNS = ("touch", "up", "close", "end", "sector", "m_vol", "vol20", "absmove")


def evidence_columns(feature: str, columns) -> list[str]:
    """The frame columns evidence.assemble reads for one VOLATILITY finding (its design, sector transfer, market-volatility failure
    contexts, replication and leak audit) plus the feature's own base columns. Handing it only these gives identical evidence; the
    other ~600 planted columns were being copied on every row filter inside it (measured: most of an assemble's time)."""
    from engine.research import evidence as EV
    from engine.research import vol_hypotheses as VH
    base = list(VH.required_columns((feature,)))
    need = list(dict.fromkeys(list(EVIDENCE_COLUMNS) + base + [c + EV.PUBLISHED_SUFFIX for c in base]))    # F28: dated availability
    return [c for c in need if c in set(columns)]


def _decision_summary(rep, sid: str) -> dict:
    d = next(x for x in rep.decisions if x.subject_id == sid)
    gates = [(g.gate, g.state, g.ok) for g in d.gates]
    stat = next((g for g in d.gates if g.gate == "out_of_sample"), None)
    return {"verdict": d.verdict.value, "n_gates": len(gates), "n_ok": int(sum(ok for _, _, ok in gates)),
            "blocking": [g for g, s, ok in gates if not ok], "states": {g: s for g, s, _ in gates},
            "details": {g.gate: g.detail[:160] for g in d.gates if not g.ok},
            "oos_margin": None if stat is None or stat.margin is None else float(stat.margin)}


# F28 attribution: the same bundle gated again with one F28 evidence part removed (never changes what the system does)
F28_ABLATIONS = ("no_rival", "no_name_units", "no_leak_suspect", "no_f28")


def fired(ev) -> bool:
    """Did any F28 part object to this evidence (a rival or name-units check not passing, a leak suspicion)?"""
    from engine.research import quality_gate as QG
    pol = QG.QualityPolicy(code_hash="cf")
    o = ev.oos
    if o is not None and o.statistical is not None:
        if QG.rival_check(o.rival, pol)[0] != QG.PASS or QG.name_units_check(o.name_units, o.statistical.n_tests_searched, pol)[0] != QG.PASS:
            return True
    return bool(ev.leak is not None and ev.leak.suspicions)


def f28_counterfactuals(b, now, code_hash: str, look: int, plan, ecfg, verdict: str | None = None) -> dict[str, str]:
    """Verdict of the SAME bundle with each F28 part stripped (rival check, identity units, the leak suspicion tier, all three), gated
    with a fresh quarantine store. In the final-look protocol a candidate is gated once, so these are exactly the verdicts the gate
    would have given without that fix; with several looks they are per-look attributions (ledger / retirement state is shared)."""
    from engine.research import evidence as EV
    ev = b.evidence
    o, lk = ev.oos, ev.leak
    if verdict is not None and not fired(ev):
        return {name: verdict for name in F28_ABLATIONS}   # no F28 part objected: stripping one cannot change the verdict

    def strip(rival: bool, units: bool, sus: bool):
        e = ev
        if o is not None and (rival or units):
            e = dataclasses.replace(e, oos=dataclasses.replace(o, rival=None if rival else o.rival, name_units=None if units else o.name_units))
        if lk is not None and sus:
            e = dataclasses.replace(e, leak=dataclasses.replace(lk, suspicions=()))
        return e
    out = {}
    for name, flags in (("no_rival", (1, 0, 0)), ("no_name_units", (0, 1, 0)), ("no_leak_suspect", (0, 0, 1)), ("no_f28", (1, 1, 1))):
        bb = EV.Bundle(b.subject_id, strip(*map(bool, flags)), b.parts, b.missing)
        rep = EV.gate([bb], now, code_hash, store=None, looks={b.subject_id: look}, plan=plan, cfg=ecfg)
        out[name] = rep.decisions[0].verdict.value
    return out


def run_system(frame: pd.DataFrame, world_id: str, seed: int, cfg: BenchConfig = BenchConfig(), *, code_hash: str = "f19",
               created_real: str = "2026-09-30T00:00:00+00:00", log=None) -> dict:
    """The current system on one world, past data only. Sees the frame and nothing else (never the key)."""
    from engine.research import evidence as EV
    from engine.research import loop as LP
    from engine.research import quality_gate as QG
    from engine.research import replication as RP
    from engine.research import volatility_lab as VL
    top, t_min = loop_screen_defaults()
    cap = top * (cfg.screen_calls or cfg.look_every)  # one screen call stands for look_every weekly calls of the loop
    planted = [c for c in frame.columns if c.startswith(PREFIX)]
    dates = pd.DatetimeIndex(frame.index.get_level_values(0).unique()).sort_values()
    plan = LP.REGATE_PLAN
    off: dict[str, Any] = {str(k): False for k in cfg.evidence_ablation}
    ecfg = dataclasses.replace(EV.EvidenceConfig(), **off)
    t0 = time.monotonic()
    cands: dict[str, dict] = {}
    screened: dict[str, int] = {}
    live: dict[str, dict] = {}
    store = QG.QuarantineStore()
    looks_meta = []
    with registered(planted), corpus_once():
        feats = scan_universe(frame.columns)
        for li, L in enumerate(cfg.looks()):
            now = dates[L] if L < len(dates) else dates[-1] + pd.Timedelta(days=7)
            lo = dates[max(0, L - cfg.frame_weeks)]
            W = frame[(frame.index.get_level_values(0) >= lo) & (frame.index.get_level_values(0) < now)]
            M = W[pd.to_datetime(W["end"]) < now]
            n_dates = int(M.index.get_level_values(0).nunique())
            lab = VL.LabConfig(min_train_dates=max(8, n_dates // 3), test_step_dates=max(2, n_dates // 8), n_boot=100)
            ts = time.monotonic()
            tab = VL.oriented_scan(M, feats, now, lab)
            scan_s = time.monotonic() - ts
            rivals = EV.rival_ranks(M, feats) if (cfg.gate and live_or_raised(tab, live, cap, t_min)) else None   # F28: the search as rivals
            tab = tab.reset_index(drop=True)
            raised = 0
            for rank, r in tab.iterrows():
                f = str(r["feature"])
                ok = np.isfinite(r["auc"]) and np.isfinite(r["lo"])
                se = max((r["hi"] - r["lo"]) / (2 * 1.645), 1e-6) if ok else float("nan")
                t = (r["auc"] - 0.5) / se if ok else float("nan")
                c = cands.setdefault(f, {"scan": [], "raised_look": None, "gate": [], "final": "NOT_RAISED", "promoted_look": None})
                c["scan"].append([li, float(r["auc"]) if ok else None, float(t) if ok else None, None if pd.isna(r["q"]) else float(r["q"]),
                                  None if pd.isna(r["sign"]) else float(r["sign"]), int(rank)])
                if ok and raised < cap and t >= t_min and f not in screened:
                    screened[f] = li
                    c["raised_look"] = li
                    c["final"] = "LIVE"
                    c["sign"] = 1.0 if (pd.isna(r["sign"]) or float(r["sign"]) >= 0) else -1.0
                    live[f] = {"looks": 0, "ledger": RP.ReplicationLedger(), "first_through": li}
                    raised += 1
            n_gated = 0
            tg = time.monotonic()
            for f in (list(live) if cfg.gate else []):
                st = live[f]
                st["looks"] += 1
                k = st["looks"]
                spec = EV.FindingSpec("D_" + f, f, cands[f]["sign"], "VOLATILITY", n_tests_searched=max(1, len(screened)),
                                      has_falsifier=True, seed=int(seed), n_scanned=len(feats) if cfg.honest_multiplicity else 0)
                b = EV.assemble(M[evidence_columns(f, M.columns)], spec, now, code_hash=code_hash, data_hash=f"bench{seed}",
                                created_real=created_real, cfg=ecfg, ledger=st["ledger"], look=k, plan=plan, rivals=rivals)
                rep = EV.gate([b], now, code_hash, store=store, looks={spec.subject_id: k}, plan=plan, cfg=ecfg)
                d = _decision_summary(rep, spec.subject_id)
                d.update(look=li, k=k, alpha=plan.alpha_at(k), n_tests_searched=max(1, len(screened)), n_scanned=len(feats),
                         n_search=spec.n_search, effect_test=b.parts.get("effect_test"), n_test=b.parts.get("n_test"),
                         missing=sorted(b.missing), leak_z=b.parts.get("future_dependence_z"),
                         rival=b.parts.get("rival"), rival_corr=b.parts.get("rival_corr"), increment_t=b.parts.get("increment_t"),
                         rival_increment_t=b.parts.get("rival_increment_t"), between_share=b.parts.get("between_share"),
                         name_p=b.parts.get("name_p"), leak_suspicions=b.parts.get("leak_suspicions"),
                         cf=f28_counterfactuals(b, now, code_hash, k, plan, ecfg, d["verdict"]))
                cands[f]["gate"].append(d)
                n_gated += 1
                why = None
                if d["verdict"] == "PROMOTE":
                    cands[f]["final"], cands[f]["promoted_look"] = "PROMOTED", li
                elif d["verdict"] == "QUARANTINED":
                    cands[f]["final"], why = "QUARANTINED", "quarantined"
                else:                                  # F26: measured futility OR repeated FAILED looks retire (plan.retire)
                    why = plan.retire(b, [g["verdict"] for g in cands[f]["gate"]])
                    if why is None and (L - cfg.looks()[st["first_through"]]) >= plan.horizon_dates:
                        why = "evidence horizon"
                    if why:
                        cands[f]["final"] = "RETIRED"
                        cands[f]["retired"] = why[:160]
                if d["verdict"] == "PROMOTE" or why:
                    live.pop(f)
            looks_meta.append({"look": li, "now": str(pd.Timestamp(now).date()), "n_dates": n_dates, "rows": int(len(M)), "raised": raised,
                               "gated": n_gated, "screened_total": len(screened), "scan_s": round(scan_s, 2),
                               "gate_s": round(time.monotonic() - tg, 2)})
            if log is not None:
                log(f"{world_id} look {li}: {looks_meta[-1]}")
    return {"world_id": world_id, "n_scanned": len(cands), "screen_top": top, "screen_t": t_min, "cap_per_screen": cap,
            "looks": looks_meta, "candidates": cands, "seconds": round(time.monotonic() - t0, 1)}


# ================================================================================================================ reference answer sheets
def live_or_raised(tab: pd.DataFrame, live: Mapping[str, Any], cap: int, t_min: float) -> bool:
    """Will anything be gated at this look (a live candidate, or the screen raising one)? The rival ranks cost ~2 s per look and
    are only built when a gate call will read them."""
    if live:
        return True
    ok = np.isfinite(tab["auc"]) & np.isfinite(tab["lo"])
    se = ((tab["hi"] - tab["lo"]) / (2 * 1.645)).clip(lower=1e-6)
    return bool((ok & ((tab["auc"] - 0.5) / se >= t_min)).any()) and cap > 0


def oracle_answers(key: dict) -> dict:
    """A perfect classifier: surfaces every candidate at the first look and promotes exactly the detectable real patterns' columns."""
    cands = {}
    for p in key["patterns"]:
        for c in p.get("columns_marginal") or p["columns"]:
            good = p["label"] == REAL and p.get("status") == DETECTABLE
            cands[c] = {"scan": [[0, 0.5 + (0.1 if good else 0.0), 3.0 if good else 0.0, 0.0 if good else 1.0, p["sign"], 0]],
                        "raised_look": 0, "gate": [], "final": "PROMOTED" if good else "RETIRED", "promoted_look": 0 if good else None,
                        "sign": p["sign"]}
    return {"world_id": key["world_id"], "n_scanned": len(cands), "looks": [], "candidates": cands, "seconds": 0.0}


def random_answers(key: dict, q: float, seed: int) -> dict:
    """A classifier that promotes each candidate with probability q, blind to everything."""
    rng = np.random.default_rng(seed)
    cands = {}
    for p in key["patterns"]:
        for c in p.get("columns_marginal") or p["columns"]:
            pr = bool(rng.random() < q)
            cands[c] = {"scan": [[0, 0.5, 0.0, 1.0 - q, 1.0, 0]], "raised_look": 0, "gate": [], "final": "PROMOTED" if pr else "RETIRED",
                        "promoted_look": 0 if pr else None, "sign": float(rng.choice([-1.0, 1.0]))}
    return {"world_id": key["world_id"], "n_scanned": len(cands), "looks": [], "candidates": cands, "seconds": 0.0}


# ================================================================================================================ scoring (after the key is opened)
def _stage(c: dict | None) -> str:
    """Where a candidate stopped: never raised by the screen, or the gate states that blocked its last look."""
    if c is None or c.get("raised_look") is None:
        return "screen: never raised (t below screen_t or over the cap)"
    if c["final"] == "PROMOTED":
        return "promoted"
    g = c["gate"][-1] if c.get("gate") else None
    if g is None:
        return "raised, never gated"
    head = {"QUARANTINED": "quarantined", "RETIRED": "retired"}.get(c["final"], "live")
    return f"{head}: {g['verdict']} blocked by " + ",".join(sorted(g["blocking"])) + (f" ({c.get('retired', '')[:40]})" if c.get("retired") else "")


def _best_gate_share(c: dict | None) -> float | None:
    if not c or not c.get("gate"):
        return None
    return max(g["n_ok"] / max(1, g["n_gates"]) for g in c["gate"])


def score_world(key: dict, answers: dict) -> tuple[list[dict], dict]:
    """Per planted pattern: right / wrong (C72) and how close (C73). Returns (rows, world summary)."""
    cands = answers["candidates"]
    n_looks = max((len(c["scan"]) for c in cands.values()), default=0)
    final_rank = {f: c["scan"][-1][5] for f, c in cands.items() if c["scan"]}
    rows = []
    by_pid = {p["pid"]: p for p in key["patterns"]}
    promoted_cols = {f for f, c in cands.items() if c["final"] == "PROMOTED"}
    for p in key["patterns"]:
        cols = list(p["columns"])                     # an interaction has none: no single candidate column represents it
        cs = [cands.get(c) for c in cols]
        best = max((c for c in cs if c is not None), key=lambda c: (c["final"] == "PROMOTED", c.get("raised_look") is not None,
                                                                    _best_gate_share(c) or 0.0), default=None)
        promoted = any(c is not None and c["final"] == "PROMOTED" for c in cs)
        surfaced = any(c is not None and c.get("raised_look") is not None for c in cs)
        last = best["scan"][-1] if best and best["scan"] else None
        q = last[3] if last else None
        conf = 1.0 - q if q is not None else 0.0
        eff = None
        if best and best.get("gate"):
            eff = best["gate"][-1].get("effect_test")
        elif last and last[1] is not None:
            eff = last[1] - 0.5
        true_sign = p.get("final_sign") or p["sign"]           # a reversed lifecycle pattern is right in its CURRENT direction
        sys_sign = best.get("sign") if best else None
        pl = best.get("promoted_look") if best else None
        r = {"world_id": key["world_id"], "split": key["split"], "tier": key.get("tier"), "eval_era": key["eval_era"], "shift": key["shift"],
             "pid": p["pid"],
             "label": p["label"], "kind": p["kind"], "band": p.get("band"), "status": p.get("status"), "oracle_power": p.get("oracle_power"),
             "marginal_power": p.get("marginal_power"), "planted_effect": p.get("planted_effect"), "surfaced": surfaced, "promoted": promoted,
             "promoted_look": pl, "raised_look": best.get("raised_look") if best else None, "earliest_look": p.get("earliest_look"),
             "stage": _stage(best), "confidence": conf, "sys_effect": eff, "sys_sign": sys_sign,
             "sign_ok": None if sys_sign is None or p["kind"] in ("mt_winner",) else bool(sys_sign == true_sign),
             "rank": final_rank.get(cols[0]) if cols else None, "t_scan": last[2] if last else None,
             "best_gate_share": _best_gate_share(best), "last_verdict": best["gate"][-1]["verdict"] if best and best.get("gate") else None,
             "n_looks_gated": len(best["gate"]) if best and best.get("gate") else 0, "parent": p.get("parent"),
             "blocking": ",".join(sorted(best["gate"][-1]["blocking"])) if best and best.get("gate") else None,
             "outcome": OUTCOME.get(best["final"], "REJECT") if best else "REJECT"}
        r["right"] = bool(promoted and r["sign_ok"] is not False) if p["label"] == REAL else False
        for ab in F28_ABLATIONS:                          # F28 attribution: promoted had that fix been off (final-look exact)
            pa = any(c is not None and (c["final"] == "PROMOTED" or any(g.get("cf", {}).get(ab) == "PROMOTE" for g in c.get("gate", [])))
                     for c in cs)
            r[f"promoted_{ab}"] = bool(pa)
            r[f"right_{ab}"] = bool(pa and r["sign_ok"] is not False) if p["label"] == REAL else False
        r["effect_error"] = None if eff is None or r["planted_effect"] is None else float(eff - r["planted_effect"])
        r["delay_looks"] = None if pl is None or r["earliest_look"] is None else int(pl - r["earliest_look"])
        credit, near = 0.0, ""
        if p["label"] == REAL:
            if promoted and r["sign_ok"] is not False:
                credit, near = 1.0, "found" + (f", {r['delay_looks']} look(s) after the earliest detectable" if (r["delay_looks"] or 0) > 0 else "")
            elif promoted:
                credit, near = 0.25, "right feature, wrong sign"
            else:
                prox = [q_ for q_ in key["patterns"] if q_["kind"] == "proxy" and q_.get("parent") == p["pid"]
                        and any(c in promoted_cols for c in q_["columns"])]
                comp = [q_ for q_ in key["patterns"] if q_["kind"] == "interaction_component" and q_.get("parent") == p["pid"]
                        and any(c in promoted_cols for c in q_["columns"])]
                if prox:
                    credit, near = 0.5, "a proxy of it was promoted instead"
                elif comp:
                    credit, near = 0.25, "a component promoted (right feature, wrong form)"
                elif best is not None and r["last_verdict"] == "NEEDS_MORE_EVIDENCE" and len(best["gate"][-1]["blocking"]) <= 1:
                    credit, near = 0.25, "one gate short at the last look"
                elif surfaced:
                    near = "surfaced, not promoted"
        elif p["label"] == NOISE and promoted:
            near = "false positive" + (" (a proxy: informative but adds nothing)" if p["kind"] == "proxy" else "")
        elif p["label"] == PART and promoted:
            near = "component promoted (right feature, wrong form)"
        r["credit"], r["near_miss"] = credit, near
        par = by_pid.get(p.get("parent") or "")
        r["parent_found"] = None if par is None else any(c in promoted_cols for c in par["columns"])
        rows.append(r)
    R = pd.DataFrame(rows)
    real = R[R["label"] == REAL]
    det = real[real["status"] == DETECTABLE]
    noise = R[R["label"] == NOISE]
    summ = {"world_id": key["world_id"], "split": key["split"], "tier": key.get("tier"), "eval_era": key["eval_era"], "shift": key["shift"],
            "n_looks": n_looks, "promoted_total": int(sum(c["final"] == "PROMOTED" for c in cands.values())),
            "outcomes": pd.Series([OUTCOME.get(c["final"], "REJECT") for c in cands.values()]).value_counts().to_dict() if cands else {},
            "real_total": int(len(real)), "real_detectable": int(len(det)), "real_right": int(det["right"].sum()),
            "real_right_any_status": int(real["right"].sum()), "promotions": int(R["promoted"].sum()),
            "real_undetectable": int((real["status"] == UNDETECTABLE).sum()), "real_not_representable": int((real["status"] == NOT_REPRESENTABLE).sum()),
            "real_promoted_any": int(real["promoted"].sum()), "noise_total": int(len(noise)), "noise_rejected": int((~noise["promoted"]).sum()),
            "false_positives": int(noise["promoted"].sum()), "fp_by_kind": noise[noise["promoted"]]["kind"].value_counts().to_dict(),
            "misses": det[~det["promoted"]][["pid", "kind", "band", "stage"]].to_dict("records"),
            "false_positive_list": noise[noise["promoted"]][["pid", "kind", "stage"]].to_dict("records"),
            "credit": float(real["credit"].sum()), "seconds": answers.get("seconds")}
    summ.update(expected_fp_bound(answers, key))
    return rows, summ


# C75 3B: the learner's outcome per candidate (the gate's final state mapped onto PROMOTE / QUARANTINE / REJECT / UNKNOWN)
OUTCOME = {"PROMOTED": "PROMOTE", "QUARANTINED": "QUARANTINE", "RETIRED": "REJECT", "NOT_RAISED": "REJECT", "LIVE": "UNKNOWN"}


def tier_table(R: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """C75 3E / 3F: per difficulty tier (0 = NULL world, where the right answer is 'no reliable signal')."""
    rows = []
    for (split, tier), g in R.groupby(["split", "tier"], dropna=False):
        sg = S[(S["split"] == split) & (S["tier"] == tier)]
        real, noise = g["label"] == REAL, g["label"] == NOISE
        det = real & (g["status"] == DETECTABLE)
        tp, fp = int((real & g["right"]).sum()), int((noise & g["promoted"]).sum())
        rows.append({"split": split, "tier": tier, "worlds": int(g["world_id"].nunique()),
                     "recall (detectable)": _fmt(*_ratio(g, det, g["right"])), "precision": f"{tp / (tp + fp):.1%}" if tp + fp else "n/a",
                     "FDR": f"{fp / (tp + fp):.1%}" if tp + fp else "n/a", "FP / world": f"{sg['false_positives'].mean():.2f}",
                     "noise rejected": _fmt(*_ratio(g, noise, ~g["promoted"])),
                     "worlds with nothing promoted": f"{int((sg['promoted_total'] == 0).sum())}/{len(sg)}"})
    return pd.DataFrame(rows)


def expected_fp_bound(answers: dict, key: dict) -> dict:
    """The false-positive count the gate's own alpha would allow on the pure-null candidates it gated: per look k the statistical gate
    passes a null with probability <= 1 - (1 - alpha_k)^(1/n) (Sidak over n searched). 'stated' uses the n the gate was given
    (candidates raised so far); 'honest' uses every feature the screen scored (the real size of the search)."""
    kind_of = {c: p["kind"] for p in key["patterns"] for c in (p.get("columns_marginal") or p["columns"])}
    stated = honest = 0.0
    for f, c in answers["candidates"].items():
        if kind_of.get(f) not in PURE_NULL_KINDS:
            continue
        for g in c.get("gate", []):
            a = float(g["alpha"])
            stated += 1 - (1 - a) ** (1.0 / max(1, int(g.get("n_search") or g["n_tests_searched"])))
            honest += 1 - (1 - a) ** (1.0 / max(1, int(g.get("n_scanned") or answers.get("n_scanned") or 1)))
    return {"expected_fp_stated_alpha": stated, "expected_fp_honest_alpha": honest}


# ================================================================================================================ aggregation
def cluster_ci(num: np.ndarray, den: np.ndarray, reps: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Pooled ratio sum(num)/sum(den) with a 95% world-cluster bootstrap interval (worlds are the independent units)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if den.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    est = num.sum() / den.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (reps, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = num[idx].sum(1) / den[idx].sum(1)
    b = b[np.isfinite(b)]
    return float(est), float(np.percentile(b, 2.5)) if len(b) else float("nan"), float(np.percentile(b, 97.5)) if len(b) else float("nan")


def _ratio(R: pd.DataFrame, mask: pd.Series, hit: pd.Series) -> tuple[float, float, float, int]:
    g = pd.DataFrame({"w": R["world_id"], "d": mask.astype(float), "h": (hit & mask).astype(float)}).groupby("w").sum()
    e, lo, hi = cluster_ci(g["h"].to_numpy(), g["d"].to_numpy())
    return e, lo, hi, int(g["d"].sum())


def _fmt(e, lo, hi, n) -> str:
    return "n/a" if not n or not np.isfinite(e) else f"{e:.1%} [{lo:.1%}, {hi:.1%}] (n={n})"


def aggregate(rows: Sequence[dict], summaries: Sequence[dict]) -> dict[str, pd.DataFrame]:
    """The score tables (development and held-out; per era, noise kind, strength band) with world-cluster bootstrap intervals."""
    R = pd.DataFrame(list(rows))
    S = pd.DataFrame(list(summaries))
    out: dict[str, pd.DataFrame] = {}
    if R.empty:
        return out
    groups = [("all", pd.Series(True, index=R.index))] + [(s, R["split"] == s) for s in ("development", "heldout")]
    tot = []
    for gname, gm in groups:
        Rg = R[gm]
        if Rg.empty:
            continue
        real, noise = Rg["label"] == REAL, Rg["label"] == NOISE
        det = real & (Rg["status"] == DETECTABLE)
        Sg = S[S["world_id"].isin(Rg["world_id"].unique())]
        fp = Sg["false_positives"].to_numpy(float)
        tp, prom = int((real & Rg["right"]).sum()), int(Rg["promoted"].sum())
        rec = _ratio(Rg, det, Rg["right"])
        tot.append({"set": gname, "worlds": int(Rg["world_id"].nunique()),
                    "real right (detectable promoted, right sign)": _fmt(*rec),
                    "precision (right real / all promotions)": f"{tp / prom:.1%} ({tp}/{prom})" if prom else "n/a (nothing promoted)",
                    "FDR": f"{1 - tp / prom:.1%}" if prom else "n/a", "FNR (detectable)": f"{1 - rec[0]:.1%}" if rec[3] else "n/a",
                    "real right / all real": f"{int((real & Rg['right']).sum())}/{int(real.sum())}",
                    "undetectable in principle": int((real & (Rg["status"] == UNDETECTABLE)).sum()),
                    "not representable": int((real & (Rg["status"] == NOT_REPRESENTABLE)).sum()),
                    "noise correctly rejected": _fmt(*_ratio(Rg, noise, ~Rg["promoted"])),
                    "false positives / world": f"{fp.mean():.2f} (max {int(fp.max()) if len(fp) else 0}; worlds with 0: {int((fp == 0).sum())}/{len(fp)})",
                    "expected FP / world at stated alpha": f"{Sg['expected_fp_stated_alpha'].mean():.3f}",
                    "expected FP / world at honest alpha": f"{Sg['expected_fp_honest_alpha'].mean():.4f}",
                    "candidate recall real": _fmt(*_ratio(Rg, real, Rg["surfaced"])),
                    "candidate recall noise": _fmt(*_ratio(Rg, noise, Rg["surfaced"])),
                    "partial credit (real)": f"{Rg.loc[real, 'credit'].mean():.3f}"})
    out["total"] = pd.DataFrame(tot)
    real = R[R["label"] == REAL]
    rows_e = []
    for (split, era), g in R.groupby(["split", "eval_era"]):
        rg, ng = g[g["label"] == REAL], g[g["label"] == NOISE]
        d = rg["status"] == DETECTABLE
        sg = S[(S["split"] == split) & (S["eval_era"] == era)]
        rows_e.append({"split": split, "era": era, "worlds": int(g["world_id"].nunique()),
                       "TP recall (detectable)": _fmt(*_ratio(rg, d, rg["right"])),
                       "FP / world": f"{sg['false_positives'].mean():.2f}", "noise rejected": _fmt(*_ratio(ng, pd.Series(True, index=ng.index), ~ng["promoted"])),
                       "detectable share": f"{d.mean():.1%}"})
    out["era"] = pd.DataFrame(rows_e)
    rows_b = []
    for (split, kind, band), g in real.groupby(["split", "kind", "band"]):
        d = g["status"] == DETECTABLE
        rows_b.append({"split": split, "kind": kind, "band": band, "n": len(g), "detectable": int(d.sum()),
                       "not representable": int((g["status"] == NOT_REPRESENTABLE).sum()), "undetectable": int((g["status"] == UNDETECTABLE).sum()),
                       "TP recall (detectable)": _fmt(*_ratio(g, d, g["right"])), "surfaced": f"{g['surfaced'].mean():.1%}",
                       "mean oracle power": f"{g['oracle_power'].mean():.2f}", "median delay looks": g["delay_looks"].median()})
    out["band"] = pd.DataFrame(rows_b)
    noise = R[R["label"].isin([NOISE, PART])]
    rows_n = []
    for (split, kind), g in noise.groupby(["split", "kind"]):
        rows_n.append({"split": split, "kind": kind, "n": len(g), "promoted (FP)": int(g["promoted"].sum()),
                       "FP rate": _fmt(*_ratio(g, pd.Series(True, index=g.index), g["promoted"])), "surfaced": f"{g['surfaced'].mean():.1%}",
                       "mean best gate share": f"{g['best_gate_share'].mean():.2f}" if g["best_gate_share"].notna().any() else "n/a",
                       "last verdicts": dict(g["last_verdict"].value_counts().head(3))})
    out["noise"] = pd.DataFrame(rows_n)
    out.update(f28_tables(R, S))
    out["closeness"] = closeness(R)
    out["calibration"] = calibration_table(R)
    out["failures"] = ranked_failures(R, S)
    out["gates"] = gate_blockers(R)
    return out


def family_of(kind: str) -> str:
    return next((f for f, ks in NOISE_FAMILIES.items() if kind in ks), "other")


def f28_tables(R: pd.DataFrame, S: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """F28's full table set (C75 Phase 10): recall per kind and per strength band alone (each with world-cluster 95% intervals), false
    positives per noise FAMILY, the null worlds, and the attribution of every F28 fix (from the counterfactual verdicts of the same
    bundles: what was promoted with that fix switched off)."""
    out: dict[str, pd.DataFrame] = {}
    real = R[R["label"] == REAL]
    for name, col in (("kind", "kind"), ("strength", "band")):
        rows = []
        for (split, v), g in real.groupby(["split", col]):
            d = g["status"] == DETECTABLE
            rows.append({"split": split, col: v, "n": len(g), "detectable": int(d.sum()),
                         "TP recall (detectable)": _fmt(*_ratio(g, d, g["right"])), "found any status": int(g["right"].sum()),
                         "surfaced (detectable)": _fmt(*_ratio(g, d, g["surfaced"]))})
        out[name] = pd.DataFrame(rows)
    nz = R[R["label"].isin([NOISE, PART])].assign(family=lambda d: d["kind"].map(family_of))
    rows = []
    for (split, fam), g in nz.groupby(["split", "family"]):
        nw = max(1, g["world_id"].nunique())
        rows.append({"split": split, "family": fam, "n": len(g), "promoted (FP)": int(g["promoted"].sum()),
                     "FP / world": f"{g['promoted'].sum() / nw:.2f}", "FP rate": _fmt(*_ratio(g, pd.Series(True, index=g.index), g["promoted"])),
                     "kinds promoted": dict(g[g["promoted"]]["kind"].value_counts())})
    out["family"] = pd.DataFrame(rows)
    nullw = S[S["tier"] == 0]
    out["null_worlds"] = pd.DataFrame([{"split": sp, "null worlds": len(g), "worlds promoting anything": int((g["promoted_total"] > 0).sum()),
                                        "promotions": int(g["promoted_total"].sum())} for sp, g in nullw.groupby("split")]) if len(nullw) else pd.DataFrame()
    rows = []
    if "promoted_no_f28" in R:
        for split, g in R.groupby("split"):
            det = (g["label"] == REAL) & (g["status"] == DETECTABLE)
            nse = g["label"] == NOISE
            nw = max(1, g["world_id"].nunique())
            for ab in ("with all F28 fixes",) + F28_ABLATIONS:
                pc, rc = ("promoted", "right") if ab.startswith("with") else (f"promoted_{ab}", f"right_{ab}")
                fp = g[nse & g[pc]]
                rows.append({"split": split, "configuration": ab, "real found (detectable)": f"{int((det & g[rc]).sum())}/{int(det.sum())}",
                             "real found (any)": int(((g["label"] == REAL) & g[rc]).sum()), "FP": len(fp), "FP / world": f"{len(fp) / nw:.2f}",
                             "proxy FP": int((fp["kind"] == "proxy").sum()), "identity_null FP": int((fp["kind"] == "identity_null").sum()),
                             "leak FP": int((fp["kind"] == "leak").sum()), "other FP": dict(fp[~fp["kind"].isin(["proxy", "identity_null", "leak"])]["kind"].value_counts())})
    out["attribution"] = pd.DataFrame(rows)
    return out


# ================================================================================================================ F28: a genuine scheduled-event signal
def plant_scheduled_event(frame: pd.DataFrame, seed: int, share: float = 0.10, lift: float = 1.8, lead_days: int = 3,
                          name: str = "ev_sched", documented: bool = True) -> pd.DataFrame:
    """F28 (F26 flagged it): a GENUINE new-information magnitude signal - a scheduled event (an earnings date) known before the
    decision - planted on a development world. On an event row the week's magnitude is `lift` times larger, so the event raises the
    touch probability AND the size of the move: within an outcome class it correlates with the coming magnitude and not with the
    past one, exactly the future-dependence signature of a leak. `documented` adds the per-row publication record
    (`<name>__published_at`, `lead_days` before the decision) that evidence.documented_availability reads. Rows' outcomes are
    rewritten consistently (touch = magnitude >= 0.10, close keeps its sign). Development seeds only; never part of make_world."""
    tuning_seeds([seed])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 28]))
    F = frame.copy()
    e = (rng.random(len(F)) < share).astype(float)
    mag = F["absmove"].to_numpy(float) * np.where(e > 0, lift, 1.0)
    touch = mag >= 0.10
    newly = touch & (F["touch"].to_numpy(float) < 0.5)
    F["absmove"] = mag
    F["close"] = F["close"].to_numpy(float) * np.where(e > 0, lift, 1.0)
    F["up"] = (F["close"] > 0).astype(float)
    F["touch"] = touch.astype(float)
    F["tday"] = np.where(newly, rng.integers(1, 6, len(F)), np.where(touch, F["tday"].to_numpy(float), 0)).astype(float)
    F[name] = e
    if documented:
        from engine.research import evidence as EV
        F[name + EV.PUBLISHED_SUFFIX] = pd.to_datetime(F.index.get_level_values(0)) - pd.Timedelta(days=int(lead_days))
    return F


def gate_blockers(R: pd.DataFrame) -> pd.DataFrame:
    """Which gate stopped what, at each candidate's last look: share of the GATED detectable real patterns each gate blocked (those
    are the gates costing true discoveries) next to the share of gated noise it blocked (the gates doing the work)."""
    g = R[R["n_looks_gated"] > 0]
    groups = {"real (detectable, not promoted)": g[(g["label"] == REAL) & (g["status"] == DETECTABLE) & ~g["promoted"]],
              "noise (gated)": g[g["label"] == NOISE]}
    gates = sorted({x for b in g["blocking"].dropna() for x in b.split(",") if x})
    rows = []
    for gate in gates:
        row = {"gate": gate}
        for lab, G in groups.items():
            n = len(G)
            row[lab] = f"{G['blocking'].fillna('').str.split(',').apply(lambda xs: gate in xs).mean():.1%} of {n}" if n else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def closeness(R: pd.DataFrame) -> pd.DataFrame:
    """C73 aggregates per split and label: Brier of the system's confidence, effect error, sign agreement, rank, near misses."""
    rows = []
    for (split, label), g in R[R["label"].isin([REAL, NOISE])].groupby(["split", "label"]):
        y = (g["label"] == REAL).astype(float)
        e = g["effect_error"].dropna()
        rows.append({"split": split, "label": label, "n": len(g), "brier": float(((g["confidence"] - y) ** 2).mean()),
                     "mean confidence": float(g["confidence"].mean()), "mean |effect error|": float(e.abs().mean()) if len(e) else None,
                     "mean effect error": float(e.mean()) if len(e) else None,
                     "sign agreement": float(g["sign_ok"].dropna().astype(float).mean()) if g["sign_ok"].notna().any() else None,
                     "median final rank": float(g["rank"].median()), "just below the line (t in [1.5, 2))": int(g["t_scan"].between(1.5, 2.0, inclusive="left").sum()),
                     "partial credit": float(g["credit"].sum()), "near misses": dict(g["near_miss"].replace("", np.nan).dropna().value_counts().head(4))})
    return pd.DataFrame(rows)


def calibration_table(R: pd.DataFrame, bins: Sequence[float] = (0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0001)) -> pd.DataFrame:
    g = R[R["label"].isin([REAL, NOISE])].copy()
    g["bin"] = pd.cut(g["confidence"], list(bins), right=False)
    t = g.groupby("bin", observed=False).agg(n=("label", "size"), share_real=("label", lambda s: float((s == REAL).mean()) if len(s) else np.nan),
                                             mean_conf=("confidence", "mean"), promoted=("promoted", "mean"))
    return t.reset_index().assign(bin=lambda d: d["bin"].astype(str))


def ranked_failures(R: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """The concrete failures that drive the next briefs, ranked by how many patterns they cost (per 100 worlds)."""
    nw = max(1, R["world_id"].nunique())
    out = []
    fp = R[(R["label"] == NOISE) & R["promoted"]]
    for (kind, stage), g in fp.groupby(["kind", "stage"]):
        out.append({"failure": f"noise promoted: {kind}", "where": stage, "eras": dict(g["eval_era"].value_counts()),
                    "count": len(g), "per_100_worlds": 100 * len(g) / nw})
    miss = R[(R["label"] == REAL) & (R["status"] == DETECTABLE) & ~R["promoted"]].copy()
    miss["where"] = miss["stage"].str.replace(r"\(.*\)", "", regex=True).str.strip()
    for (kind, where), g in miss.groupby(["kind", "where"]):
        out.append({"failure": f"detectable real missed: {kind} ({', '.join(sorted(set(g['band'].dropna())))})", "where": where,
                    "eras": dict(g["eval_era"].value_counts()), "count": len(g), "per_100_worlds": 100 * len(g) / nw})
    nr = R[(R["label"] == REAL) & (R["status"] == NOT_REPRESENTABLE)]
    for kind, g in nr.groupby("kind"):
        out.append({"failure": f"real only detectable in its true form (single-feature screen cannot represent it): {kind}",
                    "where": "candidate generation", "eras": dict(g["eval_era"].value_counts()), "count": len(g),
                    "per_100_worlds": 100 * len(g) / nw})
    late = R[(R["label"] == REAL) & R["promoted"] & (R["delay_looks"].fillna(0) >= 2)]
    if len(late):
        out.append({"failure": "real promoted >= 2 looks after it was first detectable", "where": "gate (sequential alpha / evidence floors)",
                    "eras": dict(late["eval_era"].value_counts()), "count": len(late), "per_100_worlds": 100 * len(late) / nw})
    T = pd.DataFrame(out)
    return T.sort_values("count", ascending=False).reset_index(drop=True) if len(T) else T


# ================================================================================================================ one world end to end
def run_world(seed: int, out: Path, cfg: BenchConfig = BenchConfig(), *, code_hash: str | None = None, log=None) -> dict:
    """PUBLIC ENTRY. Generate -> seal the key (C72) -> run the system on the frame only -> save its answers -> open the key -> score.
    Idempotent: a world already scored is read back, a world whose key exists without answers is re-run from its sealed key's seed
    only if the key still verifies (the key itself is never rewritten)."""
    out = Path(out)
    man = Manifest(out / "manifest.jsonl")
    wid = f"W{int(seed):05d}"
    sp = out / "scores" / f"{wid}.json"
    if sp.exists():
        return json.loads(sp.read_text(encoding="utf-8"))
    t0 = time.monotonic()
    w = make_world(seed, cfg)
    gen_s = time.monotonic() - t0
    if man.last(wid, "key") is None or not key_path(out, wid).exists():
        seal_key(w.key, out, man)
    frame = w.frame
    del w                                              # the system gets the frame; the key object is dropped before it starts
    started = time.time_ns()
    ans = run_system(frame, wid, seed, cfg, code_hash=code_hash or "f19", log=log)
    ans["generation_s"] = round(gen_s, 1)
    save_answers(ans, out, man, started)
    key, ans = open_key(out, wid, man)
    rows, summ = score_world(key, ans)
    res = {"summary": summ, "rows": rows}
    _write_new(sp, json.dumps(res, default=_json_default).encode())
    return res


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def rescore(out: Path) -> tuple[list[dict], list[dict], list[str]]:
    """Score every world again from its SEALED key and frozen answers with the current scoring code (open_key re-verifies both seals),
    so the evaluator can evolve without re-running the learner and without trusting cached scores. Returns (rows, summaries, refused)."""
    out = Path(out)
    man = Manifest(out / "manifest.jsonl")
    rows, summ, refused = [], [], []
    for ap in sorted((out / "answers").glob("W*.answers.json")):
        wid = ap.name.split(".")[0]
        try:
            key, ans = open_key(out, wid, man)
        except SealError as e:
            refused.append(f"{wid}: {e}")
            continue
        r, m = score_world(key, ans)
        rows += r
        summ.append(m)
    return rows, summ, refused


def load_scores(out: Path) -> tuple[list[dict], list[dict]]:
    rows, summ = [], []
    for p in sorted((Path(out) / "scores").glob("W*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows += d["rows"]
        summ.append(d["summary"])
    return rows, summ


def report_markdown(tables: Mapping[str, pd.DataFrame], summaries: Sequence[dict], meta: Mapping[str, Any]) -> str:
    S = pd.DataFrame(list(summaries))
    lines = ["# F19 real-vs-noise benchmark (C70-C74) - IMPLEMENTED, NOT VALIDATED", "",
             f"Worlds scored: {len(S)} ({(S['split'] == 'development').sum() if len(S) else 0} development, "
             f"{(S['split'] == 'heldout').sum() if len(S) else 0} held-out). {meta.get('note', '')}", ""]
    if len(S):
        lines += ["## C72 score: right / wrong against the sealed answer key", "",
                  f"- Real patterns found (detectable): {int(S['real_right'].sum())} of {int(S['real_detectable'].sum())}; "
                  f"all real planted: {int(S['real_total'].sum())} ({int(S['real_undetectable'].sum())} undetectable in principle, "
                  f"{int(S['real_not_representable'].sum())} detectable only in their true form).",
                  f"- Noise correctly rejected: {int(S['noise_rejected'].sum())} of {int(S['noise_total'].sum())}; "
                  f"false positives: {int(S['false_positives'].sum())} in total, {S['false_positives'].mean():.2f} per world "
                  f"(target 0; the gate's stated alpha allows {S['expected_fp_stated_alpha'].mean():.3f} per world on the pure nulls it gated, "
                  f"{S['expected_fp_honest_alpha'].mean():.4f} if the search size were every feature scored).", ""]
    for name, title in (("total", "Totals (world-cluster bootstrap 95% intervals)"), ("era", "Per era (the era covering most of the final evaluation window)"),
                        ("band", "Real patterns per kind and strength band"), ("noise", "Noise per kind"),
                        ("closeness", "C73: how close (confidence = 1 - the screen's BH q at the last look)"),
                        ("calibration", "C73: calibration of that confidence"), ("tier", "C75 3E/3F: per difficulty tier (0 = NULL world)"),
                        ("kind", "F28: real patterns per kind (all bands)"), ("strength", "F28: real patterns per strength band (all kinds)"),
                        ("family", "F28: false positives per noise family"), ("null_worlds", "F28: null worlds (the right answer is nothing)"),
                        ("attribution", "F28: each fix switched off (counterfactual verdicts of the same bundles; exact for the final-look protocol)"),
                        ("gates", "Which gate blocked what (last look)"), ("learning_curve", "C75 Phase 7: next unseen worlds after k worlds"),
                        ("failures", "Ranked concrete failures (drive the next briefs)")):
        t = tables.get(name)
        if t is None or t.empty:
            continue
        lines += [f"## {title}", "", md_table(t), ""]
    return "\n".join(lines)


# ================================================================================================================ C75 3M memorization / 3N mutation
def disguise(frame: pd.DataFrame, seed: int, shift_weeks: int = 52) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """C75 3M: the same world under new names - tickers renamed AND reordered (a rename that keeps row order hides name-order
    tie-breaks), dates shifted by whole weeks, planted columns renamed and reordered. Returns (frame, column map new -> old,
    ticker map new -> old). A learner blind to identities must give the same answers after mapping the names back."""
    rng = np.random.default_rng(np.random.SeedSequence([7, int(seed)]))
    ticks = pd.Index(frame.index.get_level_values(1).unique())
    new_t = [f"Z{int(i):03d}" for i in rng.permutation(len(ticks))]
    tmap = dict(zip(ticks, new_t))
    planted = [c for c in frame.columns if c.startswith(PREFIX)]
    perm = rng.permutation(len(planted))
    cmap = {c: f"{PREFIX}{int(perm[i]) + 500:03d}" for i, c in enumerate(planted)}
    d = pd.to_datetime(frame.index.get_level_values(0)) + pd.Timedelta(weeks=int(shift_weeks))
    t = frame.index.get_level_values(1).map(tmap)
    G = frame.rename(columns=cmap).copy()
    G.index = pd.MultiIndex.from_arrays([d, t], names=frame.index.names)
    G["end"] = pd.to_datetime(G["end"]) + pd.Timedelta(weeks=int(shift_weeks))
    order = rng.permutation(len(G))
    G = G.iloc[order]
    G = G.sort_index(level=0, sort_remaining=False, kind="stable")
    cols = [c for c in G.columns if not c.startswith(PREFIX)] + sorted(cmap.values(), key=lambda _: rng.random())
    return G[cols], {v: k for k, v in cmap.items()}, {v: k for k, v in tmap.items()}


def answers_invariance(a: dict, b: dict, back: Mapping[str, str]) -> dict:
    """Compare two answer sheets of one world (b under disguise; `back` maps b's names to a's). Invariant = same raised set, same
    final status and the same verdict at every look for every candidate."""
    diff = []
    bb = {back.get(f, f): c for f, c in b["candidates"].items()}
    for f, ca in a["candidates"].items():
        cb = bb.get(f)
        if cb is None:
            diff.append((f, "missing under disguise"))
            continue
        va, vb = [g["verdict"] for g in ca.get("gate", [])], [g["verdict"] for g in cb.get("gate", [])]
        if ca.get("raised_look") != cb.get("raised_look") or ca["final"] != cb["final"] or va != vb:
            diff.append((f, f"{ca.get('raised_look')}/{ca['final']}/{va} vs {cb.get('raised_look')}/{cb['final']}/{vb}"))
        sa = [round(x[1], 6) if x[1] is not None else None for x in ca["scan"]]
        sb = [round(x[1], 6) if x[1] is not None else None for x in cb["scan"]]
        if sa != sb:
            diff.append((f, "scan AUC differs"))
    return {"n": len(a["candidates"]), "n_diff": len(diff), "diffs": diff[:50], "invariant": not diff}


MUTATIONS: dict[str, dict[str, Any]] = {
    # C75 3N: the generator changed along one axis at a time (seeds change with every world anyway)
    "stronger": {"bands": (("obvious", 0.9), ("moderate", 0.55), ("subtle", 0.3), ("faint", 0.18), ("extremely_subtle", 0.09))},
    "weaker": {"bands": (("obvious", 0.4), ("moderate", 0.23), ("subtle", 0.13), ("faint", 0.08), ("extremely_subtle", 0.04))},
    "short_regimes": {"min_segment": 13, "single_era_share": 0.0},
    "long_regimes": {"single_era_share": 1.0},
    "noisy_names": {"ticker_sd": 1.0, "slow_sd": 0.8},
    "quiet_names": {"ticker_sd": 0.2, "slow_sd": 0.1},
    "harder_mt": {"mt_pool": 200, "mt_window": 0.5},
    "fewer_names": {"n_names": 24},
    "more_names": {"n_names": 96},
}


def mutate(cfg: BenchConfig, name: str) -> BenchConfig:
    if name not in MUTATIONS:
        raise KeyError(f"unknown mutation {name!r}; known: {sorted(MUTATIONS)}")
    out = dataclasses.replace(cfg, **MUTATIONS[name])
    errs = out.validate()
    if errs:
        raise ValueError(f"mutation {name} is invalid: {errs}")
    return out


# ================================================================================================================ C75 Phase 7 / Firewall 10
def learning_curve(summaries: Sequence[dict], order: Sequence[str] | None = None, points: Sequence[int] = (1, 5, 10, 25, 50, 100, 250, 500)
                   ) -> pd.DataFrame:
    """Performance on the NEXT unseen worlds after k worlds of history, in processing order. The gate-only path carries no memory
    between worlds (a fresh ledger and quarantine store per world), so any slope here is sampling noise: the curve is reported to
    make that explicit, not to claim learning."""
    S = pd.DataFrame(list(summaries))
    if S.empty:
        return S
    if order is not None:
        S = S.set_index("world_id").loc[[w for w in order if w in set(S["world_id"])]].reset_index()
    rows = []
    for k in points:
        if k >= len(S):
            break
        nxt = S.iloc[k:k + max(5, k)]
        det = nxt["real_detectable"].sum()
        rows.append({"history_worlds": k, "next_worlds": len(nxt), "recall_next": float(nxt["real_right"].sum() / det) if det else None,
                     "fp_per_world_next": float(nxt["false_positives"].mean()),
                     "noise_rejection_next": float(nxt["noise_rejected"].sum() / max(1, nxt["noise_total"].sum()))})
    return pd.DataFrame(rows)


def freeze_record(cfg: BenchConfig) -> dict:
    """Firewall 10 design (the final holdout is NOT created here): what must be frozen before a final benchmark is sealed - the
    benchmark configuration, the gate / plan / screen settings and the code that runs them. `assert_frozen` refuses a final run whose
    current state differs from the record."""
    from engine import provenance as PV
    from engine.research import evidence as EV
    from engine.research import loop as LP
    from engine.research import quality_gate as QG
    top, t_min = loop_screen_defaults()
    return {"benchmark_config": stable_hash(dataclasses.asdict(cfg), 16), "quality_policy": stable_hash(repr(QG.QualityPolicy()), 16),
            "sequential_plan": stable_hash(repr(LP.REGATE_PLAN), 16), "evidence_config": stable_hash(repr(EV.EvidenceConfig()), 16),
            "screen": [top, t_min, LP.REGATE_NEW_DATES], "code_hash": PV.code_stamp()["code_hash"], "final_salt": "F19-final-UNCREATED"}


def assert_frozen(record: Mapping[str, Any], cfg: BenchConfig) -> None:
    now = freeze_record(cfg)
    bad = [k for k in record if k in now and record[k] != now[k]]
    if bad:
        raise HeldOutAccess(f"final-holdout state changed since it was frozen: {bad}")


def md_table(t: pd.DataFrame) -> str:
    """A GitHub markdown table (no tabulate dependency); '|' inside cells is escaped."""
    def cell(v) -> str:
        if isinstance(v, float):
            return "" if not np.isfinite(v) else f"{v:.4g}"
        return str(v).replace("|", "/").replace("\n", " ")
    cols = [str(c) for c in t.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(cell(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(out)
