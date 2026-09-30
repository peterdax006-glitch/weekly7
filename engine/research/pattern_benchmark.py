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
persistent replication ledger, re-gated every loop.REGATE_NEW_DATES dates, retired only on futility / quarantine / the evidence
horizon), on the rolling frame the loop's feed serves (feeds.FeedConfig.frame_weeks). Deviations, all stated in the report: (1) the
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
REAL_KINDS = ("linear", "sparse", "conditional", "interaction", "drifting", "regime")
NOISE_KINDS = ("null", "identity_null", "mt_winner", "coincidence", "early_decay", "proxy", "fluke", "leak", "context", "base_null")
PURE_NULL_KINDS = ("null", "base_null", "context", "mt_winner")     # zero effect in every evaluation window: alpha bounds them
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
    first_look: int = 104                  # decision-date index of the first screen + gate
    look_every: int = 13                   # = loop.REGATE_NEW_DATES
    bands: tuple = (("strong", 0.45), ("medium", 0.28), ("weak", 0.16), ("faint", 0.08))
    real_kinds: tuple = REAL_KINDS
    noise_counts: tuple = (("null", 70), ("identity_null", 30), ("mt_winner", 50), ("coincidence", 10), ("early_decay", 10),
                           ("proxy", 12), ("fluke", 10), ("leak", 10))
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
    seed_salt: str = "F19-heldout-v1"

    def validate(self) -> list[str]:
        errs = []
        if self.n_names < 16 or self.n_dates < 60:
            errs.append("n_names >= 16 and n_dates >= 60 required")
        if not 8 <= self.first_look < self.n_dates or self.look_every < 1:
            errs.append("first_look in [8, n_dates) and look_every >= 1 required")
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
        return errs

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
    r = rankdata(np.nan_to_num(score, nan=-1e18), axis=-1)
    y = np.asarray(y, bool)
    n1 = y.sum(-1).astype(float)
    n0 = y.shape[-1] - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        a = ((r * y).sum(-1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    a[(n1 < 1) | (n0 < 1) | (y.shape[-1] < min_names)] = np.nan
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


def oracle_power(score: np.ndarray, y_draws: np.ndarray, date_mask: np.ndarray, test: tuple[int, int], crit: float) -> tuple[float, float]:
    """(power, mean effect) of a one-sided per-date AUC t-test of `score` over the test dates [a, b) where the effect exists."""
    A = date_aucs(score, y_draws)                       # draws x T
    a, b = test
    m = np.zeros(A.shape[-1], bool)
    m[a:b] = True
    m &= date_mask
    if m.sum() < 3:
        return 0.0, 0.0
    E = A[:, m] - 0.5
    t = _t(E)
    return float(np.mean(np.nan_to_num(t, nan=-np.inf) > crit)), float(np.nanmean(E))


# ================================================================================================================ the world
@dataclasses.dataclass
class World:
    seed: int
    frame: pd.DataFrame                    # what the system sees (base columns, planted columns, outcomes) - no truth
    key: dict                              # the answer key (sealed before the system runs)


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


def scan_universe(columns: Iterable[str]) -> list[str]:
    """Every feature the loop's screen scores on a frame with these columns (st_feature_screen's own filter), planted columns included."""
    from engine.research import two_stage as TS
    from engine.research import vol_hypotheses as VH
    cols = set(columns)
    base = [f for f in VH.DERIVED if not f.startswith(("ix__", "ixnull")) and f not in TS.HISTORY_DEPENDENT
            and not VH.missing_columns((f,), cols) and not f.startswith(PREFIX)]
    return base + sorted(c for c in cols if c.startswith(PREFIX))


def make_world(seed: int, cfg: BenchConfig = BenchConfig()) -> World:
    """A seeded world with KNOWN truth. Base columns come from volatility_lab.planted_frame (the research-frame schema the gate reads),
    re-scaled per era; the touch outcome is re-drawn from a logit that holds the era base rate, a persistent per-name effect and every
    planted pattern's contribution; planted columns get opaque, shuffled names."""
    from engine.research import volatility_lab as VL
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid benchmark config: " + "; ".join(errs))
    rng = np.random.default_rng(np.random.SeedSequence([20260930, int(seed)]))
    T, N = cfg.n_dates, cfg.n_names
    year = int(rng.integers(*cfg.year_range))
    first = str((pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(rng.integers(0, 330)))).date())
    F = VL.planted_frame("null", n_dates=T, n_tickers=N, seed=int(rng.integers(0, 2**31)), first=first, n_sectors=cfg.n_sectors)
    dates = pd.DatetimeIndex(F.index.get_level_values(0).unique())
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
    u_name = rng.normal(0, cfg.ticker_sd, N)
    logit = par["base"][:, None] + par["spread"][:, None] * u_name[None, :]
    cols: dict[str, np.ndarray] = {}
    pats: list[dict] = []
    oracle_forms: dict[str, tuple[np.ndarray, np.ndarray]] = {}      # pid -> (oriented oracle score, date mask where the effect exists)

    def new_col(v: np.ndarray) -> str:
        name = f"_c{len(cols):04d}"
        cols[name] = v.astype(np.float32)
        return name

    def add(label, kind, band, beta, sign, colnames, parent=None, **extra) -> dict:
        p = {"pid": f"P{len(pats):03d}", "label": label, "kind": kind, "band": band, "beta": float(beta), "sign": float(sign),
             "columns": list(colnames), "parent": parent, **extra}
        pats.append(p)
        return p

    regime_dates = np.isin(era, REGIME_ERAS)
    # ---------------------------------------------------------------- REAL patterns: every kind in every strength band
    for kind in cfg.real_kinds:
        for band, beta in cfg.bands:
            s = float(rng.choice([-1.0, 1.0]))
            x = rng.normal(0, 1, shape)
            mask = np.ones(T, bool)
            if kind == "linear":
                contrib, form = beta * s * x, s * x
                cn = [new_col(x)]
            elif kind == "sparse":
                ind = (x > 1.5).astype(float)
                contrib, form = 3.0 * beta * s * ind, s * ind
                cn = [new_col(x)]
            elif kind == "conditional":
                ctx = np.repeat((rng.random(N) < 0.5)[None, :], T, 0).astype(float)
                flip = rng.random(shape) < 0.05
                ctx = np.where(flip, 1 - ctx, ctx)
                contrib, form = 2.0 * beta * s * x * ctx, s * x * ctx
                cn = [new_col(x)]
                ctx_col = new_col(ctx)
            elif kind == "interaction":
                b2 = rng.normal(0, 1, shape)
                contrib, form = 1.5 * beta * s * x * b2, s * x * b2
                cn = [new_col(x), new_col(b2)]
            elif kind == "drifting":
                phase = float(rng.uniform(0, 2 * math.pi))
                amp = 1.0 + 0.7 * np.sin(2 * math.pi * np.arange(T) / (1.4 * T) + phase)
                contrib, form = beta * s * amp[:, None] * x, s * x
                cn = [new_col(x)]
            else:                                              # regime-limited but persistent: on in every volatile / crisis era
                contrib, form = 2.2 * beta * s * x * regime_dates[:, None], s * x
                mask = regime_dates.copy()
                cn = [new_col(x)]
            logit = logit + contrib
            p = add(REAL, kind, band, beta, s, cn if kind != "interaction" else [], columns_marginal=cn)
            oracle_forms[p["pid"]] = (form, mask)
            p["_x"] = cn[0]
            if kind == "conditional":
                add(NOISE, "context", None, 0.0, 1.0, [ctx_col], parent=p["pid"])
            if kind == "interaction":
                for c in cn:
                    add(PART, "interaction_component", band, 0.0, s, [c], parent=p["pid"])
    reals = [p for p in pats if p["label"] == REAL]
    counts = dict(cfg.noise_counts)
    # ---------------------------------------------------------------- NOISE that moves the outcome for a while (then vanishes)
    for kind, n in (("coincidence", counts.get("coincidence", 0)), ("early_decay", counts.get("early_decay", 0)),
                    ("fluke", counts.get("fluke", 0))):
        for _ in range(n):
            s = float(rng.choice([-1.0, 1.0]))
            x = rng.normal(0, 1, shape)
            tt = np.arange(T)
            if kind == "coincidence":
                w = np.where(tt < 0.25 * T, 0.35, 0.0)
            elif kind == "early_decay":
                w = 0.35 * np.clip(1 - tt / (0.5 * T), 0, 1)
            else:
                a = int(rng.integers(0, T - 10))
                w = np.where((tt >= a) & (tt < a + 10), 0.6, 0.0)
            logit = logit + s * w[:, None] * x
            add(NOISE, kind, None, float(w.max()), s, [new_col(x)], active=[int(np.flatnonzero(w)[0]), int(np.flatnonzero(w)[-1]) + 1])
    p_true = 1.0 / (1.0 + np.exp(-logit))
    touch = rng.random(shape) < p_true
    # ---------------------------------------------------------------- the outcome columns (planted_frame's recipe)
    vol20 = F["vol20"].to_numpy(float).reshape(shape)
    quiet = np.minimum(0.095, vol20 * np.sqrt(5) * np.abs(rng.normal(0, 1, shape)) * 0.8)
    absmove = np.where(touch, 0.10 + rng.exponential(0.05, shape), quiet)
    sign = np.where(rng.random(shape) < 0.5, 1.0, -1.0)
    F["touch"] = touch.reshape(-1).astype(float)
    F["absmove"] = absmove.reshape(-1)
    F["tday"] = np.where(touch, rng.integers(1, 6, shape), 0).reshape(-1).astype(float)
    F["close"] = (sign * absmove * rng.uniform(0.6, 1.0, shape)).reshape(-1)
    F["up"] = (F["close"] > 0).astype(float)
    F["end"] = pd.to_datetime(F.index.get_level_values(0)) + pd.Timedelta(days=8)
    # ---------------------------------------------------------------- NOISE that never moves the outcome
    for _ in range(counts.get("null", 0)):
        kind = int(rng.integers(0, 3))
        x = rng.normal(0, 1, shape) if kind == 0 else rng.standard_t(3, shape) if kind == 1 else rng.exponential(1.0, shape)
        add(NOISE, "null", None, 0.0, 1.0, [new_col(x)])
    for _ in range(counts.get("identity_null", 0)):
        a = rng.normal(0, 1, N)
        add(NOISE, "identity_null", None, 0.0, 1.0, [new_col(a[None, :] + 0.35 * rng.normal(0, 1, shape))])
    k_in = max(3, int(cfg.mt_window * T))
    for _ in range(counts.get("mt_winner", 0)):
        pool = rng.normal(0, 1, (cfg.mt_pool, k_in, N))
        eff = np.nanmean(date_aucs(pool, np.broadcast_to(touch[:k_in], pool.shape)) - 0.5, axis=-1)
        j = int(np.argmax(np.abs(eff)))
        x = rng.normal(0, 1, shape)
        x[:k_in] = pool[j]
        add(NOISE, "mt_winner", None, 0.0, float(np.sign(eff[j]) or 1.0), [new_col(x)], in_sample_effect=float(eff[j]),
            window=[0, k_in])
    for i in range(counts.get("proxy", 0)):
        par_p = [p for p in reals if p["kind"] != "interaction"]
        if not par_p:
            break
        src = par_p[int(rng.integers(0, len(par_p)))]
        rho = float(rng.uniform(0.6, 0.9))
        x = rho * cols[src["_x"]].astype(float) + math.sqrt(1 - rho * rho) * rng.normal(0, 1, shape)
        add(NOISE, "proxy", src["band"], 0.0, src["sign"], [new_col(x)], parent=src["pid"], rho=rho)
    gs = (0.3, 0.6, 1.0)
    for i in range(counts.get("leak", 0)):
        g = gs[i % len(gs)]
        x = rng.normal(0, 1, shape) + g * _z(absmove)          # built from the outcome window itself: correlated with the future only through overlap
        add(NOISE, "leak", None, g, 1.0, [new_col(x)])
    # ---------------------------------------------------------------- opaque names, frame, detectability, key
    order = rng.permutation(len(cols))
    rename = {old: f"{PREFIX}{int(order[i]):03d}" for i, old in enumerate(cols)}
    for old, v in cols.items():
        F[rename[old]] = v.reshape(-1)
    for p in pats:
        p["columns"] = [rename[c] for c in p["columns"]]
        if "columns_marginal" in p:
            p["columns_marginal"] = [rename[c] for c in p["columns_marginal"]]
        if "_x" in p:
            p["_x"] = rename[p["_x"]]
    base = [f for f in scan_universe(F.columns) if not f.startswith(PREFIX)]
    for f in base:
        add(NOISE, "base_null", None, 0.0, 1.0, [f])
    F.attrs.clear()
    _detectability(pats, oracle_forms, F, p_true, rng, cfg)
    wins = windows(cfg)
    L_last, a_last, b_last = wins[-1]
    test_era = pd.Series(era[a_last:b_last]).value_counts()
    key = {"world_id": f"W{int(seed):05d}", "seed": int(seed), "split": split_of(seed, cfg.seed_salt), "start": first, "year": year,
           "eras": [[int(a), e] for a, e in sched], "eval_era": str(test_era.index[0]), "shift": len(sched) > 1,
           "eval_era_share": float(test_era.iloc[0] / test_era.sum()), "base_rate": float(touch.mean()),
           "looks": [str(dates[L].date()) if L < T else str((dates[-1] + pd.Timedelta(days=7)).date()) for L in cfg.looks()],
           "config": stable_hash(dataclasses.asdict(cfg), 16), "oracle_crit": oracle_crit(), "power_floor": cfg.power_floor,
           "patterns": [{k: v for k, v in p.items() if not k.startswith("_")} for p in pats],
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
        marg = [oracle_power(p["sign"] * x, y, allm, (a, b), crit) for _, a, b in wins]
        p["marginal_power"], p["planted_effect"] = marg[-1]
        p["marginal_power_by_look"] = [round(m[0], 3) for m in marg]
        if p["pid"] in forms:
            form, mask = forms[p["pid"]]
            orc = [oracle_power(form, y, mask, (a, b), crit) for _, a, b in wins]
            p["oracle_power"], p["oracle_effect"] = orc[-1]
            p["oracle_power_by_look"] = [round(o[0], 3) for o in orc]
            first = next((i for i, o in enumerate(orc) if o[0] >= cfg.power_floor), None)
            p["earliest_look"] = first
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
    return manifest.append({"kind": "key", "world_id": key["world_id"], "sha256": _sha(data), "written_ns": time.time_ns(),
                            "mtime_ns": p.stat().st_mtime_ns})


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
    if errs:
        raise SealError("; ".join(errs))
    kb, ab = kp.read_bytes(), ap.read_bytes()
    if _sha(kb) != krow["sha256"]:
        errs.append(f"{world_id}: answer key altered (sha256 differs from the manifest)")
    if _sha(ab) != arow["sha256"]:
        errs.append(f"{world_id}: answers altered after they were saved")
    if not krow["written_ns"] < arow["started_ns"] <= arow["written_ns"]:
        errs.append(f"{world_id}: the key was not written before the system started")
    if kp.stat().st_mtime_ns > arow["written_ns"] or kp.stat().st_mtime_ns != krow["mtime_ns"]:
        errs.append(f"{world_id}: the key file was (re)written after it was sealed")
    if errs:
        raise SealError("; ".join(errs))
    return json.loads(kb), json.loads(ab)


# ================================================================================================================ the system under test
@contextlib.contextmanager
def registered(features: Sequence[str]):
    """Planted columns enter the candidate universe through the runtime derived-feature registry (as interactions do), then leave."""
    from engine.research import vol_hypotheses as VH
    added = []
    try:
        for c in features:
            if c not in VH.DERIVED:
                VH.DERIVED[c] = ((c,), (lambda F, c=c: F[c]))
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
    return int(f["screen_top"].default), float(f["screen_t"].default)


def _decision_summary(rep, sid: str) -> dict:
    d = next(x for x in rep.decisions if x.subject_id == sid)
    gates = [(g.gate, g.state, g.ok) for g in d.gates]
    stat = next((g for g in d.gates if g.gate == "out_of_sample"), None)
    return {"verdict": d.verdict.value, "n_gates": len(gates), "n_ok": int(sum(ok for _, _, ok in gates)),
            "blocking": [g for g, s, ok in gates if not ok], "states": {g: s for g, s, _ in gates},
            "oos_margin": None if stat is None or stat.margin is None else float(stat.margin)}


def run_system(frame: pd.DataFrame, world_id: str, seed: int, cfg: BenchConfig = BenchConfig(), *, code_hash: str = "f19",
               created_real: str = "2026-09-30T00:00:00+00:00") -> dict:
    """The current system on one world, past data only. Sees the frame and nothing else (never the key)."""
    from engine.research import evidence as EV
    from engine.research import loop as LP
    from engine.research import quality_gate as QG
    from engine.research import replication as RP
    from engine.research import volatility_lab as VL
    top, t_min = loop_screen_defaults()
    cap = top * cfg.look_every                        # one screen call stands for look_every weekly calls of the loop
    planted = [c for c in frame.columns if c.startswith(PREFIX)]
    dates = pd.DatetimeIndex(frame.index.get_level_values(0).unique()).sort_values()
    plan = LP.REGATE_PLAN
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
                                      has_falsifier=True, seed=int(seed))
                b = EV.assemble(M, spec, now, code_hash=code_hash, data_hash=f"bench{seed}", created_real=created_real,
                                ledger=st["ledger"], look=k, plan=plan)
                rep = EV.gate([b], now, code_hash, store=store, looks={spec.subject_id: k}, plan=plan)
                d = _decision_summary(rep, spec.subject_id)
                d.update(look=li, k=k, alpha=plan.alpha_at(k), n_tests_searched=max(1, len(screened)), n_scanned=len(feats),
                         effect_test=b.parts.get("effect_test"), n_test=b.parts.get("n_test"), missing=sorted(b.missing))
                cands[f]["gate"].append(d)
                n_gated += 1
                why = None
                if d["verdict"] == "PROMOTE":
                    cands[f]["final"], cands[f]["promoted_look"] = "PROMOTED", li
                elif d["verdict"] == "QUARANTINED":
                    cands[f]["final"], why = "QUARANTINED", "quarantined"
                else:
                    why = plan.futility(b)
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
    return {"world_id": world_id, "n_scanned": len(cands), "screen_top": top, "screen_t": t_min, "cap_per_screen": cap,
            "looks": looks_meta, "candidates": cands, "seconds": round(time.monotonic() - t0, 1)}


# ================================================================================================================ reference answer sheets
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
        true_sign = p["sign"]
        sys_sign = best.get("sign") if best else None
        pl = best.get("promoted_look") if best else None
        r = {"world_id": key["world_id"], "split": key["split"], "eval_era": key["eval_era"], "shift": key["shift"], "pid": p["pid"],
             "label": p["label"], "kind": p["kind"], "band": p.get("band"), "status": p.get("status"), "oracle_power": p.get("oracle_power"),
             "marginal_power": p.get("marginal_power"), "planted_effect": p.get("planted_effect"), "surfaced": surfaced, "promoted": promoted,
             "promoted_look": pl, "raised_look": best.get("raised_look") if best else None, "earliest_look": p.get("earliest_look"),
             "stage": _stage(best), "confidence": conf, "sys_effect": eff, "sys_sign": sys_sign,
             "sign_ok": None if sys_sign is None or p["kind"] in ("mt_winner",) else bool(sys_sign == true_sign),
             "rank": final_rank.get(cols[0]) if cols else None, "t_scan": last[2] if last else None,
             "best_gate_share": _best_gate_share(best), "last_verdict": best["gate"][-1]["verdict"] if best and best.get("gate") else None,
             "n_looks_gated": len(best["gate"]) if best and best.get("gate") else 0, "parent": p.get("parent")}
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
                elif r["last_verdict"] == "NEEDS_MORE_EVIDENCE" and len(best["gate"][-1]["blocking"]) <= 1:
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
    summ = {"world_id": key["world_id"], "split": key["split"], "eval_era": key["eval_era"], "shift": key["shift"], "n_looks": n_looks,
            "real_total": int(len(real)), "real_detectable": int(len(det)), "real_right": int(det["promoted"].sum()),
            "real_undetectable": int((real["status"] == UNDETECTABLE).sum()), "real_not_representable": int((real["status"] == NOT_REPRESENTABLE).sum()),
            "real_promoted_any": int(real["promoted"].sum()), "noise_total": int(len(noise)), "noise_rejected": int((~noise["promoted"]).sum()),
            "false_positives": int(noise["promoted"].sum()), "fp_by_kind": noise[noise["promoted"]]["kind"].value_counts().to_dict(),
            "misses": det[~det["promoted"]][["pid", "kind", "band", "stage"]].to_dict("records"),
            "false_positive_list": noise[noise["promoted"]][["pid", "kind", "stage"]].to_dict("records"),
            "credit": float(real["credit"].sum()), "seconds": answers.get("seconds")}
    summ.update(expected_fp_bound(answers, key))
    return rows, summ


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
            stated += 1 - (1 - a) ** (1.0 / max(1, int(g["n_tests_searched"])))
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
        tot.append({"set": gname, "worlds": int(Rg["world_id"].nunique()),
                    "real right (detectable promoted)": _fmt(*_ratio(Rg, det, Rg["promoted"])),
                    "real right / all real": f"{int((real & Rg['promoted']).sum())}/{int(real.sum())}",
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
                       "TP recall (detectable)": _fmt(*_ratio(rg, d, rg["promoted"])),
                       "FP / world": f"{sg['false_positives'].mean():.2f}", "noise rejected": _fmt(*_ratio(ng, pd.Series(True, index=ng.index), ~ng["promoted"])),
                       "detectable share": f"{d.mean():.1%}"})
    out["era"] = pd.DataFrame(rows_e)
    rows_b = []
    for (split, kind, band), g in real.groupby(["split", "kind", "band"]):
        d = g["status"] == DETECTABLE
        rows_b.append({"split": split, "kind": kind, "band": band, "n": len(g), "detectable": int(d.sum()),
                       "not representable": int((g["status"] == NOT_REPRESENTABLE).sum()), "undetectable": int((g["status"] == UNDETECTABLE).sum()),
                       "TP recall (detectable)": _fmt(*_ratio(g, d, g["promoted"])), "surfaced": f"{g['surfaced'].mean():.1%}",
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
    out["closeness"] = closeness(R)
    out["calibration"] = calibration_table(R)
    out["failures"] = ranked_failures(R, S)
    return out


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
def run_world(seed: int, out: Path, cfg: BenchConfig = BenchConfig(), *, code_hash: str | None = None) -> dict:
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
    ans = run_system(frame, wid, seed, cfg, code_hash=code_hash or "f19")
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
                        ("calibration", "C73: calibration of that confidence"), ("failures", "Ranked concrete failures (drive the next briefs)")):
        t = tables.get(name)
        if t is None or t.empty:
            continue
        lines += [f"## {title}", "", md_table(t), ""]
    return "\n".join(lines)


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
