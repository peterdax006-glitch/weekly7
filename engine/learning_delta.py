"""Learning-delta harness (Bible Phases 11, 21-23, 45; canon C54 and C55).

C54: the product is the LEARNING system, so a blind test is judged by how the result CHANGES when the same real year is
played again after the system has learned from the first pass. C55: the system must not be able to know it is the same
year. This module builds the harness that measures that change and refuses to report it when the disguise leaked.

The protocol, per revealed window W (never an unrevealed seal):
  Run 1     the system plays W as archived, with its current learned state S0 (basis cfg/meta + long-term memory bank).
  Learn     the system's own learner turns Run 1 into S1 (memory episodes into the bank, optional basis update). The
            harness passes the run to the learner and hand-tunes nothing.
  Run 2     the SAME real year under a NEW disguise (order-preserving fresh code names, a fresh whole-week date shift),
            played with S1. The player receives a Presentation (prices, snapshots, costs, industry division) and a
            Visible state; there is no field for a window id, real date or real ticker, and a blindness audit checks
            what was actually handed over and fails the run when anything identifying is there.
  Controls  (a) NOISE: Run 2 with S0 under a fresh disguise, so disguise noise alone is measured (with order-preserving
                codes an identity-free system reproduces Run 1 exactly; a random relabel is measured too, to size how
                much tie-break luck alone moves a result);
            (b) TRANSFER: a different revealed year B played with S1 and with S0 (same disguise for both), so what
                generalises shows up too. A same-year gain far above the transfer gain means the system is recognising
                the year (market-context fingerprints in memory, say): flagged MEMORISATION;
            (c) MEMORISER: a planted learner that stores the realised outcome of every snapshot it has seen, keyed by
                content (never by name or date). It must show a big same-year gain and about zero transfer, otherwise
                the harness could not see memorisation and its verdicts are declared void.
Learning delta = Run 2 - Run 1 per window on weekly mean, share of weeks in the 5-10% band, worst-5% week, max
drawdown, positive-week share, direction and mover hit rates; aggregated with bootstrap CIs across windows.

Everything is deterministic (seeded), takes explicit seeds, and uses no clock."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import blind_gates as BG
from . import config as K
from . import objective as O

METRICS = ("mean_week", "in_band", "worst5", "max_dd", "pos_share", "dir_hit", "mover_hit", "year_return")
PRIMARY = "mean_week"
DISGUISE_MIN_YEAR = 2100           # every date a player sees lies in the 2100s or later; the real era ends before 2030
REAL_ERA_LAST_YEAR = 2030
PRESENTATION_FIELDS = ("snaps", "closes", "opens", "bps", "divs")   # the ONLY things a player is handed about a window
VISIBLE_LTM_COLS = ("arm", "ctx", "outcome")
NOISE_TOL = 1e-9                   # order-preserving disguise noise above this on mean_week means something keys on identity
MOVER_TOUCH, MOVER_SESSIONS = 0.10, 5


class BlindnessError(RuntimeError):
    """The audit found something identifying in what a player would be handed. No result from such a run may be used."""


# ---------------------------------------------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Window:
    """A revealed window as the HARNESS holds it. Only presentation() output ever reaches a player."""
    id: str
    snaps: dict                    # date string -> snapshot frame indexed by code name (row order matters)
    closes: pd.DataFrame
    opens: pd.DataFrame | None
    bps: float
    divs: dict
    real_start: pd.Timestamp
    real_end: pd.Timestamp
    real_tickers: frozenset = frozenset()
    used_shifts: list = field(default_factory=list)      # whole-week shifts (days, relative to the archive) already shown
    used_codes: set = field(default_factory=set)         # every code name a player has already seen for this window

    def __post_init__(self):
        self.real_start, self.real_end = pd.Timestamp(self.real_start), pd.Timestamp(self.real_end)
        if self.real_end < self.real_start:
            raise ValueError(f"window {self.id}: real_end before real_start")
        if not self.used_codes:
            self.used_codes = set(self.all_codes())

    def all_codes(self):
        names = set(self.closes.columns) | set(self.divs)
        for s in self.snaps.values():
            names |= set(s.index)
        return sorted(names)

    @property
    def era(self):
        return BG.era_of(self.real_start)


@dataclass(frozen=True)
class Presentation:
    """What a player is handed about a window. There is deliberately no id, real date, real ticker or seed field."""
    snaps: dict
    closes: pd.DataFrame
    opens: pd.DataFrame | None
    bps: float
    divs: dict


@dataclass
class DisguiseRecord:
    """Harness-only record of how a presentation was disguised (never given to a player)."""
    seed: int
    shift_days: int
    order_preserving: bool
    code_map: dict
    fingerprint: str = ""


@dataclass
class Run:
    """One play of one presentation. `snaps`/`closes` are harness references for learners and metrics."""
    weekly: np.ndarray
    equity: np.ndarray
    decisions: list                # [(date string, [codes])]
    episodes: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)
    snaps: dict | None = None
    closes: pd.DataFrame | None = None


@dataclass
class Visible:
    """The learned state as a player sees it: no lineage, no window ids, no real dates."""
    cfg: dict
    meta: dict
    ltm: pd.DataFrame | None
    extra: dict


@dataclass
class LearnedState:
    cfg: dict
    meta: dict
    ltm: pd.DataFrame | None = None        # arm, ctx, outcome + harness-only real_end, source
    extra: dict = field(default_factory=dict)
    lineage: list = field(default_factory=list)

    def visible(self):
        ltm = None
        if self.ltm is not None and len(self.ltm):
            ltm = self.ltm[list(VISIBLE_LTM_COLS)].reset_index(drop=True).copy()
        return Visible(json.loads(json.dumps(self.cfg, default=str)), json.loads(json.dumps(self.meta, default=str)), ltm,
                       dict(self.extra))

    def fingerprint(self):
        h = hashlib.sha256(json.dumps({"cfg": self.cfg, "meta": self.meta}, sort_keys=True, default=str).encode())
        if self.ltm is not None and len(self.ltm):
            h.update(pd.util.hash_pandas_object(self.ltm[["arm", "outcome"]].astype({"arm": str}), index=False).values.tobytes())
        for k in sorted(self.extra):
            h.update(k.encode())
            v = self.extra[k]
            h.update(json.dumps(v, sort_keys=True, default=lambda o: sorted(o) if isinstance(o, dict) else str(o)).encode()
                     if not isinstance(v, np.ndarray) else v.tobytes())
        return h.hexdigest()[:16]

    def n_episodes(self):
        return 0 if self.ltm is None else len(self.ltm)


@dataclass(frozen=True)
class LearnContext:
    """What the harness-side learner may know about the window it learns from. Never passed on to a player."""
    window_id: str
    real_start: pd.Timestamp
    real_end: pd.Timestamp
    era: str


# ---------------------------------------------------------------------------------------------------------------
# disguise
# ---------------------------------------------------------------------------------------------------------------
def derive_seed(*parts):
    """A reproducible 63-bit seed from any hashable parts (no clock, no global RNG)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def _fresh_codes(codes, rng, order_preserving, prefix="Q", width=5, avoid=()):
    """Bijection old code -> new code. Order-preserving: sorted old names get sorted new numbers. Otherwise a random
    assignment of the same new names. New names never coincide with any name in `avoid`."""
    old = sorted(codes)
    pool = 10 ** width
    if len(old) > pool // 2:
        raise ValueError("too many names for the code space")
    nums = set()
    avoid = set(avoid)
    while len(nums) < len(old):
        for k in rng.choice(pool, size=len(old) - len(nums) + 16, replace=False):
            c = f"{prefix}{int(k):0{width}d}"
            if c not in avoid:
                nums.add(int(k))
            if len(nums) == len(old):
                break
    new = [f"{prefix}{k:0{width}d}" for k in sorted(nums)]
    if not order_preserving:
        new = [new[i] for i in rng.permutation(len(new))]
    return dict(zip(old, new))


def _shift_bounds(window, lo_w, hi_w):
    """Clip the shift range (weeks) so every shifted date stays inside [DISGUISE_MIN_YEAR, 2250]."""
    first = min(window.closes.index.min(), *(pd.Timestamp(k) for k in window.snaps)) if window.snaps else window.closes.index.min()
    last = max(window.closes.index.max(), *(pd.Timestamp(k) for k in window.snaps)) if window.snaps else window.closes.index.max()
    lo = max(lo_w, math.ceil((pd.Timestamp(f"{DISGUISE_MIN_YEAR}-01-01") - first).days / 7))
    hi = min(hi_w, math.floor((pd.Timestamp("2250-12-31") - last).days / 7))
    return lo, hi


def make_presentation(window, seed, order_preserving=True, shift_weeks=(-900, 1100), min_abs_weeks=8):
    """A fresh disguise of the archived window: new order-preserving code names and a new whole-week date shift that
    differs from every shift already shown (and from zero, the archive itself) by at least `min_abs_weeks`. Prices,
    features, row order and weekday structure are untouched, so an identity-free system must behave exactly as before.
    Returns (Presentation, DisguiseRecord). The window remembers the shift and codes so the next disguise is fresh again."""
    rng = np.random.default_rng(seed)
    lo, hi = _shift_bounds(window, *shift_weeks)
    if hi < lo:
        raise BlindnessError(f"{window.id}: no admissible date shift range ({lo}..{hi} weeks)")
    taken = [0] + list(window.used_shifts)
    for _ in range(2000):
        w = int(rng.integers(lo, hi + 1))
        if all(abs(w - t // 7) >= min_abs_weeks for t in taken):
            break
    else:
        raise BlindnessError(f"{window.id}: could not draw a date shift {min_abs_weeks} weeks away from {len(taken)} earlier ones")
    delta = pd.Timedelta(days=7 * w)
    cmap = _fresh_codes(window.all_codes(), rng, order_preserving, avoid=window.used_codes)
    snaps = {}
    for k, s in window.snaps.items():
        s2 = s.copy()
        s2.index = pd.Index([cmap[c] for c in s.index], name=s.index.name)
        s2.attrs = {}
        snaps[str((pd.Timestamp(k) + delta).date())] = s2
    closes = _relabel_frame(window.closes, cmap, delta)
    opens = None if window.opens is None else _relabel_frame(window.opens, cmap, delta)
    divs = {cmap[c]: v for c, v in window.divs.items()}
    pres = Presentation(snaps, closes, opens, float(window.bps), divs)
    window.used_shifts.append(7 * w)
    window.used_codes |= set(cmap.values())
    fp = hashlib.sha256(pd.util.hash_pandas_object(closes, index=True).values.tobytes()).hexdigest()[:16]
    return pres, DisguiseRecord(int(seed), 7 * w, bool(order_preserving), cmap, fp)


def _relabel_frame(df, cmap, delta):
    out = df.copy()
    out.columns = pd.Index([cmap[c] for c in df.columns], name=df.columns.name)
    out.index = df.index + delta
    out.attrs = {}
    return out


def archive_presentation(window):
    """Run 1 plays the window exactly as archived (its shift is 0 and its codes are the archive's)."""
    snaps = {str(pd.Timestamp(k).date()): s.copy() for k, s in window.snaps.items()}
    for s in snaps.values():
        s.attrs = {}
    return Presentation(snaps, window.closes.copy(), None if window.opens is None else window.opens.copy(), float(window.bps),
                        dict(window.divs))


# ---------------------------------------------------------------------------------------------------------------
# blindness audit (C55)
# ---------------------------------------------------------------------------------------------------------------
def has_token(text, tokens, min_len=2):
    """True when any token occurs in `text` as a whole alphanumeric token (a short id inside a hex hash is not a leak)."""
    for t in tokens:
        if t and len(t) >= min_len and re.search(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", text):
            return True
    return False


def _texts(obj, depth=0):
    """Every string reachable in a small nested structure (keys, values, attrs, index and column names)."""
    if depth > 4:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _texts(k, depth + 1)
            yield from _texts(v, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in list(obj)[:2000]:
            yield from _texts(v, depth + 1)


def _frame_texts(df):
    yield from _texts(list(df.attrs.items()))
    yield from _texts([n for n in (df.index.names if isinstance(df.index, pd.MultiIndex) else [df.index.name]) if n])
    yield from _texts([str(df.columns.name)] if df.columns.name else [])
    yield from _texts([c for c in df.columns if isinstance(c, str) and not re.fullmatch(r"[A-Za-z0-9_.\-]{1,40}", c)])


def _all_timestamps(pres):
    for k in pres.snaps:
        yield pd.Timestamp(k)
    yield pres.closes.index.min()
    yield pres.closes.index.max()
    if pres.opens is not None:
        yield pres.opens.index.min()
        yield pres.opens.index.max()


def audit_presentation(pres, window, rec, visible=None, kind="run2", order_preserving_required=True):
    """The blindness audit for one hand-over. Returns a list of BG.Finding; any 'fail' means the run is void.
    Checks: (1) structure - only the whitelisted fields exist; (2) no window id, real ticker or earlier code name in any
    string the player can read; (3) every date lies in the disguised era, the shift is a whole number of weeks and is
    new; (4) code names are a fresh bijection and (when required) keep the archive's order; (5) prices and features are
    the SAME year (returns identical after the relabel) - a rerun that is not the same year proves nothing; (6) the
    learned state carries no window id, real date, real ticker or lineage column."""
    out = []
    g = "blindness"

    def fail(msg):
        out.append(BG.Finding(g, "fail", f"[{kind}] {msg}"))

    fields_ = tuple(f.name for f in dataclasses.fields(Presentation))
    if fields_ != PRESENTATION_FIELDS:
        fail(f"presentation exposes fields {fields_}, expected {PRESENTATION_FIELDS}")
    forbidden = {window.id} | set(window.real_tickers)
    prior_codes = set(window.used_codes) - set(rec.code_map.values()) if rec is not None else set()
    strings = list(_texts(list(pres.snaps)))
    for s in pres.snaps.values():
        strings += list(_frame_texts(s))
    strings += list(_frame_texts(pres.closes)) + list(_texts(list(pres.divs.items())))
    if pres.opens is not None:
        strings += list(_frame_texts(pres.opens))
    for t in strings:
        if has_token(t, forbidden):
            fail(f"an identifying token appears in what the player can read: {t[:60]!r}")
            break
    codes = set(pres.closes.columns)
    for s in pres.snaps.values():
        codes |= set(s.index)
    codes |= set(pres.divs)
    if codes & set(window.real_tickers):
        fail(f"{len(codes & set(window.real_tickers))} real tickers are visible")
    if rec is not None:
        if codes & prior_codes:
            fail(f"{len(codes & prior_codes)} code names repeat an earlier disguise of this window")
        if set(rec.code_map.values()) != codes:
            fail("visible code names are not exactly the new mapping (unmapped names leak)")
        if len(set(rec.code_map.values())) != len(rec.code_map):
            fail("code map is not one-to-one")
        if rec.order_preserving != bool(order_preserving_required) and order_preserving_required:
            fail("order-preserving disguise required but not used")
        if rec.order_preserving:
            old = sorted(rec.code_map)
            new = [rec.code_map[c] for c in old]
            if new != sorted(new):
                fail("new code names do not keep the alphabetical order of the old ones (tie-breaks would move)")
        if rec.shift_days % 7 or rec.shift_days == 0:
            fail(f"date shift {rec.shift_days} days is zero or not a whole number of weeks")
        if rec.shift_days in window.used_shifts[:-1]:
            fail("date shift repeats an earlier disguise of this window")
    lo_ts, hi_ts = min(_all_timestamps(pres)), max(_all_timestamps(pres))
    if lo_ts.year < DISGUISE_MIN_YEAR:
        fail(f"a date in the real era is visible: {lo_ts.date()}")
    if any(pd.Timestamp(d).year <= REAL_ERA_LAST_YEAR for d in pres.snaps):
        fail("a snapshot is dated in the real era")
    real_years = {window.real_start.year, window.real_end.year}
    if lo_ts.year in real_years or hi_ts.year in real_years:
        fail("a visible date falls in the window's real year")
    # same-year fidelity: pct returns must equal the archive's, column by column after inverting the map
    if rec is not None:
        inv = {v: k for k, v in rec.code_map.items()}
        back = pres.closes.rename(columns=inv)
        try:
            back = back[window.closes.columns]
            a, b = back.pct_change().to_numpy(dtype="float64"), window.closes.pct_change().to_numpy(dtype="float64")
            same = np.allclose(np.nan_to_num(a, nan=-9.0), np.nan_to_num(b, nan=-9.0), rtol=0, atol=1e-9)
        except KeyError:
            same = False
        if not same or len(pres.closes) != len(window.closes):
            fail("the disguised prices are not the same year's returns (harness fault)")
        if list(pres.snaps.keys()) != [str((pd.Timestamp(k) + pd.Timedelta(days=rec.shift_days)).date()) for k in window.snaps]:
            fail("snapshot dates are not the archive's dates shifted by the disguise shift")
    if visible is not None:
        out += audit_visible(visible, window, kind)
    return out


def audit_visible(vis, window, kind="run2"):
    """The learned state handed to a player must carry no window id, real date, real ticker, or lineage column."""
    out, g = [], "blindness"
    if vis.ltm is not None and len(vis.ltm):
        cols = tuple(vis.ltm.columns)
        if cols != VISIBLE_LTM_COLS:
            out.append(BG.Finding(g, "fail", f"[{kind}] long-term memory exposes columns {cols}, expected {VISIBLE_LTM_COLS}"))
    text = "\n".join(list(_texts(vis.cfg)) + list(_texts(vis.meta)) + list(_texts(vis.extra))
                     + (list(map(str, vis.ltm["arm"].unique()[:5000])) if vis.ltm is not None and len(vis.ltm) else []))
    forbidden = {window.id} | set(window.real_tickers)
    if has_token(text, forbidden):
        out.append(BG.Finding(g, "fail", f"[{kind}] the learned state contains the window id or a real ticker"))
    years = {window.real_start.year, window.real_end.year}
    for y in sorted(years):
        if re.search(rf"\b{y}\b", text):
            out.append(BG.Finding(g, "fail", f"[{kind}] the learned state mentions the real year {y}"))
    if re.search(r"real_|window|run_id|test_id|hidden|sealed|seed_year|tick|symbol", " ".join(map(str, vis.extra.keys())), re.I):
        out.append(BG.Finding(g, "fail", f"[{kind}] the learned state carries a window or real-date field"))
    return out


def blindness_summary(findings):
    fails = [f for f in findings if f.severity == "fail"]
    return {"passed": not fails, "n_findings": len(findings), "fails": [str(f) for f in fails[:8]],
            "warns": [str(f) for f in findings if f.severity == "warn"][:8]}


# ---------------------------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------------------------
def pick_hits(decisions, closes, horizon=MOVER_SESSIONS, touch=MOVER_TOUCH):
    """(direction_hit, mover_hit, n_picks) from the decisions and the close path. Direction: share of picked names whose
    close-to-close return to the NEXT decision (or `horizon` sessions for the last one) was positive. Mover: share whose
    close moved at least `touch` in either direction within `horizon` sessions. Close-to-close proxies (the system fills
    at the next open) - identical for both runs, so the deltas are fair. NaN when there is nothing to score."""
    idx = closes.index
    dates = [pd.Timestamp(d) for d, _ in decisions]
    up = dn = mv = n = 0
    for j, (d, names) in enumerate(decisions):
        d = pd.Timestamp(d)
        if d not in idx:
            continue
        i = idx.get_loc(d)
        i2 = idx.get_loc(dates[j + 1]) if j + 1 < len(decisions) and dates[j + 1] in idx else min(i + horizon, len(idx) - 1)
        if i2 <= i:
            continue
        for c in names:
            if c not in closes.columns:
                continue
            p0 = closes[c].iloc[i]
            if not np.isfinite(p0) or p0 <= 0:
                continue
            p1 = closes[c].iloc[i2]
            if not np.isfinite(p1):
                continue
            path = closes[c].iloc[i + 1:min(i + horizon, len(idx) - 1) + 1] / p0 - 1
            n += 1
            up += p1 / p0 - 1 > 0
            mv += bool(len(path)) and float(np.nanmax(np.abs(path.values))) >= touch
    if n == 0:
        return float("nan"), float("nan"), 0
    return up / n, mv / n, n


def max_drawdown(equity):
    e = np.asarray(equity, float)
    if len(e) == 0:
        return 0.0
    return float((e / np.maximum.accumulate(e) - 1).min())


def run_metrics(run, closes):
    """The seven comparison metrics of a Run (plus year_return). Empty runs give zeros, never an exception."""
    w = np.asarray(run.weekly, float)
    row = O.week_row(w) if len(w) else O.week_row([])
    dh, mh, n = pick_hits(run.decisions, closes)
    eq = np.asarray(run.equity, float)
    yr = float(eq[-1] / eq[0] - 1) if len(eq) > 1 and eq[0] > 0 else (float(np.prod(1 + w) - 1) if len(w) else 0.0)
    return {"mean_week": float(row["mean_week"]), "in_band": float(row["in_band"]), "worst5": float(row["worst5"]),
            "max_dd": max_drawdown(eq) if len(eq) > 1 else float(row["max_dd"]),
            "pos_share": float((w > 0).mean()) if len(w) else 0.0, "dir_hit": dh, "mover_hit": mh, "year_return": yr,
            "n_weeks": int(len(w)), "n_picks": int(n)}


def metric_delta(a, b):
    """b - a on every comparison metric (NaN if either side is NaN)."""
    return {m: float(b[m] - a[m]) if np.isfinite(a[m]) and np.isfinite(b[m]) else float("nan") for m in METRICS}


# ---------------------------------------------------------------------------------------------------------------
# players and learners
# ---------------------------------------------------------------------------------------------------------------
class ReplayPlayer:
    """The real system's adaptive layer: engine.adaptive.replay over an archived window's snapshots."""

    def __init__(self, replay_fn=None):
        if replay_fn is None:
            from . import adaptive as A
            replay_fn = A.replay
        self.replay = replay_fn

    def patch(self, snaps, visible):
        return snaps

    def play(self, pres, visible):
        snaps = self.patch({k: v for k, v in pres.snaps.items()}, visible)
        S = self.replay(visible.cfg, snaps, pres.closes, pres.bps, pres.divs, adaptive=True, meta=visible.meta,
                        opens=pres.opens, long_term=visible.ltm)
        res = S.result()
        eq = np.array([1000.0] + [v for _, v in S.days], float)             # Session starts with $1,000
        ep = S.adapter.mem.export() if S.adapter is not None else None
        return Run(np.array(S.weeks, float), eq, [(d, list(n)) for d, n in S.decisions], ep,
                   {"n_adaptations": len(res.get("adaptations", [])), "audit_digest": res.get("audit_digest"),
                    "audit_len": res.get("audit_len")}, snaps=snaps, closes=pres.closes)


class MemoryBankLearner:
    """The system's own long-term-memory learning (C34): this window's episodes join the bank. Nothing else changes."""
    name = "memory_bank"

    def __init__(self, max_rows=None):
        self.max_rows = max_rows

    def learn(self, state, run, ctx):
        ep = run.episodes
        new = LearnedState(dict(state.cfg), dict(state.meta), state.ltm, dict(state.extra), list(state.lineage))
        if ep is None or not len(ep):
            new.lineage.append(f"{self.name}: no episodes to add")
            return new
        add = ep[list(VISIBLE_LTM_COLS)].copy()
        add["real_end"] = str(ctx.real_end.date())
        add["source"] = ctx.window_id
        new.ltm = add if state.ltm is None or not len(state.ltm) else pd.concat([state.ltm, add], ignore_index=True)
        if self.max_rows and len(new.ltm) > self.max_rows:
            new.ltm = new.ltm.iloc[-self.max_rows:].reset_index(drop=True)
        new.lineage.append(f"{self.name}: +{len(add)} episodes from {ctx.window_id}")
        return new


class BasisLearner:
    """Optional basis update. `train_fn(run, ctx, state) -> (cfg, meta, info) | None` is the system's own search (for
    example livesim_loop2.train_basis behind its firewall); the harness only applies what it adopted."""
    name = "basis"

    def __init__(self, train_fn):
        self.train_fn = train_fn

    def learn(self, state, run, ctx):
        new = LearnedState(dict(state.cfg), dict(state.meta), state.ltm, dict(state.extra), list(state.lineage))
        out = self.train_fn(run, ctx, state)
        if out is None:
            new.lineage.append(f"{self.name}: kept")
            return new
        cfg, meta, info = out
        new.cfg, new.meta = dict(cfg), dict(meta)
        new.lineage.append(f"{self.name}: adopted ({info})")
        return new


class ChainLearner:
    def __init__(self, learners):
        self.learners = list(learners)
        self.name = "+".join(l.name for l in self.learners)

    def learn(self, state, run, ctx):
        for l in self.learners:
            state = l.learn(state, run, ctx)
        return state


class NullLearner:
    """A non-learner: the state after learning is the state before. Its delta must be zero."""
    name = "none"

    def learn(self, state, run, ctx):
        return LearnedState(dict(state.cfg), dict(state.meta), state.ltm, dict(state.extra), state.lineage + ["none: unchanged"])


def snapshot_key(snap, cols=None):
    """Content fingerprint of a snapshot: the market-context columns of its first row, rounded. Independent of names and
    dates by construction, so it survives any disguise - which is exactly why it is a memorisation channel."""
    cols = cols or [c for c in snap.columns if str(c).startswith("m_")]
    vals = np.round(np.array([float(snap[c].iloc[0]) if len(snap) and c in snap else 0.0 for c in cols]), 4)
    return hashlib.sha1(vals.tobytes() + str(len(snap)).encode()).hexdigest()[:16]


def realised_forward(snap, date, closes, horizon=5):
    """Realised close-to-close return of every row of `snap` over the next `horizon` sessions (NaN if unknown). HARNESS
    use only: it reads the future of `date`, which is what a memoriser plants and a legitimate learner never does."""
    idx = closes.index
    d = pd.Timestamp(date)
    if d not in idx:
        return np.full(len(snap), np.nan)
    i = idx.get_loc(d)
    j = min(i + horizon, len(idx) - 1)
    r = closes.iloc[j] / closes.iloc[i] - 1
    return r.reindex(snap.index).to_numpy(dtype="float64")


class MemorisingReplayPlayer(ReplayPlayer):
    """Planted memoriser control on real windows: when a snapshot's content fingerprint is in its table, it replaces the
    model scores by the realised outcome it stored (so the ordinary Session then picks the hindsight winners)."""

    def patch(self, snaps, visible):
        table = visible.extra.get("memo_table") or {}
        out = {}
        for k, s in snaps.items():
            r = table.get(snapshot_key(s))
            if r is not None and len(r) == len(s):
                s = s.copy()
                r = np.nan_to_num(np.asarray(r, float), nan=0.0)
                for c in ("mu_raw", "evidence"):
                    if c in s:
                        s[c] = r
                if "p_move" in s:
                    s["p_move"] = np.abs(r)
            out[k] = s
        return out


class MemoriserLearner:
    """Stores the realised outcome of every snapshot of the run it learns from, keyed by content."""
    name = "memoriser"

    def learn(self, state, run, ctx):
        table = dict(state.extra.get("memo_table", {}))
        for k, s in (run.snaps or {}).items():
            r = realised_forward(s, k, run.closes)
            table[snapshot_key(s)] = r
        extra = {**state.extra, "memo_table": table}
        return LearnedState(dict(state.cfg), dict(state.meta), state.ltm, extra, state.lineage + [f"memoriser: {len(table)} keys"])


# ---------------------------------------------------------------------------------------------------------------
# synthetic worlds (harness self-test and unit tests)
# ---------------------------------------------------------------------------------------------------------------
def synthetic_window(seed, wid, n_stocks=40, n_weeks=30, beta=0.02, noise=0.03, first="2150-01-05", with_ctx=True):
    """A window whose forward return really is beta * signal + noise (one shared law across windows), with the columns
    the real Session needs so the real adaptive layer can also play it. Prices compound one weekly move per 5 sessions."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(first, periods=5 * n_weeks + 6)
    codes = [f"S{k:04d}" for k in rng.permutation(n_stocks)]
    codes = sorted(codes)
    sig = rng.normal(size=(n_weeks, n_stocks))
    fwd = beta * sig + noise * rng.normal(size=(n_weeks, n_stocks))
    logp = np.zeros((len(sessions), n_stocks))
    for w in range(n_weeks):
        i0 = 5 * w + 4                                       # decisions on a week's last session (a Friday)
        step = np.log1p(fwd[w]) / 5.0
        for s in range(1, 6):
            logp[i0 + s] = logp[i0 + s - 1] + step
    logp[5 * n_weeks + 5:] = logp[5 * n_weeks + 4]
    closes = pd.DataFrame(100 * np.exp(logp), index=sessions, columns=codes)
    opens = closes.shift(1).fillna(closes.iloc[0])
    ctx = rng.normal(size=(n_weeks, 6))
    snaps = {}
    for w in range(n_weeks):
        d = sessions[5 * w + 4]
        df = pd.DataFrame({"sig": sig[w], "mu_raw": sig[w], "p_move": rng.uniform(size=n_stocks),
                           "evidence": sig[w], "vol20": rng.uniform(0.1, 0.5, n_stocks), "max20": rng.uniform(0.0, 0.2, n_stocks),
                           "log_dv": rng.uniform(15, 20, n_stocks), "ev_red_flag": 0.0, "ev_offering": 0.0,
                           "r5": rng.normal(0, 0.03, n_stocks), "e_dist_52wh": rng.uniform(size=n_stocks)}, index=pd.Index(codes, name="ticker"))
        for j, c in enumerate(["m_vix", "m_vix_term", "m_spy_ma200", "m_spy_ma50", "m_breadth", "m_dispersion"]):
            df[c] = ctx[w, j] if with_ctx else 0.0
        snaps[str(d.date())] = df
    divs = {c: "D%d" % (k % 6) for k, c in enumerate(codes)}
    real = pd.Timestamp("1990-01-01") + pd.Timedelta(days=int(seed % 5000))
    return Window(wid, snaps, closes, opens, 10.0, divs, real, real + pd.Timedelta(days=364),
                  real_tickers=frozenset({f"REAL{k}" for k in range(3)}))


def score_weekly(snaps, closes, score_fn, k=5, cost=0.0005):
    """Shared by the synthetic players: weekly top-k by score_fn(snapshot, date), hold 5 sessions, equal weight."""
    idx = closes.index
    weekly, equity, decisions, e = [], [1.0], [], 1.0
    for d, s in sorted(snaps.items()):
        t = pd.Timestamp(d)
        if t not in idx:
            continue
        i = idx.get_loc(t)
        if i + 5 >= len(idx):
            continue
        sc = pd.Series(np.asarray(score_fn(s, d), float), index=s.index)
        names = sorted(sc.sort_values(ascending=False, kind="mergesort").index[:k])
        r = closes.iloc[i + 5][names] / closes.iloc[i][names] - 1
        w = float(r.mean()) - 2 * cost
        weekly.append(w)
        e *= 1 + w
        equity.append(e)
        decisions.append((str(t.date()), names))
    return Run(np.array(weekly), np.array(equity), decisions, None, {}, snaps=snaps, closes=closes)


class SyntheticPlayer:
    """Picks the top-k by weight * sig; the weight comes from the learned state (extra['w'], initially wrong-signed)."""

    def __init__(self, k=5):
        self.k = k

    def play(self, pres, visible):
        w = float(visible.extra.get("w", -0.5))
        return score_weekly(pres.snaps, pres.closes, lambda s, d: w * s["sig"].to_numpy(), self.k)


class SyntheticGeneraliser:
    """Learns the true law: a shrunk regression of realised forward return on the signal, applied to any window."""
    name = "generaliser"

    def __init__(self, prior_n=50.0):
        self.prior_n = prior_n

    def learn(self, state, run, ctx):
        xs, ys = [], []
        for d, s in run.snaps.items():
            r = realised_forward(s, d, run.closes)
            ok = np.isfinite(r)
            xs.append(s["sig"].to_numpy()[ok])
            ys.append(r[ok])
        x, y = np.concatenate(xs), np.concatenate(ys)
        slope = float((x * y).sum() / ((x * x).sum() + self.prior_n * 1e-6)) if len(x) else 0.0
        w = float(np.sign(slope))                                         # only the direction of the law is kept
        return LearnedState(dict(state.cfg), dict(state.meta), state.ltm, {**state.extra, "w": w},
                            state.lineage + [f"generaliser: slope {slope:+.4f}"])


class SyntheticMemoriserPlayer:
    """Uses the stored table when a snapshot's content matches; otherwise the weight rule. A pure content memoriser."""

    def __init__(self, k=5):
        self.k = k

    def play(self, pres, visible):
        w = float(visible.extra.get("w", -0.5))
        table = visible.extra.get("memo_table") or {}

        def sc(s, d):
            r = table.get(snapshot_key(s))
            return np.nan_to_num(r, nan=0.0) if r is not None and len(r) == len(s) else w * s["sig"].to_numpy()
        return score_weekly(pres.snaps, pres.closes, sc, self.k)


# ---------------------------------------------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------------------------------------------
def _audit_and_play(player, window, pres, rec, state, kind, log):
    vis = state.visible()
    findings = audit_presentation(pres, window, rec, vis, kind)
    summ = blindness_summary(findings)
    log.append({"run": kind, **summ})
    if not summ["passed"]:
        raise BlindnessError("; ".join(summ["fails"]))
    return player.play(pres, vis)


def run_pair(W, B, player, learner, s0, seed, controls=True, log_fn=None):
    """One learning window W (and one transfer window B). Returns a plain-dict record; raises BlindnessError if any
    hand-over fails the audit, so a leaky run never enters an aggregate."""
    t0 = time.perf_counter()
    audits = []
    ctxW = LearnContext(W.id, W.real_start, W.real_end, W.era)
    P1 = archive_presentation(W)
    f1 = audit_visible(s0.visible(), W, "run1")                # the archive is run 1's presentation; its state is audited
    audits.append({"run": "run1", **blindness_summary(f1)})
    if not audits[-1]["passed"]:
        raise BlindnessError("; ".join(audits[-1]["fails"]))
    run1 = player.play(P1, s0.visible())
    m1 = run_metrics(run1, P1.closes)
    s1 = learner.learn(s0, run1, ctxW)
    rec = {"window": W.id, "transfer": None if B is None else B.id, "era": W.era, "real_start": str(W.real_start.date()),
           "learner": getattr(learner, "name", type(learner).__name__), "s0_episodes": s0.n_episodes(), "s1_episodes": s1.n_episodes(),
           "s0_fp": s0.fingerprint(), "s1_fp": s1.fingerprint(), "lineage": s1.lineage[-3:], "run1": m1}
    P2, d2 = make_presentation(W, derive_seed(seed, W.id, "run2"))
    rec["run2"] = run_metrics(_audit_and_play(player, W, P2, d2, s1, "run2", audits), P2.closes)
    if controls:
        P2c, d2c = make_presentation(W, derive_seed(seed, W.id, "noise"))
        rec["run2_noise"] = run_metrics(_audit_and_play(player, W, P2c, d2c, s0, "noise", audits), P2c.closes)
        P2r, d2r = make_presentation(W, derive_seed(seed, W.id, "shuffle"), order_preserving=False)
        vis = s0.visible()
        f = audit_presentation(P2r, W, d2r, vis, "shuffle", order_preserving_required=False)
        audits.append({"run": "shuffle", **blindness_summary(f)})
        if not audits[-1]["passed"]:
            raise BlindnessError("; ".join(audits[-1]["fails"]))
        rec["run2_shuffle"] = run_metrics(player.play(P2r, vis), P2r.closes)
    if B is not None:
        PB, dB = make_presentation(B, derive_seed(seed, W.id, B.id, "transfer"))
        rec["transfer_s0"] = run_metrics(_audit_and_play(player, B, PB, dB, s0, "transfer_s0", audits), PB.closes)
        rb1 = _audit_and_play(player, B, PB, dB, s1, "transfer_s1", audits)
        rec["transfer_s1"] = run_metrics(rb1, PB.closes)
        rec["transfer_anachronistic"] = bool(W.real_end >= B.real_start)     # learned from a year that comes AFTER the transfer year
    rec["blindness"] = audits
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    if log_fn:
        log_fn(f"pair {W.id}->{None if B is None else B.id}: mean_week {m1['mean_week']:+.4f} -> {rec['run2']['mean_week']:+.4f} "
               f"({rec['seconds']}s)")
    return rec


# ---------------------------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------------------------
def boot_ci(x, n_boot=2000, level=0.95, seed=0):
    """Percentile bootstrap CI of the mean over windows. Returns (mean, lo, hi, n); NaNs are dropped; n<2 gives an
    honest infinite interval rather than a fake one."""
    a = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if len(a) == 1:
        return float(a[0]), float("-inf"), float("inf"), 1
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    q = (1 - level) / 2
    return float(a.mean()), float(np.quantile(means, q)), float(np.quantile(means, 1 - q)), int(len(a))


def _col(recs, a, b, m):
    """Per-record delta of metric m between two sub-records (skips records lacking either)."""
    out = []
    for r in recs:
        if a in r and b in r:
            out.append(metric_delta(r[a], r[b])[m])
    return out


def aggregate(recs, seed=0, n_boot=2000):
    """Summaries with bootstrap CIs. Rows: same-year delta (run2-run1), noise delta (noise control-run1), shuffle delta,
    learning effect (same - noise, paired by window), transfer delta (B under S1 - S0), gap (learning effect - transfer,
    paired by pair). Also per-era means of the same-year delta."""
    out = {"n_pairs": len(recs), "metrics": {}, "by_era": {}}
    for m in METRICS:
        same = _col(recs, "run1", "run2", m)
        noise = _col(recs, "run1", "run2_noise", m)
        shuf = _col(recs, "run1", "run2_shuffle", m)
        trans = _col(recs, "transfer_s0", "transfer_s1", m)
        eff, gap = [], []
        for r in recs:
            if all(k in r for k in ("run1", "run2", "run2_noise")):
                e = metric_delta(r["run1"], r["run2"])[m] - metric_delta(r["run1"], r["run2_noise"])[m]
                eff.append(e)
                if "transfer_s0" in r:
                    gap.append(e - metric_delta(r["transfer_s0"], r["transfer_s1"])[m])
        row = {}
        for name, v in (("same", same), ("noise", noise), ("shuffle", shuf), ("effect", eff), ("transfer", trans), ("gap", gap)):
            mean, lo, hi, n = boot_ci(v, n_boot, seed=derive_seed(seed, m, name) % (2 ** 31))
            row[name] = {"mean": mean, "lo": lo, "hi": hi, "n": n,
                         "share_pos": float(np.mean([x > 0 for x in v if np.isfinite(x)])) if any(np.isfinite(x) for x in v) else float("nan")}
        for name in ("same", "effect", "transfer"):
            v = {"same": same, "effect": eff, "transfer": trans}[name]
            row[name]["p_signflip"] = signflip_p(v, seed=derive_seed(seed, m, name, "p") % (2 ** 31))
        row["luck_floor"] = float(np.nanmean(np.abs(shuf))) if len(shuf) else float("nan")
        row["run1_mean"] = float(np.nanmean([r["run1"][m] for r in recs])) if recs else float("nan")
        row["run2_mean"] = float(np.nanmean([r["run2"][m] for r in recs])) if recs else float("nan")
        out["metrics"][m] = row
    eras = sorted({r["era"] for r in recs})
    for e in eras:
        sub = [r for r in recs if r["era"] == e]
        out["by_era"][e] = {"n": len(sub), **{m: float(np.nanmean(_col(sub, "run1", "run2", m))) for m in ("mean_week", "in_band", "worst5")},
                            "transfer_mean_week": float(np.nanmean(_col(sub, "transfer_s0", "transfer_s1", "mean_week"))) if any("transfer_s0" in r for r in sub) else float("nan")}
    out["noise_max_abs"] = {m: float(np.nanmax(np.abs(_col(recs, "run1", "run2_noise", m)))) if _col(recs, "run1", "run2_noise", m) else float("nan")
                            for m in METRICS}
    out["by_bank"] = {}
    for label, sub in (("empty_bank", [r for r in recs if r["s0_episodes"] == 0]), ("with_bank", [r for r in recs if r["s0_episodes"] > 0])):
        if sub:
            out["by_bank"][label] = {"n": len(sub), "mean_week_same": float(np.nanmean(_col(sub, "run1", "run2", "mean_week"))),
                                     "mean_week_transfer": float(np.nanmean(_col(sub, "transfer_s0", "transfer_s1", "mean_week")))
                                     if any("transfer_s0" in r for r in sub) else float("nan")}
    out["anachronistic_share"] = float(np.mean([r.get("transfer_anachronistic", False) for r in recs if "transfer_s0" in r])) if any("transfer_s0" in r for r in recs) else float("nan")
    out["verdict"] = verdict(out["metrics"][PRIMARY], out["n_pairs"], out["noise_max_abs"][PRIMARY])
    fl = out["metrics"][PRIMARY].get("luck_floor")
    ef = out["metrics"][PRIMARY]["effect"]["mean"]
    if fl is not None and np.isfinite(fl) and np.isfinite(ef) and abs(ef) < fl:
        out["verdict"]["notes"].append(f"the learning effect ({ef:+.5f}) is smaller than the tie-break luck floor ({fl:.5f}) that a random "
                                       "relabel alone produces; any system that reads code-name order can fake a delta of that size")
    out["verdict_in_band"] = verdict(out["metrics"]["in_band"], out["n_pairs"], out["noise_max_abs"]["in_band"])
    out["headline"] = headline_transfer(recs, seed, n_boot)
    return out


def headline_transfer(recs, seed=0, n_boot=2000, min_pairs=3):
    """Owner ruling (C54 vs C56): the HEADLINE learning number is the TRANSFER delta - what learning does for years it has
    not seen. Past-only: only pairs whose transfer window starts AFTER the learning window ended count, so learning never
    used the future of the year it is applied to. The same-year disguised delta (aggregate()['metrics'][m]['same']) is
    reported second, with its caveat."""
    have = [r for r in recs if "transfer_s0" in r and "transfer_s1" in r]
    past = [r for r in have if not r.get("transfer_anachronistic", True)]
    out = {"n_with_transfer": len(have), "n_past_only": len(past), "n_dropped_anachronistic": len(have) - len(past), "metrics": {}}
    for m in METRICS:
        v = _col(past, "transfer_s0", "transfer_s1", m)
        mean, lo, hi, n = boot_ci(v, n_boot, seed=derive_seed(seed, m, "headline") % (2 ** 31))
        out["metrics"][m] = {"mean": mean, "lo": lo, "hi": hi, "n": n, "p_signflip": signflip_p(v, seed=derive_seed(seed, m, "headline", "p") % (2 ** 31))}
    t = out["metrics"][PRIMARY]
    if t["n"] < min_pairs:
        out["verdict"] = {"label": "INCONCLUSIVE", "why": f"only {t['n']} past-only transfer pairs (< {min_pairs})"}
    elif t["lo"] > 0:
        out["verdict"] = {"label": "TRANSFER_POSITIVE", "why": f"learning raised weekly mean on unseen later years by {t['mean']:+.4g} (CI {t['lo']:+.3g}..{t['hi']:+.3g})"}
    elif t["hi"] < 0:
        out["verdict"] = {"label": "TRANSFER_NEGATIVE", "why": f"learning lowered weekly mean on unseen later years: {t['mean']:+.4g} (CI {t['lo']:+.3g}..{t['hi']:+.3g})"}
    else:
        out["verdict"] = {"label": "NO_TRANSFER", "why": f"transfer CI {t['lo']:+.3g}..{t['hi']:+.3g} contains zero"}
    return out


def signflip_p(x, n_perm=4000, seed=0):
    """Two-sided sign-flip permutation p-value that the mean of paired deltas is zero (no distributional assumption).
    Exact enumeration for up to 12 pairs, seeded Monte Carlo above. NaN with no data; 1.0 when every delta is zero."""
    a = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(a) == 0:
        return float("nan")
    obs = abs(a.mean())
    if obs == 0:
        return 1.0
    if len(a) <= 12:
        signs = np.array([[1 if (i >> j) & 1 else -1 for j in range(len(a))] for i in range(2 ** len(a))])
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, len(a)))
    perm = np.abs((signs * a).mean(axis=1))
    return float((perm >= obs - 1e-15).mean())


def pair_table(recs):
    """One row per pair for the CSV and the report: the window, its era, and the primary metric on every play."""
    rows = []
    for r in recs:
        row = {"window": r["window"], "transfer": r.get("transfer"), "era": r["era"], "s0_episodes": r["s0_episodes"],
               "s1_episodes": r["s1_episodes"], "anachronistic": r.get("transfer_anachronistic")}
        for k in ("run1", "run2", "run2_noise", "run2_shuffle", "transfer_s0", "transfer_s1"):
            for m in ("mean_week", "in_band", "worst5"):
                row[f"{k}.{m}"] = r[k][m] if k in r else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(row, n, noise_max_abs=0.0, min_pairs=3, memo_ratio=0.5):
    """One label for one metric. MEMORISATION needs the learning effect to be significantly positive AND the transfer
    gain to be under `memo_ratio` of it AND the gap CI above zero. GENERALISING needs a significantly positive transfer.
    Fewer than `min_pairs` pairs is INCONCLUSIVE whatever the point estimates say."""
    eff, tr, gap = row["effect"], row["transfer"], row["gap"]
    notes = []
    if noise_max_abs is not None and np.isfinite(noise_max_abs) and noise_max_abs > NOISE_TOL:
        notes.append(f"disguise noise is not zero (max |delta| {noise_max_abs:.2g}): something reads identity or ties break differently")
    if n < min_pairs or eff["n"] < min_pairs:
        return {"label": "INCONCLUSIVE", "why": f"only {n} pairs (< {min_pairs})", "notes": notes}
    eff_pos = eff["lo"] > 0
    trans_pos = tr["n"] >= min_pairs and tr["lo"] > 0
    memo = (eff_pos and tr["n"] >= min_pairs and np.isfinite(gap["lo"]) and gap["lo"] > 0 and tr["mean"] < memo_ratio * eff["mean"])
    if memo:
        return {"label": "MEMORISATION", "why": f"same-year effect {eff['mean']:+.4g} (CI {eff['lo']:+.3g}..{eff['hi']:+.3g}) far above "
                f"transfer {tr['mean']:+.4g}; gap CI {gap['lo']:+.3g}..{gap['hi']:+.3g}", "notes": notes}
    if trans_pos:
        return {"label": "GENERALISING", "why": f"transfer gain {tr['mean']:+.4g} (CI {tr['lo']:+.3g}..{tr['hi']:+.3g}) above zero", "notes": notes}
    if eff["hi"] < 0 or (tr["n"] >= min_pairs and tr["hi"] < 0):
        return {"label": "HARMFUL", "why": f"learning lowered the metric: effect {eff['mean']:+.4g}, transfer {tr['mean']:+.4g}", "notes": notes}
    if eff["lo"] <= 0 <= eff["hi"] and (tr["n"] < min_pairs or tr["lo"] <= 0 <= tr["hi"]):
        return {"label": "NO_EFFECT", "why": f"effect CI {eff['lo']:+.3g}..{eff['hi']:+.3g} and transfer CI contain zero", "notes": notes}
    return {"label": "UNCLEAR", "why": f"effect {eff['mean']:+.4g}, transfer {tr['mean']:+.4g} (CIs do not settle it)", "notes": notes}


# ---------------------------------------------------------------------------------------------------------------
# harness self-check: the harness must be able to see what it claims to measure
# ---------------------------------------------------------------------------------------------------------------
def harness_selfcheck(seed=0, n_windows=8, log_fn=None):
    """Three planted learners on synthetic worlds with one shared law. The harness is VALID only if the generaliser is
    labelled GENERALISING, the memoriser is labelled MEMORISATION (large same-year effect, ~0 transfer), the non-learner
    is NO_EFFECT with zero delta, and a planted leak makes the blindness audit fail."""
    s0 = LearnedState({}, {})
    wins = [synthetic_window(derive_seed(seed, "w", i), f"syn{i:02d}") for i in range(n_windows)]
    trans = [synthetic_window(derive_seed(seed, "t", i), f"tra{i:02d}") for i in range(n_windows)]
    res = {}
    for name, player, learner in (("generaliser", SyntheticPlayer(), SyntheticGeneraliser()),
                                  ("memoriser", SyntheticMemoriserPlayer(), MemoriserLearner()),
                                  ("non_learner", SyntheticPlayer(), NullLearner())):
        recs = [run_pair(w, t, player, learner, s0, derive_seed(seed, name), log_fn=None) for w, t in zip(wins, trans)]
        for w in wins + trans:
            w.used_shifts.clear()
            w.used_codes = set(w.all_codes())
        agg = aggregate(recs, seed)
        res[name] = {"verdict": agg["verdict"]["label"], "same": agg["metrics"][PRIMARY]["same"]["mean"],
                     "effect": agg["metrics"][PRIMARY]["effect"]["mean"], "transfer": agg["metrics"][PRIMARY]["transfer"]["mean"]}
        if log_fn:
            log_fn(f"selfcheck {name}: {res[name]}")
    leak = leak_probe(wins[0], seed)
    res["leak_detected"] = leak
    ok = (res["generaliser"]["verdict"] == "GENERALISING" and res["memoriser"]["verdict"] == "MEMORISATION"
          and res["non_learner"]["verdict"] == "NO_EFFECT" and abs(res["non_learner"]["same"]) < 1e-12 and leak)
    res["valid"] = bool(ok)
    return res


def leak_probe(window, seed=0):
    """Plant an id in an attribute, a real ticker as a code, a real-era date and a repeated shift; the audit must fail
    each time. True when every plant is caught (and a clean hand-over passes)."""
    caught = []
    pres, rec = make_presentation(window, derive_seed(seed, "probe"))
    clean = not [f for f in audit_presentation(pres, window, rec, LearnedState({}, {}).visible(), "probe") if f.severity == "fail"]
    k0 = next(iter(pres.snaps))
    pres.snaps[k0].attrs["window_id"] = window.id
    caught.append(any(f.severity == "fail" for f in audit_presentation(pres, window, rec)))
    pres.snaps[k0].attrs.clear()
    real = pres.closes.copy()
    real.index = real.index - pd.Timedelta(days=int(7 * 52 * 150))
    bad = Presentation(pres.snaps, real, pres.opens, pres.bps, pres.divs)
    caught.append(any(f.severity == "fail" for f in audit_presentation(bad, window, rec)))
    tick = sorted(window.real_tickers)[0]
    cl = pres.closes.copy()
    cl.columns = [tick] + list(cl.columns[1:])
    caught.append(any(f.severity == "fail" for f in audit_presentation(Presentation(pres.snaps, cl, pres.opens, pres.bps, pres.divs), window, rec)))
    vis = Visible({}, {}, None, {"real_end": "2016-12-31"})
    caught.append(any(f.severity == "fail" for f in audit_visible(vis, window)))
    return bool(clean and all(caught))


# ---------------------------------------------------------------------------------------------------------------
# real-window plumbing
# ---------------------------------------------------------------------------------------------------------------
def revealed_pool(live_dir=None):
    """Revealed windows only: cycles.json entries with revealed_year, and loop2.json windows with a `revealed` period.
    A window not yet revealed is never listed (its seal file is not even read). Returns dicts id/dir/real_start/real_end."""
    live_dir = Path(live_dir or K.STATE / "livesim")
    out = {}
    cyc = live_dir / "cycles.json"
    if cyc.exists():
        for c in json.loads(cyc.read_text()).get("cycles", []):
            y = c.get("revealed_year")
            if y:
                out[c["run_id"]] = {"id": c["run_id"], "dir": live_dir / c["run_id"], "real_start": pd.Timestamp(f"{int(y)}-01-01"),
                                    "real_end": pd.Timestamp(f"{int(y)}-12-31"), "source": "cycles"}
    l2 = live_dir / "loop2.json"
    if l2.exists():
        for w in json.loads(l2.read_text()).get("windows", []):
            rv = w.get("revealed")
            if rv and " - " in str(rv):
                a, b = str(rv).split(" - ")
                s = pd.to_datetime(a, format="%b %Y")
                e = pd.to_datetime(b, format="%b %Y") + pd.offsets.MonthEnd(0)
                out[w["run_id"]] = {"id": w["run_id"], "dir": live_dir / w["run_id"], "real_start": s, "real_end": e, "source": "loop2"}
    return [out[k] for k in sorted(out)]


def load_window(entry):
    """Load an archived revealed window from disk (weekly snapshots, closes/opens, cost, industry divisions)."""
    from . import policy
    a = Path(entry["dir"])
    files = sorted(a.glob("wsnap_*.parquet")) or sorted(a.glob("snap_*.parquet"))
    if not files:
        raise FileNotFoundError(f"{a}: no weekly snapshots")
    snaps = {f.stem.split("_", 1)[1]: pd.read_parquet(f) for f in files}
    cp = a / "closes_v2.parquet" if (a / "closes_v2.parquet").exists() else a / "closes.parquet"
    closes = pd.read_parquet(cp)
    opens = pd.read_parquet(a / "opens_v2.parquet") if (a / "opens_v2.parquet").exists() else None
    sic = pd.read_parquet(a / "sic.parquet")
    bps = float(json.loads((a / "meta.json").read_text())["cost_bps"])
    divs = {t: policy.sic_division(x) for t, x in zip(sic["ticker"], sic["sic"])}
    first, last = closes.index.min(), closes.index.max()
    snaps = {k: v for k, v in snaps.items() if first <= pd.Timestamp(k) <= last}
    return Window(entry["id"], snaps, closes, opens, bps, divs, entry["real_start"], entry["real_end"])


def causal_bank(bank, real_start):
    """C34 on the harness side: the part of the memory bank whose windows ended before `real_start`."""
    if bank is None or not len(bank):
        return None
    b = bank[pd.to_datetime(bank["real_end"]) < pd.Timestamp(real_start)]
    if not len(b):
        return None
    b = b[list(VISIBLE_LTM_COLS) + ["real_end"]].copy()
    b["source"] = "bank"
    return b.reset_index(drop=True)


def choose_pairs(pool, n_pairs, seed, min_year_gap=3):
    """Pick (learning window, transfer window) pairs: seeded, no window used twice as a learning window, transfer window at
    least `min_year_gap` calendar years away (so the two markets do not overlap), and preferring a transfer year that
    comes AFTER the learning year (learning must not use the future of the transfer year)."""
    rng = np.random.default_rng(seed)
    ids = [p["id"] for p in pool]
    order = list(rng.permutation(len(pool)))
    used_w, pairs = set(), []
    for i in order:
        if len(pairs) >= n_pairs:
            break
        w = pool[i]
        cand = [p for p in pool if p["id"] != w["id"] and abs(p["real_start"].year - w["real_start"].year) >= min_year_gap
                and p["id"] not in {q[1]["id"] for q in pairs}]
        later = [p for p in cand if p["real_start"] > w["real_end"]]
        use = later or cand
        if not use:
            continue
        pairs.append((w, use[int(rng.integers(len(use)))]))
        used_w.add(w["id"])
    return pairs


def free_ram_gb():
    import psutil
    return psutil.virtual_memory().available / 1e9


def wait_for_ram(min_gb=2.5, poll_s=60, give_up_s=1200, avail=free_ram_gb, sleep=time.sleep, log_fn=None):
    """Block until `min_gb` is free; False if it never is within `give_up_s` (the caller then reports, not runs)."""
    waited = 0
    while avail() < min_gb:
        if waited >= give_up_s:
            return False
        if log_fn:
            log_fn(f"free RAM {avail():.2f} GB < {min_gb} GB; waiting")
        sleep(poll_s)
        waited += poll_s
    return True


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def _f(x, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a" if x is None or not (isinstance(x, float) and np.isinf(x)) else ("-inf" if x < 0 else "inf")
    return f"{x * 100:+.3f}%" if pct else f"{x:+.5f}"


def render_report(summary):
    """Markdown report: verdict first, then the table the owner reads (C54), controls, per-era, blindness and caveats."""
    a, sc = summary["aggregate"], summary["selfcheck"]
    v = a["verdict"]
    h = a.get("headline")
    L = [f"# Learning delta - {summary['tag']}", ""]
    if h:
        t = h["metrics"][PRIMARY]
        L += ["## HEADLINE: transfer delta (learning applied to unseen years, past-only)", "",
              f"**{h['verdict']['label']}** - {h['verdict']['why']}", "",
              f"Weekly-mean transfer delta {_f(t['mean'])} [{_f(t['lo'])}, {_f(t['hi'])}] over {t['n']} past-only pairs "
              f"({h['n_dropped_anachronistic']} of {h['n_with_transfer']} transfer pairs dropped because the learning window came AFTER the "
              f"transfer year); sign-flip p {_f(t['p_signflip'])}.",
              "Other metrics: " + "; ".join(f"{m} {_f(r['mean'])}" for m, r in h["metrics"].items() if m != PRIMARY), "",
              "## Second: same-year disguised delta (C54/C55) - read with the caveat below", "",
              "Caveat: the same real year replayed under a new disguise is the same numbers with new labels. Any stored numeric memory "
              "recognises it (the paths are identical), so this delta can be memorisation, not learning; owner to rule (C54 vs C56). "
              "It is not the headline.", ""]
    L += [f"Same-year verdict on weekly mean: **{v['label']}** - {v['why']}",
         f"Verdict on share of weeks in the 5-10% band: **{a['verdict_in_band']['label']}** - {a['verdict_in_band']['why']}", ""]
    for n in v["notes"] + a["verdict_in_band"]["notes"]:
        L.append(f"- NOTE: {n}")
    L += ["", f"Harness self-check (planted generaliser / memoriser / non-learner / leak): "
          f"**{'VALID' if sc['valid'] else 'INVALID - no verdict below may be used'}**",
          f"- generaliser: {sc['generaliser']['verdict']}; memoriser: {sc['memoriser']['verdict']} (same-year {sc['memoriser']['same']:+.4f}, "
          f"transfer {sc['memoriser']['transfer']:+.4f}); non-learner: {sc['non_learner']['verdict']}; leak probe caught: {sc['leak_detected']}", "",
          f"Pairs: {a['n_pairs']}. Learner: {summary['learner']}. S0 basis version {summary.get('basis_version')}. "
          f"Transfer learned-from-the-future share: {a['anachronistic_share']:.0%}.", "",
          "## Learning delta per metric (Run 2 - Run 1; 95% bootstrap CI over windows)", "",
          "| metric | run1 | run2 | same-year delta | noise ctrl | shuffle ctrl | learning effect | transfer | gap |", "|---|---|---|---|---|---|---|---|---|"]
    for m, row in a["metrics"].items():
        cell = lambda k: f"{_f(row[k]['mean'])} [{_f(row[k]['lo'])}, {_f(row[k]['hi'])}]" if row[k]["n"] > 1 else _f(row[k]["mean"])
        L.append(f"| {m} | {_f(row['run1_mean'])} | {_f(row['run2_mean'])} | {cell('same')} | {cell('noise')} | {cell('shuffle')} | "
                 f"{cell('effect')} | {cell('transfer')} | {cell('gap')} |")
    L += ["", "Learning effect = same-year delta minus the no-learning rerun delta (disguise noise removed). Gap = learning effect "
          "minus transfer gain; a gap above zero with transfer near zero is MEMORISATION.", "", "## By era of the learning window", "",
          "| era | n | mean_week delta | in_band delta | worst5 delta | transfer mean_week |", "|---|---|---|---|---|---|"]
    for e, r in a["by_era"].items():
        L.append(f"| {e} | {r['n']} | {_f(r['mean_week'])} | {_f(r['in_band'])} | {_f(r['worst5'])} | {_f(r['transfer_mean_week'])} |")
    if a["by_bank"]:
        L += ["", "## By size of the starting memory bank", "", "| S0 bank | n | same-year mean_week delta | transfer mean_week delta |", "|---|---|---|---|"]
        for k, r in a["by_bank"].items():
            L.append(f"| {k} | {r['n']} | {_f(r['mean_week_same'])} | {_f(r['mean_week_transfer'])} |")
    pm = a["metrics"][PRIMARY]
    L += ["", f"Sign-flip p-values on mean_week: same-year {_f(pm['same']['p_signflip'])}, learning effect {_f(pm['effect']['p_signflip'])}, "
          f"transfer {_f(pm['transfer']['p_signflip'])}. Tie-break luck floor (mean |random relabel - run 1|): {_f(pm['luck_floor'])}."]
    if summary.get("pair_rows"):
        L += ["", "## Per pair (mean_week)", "", "| window | era | bank | run1 | run2 | noise ctrl | transfer S0 | transfer S1 |", "|---|---|---|---|---|---|---|---|"]
        for r in summary["pair_rows"]:
            L.append(f"| {r['window']} | {r['era']} | {r['s0_episodes']} | {_f(r['run1.mean_week'])} | {_f(r['run2.mean_week'])} | "
                     f"{_f(r['run2_noise.mean_week'])} | {_f(r['transfer_s0.mean_week'])} | {_f(r['transfer_s1.mean_week'])} |")
    mem = summary.get("memoriser_control")
    if mem:
        mm = mem["aggregate"]["metrics"][PRIMARY]
        L += ["", "## Memoriser control on real windows", "",
              f"Verdict {mem['aggregate']['verdict']['label']}: same-year delta {_f(mm['same']['mean'])}, transfer {_f(mm['transfer']['mean'])} "
              f"over {mem['aggregate']['n_pairs']} pairs. This must read as large-and-near-zero for the harness to be able to see memorisation."]
    L += ["", "## Blindness (C55)", "", f"Hand-overs audited: {summary['blindness']['n_audits']}; failures: {summary['blindness']['n_failed']}. "
          "Every player hand-over passed the structural, date, name and same-year-fidelity checks; a failing hand-over would abort the pair.",
          "", "## What this does not prove", ""]
    L += [f"- {x}" for x in summary["caveats"]]
    return "\n".join(L) + "\n"


CAVEATS = [
    "Runs replay archived weekly snapshots (engine.adaptive.replay): the return model is NOT retrained, so learning that lives in model "
    "fitting is not measured here. Only the adaptive layer's own learning (long-term memory bank, optional basis update) is.",
    "Snapshot features carry the market context (m_* columns), unchanged by any disguise. The memory bank matches on that context, so "
    "recognising a year through it is possible and is exactly what the transfer control and the planted memoriser exist to expose.",
    "Direction and mover hit rates use close-to-close proxies over the holding period, identical in both runs; they are not the fill-accurate figures.",
    "The learning windows are the revealed cycle years; their snapshots came from the model that was trained before them, so Run 1 is an "
    "honest blind play, but transfer windows learned from a LATER year are marked anachronistic and are a generalisation test, not a live-time claim.",
    "With few pairs the bootstrap intervals are wide; INCONCLUSIVE is the correct label until they narrow.",
]
