"""Pattern break engine (contract C62 section 10; checklist D11; canon C60/C61). Bible phase serving: pattern reliability (phase 9/10).

When an important knowledge item degrades, compare the population of periods in which it WORKED with the population in which it
FAILED across every dimension section 10 lists (feature distributions, regime, sector composition, stock type, volatility,
liquidity, trend, breadth, macro, neighbouring patterns, interaction structure, outcome magnitude, outcome duration, timing,
concentration, sample size, missingness, data quality); decide with a family-wise bar (max-|t| under circularly shifted labels,
which keeps the serial dependence of both populations) whether any difference is more than chance; turn the surviving
differences into candidate CONDITIONS; and test each condition out of sample on data strictly later than the data that proposed
it. A condition that fails either bar is never reported as the cause: the verdict is UNKNOWN and no condition is attached.

Two kinds of column exist. CAUSAL columns are known at the decision (context columns shifted by their publication lag,
trailing neighbour statistics, timing). DESCRIPTIVE columns (the item's own outcome size / run length, trailing or not, this
period's neighbour outcomes, time since its own break) differ between the populations by construction; they are reported as
symptoms and can never become a condition. Every context column is scanned for future leaks (declared negative lag, near-rank-identity with this period's
outcome, implausible separating power) and a leak raises FirewallBreach: fail closed.

Time convention (same as engine.pattern_reliability): row p is a decision at the close of period p; its outcome `value` (signed,
positive = the item worked) is realised over p -> p+1 and is known from row p+1. At `as_of` only rows with index <= as_of exist and
the outcome of the last of them is hidden.

Built on engine.pattern_reliability (`break_states` causal working->broken state machine and CUSUM onsets, `holm`, `auc`,
`nw_t`); lifted from Timelines/patterns to any item that supplies an `ItemSeries`. IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine import pattern_reliability as PR

from .core import FailureCause, FirewallBreach, Subsystem, stable_hash

DIMENSIONS = ("feature_distribution", "regime", "sector_composition", "stock_type", "volatility", "liquidity", "trend",
              "breadth", "macro", "neighbouring_patterns", "interaction_structure", "outcome_magnitude", "outcome_duration",
              "timing", "concentration", "sample_size", "missingness", "data_quality")

# a difference found on a dimension is evidence for this cause; the cause is only ever reported with a validated condition
CAUSE_BY_DIMENSION = {
    "feature_distribution": FailureCause.WRONG_CONTEXT, "regime": FailureCause.REGIME_CHANGE,
    "sector_composition": FailureCause.WRONG_CONTEXT, "stock_type": FailureCause.WRONG_CONTEXT,
    "volatility": FailureCause.REGIME_CHANGE, "liquidity": FailureCause.REGIME_CHANGE, "trend": FailureCause.REGIME_CHANGE,
    "breadth": FailureCause.REGIME_CHANGE, "macro": FailureCause.REGIME_CHANGE,
    "neighbouring_patterns": FailureCause.INTERACTION_FAILURE, "interaction_structure": FailureCause.INTERACTION_FAILURE,
    "outcome_magnitude": FailureCause.WEAKENING_EFFECT, "outcome_duration": FailureCause.TIMING_ERROR,
    "timing": FailureCause.TIMING_ERROR, "concentration": FailureCause.SELECTION_ERROR,
    "sample_size": FailureCause.INSUFFICIENT_EVIDENCE, "missingness": FailureCause.MEASUREMENT_ERROR,
    "data_quality": FailureCause.MEASUREMENT_ERROR,
}
SUBSYSTEM_BY_CAUSE = {
    FailureCause.REGIME_CHANGE: Subsystem.SELECTION, FailureCause.WRONG_CONTEXT: Subsystem.SELECTION,
    FailureCause.TIMING_ERROR: Subsystem.TIMING, FailureCause.SELECTION_ERROR: Subsystem.SELECTION,
    FailureCause.RISK_ERROR: Subsystem.RISK, FailureCause.INTERACTION_FAILURE: Subsystem.SELECTION,
}
# causal design columns that are descriptive by construction (they contain this period's realised outcome)
DESCRIPTIVE_DIMENSIONS = ("outcome_magnitude", "outcome_duration")

PARAMS = {
    "break_win": 13, "work_win": 26, "t_break": 1.5, "t_work": 1.0, "t_recover": 0.5,      # PR.break_states
    "mode": "window",               # window: centred-window t of the outcomes | state: causal episodes | outcome: winning vs losing rows
    "label_half": 8, "label_t": 1.5,
    "min_failed": 12, "min_success": 24, "min_side": 8, "min_confirm": 40,
    "disc_frac": 0.6, "embargo": 4,
    "alpha": 0.10,                  # family-wise bar for a difference between the populations
    "oos_alpha": 0.05,              # one-sided bar for the condition on later data (Holm across candidates)
    "n_perm": 300,
    "cut_quantiles": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    "max_candidates": 4, "max_cond_terms": 2, "pair_gain": 0.5,
    "cover_lo": 0.10, "cover_hi": 0.90,
    "max_pair_cols": 10, "trail": 8, "corr_win": 13,
    "peek_auc": 0.78, "peek_rho": 0.55, "peek_ratio": 1.8, "peek_hard_rho": 0.9,
    "wf_step": 13, "wf_min_train": 60,
    "boot": 400, "boot_block": 6,
    "placebo_n": 12,                # shifted-placebo runs behind an EXPLAINED verdict (family-wise control, W-05)
    "placebo_alpha": 0.10,          # the real condition must out-score the placebos: (1 + #placebos as strong) / (n + 1) <= this
}
PHRASES = {"regime": "the market regime", "volatility": "the volatility level", "liquidity": "market liquidity",
           "trend": "the market trend", "breadth": "market breadth", "macro": "the macro backdrop",
           "sector_composition": "the sector mix", "stock_type": "the type of stock", "timing": "the time of year",
           "missingness": "how much data was missing", "data_quality": "data quality", "concentration": "how concentrated it was",
           "sample_size": "the number of names", "neighbouring_patterns": "what neighbouring patterns were doing",
           "interaction_structure": "an interaction between two features", "feature_distribution": "a feature level",
           "outcome_magnitude": "recent outcome size", "outcome_duration": "recent outcome streaks"}


class BreakInputError(ValueError):
    pass


def _cfg(cfg) -> dict:
    return {**PARAMS, **(cfg or {})}


# ------------------------------------------------------------------------------------------------- data model

@dataclasses.dataclass(frozen=True)
class ContextColumn:
    """One context column of an item's frame. `lag` = periods between the moment the column describes and the moment it is
    published: the value stamped at row p is usable from row p + lag (0 = known at the close of row p). A negative lag is a
    declared future leak."""
    name: str
    dimension: str
    kind: str = "numeric"              # numeric | binary | categorical
    lag: int = 0

    def check(self) -> list[str]:
        errs = []
        if self.dimension not in DIMENSIONS:
            errs.append(f"{self.name}: unknown dimension {self.dimension!r}")
        if self.kind not in ("numeric", "binary", "categorical"):
            errs.append(f"{self.name}: unknown kind {self.kind!r}")
        if self.lag < 0:
            errs.append(f"{self.name}: negative lag {self.lag} (future information)")
        return errs


@dataclasses.dataclass
class ItemSeries:
    """Everything the break engine needs about one knowledge item: a frame indexed by strictly increasing decision dates with a
    `value` column (signed outcome) and the declared context columns; optional neighbours (other items' signed outcomes on
    the same index)."""
    item_id: str
    frame: pd.DataFrame
    columns: tuple = ()
    neighbours: pd.DataFrame | None = None
    compositions: dict = dataclasses.field(default_factory=dict)   # group -> (dimension, shares frame on the same index)

    def validate(self) -> list[str]:
        errs = []
        f = self.frame
        if "value" not in f.columns:
            errs.append("frame has no 'value' column")
        if len(f) == 0:
            errs.append("empty frame")
        if not f.index.is_monotonic_increasing or not f.index.is_unique:
            errs.append("index must be strictly increasing")
        seen = set()
        for c in self.columns:
            errs.extend(c.check())
            if c.name not in f.columns:
                errs.append(f"declared column {c.name!r} not in frame")
            if c.name in seen:
                errs.append(f"duplicate column {c.name!r}")
            seen.add(c.name)
        if "value" in f.columns and np.isinf(f["value"].astype(float).values).any():
            errs.append("infinite values")
        if self.neighbours is not None and not self.neighbours.index.equals(f.index):
            errs.append("neighbours must share the frame index")
        for g, (dim, shares) in self.compositions.items():
            if dim not in ("sector_composition", "stock_type", "concentration"):
                errs.append(f"composition {g!r}: dimension {dim!r} is not a composition dimension")
            if not shares.index.equals(f.index):
                errs.append(f"composition {g!r} must share the frame index")
            elif (shares.astype(float).values < 0).any():
                errs.append(f"composition {g!r} has negative shares")
        return errs

    def require_valid(self):
        errs = self.validate()
        if errs:
            raise BreakInputError("; ".join(errs))

    def n_visible(self, as_of) -> int:
        """Rows that exist at `as_of` (index <= as_of)."""
        if as_of is None:
            return len(self.frame)
        return int((self.frame.index <= as_of).sum())

    def n_matured(self, as_of) -> int:
        """Rows whose outcome is known at `as_of` (the last visible row's outcome is still in the future)."""
        return max(self.n_visible(as_of) - 1, 0)

    def visible(self, as_of) -> "ItemSeries":
        """A copy holding only what existed at `as_of`, with the newest outcome hidden."""
        n = self.n_visible(as_of)
        fr = self.frame.iloc[:n].copy()
        if n:
            fr.iloc[n - 1, fr.columns.get_loc("value")] = np.nan
        nb = None if self.neighbours is None else self.neighbours.iloc[:n].copy()
        if nb is not None and n:
            nb.iloc[n - 1] = np.nan
        comps = {g: (d, sh.iloc[:n].copy()) for g, (d, sh) in self.compositions.items()}
        return ItemSeries(self.item_id, fr, self.columns, nb, comps)

    def column_map(self) -> dict:
        return {c.name: c for c in self.columns}


# ------------------------------------------------------------------------------------------------- future-leak canary

@dataclasses.dataclass(frozen=True)
class LeakFinding:
    column: str
    reason: str
    score: float


def _spearman(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float("nan")
    ra, rb = sps.rankdata(a[ok]), sps.rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def scan_context_leaks(item: ItemSeries, as_of=None, cfg=None) -> list[LeakFinding]:
    """Canary for future information in the context columns. A column is a finding when (a) it declares a negative lag, (b) it
    is nearly rank-identical to this period's outcome, or (c) it is far tighter to this period's outcome than to the outcome one
    period either side AND its rank correlation or separating AUC is implausibly high. A persistent legitimate regime variable
    correlates about equally with the neighbouring outcomes, so (c) does not fire on it."""
    P = _cfg(cfg)
    vis = item.visible(as_of)
    m = vis.n_matured(None)
    y = vis.frame["value"].values[:m].astype(float)
    out = []
    for col in item.columns:
        if col.lag < 0:
            out.append(LeakFinding(col.name, "declared negative lag", float(col.lag)))
            continue
        if col.name not in vis.frame.columns or col.kind == "categorical":
            continue
        x = vis.frame[col.name].astype(float).shift(col.lag).values[:m]
        r0 = _spearman(x, y)
        if not np.isfinite(r0):
            continue
        rm = _spearman(x[1:], y[:-1])
        rp = _spearman(x[:-1], y[1:])
        near = max(abs(v) for v in (rm, rp) if np.isfinite(v)) if np.isfinite(rm) or np.isfinite(rp) else 0.0
        ok = np.isfinite(x) & np.isfinite(y)
        a = PR.auc(x[ok], y[ok] > 0) if ok.sum() >= 20 else float("nan")
        a = max(a, 1 - a) if np.isfinite(a) else 0.5
        if abs(r0) >= P["peek_hard_rho"]:
            out.append(LeakFinding(col.name, "nearly rank-identical to this period's outcome", abs(r0)))
        elif (abs(r0) >= P["peek_rho"] or a >= P["peek_auc"]) and abs(r0) >= P["peek_ratio"] * max(near, 0.05):
            out.append(LeakFinding(col.name, "far tighter to this period's outcome than to its neighbours", abs(r0)))
    return out


def assert_no_leak(item: ItemSeries, as_of=None, cfg=None) -> None:
    found = scan_context_leaks(item, as_of, cfg)
    if found:
        raise FirewallBreach("future information in context columns: " +
                             "; ".join(f"{f.column} ({f.reason}, {f.score:.2f})" for f in found))


def audit_context_builder(builder: Callable[[pd.DataFrame], pd.DataFrame], raw: pd.DataFrame, cuts: Sequence[int],
                          atol: float = 1e-9) -> list[tuple[int, str]]:
    """Truncation test: the context row a builder produces at position c from raw data cut at c must equal the row it produces
    at c from the full data. Returns the (cut, column) pairs that differ, i.e. columns that peek at later rows."""
    full = builder(raw)
    bad = []
    for c in cuts:
        part = builder(raw.iloc[: c + 1])
        for col in full.columns:
            a, b = float(part[col].iloc[-1]), float(full[col].iloc[c])
            if not (math.isnan(a) and math.isnan(b)) and not math.isclose(a, b, rel_tol=1e-7, abs_tol=atol):
                bad.append((int(c), str(col)))
    return bad


# ------------------------------------------------------------------------------------------------- episodes and populations

@dataclasses.dataclass(frozen=True)
class Episode:
    detect: int                # row at which the break became knowable
    onset: int                 # estimated first row of the failure (CUSUM inside the two windows before detection)
    recover: int | None        # row at which the item was working again, if it was

    def end(self, n: int) -> int:
        return n if self.recover is None else min(self.recover, n)

    def length(self, n: int) -> int:
        return max(self.end(n) - self.onset, 0)


def find_episodes(item: ItemSeries, as_of=None, cfg=None) -> tuple[list, np.ndarray]:
    """Causal working -> broken episodes for the item (engine.pattern_reliability.break_states on its outcome series) and the
    per-row state code (0 unproven, 1 working, 2 broken). Episodes detected after `as_of` cannot appear."""
    P = _cfg(cfg)
    vis = item.visible(as_of)
    R = vis.frame[["value"]].astype(float)
    if len(R) < 3:
        return [], np.zeros(len(R), np.int8)
    states, ev = PR.break_states(R, {k: P[k] for k in ("break_win", "work_win", "t_break", "t_work", "t_recover")})
    eps = []
    for _, r in ev.iterrows():
        rec = None if not np.isfinite(r["recover"]) else int(r["recover"])
        eps.append(Episode(int(r["detect"]), int(r["onset"]), rec))
    return eps, states.iloc[:, 0].values.astype(np.int8)


@dataclasses.dataclass(frozen=True)
class Populations:
    success: np.ndarray
    failed: np.ndarray
    mode: str
    n_matured: int
    excluded: int

    def enough(self, P) -> str | None:
        if len(self.failed) < P["min_failed"]:
            return f"only {len(self.failed)} failed rows (need {P['min_failed']})"
        if len(self.success) < P["min_success"]:
            return f"only {len(self.success)} successful rows (need {P['min_success']})"
        return None


def window_labels(v: np.ndarray, half: int, t_bar: float, limit: int) -> np.ndarray:
    """Retrospective stretch labels using only rows < `limit`: +1 where the centred window's mean outcome has t >= t_bar
    (the item was working), -1 where t <= -t_bar (it was failing), 0 in between. Legitimate for a post-mortem, and `limit`
    is what keeps the discovery labels from reading the confirmation window."""
    lab = np.zeros(limit, np.int8)
    for p in range(limit):
        seg = v[max(0, p - half):min(limit, p + half + 1)]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 5:
            continue
        sd = seg.std(ddof=1)
        if sd <= 1e-12:
            continue
        t = seg.mean() / (sd / math.sqrt(len(seg)))
        lab[p] = 1 if t >= t_bar else (-1 if t <= -t_bar else 0)
    return lab


def split_populations(item: ItemSeries, episodes: Sequence[Episode], states: np.ndarray, as_of=None, cfg=None,
                      rows: np.ndarray | None = None) -> Populations:
    """Successful vs failed rows among the matured rows (optionally restricted to `rows`, which also bounds what a label may
    read). window mode: centred-window stretches (see window_labels). state mode: failed = rows inside a broken episode from
    its onset, successful = working state outside every episode. outcome mode: failed = outcome <= 0, successful = > 0."""
    P = _cfg(cfg)
    m = item.n_matured(as_of)
    allowed = np.zeros(m, bool)
    if rows is None:
        allowed[:] = True
    else:
        r = np.asarray(rows, int)
        allowed[r[r < m]] = True
    limit = int(np.flatnonzero(allowed).max()) + 1 if allowed.any() else 0
    v = item.frame["value"].values[:m].astype(float)
    if P["mode"] == "outcome":
        good = np.isfinite(v) & allowed
        return Populations(np.flatnonzero(good & (v > 0)), np.flatnonzero(good & (v <= 0)), "outcome", m, int((~good).sum()))
    if P["mode"] == "window":
        lab = np.zeros(m, np.int8)
        lab[:limit] = window_labels(v, P["label_half"], P["label_t"], limit)
        succ, fail = (lab == 1) & allowed, (lab == -1) & allowed
        return Populations(np.flatnonzero(succ), np.flatnonzero(fail), "window", m, int(m - succ.sum() - fail.sum()))
    in_ep = np.zeros(m, bool)
    for e in episodes:
        in_ep[e.onset:e.end(m)] = True
    st = np.zeros(m, np.int8)
    st[: min(m, len(states))] = states[:m]
    fail = in_ep & allowed
    succ = (st == 1) & ~in_ep & allowed
    return Populations(np.flatnonzero(succ), np.flatnonzero(fail), "state", m, int(m - succ.sum() - fail.sum()))


# ------------------------------------------------------------------------------------------------- design matrix

@dataclasses.dataclass(frozen=True)
class ColTag:
    dimension: str
    causal: bool
    source: str = ""


@dataclasses.dataclass
class Design:
    X: pd.DataFrame
    tags: dict

    def columns(self, causal: bool | None = None, dimension: str | None = None) -> list[str]:
        out = []
        for c in self.X.columns:
            t = self.tags[c]
            if causal is not None and t.causal != causal:
                continue
            if dimension is not None and t.dimension != dimension:
                continue
            out.append(c)
        return out

    def dimensions_present(self) -> set:
        return {t.dimension for t in self.tags.values()}


def _expanding_fill(s: pd.Series) -> pd.Series:
    """NaN -> mean of the strictly earlier rows (point in time); rows with no history fall back to 0."""
    return s.fillna(s.expanding(min_periods=1).mean().shift(1)).fillna(0.0)


def _expanding_z(s: pd.Series, min_periods: int = 20) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean().shift(1)
    sd = s.expanding(min_periods=min_periods).std().shift(1).replace(0.0, np.nan)
    return ((s - mu) / sd).fillna(0.0)


def _run_lengths(sign: np.ndarray) -> np.ndarray:
    """Length of the run of equal signs that ends at each row (NaN sign = 0 and breaks the run)."""
    out = np.zeros(len(sign))
    for i, s in enumerate(sign):
        out[i] = 0 if s == 0 else (out[i - 1] + 1 if i and sign[i - 1] == s else 1)
    return out


def _full_run_lengths(sign: np.ndarray) -> np.ndarray:
    """Length of the whole run that contains each row (uses the future of that run: descriptive only)."""
    out = np.zeros(len(sign))
    i = 0
    while i < len(sign):
        j = i
        while j + 1 < len(sign) and sign[j + 1] == sign[i]:
            j += 1
        out[i:j + 1] = 0 if sign[i] == 0 else j - i + 1
        i = j + 1
    return out


def build_design(item: ItemSeries, as_of=None, cfg=None, episodes: Sequence[Episode] | None = None) -> Design:
    """One row per matured period, every column point in time (see module docstring). Covers all 18 dimensions for which the
    item supplies columns; the derived dimensions (outcome magnitude / duration, timing, missingness, neighbouring patterns,
    interaction structure) come for free."""
    P = _cfg(cfg)
    vis = item.visible(as_of)
    m = vis.n_matured(None)
    fr = vis.frame
    idx = fr.index
    cols: dict[str, pd.Series] = {}
    tags: dict[str, ColTag] = {}

    def add(name, series, dim, causal, source=""):
        s = pd.Series(np.asarray(series, float), index=idx)
        cols[name] = s
        tags[name] = ColTag(dim, causal, source or name)

    v = fr["value"].astype(float)
    for c in item.columns:
        raw = fr[c.name]
        if c.kind == "categorical":
            lab = raw.shift(c.lag)
            for lvl in sorted(lab.dropna().astype(str).unique()):
                ind = (lab.astype(str) == lvl).astype(float)
                if 0 < ind.iloc[:m].sum() < m:
                    add(f"{c.name}={lvl}", ind, c.dimension, True, c.name)
            continue
        x = raw.astype(float).shift(c.lag)
        if x.iloc[:m].isna().any():
            add(f"{c.name}__missing", x.isna().astype(float), "missingness", True, c.name)
            x = _expanding_fill(x)
        add(c.name, x, c.dimension, True, c.name)

    # outcome magnitude and duration are symptoms: the item's own outcome persistence (trailing versions included) restates
    # that it is failing rather than saying why, so none of these can become a condition (B27: own hit rate is a symptom)
    tr = P["trail"]
    add("abs_value", v.abs().fillna(0.0), "outcome_magnitude", False)
    add("loss_size", (-v.clip(upper=0.0)).fillna(0.0), "outcome_magnitude", False)
    add("abs_value_trail", v.abs().shift(1).rolling(tr, min_periods=3).mean().fillna(0.0), "outcome_magnitude", False)
    sign = np.sign(v.fillna(0.0).values)
    add("run_len", _full_run_lengths(sign), "outcome_duration", False)
    add("streak_trail", pd.Series(_run_lengths(sign), index=idx).shift(1).fillna(0.0) * pd.Series(sign, index=idx).shift(1).fillna(0.0),
        "outcome_duration", False)

    # timing
    if isinstance(idx, pd.DatetimeIndex):
        ang = 2 * math.pi * (idx.month.values - 1) / 12.0
        add("month_sin", np.sin(ang), "timing", True)
        add("month_cos", np.cos(ang), "timing", True)
    else:
        ang = 2 * math.pi * (np.arange(len(idx)) % 13) / 13.0
        add("phase_sin", np.sin(ang), "timing", True)
        add("phase_cos", np.cos(ang), "timing", True)
    add("age", np.arange(len(idx)), "timing", True)
    if episodes is not None:
        since = np.full(len(idx), np.nan)
        last = None
        det = sorted(e.detect for e in episodes)
        k = 0
        for i in range(len(idx)):
            while k < len(det) and det[k] <= i:
                last = det[k]
                k += 1
            since[i] = i - last if last is not None else i + 1
        add("since_break", since, "timing", False)      # a function of the very episodes that define the populations

    # neighbouring patterns
    if vis.neighbours is not None and vis.neighbours.shape[1] > 0:
        nb = vis.neighbours.astype(float)
        nb_mean = nb.mean(axis=1)
        add("nb_mean_now", nb_mean.fillna(0.0), "neighbouring_patterns", False)
        add("nb_mean_trail", nb_mean.shift(1).rolling(4, min_periods=2).mean().fillna(0.0), "neighbouring_patterns", True)
        add("nb_disp_trail", nb.std(axis=1).shift(1).rolling(4, min_periods=2).mean().fillna(0.0), "neighbouring_patterns", True)
        cw = P["corr_win"]
        corr = v.shift(1).rolling(cw, min_periods=cw // 2).corr(nb_mean.shift(1))
        add("nb_corr_trail", corr.fillna(0.0), "neighbouring_patterns", True)

    # composition groups (sector mix, stock type, concentration): the shares themselves, their concentration and entropy
    for g, (dim, shares) in vis.compositions.items():
        sh = shares.astype(float).clip(lower=0.0)
        tot = sh.sum(axis=1).replace(0.0, np.nan)
        pr = sh.div(tot, axis=0).fillna(0.0)
        for k in pr.columns:
            add(f"{g}:{k}", pr[k], dim, True, g)
        add(f"{g}__hhi", (pr * pr).sum(axis=1), "concentration", True, g)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(pr * np.log(pr.where(pr > 0, 1.0))).sum(axis=1)
        add(f"{g}__entropy", ent, dim, True, g)

    # interaction structure: pairwise products of standardised (past-only) causal numeric context columns
    base = [c.name for c in item.columns if c.kind != "categorical" and c.name in cols and c.dimension != "missingness"]
    base = base[: P["max_pair_cols"]]
    z = {b: _expanding_z(cols[b]) for b in base}
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            add(f"{a}*{b}", z[a] * z[b], "interaction_structure", True, f"{a}*{b}")

    X = pd.DataFrame(cols).iloc[:m].astype(float)
    return Design(X, tags)


# ------------------------------------------------------------------------------------------------- statistics

def welch_columns(Xs: np.ndarray, Xf: np.ndarray) -> np.ndarray:
    """Welch t (success minus failed) of every column; constant columns get 0."""
    ns, nf = len(Xs), len(Xf)
    if ns < 2 or nf < 2:
        return np.zeros(Xs.shape[1])
    se = np.sqrt(Xs.var(0, ddof=1) / ns + Xf.var(0, ddof=1) / nf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (Xs.mean(0) - Xf.mean(0)) / se
    t[~np.isfinite(t)] = 0.0
    t[se <= 1e-12] = 0.0
    return t


def _welch_from_labels(Xall: np.ndarray, tot: np.ndarray, totq: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """|t| for every column when `lab` (bool) marks the failed rows - by sums, so a permutation costs one matrix product."""
    nf = int(lab.sum())
    ns = len(lab) - nf
    if nf < 2 or ns < 2:
        return np.zeros(Xall.shape[1])
    lf = lab.astype(float)
    s1 = lf @ Xall
    q1 = lf @ (Xall * Xall)
    mf = s1 / nf
    ms = (tot - s1) / ns
    vf = np.maximum((q1 - nf * mf * mf) / (nf - 1), 0.0)
    vs = np.maximum(((totq - q1) - ns * ms * ms) / (ns - 1), 0.0)
    se = np.sqrt(vs / ns + vf / nf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.abs(ms - mf) / se
    t[~np.isfinite(t)] = 0.0
    t[se <= 1e-12] = 0.0
    return t


def contrast_table(X: pd.DataFrame, pops: Populations, tags: Mapping[str, ColTag], columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Per-column comparison of the successful and failed populations: means, standardised difference, Welch t and normal p,
    Kolmogorov-Smirnov distance, variance ratio and the AUC of the column at separating the populations."""
    cols = list(columns) if columns is not None else list(X.columns)
    Xs = X.iloc[pops.success][cols].values.astype(float)
    Xf = X.iloc[pops.failed][cols].values.astype(float)
    t = welch_columns(Xs, Xf)
    rows = []
    for j, c in enumerate(cols):
        a, b = Xs[:, j], Xf[:, j]
        sd = math.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) if len(a) > 1 and len(b) > 1 else 0.0
        ks = sps.ks_2samp(a, b) if len(a) > 1 and len(b) > 1 and sd > 1e-12 else None
        vb = b.var(ddof=1) if len(b) > 1 else 0.0
        rows.append({"column": c, "dimension": tags[c].dimension, "causal": tags[c].causal,
                     "mean_success": float(a.mean()), "mean_failed": float(b.mean()),
                     "smd": (float(a.mean() - b.mean()) / sd) if sd > 1e-12 else 0.0, "t": float(t[j]),
                     "p_normal": PR._p_two(t[j]), "ks": float(ks.statistic) if ks is not None else 0.0,
                     "var_ratio": float(b.var(ddof=1) / a.var(ddof=1)) if len(a) > 1 and a.var(ddof=1) > 1e-12 and vb >= 0 else float("nan"),
                     "auc": PR.auc(np.r_[a, b], np.r_[np.ones(len(a), bool), np.zeros(len(b), bool)]) if sd > 1e-12 else 0.5})
    return pd.DataFrame(rows, columns=["column", "dimension", "causal", "mean_success", "mean_failed", "smd", "t", "p_normal",
                                       "ks", "var_ratio", "auc"])


def family_wise_p(X: pd.DataFrame, pops: Populations, columns: Sequence[str], n_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(observed |t|, family-wise p) for `columns`: p_j = (1 + #{shifts whose MAX over ALL columns of |t| >= |t_j|}) / (n_perm + 1).
    Labels are circularly shifted along time within the union of both populations, so each population keeps its block
    structure and the null respects serial dependence. The max is over the whole searched family."""
    rows = np.sort(np.r_[pops.success, pops.failed])
    lab = np.isin(rows, pops.failed)
    Xall = X.iloc[rows][list(columns)].values.astype(float)
    tot, totq = Xall.sum(0), (Xall * Xall).sum(0)
    obs = _welch_from_labels(Xall, tot, totq, lab)
    rng = np.random.default_rng(seed)
    n = len(rows)
    lo = max(int(0.05 * n), 2)
    mx = np.empty(n_perm)
    for b in range(n_perm):
        k = int(rng.integers(lo, max(n - lo, lo + 1)))
        mx[b] = _welch_from_labels(Xall, tot, totq, np.roll(lab, k)).max()
    p = (1.0 + (mx[None, :] >= obs[:, None]).sum(axis=1)) / (n_perm + 1.0)
    return obs, p


def dimension_summary(table: pd.DataFrame, fw_p: Mapping[str, float], alpha: float) -> list[dict]:
    """One row per contract dimension: how many columns were compared, the strongest, its family-wise p, and a verdict.
    A dimension with no column is 'not_measured' - never silently 'no difference'."""
    out = []
    for d in DIMENSIONS:
        sub = table[table["dimension"] == d]
        if len(sub) == 0:
            out.append({"dimension": d, "n_columns": 0, "best_column": None, "best_abs_t": 0.0, "p_family": 1.0,
                        "verdict": "not_measured"})
            continue
        j = int(np.argmax(np.abs(sub["t"].values)))
        best = sub.iloc[j]
        p = float(fw_p.get(best["column"], 1.0))
        verdict = "differs" if p <= alpha else "no_difference_detected"
        if not bool(best["causal"]) and verdict == "differs":
            verdict = "symptom"
        out.append({"dimension": d, "n_columns": int(len(sub)), "best_column": str(best["column"]),
                    "best_abs_t": float(abs(best["t"])), "p_family": p, "verdict": verdict})
    return out


# ------------------------------------------------------------------------------------------------- conditions

@dataclasses.dataclass(frozen=True)
class Term:
    column: str
    op: str                    # ">=" | "<="
    cut: float

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        x = X[self.column].values.astype(float)
        with np.errstate(invalid="ignore"):
            return (x >= self.cut) if self.op == ">=" else (x <= self.cut)


@dataclasses.dataclass(frozen=True)
class Condition:
    """The item works when EVERY term holds. Built only from causal design columns."""
    terms: tuple
    dimensions: tuple = ()

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(X), bool)
        for t in self.terms:
            m &= t.mask(X)
        return m

    def describe(self) -> str:
        parts = []
        for t, d in zip(self.terms, self.dimensions or [""] * len(self.terms)):
            what = f" ({PHRASES.get(d, d)})" if d else ""
            parts.append(f"{t.column} {'at or above' if t.op == '>=' else 'at or below'} {t.cut:.4g}{what}")
        return " and ".join(parts)

    @property
    def key(self) -> str:
        return stable_hash([(t.column, t.op, round(t.cut, 8)) for t in self.terms], 12)


@dataclasses.dataclass(frozen=True)
class ConditionTest:
    """A condition on rows it never saw."""
    n_in: int
    n_out: int
    mean_in: float
    mean_out: float
    hit_in: float
    hit_out: float
    coverage: float
    t_diff: float
    p_one: float
    t_in: float
    gain: float                # mean of (value if condition holds else 0) minus mean value: what gating adds per row
    passes: bool
    why: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def evaluate_condition(cond: Condition, X: pd.DataFrame, values: np.ndarray, rows: np.ndarray, cfg=None,
                       p_override: float | None = None) -> ConditionTest:
    """Apply `cond` to `rows` (positions into X / values) and test that the item does better where it holds. Passing needs
    both sides populated, a positive mean where it holds, a better mean than where it does not, coverage away from 0 and 1,
    and a one-sided p under the OOS bar (`p_override` lets the caller substitute a Holm-adjusted p)."""
    P = _cfg(cfg)
    rows = np.asarray(rows, int)
    v = values[rows].astype(float)
    ok = np.isfinite(v)
    hold = cond.mask(X.iloc[rows])[ok]
    v = v[ok]
    n_in, n_out = int(hold.sum()), int((~hold).sum())
    nan = float("nan")
    if n_in < 2 or n_out < 2:
        return ConditionTest(n_in, n_out, nan, nan, nan, nan, n_in / max(len(v), 1), 0.0, 1.0, 0.0, 0.0, False,
                             "too few rows on one side")
    a, b = v[hold], v[~hold]
    se = math.sqrt(a.var(ddof=1) / n_in + b.var(ddof=1) / n_out)
    t = (a.mean() - b.mean()) / se if se > 1e-12 else 0.0
    p_one = float(sps.norm.sf(t))
    t_in = PR.nw_t(a, 4)
    cover = n_in / len(v)
    gain = float((v * hold).mean() - v.mean())
    p_use = p_one if p_override is None else p_override
    why = ""
    if n_in < P["min_side"] or n_out < P["min_side"]:
        why = "a side has fewer than min_side rows"
    elif not (P["cover_lo"] <= cover <= P["cover_hi"]):
        why = f"coverage {cover:.2f} outside [{P['cover_lo']}, {P['cover_hi']}]"
    elif a.mean() <= 0:
        why = "item does not work where the condition holds"
    elif a.mean() <= b.mean():
        why = "no better where the condition holds"
    elif p_use > P["oos_alpha"]:
        why = f"p={p_use:.3f} above the OOS bar {P['oos_alpha']}"
    return ConditionTest(n_in, n_out, float(a.mean()), float(b.mean()), float((a > 0).mean()), float((b > 0).mean()), cover,
                         float(t), p_one, float(t_in) if np.isfinite(t_in) else 0.0, gain, why == "", why)


def best_cut(x: np.ndarray, v: np.ndarray, op: str, cfg=None) -> tuple[float, float]:
    """(cut, Welch t of value where the term holds vs not) maximising the t over the quantile grid. NaN when no cut leaves
    min_side rows on both sides."""
    P = _cfg(cfg)
    ok = np.isfinite(x) & np.isfinite(v)
    x, v = x[ok], v[ok]
    if len(x) < 2 * P["min_side"]:
        return float("nan"), float("-inf")
    best = (float("nan"), float("-inf"))
    for q in sorted(set(np.quantile(x, P["cut_quantiles"]))):
        hold = (x >= q) if op == ">=" else (x <= q)
        a, b = v[hold], v[~hold]
        if len(a) < P["min_side"] or len(b) < P["min_side"]:
            continue
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = (a.mean() - b.mean()) / se if se > 1e-12 else 0.0
        if t > best[1]:
            best = (float(q), float(t))
    return best


def propose_conditions(design: Design, values: np.ndarray, disc_rows: np.ndarray, table: pd.DataFrame,
                       fw_p: Mapping[str, float], cfg=None) -> list[tuple[Condition, float]]:
    """Candidate conditions from the causal columns whose contrast survived the family-wise bar. Direction comes from the sign of
    the population contrast; the cut from the discovery rows only. The top singles are then paired when the conjunction
    improves the value contrast by at least `pair_gain` t-units. Returns (condition, discovery t) best first."""
    P = _cfg(cfg)
    X = design.X
    keep = table[(table["causal"]) & (table["column"].map(lambda c: fw_p.get(c, 1.0)) <= P["alpha"])]
    keep = keep.reindex(keep["t"].abs().sort_values(ascending=False).index)
    v = values[disc_rows].astype(float)
    singles: list[tuple[Condition, float]] = []
    for _, r in keep.iterrows():
        col = r["column"]
        op = ">=" if r["mean_success"] >= r["mean_failed"] else "<="
        cut, t = best_cut(X[col].values[disc_rows].astype(float), v, op, P)
        if np.isfinite(cut) and t > 0:
            singles.append((Condition((Term(col, op, cut),), (design.tags[col].dimension,)), t))
        if len(singles) >= P["max_candidates"]:
            break
    out = list(singles)
    if P["max_cond_terms"] >= 2:
        for i in range(len(singles)):
            for j in range(i + 1, len(singles)):
                c1, c2 = singles[i][0], singles[j][0]
                if c1.terms[0].column == c2.terms[0].column:
                    continue
                cond = Condition(c1.terms + c2.terms, c1.dimensions + c2.dimensions)
                hold = cond.mask(X.iloc[disc_rows])
                a, b = v[hold & np.isfinite(v)], v[~hold & np.isfinite(v)]
                if len(a) < P["min_side"] or len(b) < P["min_side"]:
                    continue
                se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
                t = (a.mean() - b.mean()) / se if se > 1e-12 else 0.0
                if t >= max(singles[i][1], singles[j][1]) + P["pair_gain"]:
                    out.append((cond, float(t)))
    out.sort(key=lambda z: -z[1])
    return out


# ------------------------------------------------------------------------------------------------- walk-forward harness

@dataclasses.dataclass(frozen=True)
class WalkForwardResult:
    hold: np.ndarray           # bool per evaluated row (NaN rows excluded)
    rows: np.ndarray
    gain: float                # mean of gated value minus mean ungated value
    t_gain: float
    mean_gated: float
    mean_all: float
    n_refits: int
    stable_cut: float          # sd of the refit cuts relative to the spread of the column (0 = same cut every time)


def walk_forward_condition(design: Design, values: np.ndarray, column: str, op: str, rows: np.ndarray, cfg=None) -> WalkForwardResult:
    """Refit the single-column cut on every earlier row and apply it to the next `wf_step` rows, over `rows` (positions in
    increasing order). This is the harness that asks: if a person had re-learned this condition as data arrived, would gating
    on it have helped? Every cut uses only rows strictly before the ones it is applied to."""
    P = _cfg(cfg)
    rows = np.sort(np.asarray(rows, int))
    x = design.X[column].values.astype(float)
    hold_all: list[bool] = []
    kept: list[int] = []
    cuts = []
    n_ref = 0
    pos = P["wf_min_train"]
    while pos < len(rows):
        train, test = rows[:pos], rows[pos:pos + P["wf_step"]]
        cut, t = best_cut(x[train], values[train].astype(float), op, P)
        n_ref += 1
        if np.isfinite(cut):
            cuts.append(cut)
            h = (x[test] >= cut) if op == ">=" else (x[test] <= cut)
        else:
            h = np.ones(len(test), bool)
        for r, hh in zip(test, h):
            if np.isfinite(values[r]):
                hold_all.append(bool(hh))
                kept.append(int(r))
        pos += P["wf_step"]
    if not kept:
        return WalkForwardResult(np.zeros(0, bool), np.zeros(0, int), 0.0, 0.0, float("nan"), float("nan"), n_ref, float("nan"))
    hold = np.array(hold_all)
    kr = np.array(kept)
    v = values[kr].astype(float)
    gated = v * hold
    diff = gated - v
    se = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else 0.0
    spread = np.nanstd(x[rows]) or 1.0
    return WalkForwardResult(hold, kr, float(diff.mean()), float(diff.mean() / se) if se > 1e-12 else 0.0,
                             float(v[hold].mean()) if hold.any() else float("nan"), float(v.mean()), n_ref,
                             float(np.std(cuts) / spread) if len(cuts) > 1 else 0.0)


def block_bootstrap_gain(values: np.ndarray, hold: np.ndarray, seed: int = 0, cfg=None) -> tuple[float, float, float]:
    """(gain, lo, hi) of gating by `hold` on the rows given, by the stationary bootstrap of the per-row gain."""
    P = _cfg(cfg)
    gain = np.asarray(values, float) * np.asarray(hold, float) - np.asarray(values, float)
    return PR.stationary_bootstrap_ci(gain, block=P["boot_block"], n_boot=P["boot"], seed=seed)


def threshold_stability(design: Design, values: np.ndarray, cond: Condition, rows: np.ndarray, cfg=None,
                        shifts: Sequence[float] = (-0.1, -0.05, 0.05, 0.1)) -> list[dict]:
    """How the OOS verdict moves when each cut is nudged by a fraction of the column's spread. A condition that only passes
    at its exact fitted cut is a curve fit, and shows up here as passes flipping."""
    P = _cfg(cfg)
    out: list[dict] = []
    if not getattr(cond, "terms", ()):
        return out                        # a fitted state has no single cut to nudge
    for s in shifts:
        terms = []
        for t in cond.terms:
            sd = float(np.nanstd(design.X[t.column].values[rows])) or 1.0
            terms.append(Term(t.column, t.op, t.cut + s * sd))
        c2 = Condition(tuple(terms), cond.dimensions)
        r = evaluate_condition(c2, design.X, values, rows, P)
        out.append({"shift": s, "passes": r.passes, "gain": r.gain, "coverage": r.coverage})
    return out


# ------------------------------------------------------------------------------------------------- the explanation

@dataclasses.dataclass(frozen=True)
class BreakExplanation:
    item_id: str
    as_of: str
    status: str                        # EXPLAINED | UNKNOWN | INSUFFICIENT_EVIDENCE | NO_BREAK
    cause: FailureCause
    subsystem: str | None
    condition: Condition | None
    oos: ConditionTest | None
    statement: str
    dimension_summary: tuple
    symptoms: tuple
    n_success: int
    n_failed: int
    n_confirm: int
    n_tested: int                      # size of the searched family (multiplicity actually paid)
    episodes: tuple
    candidates_tried: int
    detectable_smd: float = float("nan")      # smallest standardised difference this sample could have detected (power)
    compositions: tuple = ()
    best_t: float = float("-inf")             # largest out-of-sample t of any candidate condition tried (-inf: none reached the test)
    placebo: tuple = ()                       # (runs, runs at least as strong, p) of the shifted-placebo control, when it was run

    @property
    def explained(self) -> bool:
        return self.status == "EXPLAINED"

    @property
    def explanation_id(self) -> str:
        return stable_hash([self.item_id, self.as_of, self.status, self.condition.key if self.condition else None], 16)

    def as_dict(self) -> dict:
        return {"explanation_id": self.explanation_id, "item_id": self.item_id, "as_of": self.as_of, "status": self.status,
                "cause": self.cause.value, "subsystem": self.subsystem,
                "condition": None if self.condition is None else self.condition.describe(),
                "oos": None if self.oos is None else self.oos.as_dict(), "statement": self.statement,
                "dimension_summary": list(self.dimension_summary), "symptoms": list(self.symptoms),
                "n_success": self.n_success, "n_failed": self.n_failed, "n_confirm": self.n_confirm,
                "n_tested": self.n_tested, "candidates_tried": self.candidates_tried,
                "detectable_smd": self.detectable_smd, "compositions": list(self.compositions),
                "episodes": [dataclasses.asdict(e) for e in self.episodes]}


def _empty(item, as_of, status, why, summary=(), n_s=0, n_f=0, n_conf=0, n_tested=0, eps=(), sym=(), tried=0, power=float("nan"),
           comps=(), best_t=float("-inf")):
    return BreakExplanation(item.item_id, str(as_of), status, FailureCause.UNKNOWN, None, None, None, why, tuple(summary),
                            tuple(sym), n_s, n_f, n_conf, n_tested, tuple(eps), tried, power, tuple(comps), best_t)


def explain_break(item: ItemSeries, as_of=None, cfg=None, seed: int = 0) -> BreakExplanation:
    """The section-10 pipeline for one item at `as_of`:
    1 validate + future-leak canary (raises FirewallBreach), 2 episodes and populations, 3 chronological split into a
    discovery window and a later confirmation window (with an embargo), 4 contrast of the populations on every dimension in the
    discovery window with a family-wise bar, 5 candidate conditions from the causal survivors, 6 out-of-sample test of every
    candidate on the confirmation window with Holm across candidates. Anything short of both bars is UNKNOWN."""
    P = _cfg(cfg)
    item.require_valid()
    assert_no_leak(item, as_of, P)
    eps, states = find_episodes(item, as_of, P)
    m = item.n_matured(as_of)
    if not eps and P["mode"] != "outcome":
        return _empty(item, as_of, "NO_BREAK", "no break was detected in the item's matured history")
    design = build_design(item, as_of, P, eps)
    values = item.frame["value"].values[:m].astype(float)
    n_disc = int(P["disc_frac"] * m)
    disc_rows = np.arange(0, n_disc)
    conf_rows = np.arange(min(n_disc + P["embargo"], m), m)
    pops = split_populations(item, eps, states, as_of, P, rows=disc_rows)
    short = pops.enough(P)
    if short:
        return _empty(item, as_of, "INSUFFICIENT_EVIDENCE", f"discovery window: {short}", n_s=len(pops.success),
                      n_f=len(pops.failed), n_conf=len(conf_rows), eps=eps)
    if len(conf_rows) < P["min_confirm"]:
        return _empty(item, as_of, "INSUFFICIENT_EVIDENCE",
                      f"only {len(conf_rows)} rows after the discovery window (need {P['min_confirm']} to confirm)",
                      n_s=len(pops.success), n_f=len(pops.failed), n_conf=len(conf_rows), eps=eps)
    fam = {}
    tables = []
    for causal in (True, False):
        cols = design.columns(causal=causal)
        if not cols:
            continue
        tab = contrast_table(design.X, pops, design.tags, cols)
        _, p = family_wise_p(design.X, pops, cols, P["n_perm"], seed + (0 if causal else 1))
        fam.update(dict(zip(cols, p)))
        tables.append(tab)
    table = pd.concat(tables, ignore_index=True)
    summary = dimension_summary(table, fam, P["alpha"])
    symptoms = tuple({"column": r["column"], "dimension": r["dimension"], "smd": float(r["smd"]), "p_family": float(fam[r["column"]])}
                     for _, r in table[~table["causal"]].iterrows() if fam.get(r["column"], 1.0) <= P["alpha"])
    n_tested = int(sum(1 for c in design.X.columns if design.tags[c].causal))
    power = detectable_smd(len(pops.success), len(pops.failed), n_tested, P["alpha"])
    comps = tuple(composition_shift(design, g, pops, P["n_perm"], seed + 7) for g in item.compositions)
    cands: list[tuple[Any, float]] = propose_conditions(design, values, disc_rows, table, fam, P)
    cands = cands + propose_state_conditions(design, values, disc_rows, table, fam, P, seed)
    if not cands:
        return _empty(item, as_of, "UNKNOWN",
                      f"no difference between the working and failed populations beat the family-wise bar across {n_tested} "
                      f"causal comparisons (smallest standardised difference detectable here: {power:.2f}); cause unknown", summary, len(pops.success), len(pops.failed), len(conf_rows),
                      n_tested, eps, symptoms, 0, power, comps)
    tests = [evaluate_condition(c, design.X, values, conf_rows, P) for c, _ in cands]
    padj = PR.holm([t.p_one for t in tests])
    tests = [evaluate_condition(c, design.X, values, conf_rows, P, p_override=float(pa)) for (c, _), pa in zip(cands, padj)]
    good = [(c, t) for (c, _), t in zip(cands, tests) if t.passes]
    best_t = float(max((t.t_diff for t in tests), default=float("-inf")))
    if not good:
        return _empty(item, as_of, "UNKNOWN",
                      f"{len(cands)} candidate condition(s) survived discovery but none held out of sample "
                      f"({'; '.join(t.why for t in tests)}); cause unknown (smallest standardised difference detectable here: "
                      f"{power:.2f})", summary, len(pops.success), len(pops.failed),
                      len(conf_rows), n_tested, eps, symptoms, len(cands), power, comps, best_t)
    cond, test = max(good, key=lambda z: z[1].gain)
    dim = cond.dimensions[0] if cond.dimensions else "feature_distribution"
    cause = CAUSE_BY_DIMENSION.get(dim, FailureCause.WRONG_CONTEXT)
    sub = SUBSYSTEM_BY_CAUSE.get(cause)
    stmt = (f"{item.item_id} works when {cond.describe()}. Working rows {len(pops.success)} vs failed rows {len(pops.failed)}; "
            f"family-wise search over {n_tested} causal comparisons; confirmed on {len(conf_rows)} later rows: mean "
            f"{test.mean_in:+.3%} where it holds ({test.n_in} rows) vs {test.mean_out:+.3%} elsewhere ({test.n_out} rows).")
    return BreakExplanation(item.item_id, str(as_of), "EXPLAINED", cause, sub.value if sub else None, cond, test, stmt,
                            tuple(summary), symptoms, len(pops.success), len(pops.failed), len(conf_rows), n_tested,
                            tuple(eps), len(cands), power, comps, best_t)


# ------------------------------------------------------------------------------------------------- placebo control and cross-item events

def shifted_placebo(item: ItemSeries, shift: int) -> ItemSeries:
    """The same item with every context column (and neighbours) circularly shifted `shift` rows against the outcomes: the
    outcome history and its breaks are untouched but any real link between context and outcome is destroyed."""
    fr = item.frame.copy()
    for c in item.columns:
        fr[c.name] = np.roll(fr[c.name].values, shift)
    nb = None if item.neighbours is None else pd.DataFrame(np.roll(item.neighbours.values, shift, axis=0),
                                                            index=item.neighbours.index, columns=item.neighbours.columns)
    comps = {g: (d, pd.DataFrame(np.roll(sh.values, shift, axis=0), index=sh.index, columns=sh.columns))
             for g, (d, sh) in item.compositions.items()}
    return ItemSeries(item.item_id, fr, item.columns, nb, comps)


def placebo_shifts(T: int, n: int, seed: int) -> list[int]:
    """`n` distinct circular shifts, each at least T/4 rows from zero and from T (so the shifted context is out of step with
    the outcomes by a quarter of the history or more), drawn deterministically from the seed."""
    rng = np.random.default_rng(seed)
    lo, hi = T // 4, 3 * T // 4
    pool = np.arange(lo, max(hi, lo + 1))
    return [int(x) for x in rng.choice(pool, size=min(n, len(pool)), replace=False)]


def explain_break_controlled(item: ItemSeries, as_of=None, cfg=None, seed: int = 0) -> BreakExplanation:
    """explain_break plus the family-wise shifted-placebo bar (W-05). A condition that survives the out-of-sample test is
    then asked the question the test cannot answer alone: would the same pipeline have produced an equally strong condition
    from this item's outcomes with the context circularly shifted (so any real link is destroyed)? `placebo_n` shifts are run;
    the verdict stays EXPLAINED only when (1 + #placebos whose best out-of-sample t is at least the real one) / (n + 1) is at
    most `placebo_alpha`. The placebos are only run for an explained verdict, so the control costs nothing on the (majority
    of) items that are UNKNOWN anyway. Anything else is demoted to UNKNOWN with the reason recorded."""
    P = _cfg(cfg)
    ex = explain_break(item, as_of, cfg, seed)
    if not ex.explained:
        return ex
    n = int(P["placebo_n"])
    strong = 0
    runs = 0
    for i, sh in enumerate(placebo_shifts(len(item.frame), n, seed + 991)):
        try:
            pl = explain_break(shifted_placebo(item, sh), as_of, cfg, seed=seed + 1000 + i)
        except FirewallBreach:
            continue
        runs += 1
        strong += int(pl.best_t >= ex.oos.t_diff)
    p = (1.0 + strong) / (runs + 1.0)
    if runs < 2 or p <= P["placebo_alpha"]:
        return dataclasses.replace(ex, placebo=(runs, strong, float(p)))
    why = (f"{item.item_id}: the condition held out of sample (t {ex.oos.t_diff:.2f}) but {strong} of {runs} shifted-context placebos "
           f"found one at least as strong (p={p:.2f} > {P['placebo_alpha']}); treated as a chance condition, cause unknown")
    return dataclasses.replace(_empty(item, as_of, "UNKNOWN", why, ex.dimension_summary, ex.n_success, ex.n_failed, ex.n_confirm,
                                      ex.n_tested, ex.episodes, ex.symptoms, ex.candidates_tried, ex.detectable_smd, ex.compositions,
                                      ex.best_t), placebo=(runs, strong, float(p)))


def placebo_false_condition_rate(item: ItemSeries, as_of=None, n: int = 12, cfg=None, seed: int = 0, controlled: bool = True) -> dict:
    """Fraction of placebo runs that still produce an EXPLAINED verdict. It is the empirical false-condition rate of the
    whole pipeline on this item; it must sit near or under the configured alpha for the engine to be trusted. `controlled`
    measures the pipeline as shipped (explain_break_controlled); False measures the bare explain_break."""
    explain = explain_break_controlled if controlled else explain_break
    rng = np.random.default_rng(seed)
    T = len(item.frame)
    hits = 0
    statuses: dict[str, int] = {}
    for i in range(n):
        sh = int(rng.integers(T // 4, 3 * T // 4))
        try:
            r = explain(shifted_placebo(item, sh), as_of, cfg, seed=seed + i)
            statuses[r.status] = statuses.get(r.status, 0) + 1
            hits += int(r.explained)
        except FirewallBreach:
            statuses["LEAK_REFUSED"] = statuses.get("LEAK_REFUSED", 0) + 1
    return {"runs": n, "explained": hits, "rate": hits / max(n, 1), "statuses": statuses}


def shared_break_events(episodes_by_item: Mapping[str, Sequence[Episode]], gap: int = 4, min_items: int = 3) -> list[dict]:
    """Group onsets of different items that fall within `gap` rows of one another. A break shared by many items points at a
    market-wide cause rather than an item-specific one."""
    pts = sorted((e.onset, k) for k, es in episodes_by_item.items() for e in es)
    groups: list[list[tuple[int, str]]] = []
    for o, k in pts:
        if groups and o - groups[-1][-1][0] <= gap:
            groups[-1].append((o, k))
        else:
            groups.append([(o, k)])
    total = max(len(episodes_by_item), 1)
    return [{"start": g[0][0], "end": g[-1][0], "items": sorted({k for _, k in g}), "share": len({k for _, k in g}) / total}
            for g in groups if len({k for _, k in g}) >= min_items]


def episode_profile(episodes: Sequence[Episode], n: int) -> dict:
    """Summary of an item's break history: count, lengths, share of time spent broken, recoveries."""
    if not episodes:
        return {"n": 0, "mean_length": 0.0, "share_broken": 0.0, "recovered": 0, "open": 0}
    lens = [e.length(n) for e in episodes]
    return {"n": len(episodes), "mean_length": float(np.mean(lens)), "share_broken": float(min(sum(lens) / max(n, 1), 1.0)),
            "recovered": sum(e.recover is not None for e in episodes), "open": sum(e.recover is None for e in episodes)}


def render_report(exps: Sequence[BreakExplanation]) -> str:
    """Plain-language report: one block per item, the verdict, the evidence counts, the per-dimension table."""
    lines = ["# Break explanations", "IMPLEMENTED - NOT VALIDATED", ""]
    for e in exps:
        lines.append(f"## {e.item_id} - {e.status} ({e.cause.value})")
        lines.append(e.statement)
        lines.append(f"populations: {e.n_success} working / {e.n_failed} failed; confirmation rows {e.n_confirm}; "
                     f"causal comparisons {e.n_tested}; candidates tried {e.candidates_tried}")
        for d in e.dimension_summary:
            lines.append(f"- {d['dimension']}: {d['verdict']} (columns {d['n_columns']}, best {d['best_column']}, "
                         f"|t| {d['best_abs_t']:.2f}, family p {d['p_family']:.3f})")
        for s in e.symptoms:
            lines.append(f"- symptom (not a cause): {s['column']} differs (smd {s['smd']:+.2f})")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- power, compositions, categories

def detectable_smd(n_s: int, n_f: int, n_tests: int, alpha: float = 0.10, power: float = 0.8, design_effect: float = 2.0) -> float:
    """Smallest standardised mean difference the populations could have shown at the family-wise bar with `power`:
    (z_{1-alpha/2m} + z_power) * sqrt(deff * (1/n_s + 1/n_f)). It is what makes UNKNOWN honest: 'nothing found' from a sample
    that could only see huge effects is not evidence that nothing is there. The design effect stands in for serial
    dependence (periods inside a stretch are not independent)."""
    if n_s < 2 or n_f < 2:
        return float("inf")
    m = max(int(n_tests), 1)
    z_a = float(sps.norm.isf(alpha / (2.0 * m)))
    z_b = float(sps.norm.ppf(power))
    return float((z_a + z_b) * math.sqrt(design_effect * (1.0 / n_s + 1.0 / n_f)))


def composition_shift(design: Design, group: str, pops: Populations, n_perm: int = 300, seed: int = 0) -> dict:
    """Did the MIX change? Total-variation distance between the mean composition (sector shares, stock-type shares) in the
    working and failed populations, with a p-value from circularly shifted labels. Also names the category that moved most
    and the concentration change. Compositions are compared as wholes because a shift spread over many small categories is
    invisible to a per-column test."""
    cols = [c for c in design.X.columns if c.startswith(f"{group}:")]
    if not cols or len(pops.success) < 2 or len(pops.failed) < 2:
        return {"group": group, "tv": float("nan"), "p": 1.0, "largest_mover": None, "categories": len(cols)}
    rows = np.sort(np.r_[pops.success, pops.failed])
    lab = np.isin(rows, pops.failed)
    A = design.X.iloc[rows][cols].values.astype(float)

    def tv(l):
        return 0.5 * float(np.abs(A[~l].mean(0) - A[l].mean(0)).sum())

    obs = tv(lab)
    rng = np.random.default_rng(seed)
    n = len(rows)
    lo = max(int(0.05 * n), 2)
    null = np.array([tv(np.roll(lab, int(rng.integers(lo, max(n - lo, lo + 1))))) for _ in range(n_perm)])
    p = float((1 + (null >= obs).sum()) / (n_perm + 1))
    move = A[~lab].mean(0) - A[lab].mean(0)
    hhi_col = f"{group}__hhi"
    hhi = design.X[hhi_col].values.astype(float) if hhi_col in design.X.columns else None
    return {"group": group, "tv": obs, "p": p, "largest_mover": cols[int(np.argmax(np.abs(move)))],
            "categories": len(cols), "shift_null_mean": float(null.mean()),
            "hhi_success": float(hhi[pops.success].mean()) if hhi is not None else float("nan"),
            "hhi_failed": float(hhi[pops.failed].mean()) if hhi is not None else float("nan")}


def categorical_association(labels: Sequence[Any], pops: Populations) -> dict:
    """Chi-square test and Cramer's V of a categorical variable (a sector or stock-type label per row) against the population
    label, using only the rows in the two populations. Small expected counts are pooled into 'other' first."""
    lab = np.asarray(labels, dtype=object)
    s, f = lab[pops.success], lab[pops.failed]
    cats, counts = np.unique(np.r_[s, f].astype(str), return_counts=True)
    keep = set(cats[counts >= 5])
    if len(keep) < 2:
        return {"chi2": 0.0, "p": 1.0, "cramers_v": 0.0, "categories": len(keep)}
    fold = lambda a: np.array([x if x in keep else "other" for x in a.astype(str)])
    s, f = fold(s), fold(f)
    levels = sorted(set(s) | set(f))
    tab = np.array([[np.sum(s == l) for l in levels], [np.sum(f == l) for l in levels]], float)
    tab = tab[:, tab.sum(0) > 0]
    if tab.shape[1] < 2:
        return {"chi2": 0.0, "p": 1.0, "cramers_v": 0.0, "categories": tab.shape[1]}
    chi2, p, _, _ = sps.chi2_contingency(tab, correction=False)
    v = math.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))
    return {"chi2": float(chi2), "p": float(p), "cramers_v": float(v), "categories": int(tab.shape[1])}


def distribution_comparison(a: np.ndarray, b: np.ndarray, qs: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    """How the whole distribution of one column differs between the populations, not just its mean: quantile shifts in units
    of the pooled sd, Kolmogorov-Smirnov, Mann-Whitney, variance ratio and skew difference. The quantile profile is what
    distinguishes 'the level moved' from 'the tails fattened'."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {"n": (len(a), len(b))}
    sd = math.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) or 1.0
    ks = sps.ks_2samp(a, b)
    mw = sps.mannwhitneyu(a, b, alternative="two-sided")
    return {"n": (len(a), len(b)), "quantile_shift": {q: float((np.quantile(a, q) - np.quantile(b, q)) / sd) for q in qs},
            "ks": float(ks.statistic), "ks_p": float(ks.pvalue), "mw_p": float(mw.pvalue),
            "var_ratio": float(b.var(ddof=1) / a.var(ddof=1)) if a.var(ddof=1) > 1e-12 else float("nan"),
            "skew_diff": float(sps.skew(a) - sps.skew(b))}


# ------------------------------------------------------------------------------------------------- timing structure

def event_study(design: Design, episodes: Sequence[Episode], columns: Sequence[str], window: int = 8) -> pd.DataFrame:
    """Mean standardised value of each column at offsets -window..+window around the episode onsets. A real cause moves
    BEFORE or AT the onset; a column that only moves after the onset is a consequence. Standardisation uses the whole
    column (descriptive analysis), and offsets that fall outside the matured rows are skipped."""
    X = design.X
    n = len(X)
    rows = []
    for c in columns:
        x = X[c].values.astype(float)
        sd = np.nanstd(x)
        z = (x - np.nanmean(x)) / sd if sd > 1e-12 else np.zeros_like(x)
        row: dict[Any, Any] = {"column": c}
        for off in range(-window, window + 1):
            vals = [z[e.onset + off] for e in episodes if 0 <= e.onset + off < n]
            row[off] = float(np.mean(vals)) if vals else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")


def lead_lag_verdict(profile: pd.Series, window: int = 8, bar: float = 0.5) -> str:
    """'leads' if the column is already displaced before the onset, 'coincident' if it moves at it, 'lags' if only after,
    'flat' if it never moves by `bar` sd."""
    pre = float(np.nanmean(np.abs(profile.loc[[o for o in profile.index if -window <= o < -1]])))
    at = float(np.nanmean(np.abs(profile.loc[[o for o in profile.index if -1 <= o <= 1]])))
    post = float(np.nanmean(np.abs(profile.loc[[o for o in profile.index if 1 < o <= window]])))
    if max(pre, at, post) < bar:
        return "flat"
    if pre >= bar:
        return "leads"
    return "coincident" if at >= post else "lags"


def era_consistency(design: Design, pops: Populations, columns: Sequence[str], n_eras: int = 3) -> pd.DataFrame:
    """Split the two populations into `n_eras` consecutive time blocks and recompute each column's standardised difference in
    every block that has both populations. A difference that flips sign between eras is a coincidence of one episode."""
    have = len(pops.success) > 0 and len(pops.failed) > 0
    lo = int(min(pops.success.min(), pops.failed.min())) if have else 0
    hi = int(max(pops.success.max(), pops.failed.max())) + 1 if have else 0
    edges = np.linspace(lo, hi, n_eras + 1).astype(int)
    out = []
    for c in columns:
        x = design.X[c].values.astype(float)
        row: dict[str, Any] = {"column": c}
        for k in range(n_eras):
            s = pops.success[(pops.success >= edges[k]) & (pops.success < edges[k + 1])]
            f = pops.failed[(pops.failed >= edges[k]) & (pops.failed < edges[k + 1])]
            if len(s) < 3 or len(f) < 3:
                row[f"era{k}"] = float("nan")
                continue
            sd = math.sqrt(0.5 * (x[s].var(ddof=1) + x[f].var(ddof=1)))
            row[f"era{k}"] = float((x[s].mean() - x[f].mean()) / sd) if sd > 1e-12 else 0.0
        vals = np.array([row[f"era{k}"] for k in range(n_eras)], float)
        vals = vals[np.isfinite(vals)]
        row["n_eras"] = int(len(vals))
        row["same_sign"] = bool(len(vals) >= 2 and (np.all(vals > 0) or np.all(vals < 0)))
        out.append(row)
    return pd.DataFrame(out)


def conditional_profile(cond: Condition, design: Design, values: np.ndarray, rows: np.ndarray, n_bins: int = 4) -> pd.DataFrame:
    """Value statistics where the condition holds vs not, and across quantile bins of its first column - to see whether the
    effect is a threshold, a gradient or a single lucky cell."""
    rows = np.asarray(rows, int)
    v = values[rows].astype(float)
    hold = cond.mask(design.X.iloc[rows])

    def cell(name, sel):
        sel = sel & np.isfinite(v)
        return {"cell": name, "n": int(sel.sum()), "mean": float(v[sel].mean()) if sel.any() else float("nan"),
                "hit": float((v[sel] > 0).mean()) if sel.any() else float("nan")}

    out = [cell("holds", hold), cell("fails", ~hold)]
    x = design.X[cond.terms[0].column].values.astype(float)[rows]
    ok = np.isfinite(x) & np.isfinite(v)
    if ok.sum() >= 4 * n_bins:
        edges = np.quantile(x[ok], np.linspace(0, 1, n_bins + 1))
        for k in range(n_bins):
            hi = edges[k + 1] if k < n_bins - 1 else np.inf
            out.append(cell(f"q{k + 1}", ok & (x >= edges[k]) & (x < hi)))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------- the per-condition OOS dossier

def condition_dossier(item: ItemSeries, ex: BreakExplanation, as_of=None, cfg=None, seed: int = 0) -> dict:
    """The full out-of-sample harness for an EXPLAINED break: the fixed-cut test on the confirmation window is already in
    `ex.oos`; this adds (a) the walk-forward gate on each term column re-fitted as data arrived, (b) threshold stability,
    (c) a block-bootstrap interval of the gating gain, (d) era consistency of the term columns and (e) their lead/lag
    relation to the onsets. A condition is 'robust' only when every part agrees. UNKNOWN explanations return {}."""
    if ex.condition is None:
        return {}
    P = _cfg(cfg)
    eps, states = find_episodes(item, as_of, P)
    design = build_design(item, as_of, P, eps)
    m = len(design.X)
    values = item.frame["value"].values[:m].astype(float)
    n_disc = int(P["disc_frac"] * m)
    conf_rows = np.arange(min(n_disc + P["embargo"], m), m)
    wf = []
    for t in ex.condition.terms:
        r = walk_forward_condition(design, values, t.column, t.op, np.arange(m), P)
        if len(r.rows) > 20:
            _, lo, hi = block_bootstrap_gain(values[r.rows], r.hold, seed, P)
        else:
            lo = hi = float("nan")
        wf.append({"column": t.column, "gain": r.gain, "t_gain": r.t_gain, "gain_lo": float(lo), "gain_hi": float(hi),
                   "n_refits": r.n_refits, "cut_instability": r.stable_cut})
    stab = threshold_stability(design, values, ex.condition, conf_rows, P)
    cols = [t.column for t in ex.condition.terms]
    pops = split_populations(item, eps, states, as_of, P)
    eras = era_consistency(design, pops, cols) if len(pops.success) and len(pops.failed) else pd.DataFrame()
    es = event_study(design, eps, cols) if eps else pd.DataFrame()
    timing = {c: lead_lag_verdict(es.loc[c]) for c in cols} if len(es) else {}
    robust = bool(all(w["gain"] > 0 and w["gain_lo"] > 0 for w in wf) and sum(s["passes"] for s in stab) >= len(stab) - 1
                  and (len(eras) == 0 or bool(eras["same_sign"].all())))
    return {"walk_forward": wf, "stability": stab, "eras": eras.to_dict("records"), "timing": timing, "robust": robust}


# ------------------------------------------------------------------------------------------------- across items and over time

def shared_drivers(exps: Sequence[BreakExplanation]) -> list[dict]:
    """Conditions that recur across items: same column, same direction. A driver explaining several items' breaks is more
    credible (independent replications) and points at the market rather than the item."""
    seen: dict[tuple, list[str]] = {}
    for e in exps:
        if e.condition is None:
            continue
        for t in e.condition.terms:
            seen.setdefault((t.column, t.op), []).append(e.item_id)
    return sorted(({"column": k[0], "op": k[1], "items": sorted(set(v)), "n_items": len(set(v))}
                   for k, v in seen.items() if len(set(v)) >= 2), key=lambda d: -d["n_items"])


def pooled_explain(items: Sequence[ItemSeries], as_of=None, cfg=None, seed: int = 0) -> dict:
    """Run the break engine on every item; return the explanations, the unknown-cause share, and shared drivers. A leaking item
    is refused and named, never skipped silently."""
    exps, refused = [], []
    for i, it in enumerate(items):
        try:
            exps.append(explain_break_controlled(it, as_of, cfg, seed + i))
        except FirewallBreach as err:
            refused.append({"item_id": it.item_id, "why": str(err)})
    broken = [e for e in exps if e.status in ("EXPLAINED", "UNKNOWN")]
    return {"explanations": exps, "refused": refused, "n_broken": len(broken),
            "unknown_share": (sum(e.status == "UNKNOWN" for e in broken) / len(broken)) if broken else float("nan"),
            "shared_drivers": shared_drivers(exps)}


class BreakLedger:
    """Append-only, hash-chained record of break explanations, so a verdict can be traced and never quietly rewritten.
    `latest(item, as_of)` returns only what had been recorded by `as_of`."""

    def __init__(self):
        self._rows: list[dict] = []

    def __len__(self):
        return len(self._rows)

    def append(self, ex: BreakExplanation, recorded_at) -> dict:
        prev = self._rows[-1]["chain"] if self._rows else "genesis"
        body = ex.as_dict()
        row = {"recorded_at": str(recorded_at), "explanation": body, "prev": prev,
               "chain": stable_hash([prev, str(recorded_at), body], 20)}
        self._rows.append(row)
        return row

    def latest(self, item_id: str, as_of=None) -> dict | None:
        cands = [r for r in self._rows if r["explanation"]["item_id"] == item_id and (as_of is None or r["recorded_at"] <= str(as_of))]
        return cands[-1] if cands else None

    def verify(self) -> list[str]:
        errs, prev = [], "genesis"
        for i, r in enumerate(self._rows):
            if r["prev"] != prev or r["chain"] != stable_hash([prev, r["recorded_at"], r["explanation"]], 20):
                errs.append(f"row {i}: chain broken")
            prev = r["chain"]
        return errs

    def unknown_share(self, as_of=None) -> float:
        """Share of the latest verdict per item that is UNKNOWN (C61: admit unknown causes, and track how often)."""
        latest = {}
        for r in self._rows:
            if as_of is None or r["recorded_at"] <= str(as_of):
                latest[r["explanation"]["item_id"]] = r["explanation"]["status"]
        decided = [s for s in latest.values() if s in ("EXPLAINED", "UNKNOWN")]
        return decided.count("UNKNOWN") / len(decided) if decided else float("nan")


# ------------------------------------------------------------------------------------------------- multi-column (state) conditions

def kmeans(Z: np.ndarray, k: int, seed: int, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Seeded Lloyd's algorithm with farthest-point initialisation. Returns (centres, labels). Deterministic."""
    rng = np.random.default_rng(seed)
    n = len(Z)
    centres = [Z[int(rng.integers(0, n))]]
    while len(centres) < k:
        d = np.min(((Z[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        centres.append(Z[int(np.argmax(d))])
    C = np.array(centres, float)
    lab = np.zeros(n, int)
    for _ in range(iters):
        d = ((Z[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if _ and np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = Z[lab == j].mean(0)
    return C, lab


@dataclasses.dataclass(frozen=True)
class StateCondition:
    """The item works when the market sits in one of the fitted states: nearest centre in the standardised space of `columns`.
    Catches regimes no single threshold captures. Centres and scales are fixed at fit time from discovery rows only."""
    columns: tuple
    centres: tuple             # k tuples
    mean: tuple
    scale: tuple
    good: tuple                # state indices in which the item works
    dimensions: tuple = ("regime",)
    terms: tuple = ()

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[list(self.columns)].values.astype(float) - np.array(self.mean)) / np.array(self.scale)
        d = ((Z[:, None, :] - np.array(self.centres)[None]) ** 2).sum(-1)
        return np.isin(d.argmin(1), self.good)

    def describe(self) -> str:
        return f"the market is in state(s) {list(self.good)} of {len(self.centres)} defined by {', '.join(self.columns)}"

    @property
    def key(self) -> str:
        return stable_hash([self.columns, [[round(x, 6) for x in c] for c in self.centres], self.good], 12)


def propose_state_conditions(design: Design, values: np.ndarray, disc_rows: np.ndarray, table: pd.DataFrame,
                             fw_p: Mapping[str, float], cfg=None, seed: int = 0, ks: Sequence[int] = (2, 3)) -> list[tuple[StateCondition, float]]:
    """State candidates built ONLY from causal numeric columns that already beat the family-wise bar (so this adds no new
    search over raw columns), k-means on discovery rows, the good states being those where the discovery mean outcome is
    positive. Scored by the Welch t of outcome inside vs outside the good states."""
    P = _cfg(cfg)
    cols = [c for c in table[table["causal"]]["column"]
            if fw_p.get(c, 1.0) <= P["alpha"] and design.X[c].nunique() > 5]
    if len(cols) < 2:
        return []
    cols = cols[: P["max_pair_cols"]]
    A = design.X.iloc[disc_rows][cols].values.astype(float)
    mu, sd = A.mean(0), A.std(0)
    sd[sd <= 1e-12] = 1.0
    Z = (A - mu) / sd
    v = values[disc_rows].astype(float)
    out = []
    for k in ks:
        C, lab = kmeans(Z, k, seed)
        means = [np.nanmean(v[lab == j]) if (lab == j).sum() >= P["min_side"] else np.nan for j in range(k)]
        good = tuple(j for j, mv in enumerate(means) if np.isfinite(mv) and mv > 0)
        if not good or len(good) == k:
            continue
        cond = StateCondition(tuple(cols), tuple(tuple(map(float, c)) for c in C), tuple(map(float, mu)), tuple(map(float, sd)), good)
        hold = np.isin(lab, good) & np.isfinite(v)
        a, b = v[hold], v[~hold & np.isfinite(v)]
        if len(a) < P["min_side"] or len(b) < P["min_side"]:
            continue
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        out.append((cond, float((a.mean() - b.mean()) / se) if se > 1e-12 else 0.0))
    return sorted(out, key=lambda z: -z[1])


# ------------------------------------------------------------------------------------------------- tables for people and for scoring

def stretch_table(item: ItemSeries, as_of=None, cfg=None) -> pd.DataFrame:
    """Consecutive stretches (working / failing / mixed) of the matured history with their length, mean outcome and hit rate.
    The first thing a reviewer reads before trusting any explanation of the breaks."""
    P = _cfg(cfg)
    m = item.n_matured(as_of)
    v = item.frame["value"].values[:m].astype(float)
    lab = window_labels(v, P["label_half"], P["label_t"], m)
    rows, start = [], 0
    for i in range(1, m + 1):
        if i == m or lab[i] != lab[start]:
            seg = v[start:i]
            rows.append({"start": start, "end": i, "length": i - start,
                         "kind": {1: "working", -1: "failing", 0: "mixed"}[int(lab[start])],
                         "mean": float(np.nanmean(seg)), "hit": float(np.nanmean(seg > 0))})
            start = i
    return pd.DataFrame(rows, columns=["start", "end", "length", "kind", "mean", "hit"])


def research_hints(table: pd.DataFrame, fw_p: Mapping[str, float], alpha: float = 0.10, top: int = 5) -> list[dict]:
    """Near misses that did NOT clear the family-wise bar, offered only as directions for the research queue (never as
    causes): the causal columns with the smallest family-wise p above the bar."""
    sub = table[table["causal"]].copy()
    sub["p_family"] = sub["column"].map(lambda c: fw_p.get(c, 1.0))
    sub = sub[sub["p_family"] > alpha].sort_values("p_family").head(top)
    return [{"column": r["column"], "dimension": r["dimension"], "smd": float(r["smd"]), "p_family": float(r["p_family"]),
             "status": "hint_not_evidence"} for _, r in sub.iterrows()]


def score_explanation(ex: BreakExplanation, truth_columns: Sequence[str] | None) -> dict:
    """Score one verdict against a planted truth. `truth_columns` = the columns that really drive the break, or None when the
    break is random. Returns correct (right column found, or UNKNOWN when there is none), and a fabricated flag when a condition
    was reported for a random break or names a wrong column - the worst outcome, counted separately."""
    found = {t.column for t in ex.condition.terms} if ex.condition is not None and ex.condition.terms else set()
    if isinstance(ex.condition, StateCondition):
        found = set(ex.condition.columns)
    if truth_columns is None:
        return {"correct": ex.status in ("UNKNOWN", "INSUFFICIENT_EVIDENCE", "NO_BREAK"), "fabricated": bool(found)}
    truth = set(truth_columns)
    return {"correct": bool(found & truth), "fabricated": bool(found - truth), "found": sorted(found),
            "missed": ex.status != "EXPLAINED"}


def run_planted_study(make_item_fn: Callable[[int], tuple], n: int = 10, cfg=None, seed: int = 0) -> dict:
    """Run the pipeline on `n` planted worlds. `make_item_fn(i) -> (ItemSeries, truth_columns_or_None)`. Reports recall on
    real drivers, the fabrication rate on everything, and the UNKNOWN rate on random breaks. The numbers a validation wave
    needs; nothing here tunes anything."""
    rows = []
    for i in range(n):
        item, truth = make_item_fn(i)
        try:
            ex = explain_break_controlled(item, None, cfg, seed + i)
        except FirewallBreach:
            rows.append({"i": i, "truth": truth, "status": "LEAK_REFUSED", "correct": False, "fabricated": False})
            continue
        rows.append({"i": i, "truth": truth, "status": ex.status, **score_explanation(ex, truth)})
    df = pd.DataFrame(rows)
    real = df[df["truth"].map(lambda t: t is not None)]
    rand = df[df["truth"].map(lambda t: t is None)]
    return {"n": n, "recall_real": float(real["correct"].mean()) if len(real) else float("nan"),
            "fabrication_rate": float(df["fabricated"].mean()) if len(df) else float("nan"),
            "unknown_rate_random": float((rand["status"] == "UNKNOWN").mean()) if len(rand) else float("nan"),
            "table": df}


class BreakEngine:
    """Stateful wrapper: holds the parameters, the ledger and the registered items; `run(as_of)` explains every item whose
    break is knowable at `as_of` and records the verdicts. Leaking items are refused loudly, once per run."""

    def __init__(self, cfg=None, seed: int = 0):
        self.cfg = _cfg(cfg)
        self.seed = seed
        self.items: dict[str, ItemSeries] = {}
        self.ledger = BreakLedger()
        self.refused: list[dict] = []

    def register(self, item: ItemSeries) -> None:
        errs = item.validate()
        if errs:
            raise BreakInputError(f"{item.item_id}: " + "; ".join(errs))
        self.items[item.item_id] = item

    def run(self, as_of) -> list[BreakExplanation]:
        out = []
        for i, (kid, item) in enumerate(sorted(self.items.items())):
            try:
                ex = explain_break_controlled(item, as_of, self.cfg, self.seed + i)
            except FirewallBreach as err:
                self.refused.append({"item_id": kid, "as_of": str(as_of), "why": str(err)})
                continue
            self.ledger.append(ex, as_of)
            out.append(ex)
        return out

    def status_counts(self, as_of=None) -> dict:
        latest = {}
        for r in self.ledger._rows:
            if as_of is None or r["recorded_at"] <= str(as_of):
                latest[r["explanation"]["item_id"]] = r["explanation"]["status"]
        counts: dict[str, int] = {}
        for s in latest.values():
            counts[s] = counts.get(s, 0) + 1
        return counts


# ------------------------------------------------------------------------------------------------- stability, comparison, coverage

def explanation_stability(item: ItemSeries, as_ofs: Sequence[Any], cfg=None, seed: int = 0) -> dict:
    """Explain the same item at several dates and see whether the story holds still. Reports the status at each date, how often
    the top condition column is the same, and how often the verdict is UNKNOWN. A cause that changes every time new data
    arrives was probably never a cause."""
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(as_ofs):
        try:
            ex = explain_break(item, a, cfg, seed + i)
        except FirewallBreach as err:
            rows.append({"as_of": str(a), "status": "LEAK_REFUSED", "column": None, "why": str(err)})
            continue
        col = None
        if ex.condition is not None:
            col = ex.condition.terms[0].column if ex.condition.terms else "state:" + ",".join(getattr(ex.condition, "columns", ()))
        rows.append({"as_of": str(a), "status": ex.status, "column": col, "why": ex.statement[:120]})
    cols = [r["column"] for r in rows if r["column"]]
    top = max(set(cols), key=cols.count) if cols else None
    return {"rows": rows, "n": len(rows), "top_column": top, "top_share": (cols.count(top) / len(cols)) if cols else float("nan"),
            "explained": sum(r["status"] == "EXPLAINED" for r in rows), "unknown": sum(r["status"] == "UNKNOWN" for r in rows),
            "stable": bool(cols and cols.count(top) == len(cols))}


def compare_explanations(a: BreakExplanation, b: BreakExplanation) -> dict:
    """Side-by-side of two verdicts (different dates, seeds or items): did status, cause, condition column and direction agree?"""
    def key(e):
        if e.condition is None:
            return None
        return tuple((t.column, t.op) for t in e.condition.terms) or ("state",)
    return {"same_status": a.status == b.status, "same_cause": a.cause == b.cause, "same_condition": key(a) == key(b),
            "gain_a": None if a.oos is None else a.oos.gain, "gain_b": None if b.oos is None else b.oos.gain,
            "n_failed": (a.n_failed, b.n_failed)}


def coverage_report(item: ItemSeries, cfg=None) -> dict:
    """Which of the 18 dimensions the item can be tested on, from which columns, and which are unmeasured. Unmeasured dimensions
    are a data-collection to-do list: an UNKNOWN break cause on an item with six unmeasured dimensions says less than one on
    an item with none."""
    design = build_design(item, None, cfg, [])
    by_dim = {d: [c for c in design.X.columns if design.tags[c].dimension == d] for d in DIMENSIONS}
    causal = {d: [c for c in cols if design.tags[c].causal] for d, cols in by_dim.items()}
    missing = [d for d in DIMENSIONS if not by_dim[d]]
    only_symptoms = [d for d in DIMENSIONS if by_dim[d] and not causal[d]]
    return {"columns": {d: len(v) for d, v in by_dim.items()}, "unmeasured": missing, "symptom_only": only_symptoms,
            "n_causal_columns": sum(len(v) for v in causal.values()), "measured_share": 1.0 - len(missing) / len(DIMENSIONS)}


def categorical_report(item: ItemSeries, as_of=None, cfg=None) -> list[dict]:
    """Chi-square association of every raw categorical context column (sector, stock type, exchange...) with the working /
    failed populations of the discovery window. Reported per column with Cramer's V; family-wise Holm across the columns."""
    P = _cfg(cfg)
    eps, states = find_episodes(item, as_of, P)
    m = item.n_matured(as_of)
    pops = split_populations(item, eps, states, as_of, P, rows=np.arange(int(P["disc_frac"] * m)))
    out = []
    for c in item.columns:
        if c.kind != "categorical" or len(pops.success) < 5 or len(pops.failed) < 5:
            continue
        lab = item.frame[c.name].astype(str).values[:m]
        r = categorical_association(lab, pops)
        out.append({"column": c.name, "dimension": c.dimension, **r})
    if out:
        adj = PR.holm([r["p"] for r in out])
        for r, a in zip(out, adj):
            r["p_holm"] = float(a)
    return out


def dossier_text(ex: BreakExplanation, dossier: Mapping[str, Any]) -> str:
    """The out-of-sample harness result as plain language."""
    if not dossier:
        return f"{ex.item_id}: no condition to test ({ex.status})"
    lines = [f"{ex.item_id}: {cast(Condition, ex.condition).describe()}", f"robust: {dossier['robust']}"]
    for w in dossier["walk_forward"]:
        lines.append(f"- walk-forward gate on {w['column']}: gain {w['gain']:+.3%} per period (90% CI {w['gain_lo']:+.3%} to "
                     f"{w['gain_hi']:+.3%}), {w['n_refits']} refits, cut instability {w['cut_instability']:.2f}")
    if dossier["stability"]:
        lines.append(f"- threshold nudges that still pass: {sum(s['passes'] for s in dossier['stability'])}/{len(dossier['stability'])}")
    for c, v in dossier["timing"].items():
        lines.append(f"- {c} relative to the break onsets: {v}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- neighbours and interactions

def neighbour_lead_lag(item: ItemSeries, as_of=None, max_lag: int = 4) -> pd.DataFrame:
    """Rank correlation between the item's outcome at row t and each neighbour's outcome at row t - lag, lags 0..max_lag, over the
    matured rows. A neighbour that LEADS the item (large correlation at lag >= 1) is usable information at decision time;
    one that only correlates at lag 0 shares the item's shocks and explains nothing in advance. The lag-0 column is
    descriptive (contemporaneous) and is labelled so."""
    vis = item.visible(as_of)
    if vis.neighbours is None or vis.neighbours.shape[1] == 0:
        return pd.DataFrame(columns=["neighbour", "lag", "rho", "n", "usable_in_advance"])
    m = vis.n_matured(None)
    y = vis.frame["value"].values[:m].astype(float)
    rows = []
    for c in vis.neighbours.columns:
        z = vis.neighbours[c].values[:m].astype(float)
        for lag in range(0, max_lag + 1):
            a, b = (y[lag:], z[: m - lag]) if lag else (y, z)
            ok = np.isfinite(a) & np.isfinite(b)
            rows.append({"neighbour": c, "lag": lag, "rho": _spearman(a[ok], b[ok]) if ok.sum() >= 20 else float("nan"),
                         "n": int(ok.sum()), "usable_in_advance": lag >= 1})
    return pd.DataFrame(rows)


def interaction_scan(design: Design, values: np.ndarray, rows: np.ndarray, columns: Sequence[str] | None = None,
                     max_cols: int = 8) -> pd.DataFrame:
    """Does the outcome depend on two causal columns TOGETHER? For every pair, an incremental F test of value ~ a + b + a*b
    against value ~ a + b on the given rows (z-scored inputs), Holm-adjusted across all pairs searched. A significant product
    term with weak main effects is the signature of an interaction failure (the item breaks only when both conditions
    hold), which a per-column contrast cannot see."""
    cols = list(columns) if columns is not None else [c for c in design.columns(causal=True) if "*" not in c and "__" not in c]
    cols = cols[:max_cols]
    rows = np.asarray(rows, int)
    y = values[rows].astype(float)
    ok = np.isfinite(y)
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            xa = design.X[a].values[rows].astype(float)
            xb = design.X[b].values[rows].astype(float)
            good = ok & np.isfinite(xa) & np.isfinite(xb)
            if good.sum() < 40 or xa[good].std() < 1e-12 or xb[good].std() < 1e-12:
                continue
            za = (xa[good] - xa[good].mean()) / xa[good].std()
            zb = (xb[good] - xb[good].mean()) / xb[good].std()
            yy = y[good]
            A0 = np.column_stack([np.ones(len(yy)), za, zb])
            A1 = np.column_stack([A0, za * zb])
            c0, *_ = np.linalg.lstsq(A0, yy, rcond=None)
            c1, *_ = np.linalg.lstsq(A1, yy, rcond=None)
            r0, r1 = float(((yy - A0 @ c0) ** 2).sum()), float(((yy - A1 @ c1) ** 2).sum())
            dof = len(yy) - 4
            F = (r0 - r1) / (r1 / dof) if r1 > 1e-18 and dof > 0 else 0.0
            out.append({"a": a, "b": b, "F": float(F), "p": float(sps.f.sf(F, 1, max(dof, 1))), "beta_product": float(c1[3]),
                        "beta_a": float(c1[1]), "beta_b": float(c1[2]), "n": int(good.sum())})
    df = pd.DataFrame(out, columns=["a", "b", "F", "p", "beta_product", "beta_a", "beta_b", "n"])
    if len(df):
        df["p_holm"] = PR.holm(df["p"].values)
        df["pure_interaction"] = (df["p_holm"] < 0.05) & (df["beta_product"].abs() > df[["beta_a", "beta_b"]].abs().max(axis=1))
    return df.sort_values("p").reset_index(drop=True)


def contrast_frame(item: ItemSeries, as_of=None, cfg=None, seed: int = 0) -> pd.DataFrame:
    """The full per-column comparison of working vs failed rows in the discovery window with family-wise p-values, as one table
    (what `explain_break` computes internally, exposed for review and for the validation wave). Empty when there is not enough
    evidence to form both populations."""
    P = _cfg(cfg)
    eps, states = find_episodes(item, as_of, P)
    m = item.n_matured(as_of)
    design = build_design(item, as_of, P, eps)
    pops = split_populations(item, eps, states, as_of, P, rows=np.arange(int(P["disc_frac"] * m)))
    if pops.enough(P):
        return pd.DataFrame()
    parts = []
    for causal in (True, False):
        cols = design.columns(causal=causal)
        if cols:
            tab = contrast_table(design.X, pops, design.tags, cols)
            _, p = family_wise_p(design.X, pops, cols, P["n_perm"], seed + (0 if causal else 1))
            tab["p_family"] = p
            parts.append(tab)
    return pd.concat(parts, ignore_index=True).sort_values("p_family").reset_index(drop=True)


def min_detectable_gain(n_in: int, n_out: int, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest difference in mean outcome (holds minus not) the out-of-sample test could have confirmed: (z_{1-alpha} + z_power) *
    sd * sqrt(1/n_in + 1/n_out), one-sided. Read next to `ConditionTest`: a condition that failed OOS on a confirmation window
    too small to see any plausible effect is 'untested', not 'refuted'."""
    if n_in < 2 or n_out < 2 or sd <= 0:
        return float("inf")
    return float((sps.norm.isf(alpha) + sps.norm.ppf(power)) * sd * math.sqrt(1.0 / n_in + 1.0 / n_out))


def population_table(item: ItemSeries, pops: Populations) -> pd.DataFrame:
    """Plain description of the two populations: rows, mean / sd / hit rate of the outcome, first and last row, and the longest
    unbroken stretch. What a reviewer reads before looking at any column-level result."""
    v = item.frame["value"].values.astype(float)
    rows = []
    for name, idx in (("working", pops.success), ("failed", pops.failed)):
        x = v[idx] if len(idx) else np.zeros(0)
        run = best = 0
        for a, b in zip(idx[:-1], idx[1:]):
            run = run + 1 if b == a + 1 else 0
            best = max(best, run)
        rows.append({"population": name, "rows": int(len(idx)), "mean": float(np.nanmean(x)) if len(x) else float("nan"),
                     "sd": float(np.nanstd(x, ddof=1)) if len(x) > 1 else float("nan"),
                     "hit": float(np.nanmean(x > 0)) if len(x) else float("nan"),
                     "first_row": int(idx.min()) if len(idx) else None, "last_row": int(idx.max()) if len(idx) else None,
                     "longest_stretch": int(best + 1) if len(idx) else 0})
    return pd.DataFrame(rows)
