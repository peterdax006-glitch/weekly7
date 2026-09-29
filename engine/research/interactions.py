"""Interaction discovery (C66 section 26; Bible phase 25 'interactions'). IMPLEMENTED - NOT VALIDATED.

Searches the eight section-26 interaction families (volume x volatility, momentum x event, reversal x gap, sector x stock
strength, market volatility x liquidity, pattern x regime, pattern x pattern, event x pattern) and separates an *interesting
interaction* from one *found because millions of combinations were tried*.

Every combination that is actually tested is entered in a TrialLedger (family, both atoms, functional form, quantile,
horizon), and every multiple-testing correction is computed against the CUMULATIVE count of that ledger for the same data,
not against the survivors: p-values of trials tried in earlier searches on the same data are padded in as non-discoveries,
so searching again and again raises the bar instead of lowering it (the counting rule of section 26).

Pipeline, in the order of the Stage ladder of engine.research.core:
  CHEAP_SCREEN    discovery window only. Fama-MacBeth style incremental test of the interaction term after both main effects
                  (or, when one atom is market-level, a Newey-West test that the cross-sectional slope of the other atom moves
                  with it). BH / BY / Bonferroni over the whole ledger count (engine.pattern_stats).
  STRONGER_TESTS  shuffled controls (labels permuted within date) rerun the WHOLE trial set: max-|t| family-wise p-values and a
                  pipeline-level false-positive rate; complexity penalty (engine.learning.complexity.required_t / within_budget).
  CROSS_YEAR      per-year sign agreement, drop-the-best-year robustness, purged walk-forward folds (engine.antioverfit).
  FRESH_HOLDOUT   held-out validation window, fresh never-reused seeds (ticker subsamples and block bootstraps), cross-stock
                  groups, the simple-vs-complex OOS comparison (complexity.compare), then a one-shot HoldoutVault window.
The result is a Fate per trial and a lottery diagnostic for the whole search. Nothing here promotes: a VALIDATED interaction
is Epistemic.SUPPORTED research knowledge, and reaches a decision only as a MaturedRecord through MaturedRecord.gate(now).

Time (rule 3): prepare() refuses any date at or after `now` (FirewallBreach) and drops the last `horizon` sessions whose forward
label would not yet have matured. Determinism (rule 4): every random draw comes from SeedLedger, which refuses reuse.
Builds on engine.pattern_stats (BH/BY/Bonferroni, t->p, jaccard), engine.antioverfit (within-date permutation, walk-forward),
engine.learning.complexity (RuleSpec, required_t, within_budget, Candidate, compare) and engine.pattern_reliability.nw_t.
Public entry: step(state, inputs, now, cfg)."""
from __future__ import annotations

import dataclasses as dc
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from engine.antioverfit import permute_within_date, verify_walk_forward, walk_forward_splits
from engine.learning import complexity as CX
from engine.learning.core import (Epistemic, FirewallBreach, Provenance, _StrEnum, current_code_hash, require_past,
                                  stable_hash)
from engine.pattern_reliability import nw_t
from engine.pattern_stats import bh_qvalues, bonferroni, by_qvalues, jaccard, t_to_p
from engine.research.core import MaturedRecord, Stage


class InteractionError(ValueError):
    """Bad inputs or configuration (distinct from FirewallBreach, which is a time/identity violation)."""


class SeedReuse(InteractionError):
    """A seed purpose+key was requested twice, or a validation seed collided with a discovery seed."""


class HoldoutSpent(InteractionError):
    """A candidate was already tested on this fresh holdout window: the window is no longer fresh for it."""


class Form(_StrEnum):
    PRODUCT = "PRODUCT"            # rank(a) * rank(b) after both main effects
    CORNER_HH = "CORNER_HH"        # a and b both in their top quantile
    CORNER_HL = "CORNER_HL"        # a in its top quantile, b in its bottom quantile
    CORNER_H = "CORNER_H"          # a in its top quantile while binary b is on
    MODULATION = "MODULATION"      # cross-sectional slope of the row atom moves with the market-level atom


class Fate(_StrEnum):
    NOISE = "NOISE"                                  # raw p >= alpha_raw: nothing to explain
    INTERESTING_ONLY = "INTERESTING_ONLY"            # raw p < alpha_raw but expected from the number of combinations tried
    SURVIVED_CORRECTION = "SURVIVED_CORRECTION"      # passes the screen and shuffled controls, not yet through later stages
    REJECTED_COMPLEXITY = "REJECTED_COMPLEXITY"      # |t| below the bar its complexity units demand
    REJECTED_CROSS_YEAR = "REJECTED_CROSS_YEAR"
    REJECTED_VALIDATION = "REJECTED_VALIDATION"      # did not hold in the held-out window
    REJECTED_CROSS_STOCK = "REJECTED_CROSS_STOCK"
    REJECTED_SEEDS = "REJECTED_SEEDS"                # significance depends on which stocks/dates were drawn
    REJECTED_OOS_COMPLEXITY = "REJECTED_OOS_COMPLEXITY"   # the simple main-effects rule does as well out of sample
    REJECTED_HOLDOUT = "REJECTED_HOLDOUT"
    NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"      # a required test could not run (too little data)
    VALIDATED = "VALIDATED"


_TERMINAL_REJECTIONS = frozenset({Fate.REJECTED_COMPLEXITY, Fate.REJECTED_CROSS_YEAR, Fate.REJECTED_VALIDATION,
                                  Fate.REJECTED_CROSS_STOCK, Fate.REJECTED_SEEDS, Fate.REJECTED_OOS_COMPLEXITY,
                                  Fate.REJECTED_HOLDOUT})


@dc.dataclass(frozen=True)
class SearchConfig:
    horizon: int = 5                       # sessions the forward label looks ahead (labels overlap: HAC lags follow it)
    min_names: int = 8                     # cross-section smaller than this on a date is skipped
    min_dates_window: int = 60             # a window with fewer dates is not tested
    frac_discovery: float = 0.5
    frac_validation: float = 0.3           # the rest, less two purge gaps, is the fresh holdout
    quantiles: tuple[float, ...] = (0.33,)
    alpha_raw: float = 0.05
    alpha_screen: float = 0.10             # BH q for the cheap screen
    alpha_fwer: float = 0.10               # max-|t| family-wise level for the shuffled controls
    correction: str = "bh"                 # bh | by | bonferroni
    n_shuffles: int = 40
    alpha_valid: float = 0.05              # one-sided, sign of the discovery effect, on the held-out window
    n_fresh_seeds: int = 7
    seed_pass_frac: float = 0.6
    subsample_frac: float = 0.7
    n_stock_groups: int = 4
    min_group_agree: float = 0.75
    min_years: int = 3
    min_year_dates: int = 30
    min_year_agree: float = 0.7
    robust_t: float = 1.28
    wf_folds: int = 4
    min_wf_agree: float = 0.6
    max_candidates: int = 20000
    dispersion_control: bool = True        # slopes are divided by the date's cross-sectional sd of y (vol-scaling confound)
    overlap_max: float = 0.9               # two binary atoms overlapping more than this are one atom
    min_atom_freq: float = 0.01
    base_seed: int = 7
    complexity: CX.ComplexityConfig = CX.DEFAULT_CCFG

    def validate(self) -> list[str]:
        errs = []
        if self.horizon < 1:
            errs.append("horizon < 1")
        if not (0.2 <= self.frac_discovery <= 0.8 and 0.1 <= self.frac_validation <= 0.6
                and self.frac_discovery + self.frac_validation < 0.95):
            errs.append("window fractions leave no room for a holdout")
        if any(not (0.05 <= q <= 0.5) for q in self.quantiles) or not self.quantiles:
            errs.append("quantiles must be in [0.05, 0.5]")
        if self.correction not in ("bh", "by", "bonferroni"):
            errs.append("unknown correction")
        for name in ("alpha_raw", "alpha_screen", "alpha_fwer", "alpha_valid", "seed_pass_frac", "subsample_frac",
                     "min_group_agree", "min_year_agree", "min_wf_agree"):
            if not 0.0 < getattr(self, name) < 1.0:
                errs.append(f"{name} outside (0,1)")
        if self.n_shuffles < 10:
            errs.append("n_shuffles < 10 cannot resolve a family-wise p at 0.10")
        if self.n_fresh_seeds < 3 or self.n_stock_groups < 2 or self.min_names < 3:
            errs.append("too few seeds / groups / names")
        return errs

    @property
    def lags(self) -> int:
        return self.horizon + 1


# ---------------------------------------------------------------------------------------------------- identity hygiene
_YEAR_RE = re.compile(r"(?<![0-9])(19|20)[0-9]{2}(?![0-9])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9.]+")


def assert_identity_free(payload: Any, tickers: Iterable[str] = (), where: str = "payload") -> None:
    """Section 30/31 for everything that leaves the research side: no year, no ISO date and no ticker may appear in any
    string or key. Numbers are fine. Tickers are matched as whole upper-case tokens of length >= 2."""
    names = {t for t in map(str, tickers) if len(t) >= 2}

    def walk(o, path):
        if isinstance(o, Mapping):
            for k, v in o.items():
                walk(str(k), path + "/key")
                walk(v, f"{path}/{k}")
        elif isinstance(o, (list, tuple, set, frozenset)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, str):
            if _YEAR_RE.search(o):
                raise FirewallBreach(f"{where}{path}: contains a year/date-like token {o!r}")
            if names and any(tok in names for tok in _TOKEN_RE.findall(o)):
                raise FirewallBreach(f"{where}{path}: contains a ticker in {o!r}")
    walk(payload, "")


# ---------------------------------------------------------------------------------------------------- roles and families
DEFAULT_ROLE_PREFIXES: dict[str, tuple[str, ...]] = {   # first matching role wins, so the order encodes precedence
    "market_vol": ("m_vix", "m_vol", "m_rv", "m_atr"),
    "sector_strength": ("sector", "sec_"),
    "liquidity": ("liq", "spread", "amihud", "adv", "dollar"),
    "volume": ("volume", "rvol", "turnover", "vol_ratio", "vlm"),
    "volatility": ("atr", "volat", "vola", "hv", "rv_", "range", "std"),
    "gap": ("gap", "overnight"),
    "reversal": ("rev", "rsi", "dist_", "zscore", "oversold", "overbought"),
    "momentum": ("mom", "roc", "trend", "ret_"),
    "stock_strength": ("rs_", "rel", "strength"),
}


def infer_roles(columns: Iterable[str], prefixes: Mapping[str, Sequence[str]] = DEFAULT_ROLE_PREFIXES) -> dict[str, tuple[str, ...]]:
    """Assign feature columns to section-26 roles by name prefix. Explicit roles on InteractionInputs override this; a column
    that matches nothing plays no role and is never searched (a column is not tried just because it exists)."""
    out: dict[str, list[str]] = {}
    for c in columns:
        low = str(c).lower()
        for role, pre in prefixes.items():
            if any(low.startswith(p) for p in pre):
                out.setdefault(role, []).append(str(c))
                break
    return {r: tuple(v) for r, v in out.items()}


@dc.dataclass(frozen=True)
class Family:
    name: str
    left: str
    right: str


FAMILIES: tuple[Family, ...] = (
    Family("volume_x_volatility", "volume", "volatility"),
    Family("momentum_x_event", "momentum", "event"),
    Family("reversal_x_gap", "reversal", "gap"),
    Family("sector_x_stock_strength", "sector_strength", "stock_strength"),
    Family("market_vol_x_liquidity", "market_vol", "liquidity"),
    Family("pattern_x_regime", "pattern", "regime"),
    Family("pattern_x_pattern", "pattern", "pattern"),
    Family("event_x_pattern", "event", "pattern"),
)


@dc.dataclass(frozen=True)
class Atom:
    name: str
    role: str
    kind: str          # "cont" (cross-sectional rank in [-0.5, 0.5]) or "bin" (0/1)
    level: str         # "row" (varies across stocks on a date) or "date" (market-level: one value per date)


@dc.dataclass(frozen=True)
class Trial:
    """One tested combination. Everything that could have been chosen differently is in the id, so a second functional form or
    quantile of the same pair is a second trial, counted as such."""
    family: str
    a: str
    b: str
    form: Form
    q: float
    horizon: int
    kinds: str          # e.g. "cont*bin"; the b atom is the date-level one when form is MODULATION

    @property
    def trial_id(self) -> str:
        return "I" + stable_hash({"f": self.family, "a": self.a, "b": self.b, "form": str(self.form), "q": self.q,
                                  "h": self.horizon}, 12)

    def spec(self) -> CX.RuleSpec:
        """Structural complexity: two features, one interaction, a threshold for corner forms and a gate per binary atom."""
        gates = self.kinds.count("bin")
        thresholds = 1 if self.form in (Form.CORNER_HH, Form.CORNER_HL, Form.CORNER_H) else 0
        return CX.RuleSpec(self.trial_id, n_features=2, n_conditions=gates, n_thresholds=thresholds, n_interactions=1,
                           tree_depth=1 + (1 if gates or thresholds else 0), n_free_params=1)

    def simple_spec(self) -> CX.RuleSpec:
        """The same two features without the interaction: what the interaction has to beat."""
        gates = self.kinds.count("bin")
        return CX.RuleSpec(self.trial_id + "_main", n_features=2, n_conditions=gates, n_thresholds=0,
                           tree_depth=1 + (1 if gates else 0), n_free_params=1)

    def text(self) -> str:
        return f"{self.family}: {self.a} x {self.b} [{self.form}, q={self.q:g}, h={self.horizon}]"


# ---------------------------------------------------------------------------------------------------- ledgers
class TrialLedger:
    """Counts EVERY combination ever tested, per data key (the data's start, columns and horizon - not its end, so a later
    search on extended data keeps the earlier count). m_total is what multiple-testing corrections divide by."""

    def __init__(self) -> None:
        self._seen: dict[str, dict[str, int]] = {}
        self._validated: dict[str, set[str]] = {}
        self.runs = 0

    def register(self, data_key: str, trials: Sequence[Trial]) -> int:
        book = self._seen.setdefault(data_key, {})
        for t in trials:
            book[t.trial_id] = book.get(t.trial_id, 0) + 1
        return len(book)

    def m_total(self, data_key: str) -> int:
        return len(self._seen.get(data_key, {}))

    def looks(self, data_key: str) -> int:
        return sum(self._seen.get(data_key, {}).values())

    def repeats(self, data_key: str) -> int:
        return sum(1 for n in self._seen.get(data_key, {}).values() if n > 1)

    def note_validated(self, data_key: str, ids: Iterable[str]) -> int:
        s = self._validated.setdefault(data_key, set())
        s.update(ids)
        return len(s)

    def validation_count(self, data_key: str) -> int:
        return len(self._validated.get(data_key, ()))

    def to_dict(self) -> dict:
        return {"seen": {k: dict(v) for k, v in self._seen.items()},
                "validated": {k: sorted(v) for k, v in self._validated.items()}, "runs": self.runs}

    @classmethod
    def from_dict(cls, d: Mapping) -> "TrialLedger":
        led = cls()
        led._seen = {k: {t: int(n) for t, n in v.items()} for k, v in d.get("seen", {}).items()}
        led._validated = {k: set(v) for k, v in d.get("validated", {}).items()}
        led.runs = int(d.get("runs", 0))
        return led


class SeedLedger:
    """Deterministic, single-use seeds. A (purpose, key) pair yields one seed once; asking again is SeedReuse, and a
    'fresh' (validation) seed that equals any discovery seed is refused, so validation never replays the search's luck."""

    def __init__(self, base: int) -> None:
        self.base = int(base)
        self._used: dict[int, tuple[str, str]] = {}
        self._pairs: set[tuple[str, str]] = set()

    def take(self, purpose: str, key: Any) -> int:
        k = str(key)
        if (purpose, k) in self._pairs:
            raise SeedReuse(f"seed ({purpose}, {k}) already used")
        seed = int(stable_hash({"b": self.base, "p": purpose, "k": k}, 12), 16) % (2 ** 32)
        prior = self._used.get(seed)
        if prior is not None and (prior[0] == "fresh") != (purpose == "fresh"):
            raise SeedReuse(f"{purpose} seed {seed} collides with a {prior[0]} seed")
        self._pairs.add((purpose, k))
        self._used[seed] = (purpose, k)
        return seed

    def rng(self, purpose: str, key: Any) -> np.random.Generator:
        return np.random.default_rng(self.take(purpose, key))

    def used(self, purpose: str | None = None) -> list[int]:
        return sorted(s for s, (p, _) in self._used.items() if purpose is None or p == purpose)


class HoldoutVault:
    """One-shot fresh windows. A window is identified by its data key; each trial id may be opened against it once. The
    cumulative count of ids opened is returned so the caller corrects for every look the window has ever had."""

    def __init__(self) -> None:
        self._opened: dict[str, set[str]] = {}

    def open(self, window_key: str, trial_ids: Iterable[str]) -> int:
        ids = set(trial_ids)
        book = self._opened.setdefault(window_key, set())
        spent = book & ids
        if spent:
            raise HoldoutSpent(f"{len(spent)} candidate(s) were already tested on this holdout window")
        book |= ids
        return len(book)

    def was_opened(self, window_key: str, trial_id: str) -> bool:
        return trial_id in self._opened.get(window_key, ())

    def count(self, window_key: str) -> int:
        return len(self._opened.get(window_key, ()))


# ---------------------------------------------------------------------------------------------------- inputs and panel
@dc.dataclass
class InteractionInputs:
    """What one search reads. X/y follow the repo panel convention (MultiIndex (date, ticker); y = forward return over
    `horizon` sessions). masks/events are boolean frames on the same index (pattern firings / event flags); regimes is a
    per-date label series; sectors maps ticker -> sector (used only to build the cross-stock groups)."""
    X: pd.DataFrame
    y: pd.Series
    horizon: int = 5
    roles: Mapping[str, Sequence[str]] | None = None
    masks: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    regimes: pd.Series | None = None
    sectors: pd.Series | None = None


def stock_groups(names: Sequence[str], k: int, sectors: pd.Series | None = None) -> np.ndarray:
    """Group id per ticker: the sector when known, otherwise a stable hash of the name (independent of the data, so the
    groups never depend on which outcomes were seen)."""
    if sectors is not None:
        sec = pd.Series(sectors).reindex(list(names)).fillna("?").astype(str)
        return pd.factorize(sec.values)[0].astype(np.int64)
    return np.array([int(stable_hash(str(n), 8), 16) % k for n in names], dtype=np.int64)


@dc.dataclass
class Panel:
    """Date-sorted arrays for one prepared search. Row atoms are per-date centred ranks (cont) or 0/1 (bin); date atoms hold
    one value per session in `all_dates`. select() keeps `all_dates` so date positions stay comparable across subsets."""
    all_dates: np.ndarray
    date_pos: np.ndarray
    tcode: np.ndarray
    tnames: np.ndarray
    tgroup: np.ndarray
    y: np.ndarray
    row_atoms: dict
    date_atoms: dict
    meta: dict
    horizon: int
    dropped_unmatured: int = 0
    data_key: str = ""
    last_date: str = ""

    def __post_init__(self) -> None:
        if len(self.date_pos):
            self.upos, self.starts, self.counts = np.unique(self.date_pos, return_index=True, return_counts=True)
        else:
            self.upos = self.starts = self.counts = np.zeros(0, dtype=np.int64)

    @property
    def n_rows(self) -> int:
        return int(len(self.y))

    @property
    def n_dates(self) -> int:
        return int(len(self.upos))

    def select(self, mask: np.ndarray) -> "Panel":
        m = np.asarray(mask, bool)
        return Panel(self.all_dates, self.date_pos[m], self.tcode[m], self.tnames, self.tgroup, self.y[m],
                     {k: v[m] for k, v in self.row_atoms.items()}, self.date_atoms, self.meta, self.horizon,
                     self.dropped_unmatured, self.data_key, self.last_date)

    def window(self, lo: int, hi: int) -> "Panel":
        return self.select((self.date_pos >= lo) & (self.date_pos < hi))

    def with_y(self, y: np.ndarray) -> "Panel":
        out = Panel(self.all_dates, self.date_pos, self.tcode, self.tnames, self.tgroup, np.asarray(y, float),
                    self.row_atoms, self.date_atoms, self.meta, self.horizon, self.dropped_unmatured, self.data_key, self.last_date)
        return out

    def by_role(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for n, a in self.meta.items():
            out.setdefault(a.role, []).append(n)
        return out

    def year_of_pos(self) -> np.ndarray:
        return self.all_dates.astype("datetime64[Y]").astype(int) + 1970


def _rank_centered(s: pd.Series) -> np.ndarray:
    r = s.groupby(level=0).rank(pct=True) - 0.5
    return r.fillna(0.0).to_numpy(float)


def _is_date_level(s: pd.Series) -> bool:
    return bool(s.groupby(level=0).nunique(dropna=True).max() <= 1)


def _per_date(s: pd.Series, all_dates: pd.DatetimeIndex) -> np.ndarray:
    """One value per session for a market-level column; gaps are filled forward only (never from the future)."""
    v = s.groupby(level=0).first().reindex(all_dates).ffill()
    return v.to_numpy(float)


def prepare(inp: InteractionInputs, now, cfg: SearchConfig) -> Panel:
    """Validate inputs, fail closed on anything dated at/after `now`, drop labels that have not matured, build the atoms."""
    bad = cfg.validate()
    if bad:
        raise InteractionError("; ".join(bad))
    X = inp.X
    if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels != 2:
        raise InteractionError("X must be indexed by (date, ticker)")
    if len(X) == 0:
        raise InteractionError("empty panel")
    if X.index.has_duplicates:
        raise InteractionError("duplicate (date, ticker) rows")
    if int(inp.horizon) != cfg.horizon:
        raise InteractionError(f"inputs horizon {inp.horizon} != config horizon {cfg.horizon}")
    X = X.sort_index()
    dates = pd.DatetimeIndex(X.index.get_level_values(0))
    require_past(dates.max(), now, "interaction panel")
    for name, extra in (("regimes", inp.regimes), ("y", inp.y)):
        if extra is not None and len(extra):
            idx = extra.index.get_level_values(0) if isinstance(extra.index, pd.MultiIndex) else extra.index
            require_past(pd.DatetimeIndex(idx).max(), now, name)
    y = inp.y.reindex(X.index)
    if y.isna().all():
        raise InteractionError("y has no values on the X index")
    all_dates = pd.DatetimeIndex(np.sort(dates.unique()))
    if len(all_dates) <= cfg.horizon + 1:
        raise InteractionError("fewer sessions than the horizon allows")
    matured = all_dates[: len(all_dates) - cfg.horizon]           # the last `horizon` labels look past the data we have
    keep = np.asarray(dates.isin(matured)) & y.notna().to_numpy()
    dropped = int((~keep).sum())
    Xk, yk = X[keep], y[keep]
    if len(Xk) == 0:
        raise InteractionError("no matured, labelled rows")
    all_dates = pd.DatetimeIndex(np.sort(pd.DatetimeIndex(Xk.index.get_level_values(0)).unique()))
    dpos = all_dates.get_indexer(pd.DatetimeIndex(Xk.index.get_level_values(0)))
    tick = Xk.index.get_level_values(1).astype(str).to_numpy()
    tcode, tnames = pd.factorize(tick)
    roles = {r: tuple(c) for r, c in (inp.roles if inp.roles is not None else infer_roles(Xk.columns)).items()}
    row_atoms: dict[str, np.ndarray] = {}
    date_atoms: dict[str, np.ndarray] = {}
    meta: dict[str, Atom] = {}
    for role, cols in roles.items():
        for c in cols:
            if c not in Xk.columns:
                raise InteractionError(f"role {role!r} names missing column {c!r}")
            s = Xk[c].astype(float)
            if s.notna().mean() < 0.5:
                continue
            if _is_date_level(s):
                v = _per_date(s, all_dates)
                if np.isfinite(v).sum() >= 20 and np.nanstd(v) > 1e-12:
                    date_atoms[c] = np.nan_to_num(v, nan=float(np.nanmean(v)))
                    meta[c] = Atom(c, role, "cont", "date")
            else:
                v = _rank_centered(s)
                if v.std() > 1e-9:
                    row_atoms[c] = v
                    meta[c] = Atom(c, role, "cont", "row")
    for role, frame in (("pattern", inp.masks), ("event", inp.events)):
        if frame is None:
            continue
        fr = frame.reindex(Xk.index).fillna(False).astype(bool)
        for c in fr.columns:
            v = fr[c].to_numpy(float)
            if inp.masks is not None and inp.events is not None and c in inp.masks.columns and c in inp.events.columns \
                    and role == "event":
                raise InteractionError(f"column {c!r} is both a pattern mask and an event flag")
            if cfg.min_atom_freq <= v.mean() <= 1 - cfg.min_atom_freq:
                row_atoms[str(c)] = v
                meta[str(c)] = Atom(str(c), role, "bin", "row")
    if inp.regimes is not None:
        lab = pd.Series(inp.regimes).copy()
        lab.index = pd.DatetimeIndex(lab.index)
        lab = lab[~lab.index.duplicated(keep="last")].reindex(all_dates, method="ffill")
        for value in sorted(lab.dropna().unique(), key=str):
            d = (lab == value).to_numpy(float)
            if max(10, 0.05 * len(d)) <= d.sum() <= 0.9 * len(d):
                nm = f"regime={value}"
                date_atoms[nm] = d
                meta[nm] = Atom(nm, "regime", "bin", "date")
    key = stable_hash({"first": str(all_dates[0]), "atoms": sorted(meta), "h": cfg.horizon}, 12)
    groups = stock_groups(list(tnames), cfg.n_stock_groups, inp.sectors)
    return Panel(all_dates.to_numpy(), dpos.astype(np.int64), tcode.astype(np.int64), np.asarray(tnames, dtype=object),
                 groups, yk.to_numpy(float), row_atoms, date_atoms, meta, cfg.horizon, dropped, key, str(dates.max())[:10])


@dc.dataclass(frozen=True)
class Windows:
    """Date-position ranges [lo, hi): discovery, validation and the fresh holdout, separated by purge gaps of horizon+1."""
    disc: tuple[int, int]
    valid: tuple[int, int]
    hold: tuple[int, int] | None
    gap: int

    def check(self, horizon: int) -> list[str]:
        errs = []
        if self.valid[0] - self.disc[1] < horizon:
            errs.append("discovery -> validation purge gap shorter than the horizon")
        if self.hold is not None and self.hold[0] - self.valid[1] < horizon:
            errs.append("validation -> holdout purge gap shorter than the horizon")
        if self.disc[0] >= self.disc[1] or self.valid[0] >= self.valid[1]:
            errs.append("empty window")
        return errs


def split_windows(n_dates: int, cfg: SearchConfig) -> Windows:
    g = cfg.horizon + 1
    d1 = int(n_dates * cfg.frac_discovery)
    v0 = d1 + g
    v1 = v0 + int(n_dates * cfg.frac_validation)
    h0 = v1 + g
    hold = (h0, n_dates) if n_dates - h0 >= cfg.min_dates_window // 2 else None
    w = Windows((0, d1), (v0, min(v1, n_dates)), hold, g)
    bad = w.check(cfg.horizon)
    if bad:
        raise InteractionError("; ".join(bad))
    return w


# ---------------------------------------------------------------------------------------------------- enumerating trials
def enumerate_trials(P: Panel, cfg: SearchConfig, seeds: SeedLedger | None = None, run_key: Any = 0,
                     families: Sequence[Family] = FAMILIES) -> tuple[list[Trial], list[dict]]:
    """Every testable combination of every family, in a fixed order. Combinations that cannot be tested (two market-level
    atoms have no cross-section; two binary atoms that are the same rows are one atom) are returned in `skipped` with the
    reason - they are NOT counted as tried. If the space exceeds max_candidates a seeded subsample is tested and m is the
    number actually tested (the untested rest were not tried)."""
    by_role = P.by_role()
    seen: set = set()
    trials: list[Trial] = []
    skipped: list[dict] = []

    def add(fam: str, a: str, b: str, form: Form, q: float, kinds: str) -> None:
        key = (frozenset((a, b)), str(form), q)
        if key in seen:
            skipped.append({"family": fam, "a": a, "b": b, "reason": "same pair already in an earlier family"})
            return
        seen.add(key)
        trials.append(Trial(fam, a, b, form, q, cfg.horizon, kinds))

    for fam in families:
        left, right = by_role.get(fam.left, []), by_role.get(fam.right, [])
        if fam.left == fam.right:
            pairs = [(left[i], left[j]) for i in range(len(left)) for j in range(i + 1, len(left))]
        else:
            pairs = [(a, b) for a in left for b in right]
        for a, b in pairs:
            if a == b:
                continue
            ma, mb = P.meta[a], P.meta[b]
            if ma.level == "date" and mb.level == "date":
                skipped.append({"family": fam.name, "a": a, "b": b, "reason": "both atoms market-level: no cross-section"})
                continue
            if ma.level == "date" or mb.level == "date":
                row, day = (a, b) if ma.level == "row" else (b, a)
                add(fam.name, row, day, Form.MODULATION, 0.0, f"{P.meta[row].kind}*{P.meta[day].kind}")
                continue
            if ma.kind == "bin" and mb.kind == "bin":
                if jaccard(P.row_atoms[a] > 0.5, P.row_atoms[b] > 0.5) > cfg.overlap_max:
                    skipped.append({"family": fam.name, "a": a, "b": b, "reason": "binary atoms are the same rows"})
                    continue
                add(fam.name, a, b, Form.PRODUCT, 0.0, "bin*bin")
                continue
            if ma.kind == "bin":                    # canonical order: the continuous atom first
                a, b, ma, mb = b, a, mb, ma
            if mb.kind == "bin":
                add(fam.name, a, b, Form.PRODUCT, 0.0, "cont*bin")
                for q in cfg.quantiles:
                    add(fam.name, a, b, Form.CORNER_H, q, "cont*bin")
            else:
                add(fam.name, a, b, Form.PRODUCT, 0.0, "cont*cont")
                for q in cfg.quantiles:
                    add(fam.name, a, b, Form.CORNER_HH, q, "cont*cont")
                    add(fam.name, a, b, Form.CORNER_HL, q, "cont*cont")
    if len(trials) > cfg.max_candidates:
        rng = (seeds.rng("enumerate", run_key) if seeds is not None else np.random.default_rng(cfg.base_seed))
        pick = np.sort(rng.choice(len(trials), size=cfg.max_candidates, replace=False))
        skipped.append({"family": "*", "a": "", "b": "", "reason": f"space of {len(trials)} truncated to {cfg.max_candidates}"})
        trials = [trials[i] for i in pick]
    return trials, skipped


# ---------------------------------------------------------------------------------------------------- statistic engine
def _center(v: np.ndarray, P: Panel) -> np.ndarray:
    """Subtract each date's cross-sectional mean (rows are date-sorted, so segment sums are a reduceat)."""
    return v - np.repeat(np.add.reduceat(v, P.starts) / P.counts, P.counts)


@dc.dataclass(frozen=True)
class Stat:
    beta: float
    se: float
    t: float
    n: int
    p: float

    @property
    def ok(self) -> bool:
        return math.isfinite(self.t)


_NO_STAT = Stat(float("nan"), float("nan"), float("nan"), 0, 1.0)


@dc.dataclass(frozen=True)
class TrialSeries:
    """The per-date evidence of one trial: pos = date position, v = the per-date interaction coefficient (or cross-sectional
    slope), z = the market-level covariate the slope is regressed on (None for row-level interactions)."""
    pos: np.ndarray
    v: np.ndarray
    z: np.ndarray | None = None

    def __len__(self) -> int:
        return int(len(self.v))

    def slice(self, lo: int, hi: int) -> "TrialSeries":
        m = (self.pos >= lo) & (self.pos < hi)
        return TrialSeries(self.pos[m], self.v[m], None if self.z is None else self.z[m])

    def take(self, idx: np.ndarray) -> "TrialSeries":
        return TrialSeries(self.pos[idx], self.v[idx], None if self.z is None else self.z[idx])


def _term(P: Panel, tr: Trial) -> np.ndarray:
    a, b = P.row_atoms[tr.a], P.row_atoms[tr.b]
    if tr.form is Form.PRODUCT:
        return a * b
    top = 0.5 - tr.q
    if tr.form is Form.CORNER_HH:
        return ((a > top) & (b > top)).astype(float)
    if tr.form is Form.CORNER_HL:
        return ((a > top) & (b < -top)).astype(float)
    if tr.form is Form.CORNER_H:
        return ((a > top) & (b > 0.5)).astype(float)
    raise InteractionError(f"{tr.form} is not a row-level term")


def _dispersion(P: Panel, yc: np.ndarray) -> np.ndarray:
    return np.sqrt(np.add.reduceat(yc * yc, P.starts) / P.counts)


def fm_series(P: Panel, cols: Sequence[np.ndarray], term: np.ndarray, cfg: SearchConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per-date cross-sectional regression of y on [cols..., term], all centred within the date; returns the date positions
    and the coefficient of `term`. A date is skipped when its cross-section is short or `term` is (nearly) collinear with the
    main effects, i.e. when it carries no independent information about the interaction."""
    if P.n_rows == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    R = [_center(np.asarray(c, float), P) for c in cols] + [_center(np.asarray(term, float), P)]
    yc = _center(P.y, P)
    k, D = len(R), len(P.starts)
    G, r = np.empty((D, k, k)), np.empty((D, k))
    for i in range(k):
        r[:, i] = np.add.reduceat(R[i] * yc, P.starts)
        for j in range(i, k):
            G[:, i, j] = G[:, j, i] = np.add.reduceat(R[i] * R[j], P.starts)
    G = G + (1e-9 * np.trace(G, axis1=1, axis2=2) + 1e-12)[:, None, None] * np.eye(k)
    beta = np.linalg.solve(G, r[:, :, None])[:, k - 1, 0]
    if k > 1:
        sol = np.linalg.solve(G[:, : k - 1, : k - 1], G[:, : k - 1, k - 1:])[:, :, 0]
        partial = G[:, k - 1, k - 1] - (G[:, k - 1, : k - 1] * sol).sum(1)
    else:
        partial = G[:, 0, 0]
    ok = (P.counts >= cfg.min_names) & (G[:, k - 1, k - 1] > 1e-9 * P.counts) & (partial > 1e-6 * G[:, k - 1, k - 1])
    if cfg.dispersion_control:
        sd = _dispersion(P, yc)
        ok &= sd > 1e-12
        beta = beta / np.where(sd > 1e-12, sd, 1.0)
    ok &= np.isfinite(beta)
    return P.upos[ok], beta[ok]


def slope_series(P: Panel, b: np.ndarray, cfg: SearchConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per-date univariate cross-sectional slope of y on the row atom `b`."""
    if P.n_rows == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    bc, yc = _center(np.asarray(b, float), P), _center(P.y, P)
    sbb = np.add.reduceat(bc * bc, P.starts)
    ok = (P.counts >= cfg.min_names) & (sbb > 1e-9 * P.counts)
    s = np.add.reduceat(bc * yc, P.starts) / np.where(ok, sbb, 1.0)
    if cfg.dispersion_control:
        sd = _dispersion(P, yc)
        ok &= sd > 1e-12
        s = s / np.where(sd > 1e-12, sd, 1.0)
    ok &= np.isfinite(s)
    return P.upos[ok], s[ok]


def trial_series(P: Panel, tr: Trial, cfg: SearchConfig) -> TrialSeries:
    """The per-date evidence for one trial on this panel (empty when the panel cannot support the test)."""
    if tr.form is Form.MODULATION:
        pos, s = slope_series(P, P.row_atoms[tr.a], cfg)
        if len(pos) == 0:
            return TrialSeries(pos, s, np.zeros(0))
        z = P.date_atoms[tr.b][pos].astype(float)
        if P.meta[tr.b].kind == "cont":
            sd = z.std()
            z = (z - z.mean()) / sd if sd > 1e-12 else np.zeros_like(z)
        return TrialSeries(pos, s, z)
    pos, v = fm_series(P, [P.row_atoms[tr.a], P.row_atoms[tr.b]], _term(P, tr), cfg)
    return TrialSeries(pos, v, None)


def series_stat(ts: TrialSeries, lags: int, min_n: int = 20) -> Stat:
    """Newey-West test: mean of the per-date coefficient, or (with a covariate) the slope of the per-date slope on it. Market
    -level covariates are serially dependent and labels overlap, so an iid t would overstate the evidence."""
    n = len(ts)
    if n < min_n:
        return _NO_STAT
    if ts.z is None:
        t = nw_t(ts.v, lags=lags)
        m = float(ts.v.mean())
        if not math.isfinite(t) or abs(t) < 1e-12:
            return Stat(m, float("nan"), float("nan"), n, 1.0)
        return Stat(m, abs(m / t), float(t), n, float(t_to_p(t)))
    x = ts.z - ts.z.mean()
    sxx = float(x @ x)
    if sxx <= 1e-9 or (np.unique(ts.z).size == 2 and min(int((ts.z > ts.z.mean()).sum()), int((ts.z <= ts.z.mean()).sum())) < 5):
        return Stat(float("nan"), float("nan"), float("nan"), n, 1.0)
    slope = float(x @ (ts.v - ts.v.mean())) / sxx
    u = ts.v - ts.v.mean() - slope * x
    g = x * u
    s = float(g @ g)
    for lag in range(1, min(lags, n - 1) + 1):
        s += 2.0 * (1.0 - lag / (lags + 1.0)) * float(g[lag:] @ g[:-lag])
    var = s / (sxx * sxx)
    if var <= 1e-30:
        return Stat(slope, float("nan"), float("nan"), n, 1.0)
    se = math.sqrt(var)
    return Stat(slope, se, slope / se, n, float(t_to_p(slope / se)))


def trial_stat(P: Panel, tr: Trial, cfg: SearchConfig) -> Stat:
    return series_stat(trial_series(P, tr, cfg), cfg.lags)


def one_sided_p(t: float, sign: float) -> float:
    """P(T >= t) in the direction `sign` found in discovery: a held-out effect of the opposite sign is not a weak success."""
    return float(sps.norm.sf(sign * t)) if math.isfinite(t) else 1.0


# ---------------------------------------------------------------------------------------------------- multiple testing
def corrected_q(p: Sequence[float], m_total: int, method: str = "bh") -> np.ndarray:
    """Adjusted p-values against the LEDGER count: p is padded with 1.0 up to m_total (every trial ever tried on this data that
    is not in `p` counts as a non-discovery), then BH / BY / Bonferroni is applied to the padded vector."""
    arr = np.nan_to_num(np.asarray(p, float), nan=1.0)
    m = max(int(m_total), len(arr))
    padded = np.concatenate([arr, np.ones(m - len(arr))])
    if method == "bh":
        q = bh_qvalues(padded)
    elif method == "by":
        q = by_qvalues(padded)
    elif method == "bonferroni":
        q = bonferroni(padded)
    else:
        raise InteractionError(f"unknown correction {method!r}")
    return q[: len(arr)]


def detectable_t(m_total: int, alpha: float = 0.05) -> float:
    """The |t| a single interaction needs to survive Bonferroni over m_total combinations: the honest size of the bar."""
    return float(sps.norm.isf(alpha / (2.0 * max(int(m_total), 1))))


@dc.dataclass(frozen=True)
class ShuffleControl:
    """Whole-search reruns on labels permuted within date. No interaction can be real in them, so whatever the pipeline
    'finds' there measures the search itself."""
    n: int
    null_t: np.ndarray                 # (n, n_trials)
    null_max: np.ndarray               # max |t| of each rerun
    null_hits: np.ndarray              # raw p < alpha_raw count of each rerun
    expected_hits: float
    pipeline_fp_rate: float            # share of reruns in which the full screen would have 'discovered' something

    def adj_p(self, t: float) -> float:
        """Westfall-Young max-|t| family-wise p-value: how often the best of ALL trials on shuffled labels beats this one."""
        if not math.isfinite(t):
            return 1.0
        return float((1 + int((self.null_max >= abs(t)).sum())) / (self.n + 1))

    def summary(self) -> dict:
        return {"n": self.n, "null_max_t_median": float(np.median(self.null_max)) if self.n else float("nan"),
                "null_max_t_p95": float(np.quantile(self.null_max, 0.95)) if self.n else float("nan"),
                "mean_raw_hits": float(self.null_hits.mean()) if self.n else float("nan"),
                "expected_raw_hits": self.expected_hits, "pipeline_fp_rate": self.pipeline_fp_rate}


def run_shuffled_controls(Pd: Panel, trials: Sequence[Trial], cfg: SearchConfig, seeds: SeedLedger, run_key: Any,
                          m_total: int) -> ShuffleControl:
    """Permute y within each date `n_shuffles` times and rerun every trial. `pipeline_fp_rate` applies the cheap screen (ledger
    -count correction) plus the family-wise test (against the OTHER reruns, leave-one-out) to each rerun: the share of reruns
    that still 'discover' something is the search machinery's own false-positive rate."""
    nt = len(trials)
    null_t = np.full((cfg.n_shuffles, nt), np.nan)
    for r in range(cfg.n_shuffles):
        rng = seeds.rng("shuffle", (run_key, r))
        yp = permute_within_date(Pd.y, Pd.date_pos, rng)
        Ps = Pd.with_y(yp)
        for i, tr in enumerate(trials):
            null_t[r, i] = trial_stat(Ps, tr, cfg).t
    absn = np.abs(null_t)
    null_max = np.nanmax(np.where(np.isfinite(absn), absn, 0.0), axis=1) if nt else np.zeros(cfg.n_shuffles)
    p = np.where(np.isfinite(null_t), t_to_p(np.nan_to_num(null_t)), 1.0)
    hits = (p < cfg.alpha_raw).sum(axis=1) if nt else np.zeros(cfg.n_shuffles, int)
    fp = 0
    for r in range(cfg.n_shuffles):
        q = corrected_q(p[r], m_total, cfg.correction)
        other = np.delete(null_max, r)
        found = False
        for i in np.flatnonzero(q <= cfg.alpha_screen):
            if (1 + int((other >= absn[r, i]).sum())) / (len(other) + 1) <= cfg.alpha_fwer:
                found = True
                break
        fp += int(found)
    return ShuffleControl(cfg.n_shuffles, null_t, null_max, hits, nt * cfg.alpha_raw, fp / max(cfg.n_shuffles, 1))


def lottery_diagnostic(p: Sequence[float], m_total: int, n_survivors: int, alpha: float = 0.05) -> dict:
    """Was the search a lottery? Compares the raw hits to the number chance alone hands out among the combinations tried
    (binomial), estimates the true-signal share from the p-value histogram (Storey pi0 at lambda 0.5), and tests the p-values
    for uniformity. Trials are dependent, so the binomial band is indicative; the shuffled controls are the authority."""
    arr = np.nan_to_num(np.asarray(p, float), nan=1.0)
    m = len(arr)
    if m == 0:
        return {"m_tested": 0, "m_total": int(m_total), "verdict": "EMPTY", "raw_hits": 0, "expected_raw_hits": 0.0,
                "bonferroni_t": detectable_t(max(m_total, 1), alpha)}
    hits = int((arr < alpha).sum())
    expected = m * alpha
    pi0 = min(1.0, float((arr > 0.5).mean()) / 0.5)
    upper = expected + 2.0 * math.sqrt(m * alpha * (1 - alpha))
    if n_survivors > 0:
        verdict = "SIGNAL"
    elif hits <= upper:
        verdict = "LOTTERY"
    else:
        verdict = "AMBIGUOUS"
    return {"m_tested": m, "m_total": int(m_total), "raw_hits": hits, "expected_raw_hits": expected,
            "hits_binomial_p": float(sps.binom.sf(hits - 1, m, alpha)) if hits else 1.0, "pi0": pi0,
            "est_true_share": 1.0 - pi0, "uniform_ks_p": float(sps.kstest(arr, "uniform").pvalue) if m >= 5 else float("nan"),
            "bonferroni_t": detectable_t(max(m_total, m), alpha), "survivors": int(n_survivors), "verdict": verdict}


# ---------------------------------------------------------------------------------------------------- transfer checks
def _sgn(x: float) -> float:
    return 1.0 if x >= 0 else -1.0


def cross_year_check(ts: TrialSeries, P: Panel, cfg: SearchConfig, sign: float) -> dict:
    """Does the effect show up year after year, or is one year doing all the work? `usable` is False when there are too few
    years with enough sessions to say: that is 'unknown', never a pass."""
    years = P.year_of_pos()[ts.pos] if len(ts) else np.zeros(0, int)
    per: dict[int, float] = {}
    for y in np.unique(years):
        m = years == y
        if m.sum() >= cfg.min_year_dates:
            st = series_stat(TrialSeries(ts.pos[m], ts.v[m], None if ts.z is None else ts.z[m]), cfg.lags, min_n=cfg.min_year_dates)
            if st.ok:
                per[int(y)] = sign * st.t
    out = {"n_years": len(per), "usable": len(per) >= cfg.min_years, "per_year_signed_t": per}
    if not per:
        return {**out, "agree": float("nan"), "drop_best_t": float("nan"), "pass": False}
    out["agree"] = float(np.mean([t > 0 for t in per.values()]))
    best = max(per, key=per.get)
    rest = years != best
    st = series_stat(TrialSeries(ts.pos[rest], ts.v[rest], None if ts.z is None else ts.z[rest]), cfg.lags)
    out["drop_best_t"] = sign * st.t if st.ok else float("nan")
    out["pass"] = bool(out["usable"] and out["agree"] >= cfg.min_year_agree and out["drop_best_t"] >= cfg.robust_t)
    return out


def walk_forward_check(ts: TrialSeries, P: Panel, lo: int, hi: int, cfg: SearchConfig, sign: float) -> dict:
    """Purged expanding-window folds over [lo, hi) from engine.antioverfit: in how many test blocks does the effect have the
    discovery sign? The splits are audited with verify_walk_forward; a leaking split is an error, not a weak result."""
    dates = P.all_dates[lo:hi]
    splits = walk_forward_splits(dates, n_folds=cfg.wf_folds, horizon=cfg.horizon, min_train_dates=max(20, (hi - lo) // 5))
    if not splits:
        return {"usable": False, "agree": float("nan"), "n_folds": 0, "pass": False}
    bad = verify_walk_forward(splits, dates, cfg.horizon)
    if bad:
        raise InteractionError("walk-forward splits leak: " + "; ".join(bad))
    signed = []
    for s in splits:
        a = int(np.searchsorted(P.all_dates, np.datetime64(s["test_start"])))
        b = int(np.searchsorted(P.all_dates, np.datetime64(s["test_end"]), side="right"))
        st = series_stat(ts.slice(a, b), cfg.lags, min_n=12)
        if st.ok:
            signed.append(sign * st.t)
    agree = float(np.mean([t > 0 for t in signed])) if signed else float("nan")
    return {"usable": len(signed) >= 2, "agree": agree, "n_folds": len(signed),
            "pass": bool(len(signed) >= 2 and agree >= cfg.min_wf_agree)}


def cross_stock_check(Pv: Panel, tr: Trial, cfg: SearchConfig, sign: float) -> dict:
    """Split the universe into stock groups (sector, or a stable name hash). The effect must keep its sign in most groups and
    survive leaving any one group out. Groups with too few names for a cross-section are skipped and reported."""
    signed: dict[int, float] = {}
    logo: dict[int, float] = {}
    for g in np.unique(Pv.tgroup):
        members = Pv.tgroup == g
        if int(members.sum()) < cfg.min_names:
            continue
        rows = members[Pv.tcode]
        st = trial_stat(Pv.select(rows), tr, cfg)
        rest = trial_stat(Pv.select(~rows), tr, cfg)
        if st.ok:
            signed[int(g)] = sign * st.t
        if rest.ok:
            logo[int(g)] = sign * rest.t
    out = {"n_groups": len(signed), "usable": len(signed) >= 2, "per_group_signed_t": signed}
    out["agree"] = float(np.mean([t > 0 for t in signed.values()])) if signed else float("nan")
    out["min_leave_one_out_t"] = min(logo.values()) if logo else float("nan")
    out["pass"] = bool(out["usable"] and out["agree"] >= cfg.min_group_agree and out["min_leave_one_out_t"] >= cfg.robust_t)
    return out


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=int(math.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def fresh_seed_check(Pv: Panel, tr: Trial, cfg: SearchConfig, sign: float, seeds: SeedLedger, run_key: Any) -> dict:
    """Redo the held-out test under `n_fresh_seeds` fresh seeds: each draws a random share of the stocks and a moving-block
    bootstrap of the dates. A real interaction is significant in most draws; one that depended on particular stocks or
    particular weeks is not. Seeds are single-use and may never equal a discovery seed."""
    names = np.unique(Pv.tcode)
    signed = []
    for k in range(cfg.n_fresh_seeds):
        rng = seeds.rng("fresh", (tr.trial_id, run_key, k))
        chosen = names[rng.random(len(names)) < cfg.subsample_frac]
        ts = trial_series(Pv.select(np.isin(Pv.tcode, chosen)), tr, cfg)
        if len(ts) < 20:
            continue
        st = series_stat(ts.take(_block_indices(len(ts), 2 * cfg.horizon, rng)), cfg.horizon)
        if st.ok:
            signed.append(sign * st.t)
    passed = float(np.mean([t >= 1.645 for t in signed])) if signed else float("nan")
    return {"n_draws": len(signed), "usable": len(signed) >= max(3, cfg.n_fresh_seeds // 2), "pass_share": passed,
            "median_signed_t": float(np.median(signed)) if signed else float("nan"),
            "pass": bool(len(signed) >= max(3, cfg.n_fresh_seeds // 2) and passed >= cfg.seed_pass_frac)}


def _oos_design(P: Panel, tr: Trial, ref_mean: float, ref_sd: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(main-effects columns, main-effects + interaction columns) on this panel. Market-level covariates are standardised with
    the DISCOVERY window's mean/sd so validation data never informs its own scaling."""
    if tr.form is Form.MODULATION:
        b = P.row_atoms[tr.a]
        z = P.date_atoms[tr.b][P.date_pos].astype(float)
        if P.meta[tr.b].kind == "cont":
            z = (z - ref_mean) / (ref_sd if ref_sd > 1e-12 else 1.0)
        return [b], [b, b * z]
    a, b = P.row_atoms[tr.a], P.row_atoms[tr.b]
    return [a, b], [a, b, _term(P, tr)]


def _daily_returns(P: Panel, cols: Sequence[np.ndarray], beta: np.ndarray) -> pd.Series:
    """Return of a dollar-neutral portfolio weighted by the model's prediction, per date (gross exposure 1)."""
    pred = sum(_center(np.asarray(c, float), P) * b for c, b in zip(cols, beta))
    gross = np.add.reduceat(np.abs(pred), P.starts)
    ret = np.add.reduceat(pred * P.y, P.starts) / np.where(gross > 1e-12, gross, 1.0)
    return pd.Series(np.where(gross > 1e-12, ret, 0.0), index=pd.DatetimeIndex(P.all_dates[P.upos]))


def oos_compare(Pd: Panel, Pv: Panel, tr: Trial, cfg: SearchConfig) -> dict:
    """Does the interaction beat the same two features WITHOUT it, out of sample, once complexity is priced? Both models are fit
    on the discovery window only; their per-date long-short returns on the validation window (folds = calendar quarters) go to
    complexity.compare. That judges incremental gain, transfer across folds, stability, tail safety and optimism together."""
    ref = Pd.date_atoms[tr.b][Pd.upos] if tr.form is Form.MODULATION else np.zeros(1)
    mean, sd = float(ref.mean()), float(ref.std())
    simple_d, full_d = _oos_design(Pd, tr, mean, sd)
    simple_v, full_v = _oos_design(Pv, tr, mean, sd)
    yc = _center(Pd.y, Pd)
    series = {}
    for name, cd, cv in (("simple", simple_d, simple_v), ("full", full_d, full_v)):
        M = np.column_stack([_center(np.asarray(c, float), Pd) for c in cd])
        beta = np.linalg.lstsq(M, yc, rcond=None)[0]
        series[name] = (_daily_returns(Pv, cv, beta), float(_daily_returns(Pd, cd, beta).mean()))
    folds = pd.Series(pd.PeriodIndex(series["full"][0].index, freq="Q").astype(str), index=series["full"][0].index)
    cand_s = CX.Candidate(tr.simple_spec(), series["simple"][0], folds, series["simple"][1])
    cand_c = CX.Candidate(tr.spec(), series["full"][0], folds, series["full"][1])
    v = CX.compare(cand_s, cand_c, cfg.complexity)
    return {"verdict": str(v.verdict), "gain": v.gain, "gain_t": v.gain_t, "t_required": v.t_required,
            "transfer": v.transfer, "delta_units": v.delta_units, "n": v.n, "reasons": list(v.reasons),
            "usable": v.verdict != CX.Verdict.NEED_DATA, "pass": v.verdict == CX.Verdict.COMPLEX}


# ---------------------------------------------------------------------------------------------------- findings and report
_EPISTEMIC = {Fate.NOISE: Epistemic.UNKNOWN, Fate.INTERESTING_ONLY: Epistemic.HYPOTHESIS,
              Fate.SURVIVED_CORRECTION: Epistemic.HYPOTHESIS, Fate.NEEDS_MORE_EVIDENCE: Epistemic.HYPOTHESIS,
              Fate.REJECTED_COMPLEXITY: Epistemic.HYPOTHESIS, Fate.REJECTED_CROSS_YEAR: Epistemic.HYPOTHESIS,
              Fate.REJECTED_OOS_COMPLEXITY: Epistemic.HYPOTHESIS, Fate.REJECTED_VALIDATION: Epistemic.CONTRADICTED,
              Fate.REJECTED_CROSS_STOCK: Epistemic.CONTRADICTED, Fate.REJECTED_SEEDS: Epistemic.CONTRADICTED,
              Fate.REJECTED_HOLDOUT: Epistemic.CONTRADICTED, Fate.VALIDATED: Epistemic.SUPPORTED}

_VALIDATED_KEYS = ("disc_t", "q", "maxt_p", "valid_t", "hold_t", "oos_gain_t")


@dc.dataclass(frozen=True)
class Finding:
    trial: Trial
    fate: Fate
    stage: Stage
    units: float
    stats: Mapping[str, float]
    reasons: tuple[str, ...]

    @property
    def epistemic(self) -> Epistemic:
        return _EPISTEMIC[self.fate]

    def validate(self) -> list[str]:
        errs = []
        if not self.reasons:
            errs.append("a finding without a reason is not auditable")
        if self.fate is Fate.VALIDATED:
            missing = [k for k in _VALIDATED_KEYS if not math.isfinite(self.stats.get(k, float("nan")))]
            if missing:
                errs.append(f"VALIDATED without evidence for {missing}")
            if self.stage is not Stage.FRESH_HOLDOUT:
                errs.append("VALIDATED must have reached the fresh holdout stage")
        if self.fate in _TERMINAL_REJECTIONS and self.stage is Stage.CHEAP_SCREEN:
            errs.append("a later-stage rejection recorded at the cheap screen")
        if self.units <= 0:
            errs.append("units must be positive")
        return errs

    def payload(self) -> dict:
        """Identity-free content for a MaturedRecord: no dates, years or tickers, only the structure and the evidence."""
        return {"family": self.trial.family, "a": self.trial.a, "b": self.trial.b, "form": str(self.trial.form),
                "quantile": self.trial.q, "horizon": self.trial.horizon, "fate": str(self.fate), "units": self.units,
                "stats": {k: (None if not math.isfinite(v) else float(v)) for k, v in self.stats.items()},
                "epistemic": str(self.epistemic)}


@dc.dataclass(frozen=True)
class SearchReport:
    run_no: int
    data_key: str
    n_trials: int
    m_total: int
    findings: tuple[Finding, ...]
    skipped: tuple[Mapping[str, Any], ...]
    lottery: Mapping[str, Any]
    control: Mapping[str, Any]
    windows: Mapping[str, Any]
    dropped_unmatured: int
    evidence_end: str
    config_hash: str
    notes: tuple[str, ...] = ()

    def by_fate(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[str(f.fate)] = out.get(str(f.fate), 0) + 1
        return out

    def validated(self) -> list[Finding]:
        return [f for f in self.findings if f.fate is Fate.VALIDATED]

    def audit(self) -> list[str]:
        """Invariants a report must satisfy whatever the data was (used by the tests and by the wave-2 loop)."""
        errs = []
        if self.m_total < self.n_trials:
            errs.append("m_total below the number of trials actually tested")
        for f in self.findings:
            errs += [f"{f.trial.trial_id}: {e}" for e in f.validate()]
        ids = [f.trial.trial_id for f in self.findings]
        if len(ids) != len(set(ids)):
            errs.append("duplicate trial ids in one report")
        if self.lottery.get("verdict") == "LOTTERY" and self.validated():
            errs.append("search called a lottery yet reports validated interactions")
        return errs


def empty_report(run_no: int, data_key: str, note: str, config_hash: str = "") -> SearchReport:
    return SearchReport(run_no, data_key, 0, 0, (), (), {"verdict": "EMPTY", "m_tested": 0}, {}, {}, 0, "", config_hash, (note,))


def render_text(rep: SearchReport, top: int = 12) -> str:
    """Plain-language report: what was tried, how much chance alone explains, what survived, and why the rest did not."""
    lo = rep.lottery
    lines = [f"INTERACTION SEARCH run {rep.run_no}  data {rep.data_key}",
             f"combinations tested this run {rep.n_trials}; ledger total for this data {rep.m_total}",
             f"raw hits {lo.get('raw_hits', 0)} vs {lo.get('expected_raw_hits', 0):.1f} expected from chance alone; "
             f"verdict {lo.get('verdict')}; Bonferroni bar |t| >= {lo.get('bonferroni_t', float('nan')):.2f}"]
    if rep.control:
        c = rep.control
        lines.append(f"shuffled controls: median max|t| {c['null_max_t_median']:.2f}, 95th {c['null_max_t_p95']:.2f}; "
                     f"the search 'discovers' something in {c['pipeline_fp_rate']:.0%} of label-shuffled reruns")
    lines.append("fates: " + ", ".join(f"{k}={v}" for k, v in sorted(rep.by_fate().items())))
    lines.append(f"labels not yet matured and dropped: {rep.dropped_unmatured}; skipped combinations: {len(rep.skipped)}")
    ranked = sorted(rep.findings, key=lambda f: (f.fate is not Fate.VALIDATED, -abs(f.stats.get("disc_t", 0.0) or 0.0)))
    for f in ranked[:top]:
        lines.append(f"  {f.fate:<24} t={f.stats.get('disc_t', float('nan')):+.2f} q={f.stats.get('q', float('nan')):.3g} "
                     f"{f.trial.text()}")
        lines.append(f"      {f.reasons[0]}")
    lines += [f"NOTE {n}" for n in rep.notes]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- state and the search
class InteractionState:
    """Everything that must persist between searches: the ledger of tried combinations, single-use seeds, the one-shot
    holdout vault and a compact history. Mutated in place by step()."""

    def __init__(self, base_seed: int = 7) -> None:
        self.ledger = TrialLedger()
        self.seeds = SeedLedger(base_seed)
        self.vault = HoldoutVault()
        self.history: list[dict] = []


@dc.dataclass
class _Work:
    trial: Trial
    disc: Stat
    units: float
    q: float = 1.0
    maxt_p: float = 1.0
    fate: Fate = Fate.NOISE
    stage: Stage = Stage.CHEAP_SCREEN
    reasons: list = dc.field(default_factory=list)
    stats: dict = dc.field(default_factory=dict)

    def close(self, fate: Fate, stage: Stage, reason: str) -> None:
        self.fate, self.stage = fate, stage
        self.reasons.append(reason)

    def freeze(self) -> Finding:
        st = {"disc_beta": self.disc.beta, "disc_t": self.disc.t, "disc_p": self.disc.p, "q": self.q, "maxt_p": self.maxt_p}
        st.update(self.stats)
        return Finding(self.trial, self.fate, self.stage, self.units, st, tuple(self.reasons) or ("no reason recorded",))


def config_hash(cfg: SearchConfig) -> str:
    d = dc.asdict(cfg)
    d["quantiles"] = list(cfg.quantiles)
    return stable_hash(d, 12)


def _screen(Pd: Panel, trials: Sequence[Trial], cfg: SearchConfig, state: InteractionState, run_key: int,
            m_total: int) -> tuple[list[_Work], ShuffleControl, np.ndarray]:
    """Stage CHEAP_SCREEN + STRONGER_TESTS on the discovery window. Returns one _Work per trial."""
    stats = [trial_stat(Pd, tr, cfg) for tr in trials]
    p = np.array([s.p if s.ok else 1.0 for s in stats])
    q = corrected_q(p, m_total, cfg.correction)
    control = run_shuffled_controls(Pd, trials, cfg, state.seeds, run_key, m_total)
    n_eff = Pd.n_dates / max(cfg.horizon, 1)
    works = []
    for tr, st, qi in zip(trials, stats, q):
        w = _Work(tr, st, tr.spec().units(cfg.complexity.weights), q=float(qi))
        works.append(w)
        if not st.ok:
            w.close(Fate.NOISE, Stage.CHEAP_SCREEN, "statistic undefined on the discovery window (degenerate or too few dates)")
            continue
        w.maxt_p = control.adj_p(st.t)
        if st.p >= cfg.alpha_raw:
            w.close(Fate.NOISE, Stage.CHEAP_SCREEN, f"raw p={st.p:.3g}: no effect to explain")
        elif qi > cfg.alpha_screen:
            w.close(Fate.INTERESTING_ONLY, Stage.CHEAP_SCREEN,
                    f"raw p={st.p:.3g} but q={qi:.3g} after correcting for {m_total} combinations tried; chance alone "
                    f"gives about {len(trials) * cfg.alpha_raw:.1f} raw hits in this search")
        elif w.maxt_p > cfg.alpha_fwer:
            w.close(Fate.INTERESTING_ONLY, Stage.STRONGER_TESTS,
                    f"passes the BH screen (q={qi:.3g}) but the best of all trials on shuffled labels beats |t|={abs(st.t):.2f} "
                    f"with family-wise p={w.maxt_p:.3g}")
        else:
            delta = w.units - tr.simple_spec().units(cfg.complexity.weights)
            need = CX.required_t(delta, cfg.complexity)
            budget = CX.max_units(n_eff)
            w.stats.update(t_required=need, delta_units=delta)
            if abs(st.t) < need:
                w.close(Fate.REJECTED_COMPLEXITY, Stage.STRONGER_TESTS,
                        f"|t|={abs(st.t):.2f} is below the {need:.2f} that {delta:.1f} extra complexity units demand")
            elif delta > budget:
                w.close(Fate.REJECTED_COMPLEXITY, Stage.STRONGER_TESTS,
                        f"{delta:.1f} extra complexity units exceed the {budget:.1f} that {n_eff:.0f} independent periods can support")
            else:
                w.close(Fate.SURVIVED_CORRECTION, Stage.STRONGER_TESTS,
                        f"|t|={abs(st.t):.2f}, q={qi:.3g}, family-wise p={w.maxt_p:.3g} against {control.n} shuffled reruns")
    return works, control, p


def _validate_survivors(P: Panel, win: Windows, works: list[_Work], cfg: SearchConfig, state: InteractionState,
                        run_key: int) -> None:
    """Stages CROSS_YEAR then FRESH_HOLDOUT for every _Work still SURVIVED_CORRECTION. Mutates the works."""
    live = [w for w in works if w.fate is Fate.SURVIVED_CORRECTION]
    if not live:
        return
    Pdv = P.window(0, win.valid[1])
    series = {w.trial.trial_id: trial_series(Pdv, w.trial, cfg) for w in live}
    for w in live:
        sign = _sgn(w.disc.beta)
        ts = series[w.trial.trial_id]
        cy = cross_year_check(ts, P, cfg, sign)
        w.stats.update(cy_agree=cy["agree"], cy_drop_best_t=cy["drop_best_t"], cy_years=float(cy["n_years"]))
        wf = walk_forward_check(ts.slice(*win.disc), P, win.disc[0], win.disc[1], cfg, sign)
        w.stats.update(wf_agree=wf["agree"])
        if not cy["usable"]:
            w.close(Fate.NEEDS_MORE_EVIDENCE, Stage.CROSS_YEAR, f"only {cy['n_years']} usable year(s): cross-year transfer is untested")
        elif not cy["pass"]:
            w.close(Fate.REJECTED_CROSS_YEAR, Stage.CROSS_YEAR,
                    f"sign agrees in {cy['agree']:.0%} of {cy['n_years']} years and t is {cy['drop_best_t']:.2f} without the best year")
        elif wf["usable"] and not wf["pass"]:
            w.close(Fate.REJECTED_CROSS_YEAR, Stage.CROSS_YEAR, f"walk-forward agreement {wf['agree']:.0%} over {wf['n_folds']} purged folds")
    live = [w for w in works if w.fate is Fate.SURVIVED_CORRECTION]
    if not live:
        return
    Pd, Pv = P.window(*win.disc), P.window(*win.valid)
    vstat = [trial_stat(Pv, w.trial, cfg) for w in live]
    signs = [_sgn(w.disc.beta) for w in live]
    pv = np.array([one_sided_p(s.t, sg) for s, sg in zip(vstat, signs)])
    m_valid = state.ledger.validation_count(P.data_key) + len(live)
    qv = corrected_q(pv, m_valid, "bh")
    state.ledger.note_validated(P.data_key, [w.trial.trial_id for w in live])
    for w, st, sg, qi in zip(live, vstat, signs, qv):
        w.stats.update(valid_t=st.t if st.ok else float("nan"), valid_beta=st.beta, valid_q=float(qi))
        if not st.ok or qi > cfg.alpha_valid:
            w.close(Fate.REJECTED_VALIDATION, Stage.FRESH_HOLDOUT,
                    f"held-out window: t={st.t:.2f} in the discovery direction, q={qi:.3g} (needs <= {cfg.alpha_valid})")
    for w in [w for w in live if w.fate is Fate.SURVIVED_CORRECTION]:
        sign = _sgn(w.disc.beta)
        cs = cross_stock_check(Pv, w.trial, cfg, sign)
        w.stats.update(cs_agree=cs["agree"], cs_min_logo_t=cs["min_leave_one_out_t"])
        if not cs["usable"]:
            w.close(Fate.NEEDS_MORE_EVIDENCE, Stage.FRESH_HOLDOUT, f"only {cs['n_groups']} stock group(s) large enough: cross-stock transfer untested")
            continue
        if not cs["pass"]:
            w.close(Fate.REJECTED_CROSS_STOCK, Stage.FRESH_HOLDOUT,
                    f"sign holds in {cs['agree']:.0%} of {cs['n_groups']} stock groups; weakest leave-one-group-out t={cs['min_leave_one_out_t']:.2f}")
            continue
        fs = fresh_seed_check(Pv, w.trial, cfg, sign, state.seeds, run_key)
        w.stats.update(seed_pass=fs["pass_share"], seed_median_t=fs["median_signed_t"])
        if not fs["usable"]:
            w.close(Fate.NEEDS_MORE_EVIDENCE, Stage.FRESH_HOLDOUT, "too few fresh-seed draws produced a testable panel")
            continue
        if not fs["pass"]:
            w.close(Fate.REJECTED_SEEDS, Stage.FRESH_HOLDOUT,
                    f"significant in {fs['pass_share']:.0%} of {fs['n_draws']} fresh stock/date draws (needs {cfg.seed_pass_frac:.0%})")
            continue
        oc = oos_compare(Pd, Pv, w.trial, cfg)
        w.stats.update(oos_gain=oc["gain"], oos_gain_t=oc["gain_t"], oos_transfer=oc["transfer"])
        if not oc["usable"]:
            w.close(Fate.NEEDS_MORE_EVIDENCE, Stage.FRESH_HOLDOUT, "too few out-of-sample periods to compare with the main-effects rule")
        elif not oc["pass"]:
            w.close(Fate.REJECTED_OOS_COMPLEXITY, Stage.FRESH_HOLDOUT,
                    f"out of sample the main-effects rule does as well ({oc['verdict']}): " + "; ".join(oc["reasons"][:1]))
    _holdout(P, win, [w for w in works if w.fate is Fate.SURVIVED_CORRECTION], cfg, state)


def _holdout(P: Panel, win: Windows, live: list[_Work], cfg: SearchConfig, state: InteractionState) -> None:
    """The one-shot fresh window. Candidates already opened on it are not re-tested (the window is spent for them)."""
    if not live:
        return
    if win.hold is None:
        for w in live:
            w.close(Fate.NEEDS_MORE_EVIDENCE, Stage.FRESH_HOLDOUT, "passed everything except the fresh holdout, which is too short to test")
        return
    key = stable_hash({"k": P.data_key, "lo": str(P.all_dates[win.hold[0]]), "hi": str(P.all_dates[win.hold[1] - 1])}, 12)
    fresh = [w for w in live if not state.vault.was_opened(key, w.trial.trial_id)]
    for w in live:
        if w not in fresh:
            w.close(Fate.NEEDS_MORE_EVIDENCE, Stage.FRESH_HOLDOUT, "this candidate already used up the fresh holdout window")
    if not fresh:
        return
    m_cum = state.vault.open(key, [w.trial.trial_id for w in fresh])
    Ph = P.window(*win.hold)
    for w in fresh:
        st = trial_stat(Ph, w.trial, cfg)
        p = min(1.0, one_sided_p(st.t, _sgn(w.disc.beta)) * m_cum)
        w.stats.update(hold_t=st.t if st.ok else float("nan"), hold_beta=st.beta, hold_p=p)
        if st.ok and p <= cfg.alpha_valid:
            w.close(Fate.VALIDATED, Stage.FRESH_HOLDOUT,
                    f"held-out, cross-year, cross-stock, fresh-seed and out-of-sample tests passed; fresh holdout t={st.t:.2f} "
                    f"(Bonferroni over {m_cum} holdout looks p={p:.3g})")
        else:
            w.close(Fate.REJECTED_HOLDOUT, Stage.FRESH_HOLDOUT,
                    f"fresh holdout t={st.t:.2f}, adjusted p={p:.3g} over {m_cum} looks: it did not survive data nobody had touched")


def step(state: InteractionState, inputs: InteractionInputs, now, cfg: SearchConfig | None = None,
         families: Sequence[Family] = FAMILIES) -> SearchReport:
    """One interaction search on data strictly before `now`. Registers every tested combination, corrects against the
    cumulative count, and returns a SearchReport. Too little data returns an empty, explained report (never a guess)."""
    cfg = cfg or SearchConfig(horizon=inputs.horizon)
    run_key = state.ledger.runs
    chash = config_hash(cfg)
    try:
        P = prepare(inputs, now, cfg)
    except InteractionError as e:
        if str(e) in ("empty panel", "fewer sessions than the horizon allows", "no matured, labelled rows"):
            state.ledger.runs += 1
            return empty_report(run_key, "", f"no search run: {e}", chash)
        raise
    if P.all_dates.size < 3 * cfg.min_dates_window // 2 + 2 * (cfg.horizon + 1):
        state.ledger.runs += 1
        return empty_report(run_key, P.data_key, f"only {P.all_dates.size} matured sessions: below the minimum to hold out anything", chash)
    win = split_windows(int(P.all_dates.size), cfg)
    Pd = P.window(*win.disc)
    trials, skipped = enumerate_trials(P, cfg, state.seeds, run_key, families)
    if not trials:
        state.ledger.runs += 1
        return empty_report(run_key, P.data_key, "no testable combinations among the supplied atoms", chash)
    m_total = state.ledger.register(P.data_key, trials)
    works, control, pvals = _screen(Pd, trials, cfg, state, run_key, m_total)
    survivors = sum(1 for w in works if w.fate is Fate.SURVIVED_CORRECTION)
    lottery = lottery_diagnostic(pvals, m_total, survivors, cfg.alpha_raw)
    _validate_survivors(P, win, works, cfg, state, run_key)
    lottery = {**lottery, "validated": sum(1 for w in works if w.fate is Fate.VALIDATED)}
    if lottery["validated"] == 0 and lottery["verdict"] == "SIGNAL":
        lottery["verdict"] = "SIGNAL_NOT_VALIDATED"
    findings = tuple(w.freeze() for w in works)
    notes = []
    if state.ledger.repeats(P.data_key):
        notes.append(f"{state.ledger.repeats(P.data_key)} combinations were tried before on this data; the correction counts all {m_total}")
    if win.hold is None:
        notes.append("no fresh holdout window: nothing can be VALIDATED")
    if control.pipeline_fp_rate > 2 * max(cfg.alpha_fwer, cfg.alpha_screen):
        notes.append(f"the search machinery false-positive rate is {control.pipeline_fp_rate:.0%} on shuffled labels: do not trust its findings")
    rep = SearchReport(run_key, P.data_key, len(trials), m_total, findings, tuple(skipped), lottery, control.summary(),
                       {"discovery_dates": Pd.n_dates, "validation_dates": win.valid[1] - win.valid[0],
                        "holdout_dates": 0 if win.hold is None else win.hold[1] - win.hold[0], "purge_gap": win.gap},
                       P.dropped_unmatured, P.last_date, chash, tuple(notes))
    state.history.append({"run": run_key, "data_key": P.data_key, "n_trials": len(trials), "m_total": m_total,
                          "fates": rep.by_fate(), "verdict": lottery["verdict"]})
    state.ledger.runs += 1
    return rep


def to_matured_records(rep: SearchReport, created_real: str, tickers: Iterable[str] = (), seed: int | None = None) -> list[MaturedRecord]:
    """VALIDATED interactions as MATURED_RESEARCH_STATE records. `matured_at` is the newest date of the data searched, so
    MaturedRecord.gate(now) refuses any decision at or before it - in particular a disguised replay of the same year, which is
    the same-year rerun leak. The payload is scrubbed of dates, years and tickers before the record exists."""
    if not rep.evidence_end:
        return []
    out = []
    tick = tuple(tickers)
    for f in rep.validated():
        bad = f.validate()
        if bad:
            raise InteractionError("refusing to release an invalid finding: " + "; ".join(bad))
        payload = f.payload()
        assert_identity_free(payload, tick, where="interaction payload")
        prov = Provenance(created_real=created_real, learned_at=rep.evidence_end, code_hash=current_code_hash() or "unknown",
                          config_hash=rep.config_hash, experiment_id=rep.data_key, run_id=f"interactions-{rep.run_no}",
                          seed=seed, outcomes_seen_through=rep.evidence_end)
        out.append(MaturedRecord("IR" + stable_hash({"t": f.trial.trial_id, "k": rep.data_key}, 12), rep.evidence_end,
                                 payload, prov))
    return out
