"""Bible Phase 3.1 - candidate generation for the pattern miner (canons C35, C37, C43).

What the miner is allowed to try, made explicit, deterministic and auditable:
  * the FEATURE UNIVERSE: the 54 engine columns (46 per-stock features + 8 market-context columns), the 25 candle /
    micro signals from engine.candles, each transformed to quintiles - cross-sectionally per date for stock features,
    and by an EXPANDING (point-in-time) time-series rank for market context, whose cross-sectional rank is constant;
  * the candidate FAMILIES: every single (feature x quintile), pairs (top-scoring singles crossed, plus SEEDED
    random pairs drawn without replacement so small unexpected interactions get a chance), 'A AND B UNLESS C'
    exceptions, and bank re-tests of patterns found in earlier windows;
  * DETERMINISTIC IDS: a candidate's id is its identity hash (engine.pattern_identity), independent of the order it was
    generated in, of the seed, and of the process (no builtin hash()).
`coverage_audit` proves - or refutes - that every feature in the universe is actually reachable by the search.
Nothing here reads outcomes; scores that steer pair selection are passed in by the caller."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .pattern_identity import Expression, IdentityError, N_LEVELS, Term, pattern_id

ENGINE_FEATURES = (
    "r1", "r5", "r20", "r60", "r120", "mom_12_1", "dist_52wh", "dist_ma50", "dist_ma200", "vol20", "vol_ratio",
    "atr_pct", "max20", "min20", "skew60", "log_dv", "vol_surge1", "vol_surge5", "overnight20", "intraday20", "frog",
    "range_compress", "gap_today", "close_loc", "ind_mom20", "rel_ind20", "ind_mom60", "rel_ind60", "days_since_earn",
    "ear", "ear_volsurge", "days_to_earn", "earn_in_week", "ev_offering", "ev_shelf", "ev_activist", "ev_activist_amend",
    "ev_agreement", "ev_red_flag", "news5", "r5_nonews", "r5_news", "ins_buyers30", "ins_value30", "ins_officer30",
    "ins_opportunistic30")
MARKET_CONTEXT = ("m_spy_ma50", "m_spy_ma200", "m_spy_r5", "m_vix", "m_vix_chg5", "m_vix_term", "m_breadth",
                  "m_dispersion")
CANDLE_SIGNALS = (
    "cd_body", "cd_upwick", "cd_lowwick", "cd_pos", "cd_range", "cw_body", "cw_upwick", "cw_lowwick", "cw_pos",
    "cw_range", "cm_body", "cm_upwick", "cm_lowwick", "cm_pos", "cm_range", "streak", "reversal_1d",
    "reversal_vs_week", "inside_day", "outside_day", "engulf", "gap", "gap_filled", "week_vs_month_pos",
    "day_vs_week_body")
FAMILY_OF = {"engine": ENGINE_FEATURES, "context": MARKET_CONTEXT, "candle": CANDLE_SIGNALS}

CAND_DEFAULT = {"seed": 7, "top_singles": 60, "max_pairs": 4000, "max_unless": 600, "unless_top_pairs": 40,
                "unless_thirds": 15, "unless_levels": (0, 4), "min_history": 60, "min_random_frac": 0.25,
                "context_top": 20, "context_share": 0.3}


# ---------------------------------------------------------------- the universe
@dataclass(frozen=True)
class Universe:
    """Named groups of feature columns. `all` is sorted, so nothing downstream depends on column order."""
    engine: tuple = ENGINE_FEATURES
    candle: tuple = CANDLE_SIGNALS
    context: tuple = MARKET_CONTEXT

    def __post_init__(self):
        allf = list(self.engine) + list(self.candle) + list(self.context)
        dup = sorted({f for f in allf if allf.count(f) > 1})
        if dup:
            raise IdentityError(f"feature(s) named in two groups: {dup}")
        for f in allf:
            if not f or " " in f or "&" in f:
                raise IdentityError(f"feature name {f!r} cannot be written into an expression")

    @property
    def all(self) -> tuple:
        return tuple(sorted(list(self.engine) + list(self.candle) + list(self.context)))

    @property
    def stock_features(self) -> tuple:
        return tuple(sorted(list(self.engine) + list(self.candle)))

    def group_of(self, feature: str) -> str:
        for g in ("engine", "candle", "context"):
            if feature in getattr(self, g):
                return g
        raise KeyError(feature)

    def is_context(self, feature: str) -> bool:
        return feature in self.context

    def transform_of(self, expr: Expression) -> str:
        """'ts_quintile' if any term is market context (its level comes from an expanding time-series rank, so the
        whole condition is only meaningful with that transform), else cross-sectional."""
        return "ts_quintile" if any(self.is_context(f) for f in expr.features) else "xs_quintile"

    def restrict(self, available: Iterable[str]) -> "Universe":
        have = set(available)
        return Universe(tuple(f for f in self.engine if f in have), tuple(f for f in self.candle if f in have),
                        tuple(f for f in self.context if f in have))


def universe_from_columns(columns: Iterable[str], candle_names: Iterable[str] = CANDLE_SIGNALS) -> Universe:
    """Classify whatever columns a panel really has: known context, known candle, everything else engine."""
    cols = list(columns)
    ctx = tuple(c for c in cols if c.startswith("m_"))
    cs = set(candle_names)
    candle = tuple(c for c in cols if c in cs)
    engine = tuple(c for c in cols if c not in set(ctx) | set(candle) and c not in ("date", "ticker"))
    return Universe(engine, candle, ctx)


# ---------------------------------------------------------------- quantile transforms
def xs_quintile_codes(X: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Cross-sectional quintile per date: percentile rank within the day's stocks, NaN -> the middle. Identical to
    PatternMiner._quintiles so masks built here equal the miner's."""
    Q = np.empty((len(X), len(cols)), dtype=np.int8)
    for j, c in enumerate(cols):
        r = X[c].groupby(level=0).rank(pct=True).fillna(0.5).values
        Q[:, j] = np.minimum((r * N_LEVELS).astype(np.int8), N_LEVELS - 1)
    return Q


def ts_quintile_series(values: pd.Series, min_history: int = 60) -> pd.Series:
    """Expanding, point-in-time quintile of a per-date series: each date is ranked only against dates up to and
    including itself, so nothing after a date can change its level. Dates with fewer than `min_history` valid past
    values (and NaN dates) get -1 = unknown, which matches no level and is no exception either."""
    v = values.astype(float)
    out = np.full(len(v), -1, dtype=np.int8)
    hist = []                                       # sorted list of past values
    import bisect
    for i, x in enumerate(v.values):
        if not np.isfinite(x):
            continue
        n_before = len(hist)
        bisect.insort(hist, x)
        if n_before + 1 < min_history:
            continue
        lo = bisect.bisect_left(hist, x)
        hi = bisect.bisect_right(hist, x)
        pct = (lo + hi) / 2.0 / len(hist)           # mid-rank: ties do not all pile into the top quintile
        out[i] = min(int(pct * N_LEVELS), N_LEVELS - 1)
    return pd.Series(out, index=v.index)


def ts_quintile_codes(X: pd.DataFrame, cols: Sequence[str], min_history: int = 60) -> np.ndarray:
    """Time-series quintile per row for per-date columns: computed on the one-value-per-date series, then broadcast."""
    Q = np.full((len(X), len(cols)), -1, dtype=np.int8)
    dates = X.index.get_level_values(0)
    for j, c in enumerate(cols):
        per_date = X[c].groupby(level=0).first()
        lv = ts_quintile_series(per_date.sort_index(), min_history)
        Q[:, j] = lv.reindex(dates).fillna(-1).values.astype(np.int8)
    return Q


@dataclass
class QuantileMatrix:
    """Quantised feature matrix for row masks. `features` names the columns of Q; `missing` lists universe features
    the panel did not have (reported, never silently dropped)."""
    features: tuple
    Q: np.ndarray
    universe: Universe
    missing: tuple = ()

    def mask(self, expr: Expression) -> np.ndarray:
        return expr.mask(self.Q, self.features)

    def occupancy(self) -> pd.DataFrame:
        """Rows per level per feature: a feature with fewer than N_LEVELS occupied levels cannot host every single."""
        rows = []
        for j, f in enumerate(self.features):
            counts = np.bincount(np.maximum(self.Q[:, j], 0)[self.Q[:, j] >= 0], minlength=N_LEVELS)
            rows.append({"feature": f, "group": self.universe.group_of(f),
                         **{f"q{q}": int(counts[q]) for q in range(N_LEVELS)},
                         "unknown": int((self.Q[:, j] < 0).sum()),
                         "levels_occupied": int((counts > 0).sum())})
        return pd.DataFrame(rows)


def quantise(X: pd.DataFrame, universe: Universe, min_history: int = 60) -> QuantileMatrix:
    """Build the quantile matrix for every universe feature present in X (index (date, ticker))."""
    if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels != 2:
        raise IdentityError("X must be indexed by (date, ticker)")
    missing = tuple(f for f in universe.all if f not in X.columns)
    u = universe.restrict(X.columns)
    stock, ctx = list(u.stock_features), list(u.context)
    Qs = xs_quintile_codes(X, stock) if stock else np.empty((len(X), 0), np.int8)
    Qc = ts_quintile_codes(X, ctx, min_history) if ctx else np.empty((len(X), 0), np.int8)
    return QuantileMatrix(tuple(stock + ctx), np.hstack([Qs, Qc]), u, missing)


# ---------------------------------------------------------------- candidates
@dataclass(frozen=True)
class Candidate:
    expression: Expression
    family: str                # single | pair | unless | bank
    origin: str                # all_single | top_pair | random_pair | unless | bank
    transform: str = "xs_quintile"
    target: str = "excess_5d"

    @property
    def id(self) -> str:
        return pattern_id(self.expression, self.transform, self.target)

    @property
    def text(self) -> str:
        return self.expression.text


class CandidateSet:
    """Ordered, de-duplicated candidates keyed by deterministic id. When two origins propose the same idea the first
    keeps its origin and the later one is counted in `collisions` (so the audit can show the overlap, not hide it)."""

    def __init__(self):
        self._c = {}
        self.collisions = {}

    def __len__(self):
        return len(self._c)

    def __iter__(self):
        return iter(self._c.values())

    def __contains__(self, item):
        return (item if isinstance(item, str) else item.id) in self._c

    def add(self, cand: Candidate) -> bool:
        if cand.id in self._c:
            key = (self._c[cand.id].origin, cand.origin)
            self.collisions[key] = self.collisions.get(key, 0) + 1
            return False
        self._c[cand.id] = cand
        return True

    def extend(self, cands: Iterable[Candidate]) -> int:
        return sum(self.add(c) for c in cands)

    def ids(self) -> list:
        return list(self._c)

    def texts(self) -> list:
        return [c.text for c in self._c.values()]

    def counts(self) -> dict:
        out = {}
        for c in self._c.values():
            out[c.origin] = out.get(c.origin, 0) + 1
        return dict(sorted(out.items()))

    def features_used(self) -> set:
        s = set()
        for c in self._c.values():
            s |= set(c.expression.features)
        return s

    def digest(self) -> str:
        """Fingerprint of the id set, order-independent: same candidates -> same digest, whatever the generation order."""
        return hashlib.sha256("\n".join(sorted(self._c)).encode()).hexdigest()

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"id": c.id, "text": c.text, "family": c.family, "origin": c.origin,
                              "transform": c.transform, "order": c.expression.order} for c in self._c.values()])


def _cand(expr: Expression, uni: Universe, family: str, origin: str, target: str) -> Candidate:
    return Candidate(expr, family, origin, uni.transform_of(expr), target)


def single_candidates(uni: Universe, target: str = "excess_5d") -> list:
    """Every feature at every level: |universe| x 5 singles, in sorted feature order."""
    return [_cand(Expression.single(f, q), uni, "single", "all_single", target)
            for f in uni.all for q in range(N_LEVELS)]


def top_terms(scores: Mapping[str, float], k: int) -> list:
    """The k strongest single terms from {single_text: |t|}. Ties break on the text so the choice is reproducible."""
    ranked = sorted(scores.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    return [Expression.parse(t).base[0] for t, _ in ranked[:k]]


def top_pair_candidates(terms: Sequence[Term], uni: Universe, target: str = "excess_5d") -> list:
    """All cross-feature pairs among the strongest single terms (two levels of one feature is empty, so skipped).
    Ordered by RANK SUM of the two terms (strongest-with-strongest first), so a pair budget smaller than the pool keeps
    the top-k x top-k square instead of only the first few terms crossed with everything."""
    idx = sorted(((i + j, i, j) for i, j in itertools.combinations(range(len(terms)), 2)))
    out = []
    for _, i, j in idx:
        a, b = terms[i], terms[j]
        if a.feature == b.feature:
            continue
        out.append(_cand(Expression.make([a, b]), uni, "pair", "top_pair", target))
    return out


def context_pair_candidates(terms: Sequence[Term], uni: Universe, target: str = "excess_5d") -> list:
    """The strongest STOCK terms crossed with every market-context term (all five time-series levels of every context
    column). A context column has the same value for every stock on a date, so against a per-date demeaned outcome its
    single-term effect is exactly zero: it can only matter in interaction with a stock feature, and it can never earn a
    place among the 'strongest singles'. This family is how a regime pattern gets tested on purpose, not by luck."""
    ctx = [f for f in uni.all if uni.is_context(f)]
    out = []
    for t in terms:
        if uni.is_context(t.feature):
            continue
        for f in ctx:
            for q in range(N_LEVELS):
                out.append(_cand(Expression.make([t, Term(f, q)]), uni, "pair", "context_pair", target))
    return out


def _pair_space(features: Sequence[str]):
    combos = list(itertools.combinations(sorted(features), 2))
    return combos, len(combos) * N_LEVELS * N_LEVELS


def random_pair_candidates(uni: Universe, n: int, rng: np.random.Generator, exclude: Optional[set] = None,
                           target: str = "excess_5d", empty: Optional[set] = None) -> list:
    """n distinct random pairs drawn WITHOUT replacement from the whole pair space (feature pairs x 25 level combos),
    skipping any id in `exclude`. engine.patterns draws with replacement and then de-duplicates, silently returning
    fewer than asked; here the count is honoured until the space is exhausted. `empty` = {(feature, level)} holding no
    testable rows: a pair containing one can never pass the miner's gate, so the budget is not spent on it."""
    combos, size = _pair_space(uni.all)
    exclude = exclude or set()
    empty = empty or set()
    out, seen = [], set()
    if n <= 0 or size == 0:
        return out
    order = rng.permutation(size)                    # a full permutation: size is ~10^5, and it makes exhaustion exact
    for flat in order:
        if len(out) >= n:
            break
        pi, lv = divmod(int(flat), N_LEVELS * N_LEVELS)
        (fa, fb), (qa, qb) = combos[pi], divmod(lv, N_LEVELS)
        if (fa, qa) in empty or (fb, qb) in empty:
            continue
        cand = _cand(Expression.make([Term(fa, qa), Term(fb, qb)]), uni, "pair", "random_pair", target)
        if cand.id in exclude or cand.id in seen:
            continue
        seen.add(cand.id)
        out.append(cand)
    return out


def unless_candidates(ranked_pairs: Sequence[Expression], uni: Universe, rng: np.random.Generator,
                      top_pairs: int = 40, thirds: int = 15, levels: Sequence[int] = (0, 4), max_total: int = 600,
                      target: str = "excess_5d", empty: Optional[set] = None) -> list:
    """'A AND B UNLESS C': for each of the strongest pairs, `thirds` randomly chosen third features, each excluded at
    the given extreme levels. The third feature must differ from both pair features. An exception on an empty level
    (`empty`) excludes nothing, so it would only duplicate its parent pair and is skipped."""
    out = []
    empty = empty or set()
    feats = list(uni.all)
    for pair in ranked_pairs[:top_pairs]:
        if len(pair.base) != 2 or pair.unless:
            continue
        for j3 in rng.permutation(len(feats))[:thirds]:
            f3 = feats[int(j3)]
            if f3 in pair.features:
                continue
            for q3 in levels:
                if len(out) >= max_total:
                    return out
                if (f3, q3) in empty:
                    continue
                out.append(_cand(Expression.make(pair.base, [Term(f3, q3)]), uni, "unless", "unless", target))
    return out


def bank_candidates(prior: Optional[pd.DataFrame], uni: Universe, target: str = "excess_5d") -> tuple:
    """Re-test candidates from an earlier window's bank: rows with `names` = ['s', f, q] / ['p', f1, q1, f2, q2] /
    ['u', f1, q1, f2, q2, f3, q3] (PatternMiner.export_bank). Returns (candidates, skipped) where skipped lists rows
    whose feature is not in this universe - reported, because a pattern silently dropped is a pattern forgotten."""
    cands, skipped = [], []
    if prior is None or len(prior) == 0:
        return cands, skipped
    known = set(uni.all)
    for nk in prior["names"]:
        nk = list(nk)
        try:
            kind = nk[0]
            if kind == "s":
                parts = [(nk[1], nk[2])]
                base, exc = parts, []
            elif kind == "p":
                base, exc = [(nk[1], nk[2]), (nk[3], nk[4])], []
            elif kind == "u":
                base, exc = [(nk[1], nk[2]), (nk[3], nk[4])], [(nk[5], nk[6])]
            else:
                raise IdentityError(f"unknown kind {kind!r}")
            if any(f not in known for f, _ in base + exc):
                skipped.append((tuple(map(str, nk)), "feature not in universe"))
                continue
            expr = Expression.make([Term(f, int(q)) for f, q in base], [Term(f, int(q)) for f, q in exc])
        except (IdentityError, IndexError, ValueError, TypeError) as ex:
            skipped.append((tuple(map(str, nk)), str(ex)))
            continue
        cands.append(_cand(expr, uni, "bank", "bank", target))
    return cands, skipped


# ---------------------------------------------------------------- the staged generator
class CandidateGenerator:
    """Stages mirror the miner: singles are tested first, their scores pick the pair pool, pair scores pick the
    exception bases. Each stage is a pure function of (universe, seed, scores) so a run can be replayed exactly."""

    def __init__(self, universe: Universe, params: Optional[dict] = None, target: str = "excess_5d",
                 empty: Optional[set] = None):
        self.uni = universe
        self.empty = set(empty or ())
        self.p = {**CAND_DEFAULT, **(params or {})}
        self.target = target
        self.rng = np.random.default_rng(self.p["seed"])       # one stream, consumed in a fixed stage order
        self.set = CandidateSet()
        self.skipped_bank = []

    def _is_empty(self, cand: Candidate) -> bool:
        e = cand.expression
        return any((t.feature, t.level) in self.empty for t in e.base + e.unless)

    def stage_singles(self) -> list:
        c = single_candidates(self.uni, self.target)
        self.set.extend(c)
        return c

    def stage_pairs(self, single_scores: Mapping[str, float]) -> list:
        """single_scores: {'feat q3': |t|} from the singles just tested."""
        terms = top_terms(single_scores, self.p["top_singles"])
        # the strongest singles may not crowd out the random draw: at least min_random_frac of the budget stays random,
        # otherwise a feature outside the top-k singles could never be paired (the default 4,000 leaves ~2,250 random)
        floor = int(self.p["max_pairs"] * self.p["min_random_frac"])
        room = max(0, self.p["max_pairs"] - floor)
        stock = [t for t in terms if not self.uni.is_context(t.feature)]
        ctxp = context_pair_candidates(stock[: self.p["context_top"]], self.uni, self.target)
        ctxp = [c for c in ctxp if not self._is_empty(c)][: int(room * self.p["context_share"])]
        top = [c for c in top_pair_candidates(stock, self.uni, self.target) if not self._is_empty(c)][: room - len(ctxp)]
        top = ctxp + top
        self.set.extend(top)
        n_rand = max(0, self.p["max_pairs"] - len(top))
        rnd = random_pair_candidates(self.uni, n_rand, self.rng, set(self.set.ids()), self.target, self.empty)
        self.set.extend(rnd)
        return top + rnd

    def stage_unless(self, pair_scores: Mapping[str, float]) -> list:
        ranked = [Expression.parse(t) for t, _ in sorted(pair_scores.items(), key=lambda kv: (-abs(kv[1]), kv[0]))]
        c = unless_candidates(ranked, self.uni, self.rng, self.p["unless_top_pairs"], self.p["unless_thirds"],
                              self.p["unless_levels"], self.p["max_unless"], self.target, self.empty)
        self.set.extend(c)
        return c

    def stage_bank(self, prior: Optional[pd.DataFrame]) -> list:
        c, self.skipped_bank = bank_candidates(prior, self.uni, self.target)
        self.set.extend(c)
        return c


def plan_size(universe: Universe, params: Optional[dict] = None, n_bank: int = 0) -> dict:
    """Upper bound of candidates per run before any data is seen (the blueprint quotes ~4,300-4,400)."""
    p = {**CAND_DEFAULT, **(params or {})}
    singles = len(universe.all) * N_LEVELS
    pair_space = _pair_space(universe.all)[1]
    pairs = min(p["max_pairs"], pair_space)
    unless = min(p["max_unless"], p["unless_top_pairs"] * p["unless_thirds"] * len(p["unless_levels"]))
    return {"singles": singles, "pairs": pairs, "unless": unless, "bank": n_bank,
            "total": singles + pairs + unless + n_bank, "pair_space": pair_space}


def enumerate_static(universe: Universe, seed: int = 7, params: Optional[dict] = None, single_t: Optional[Mapping] = None):
    """Data-free run of every stage: with no scores, a seeded pseudo-score stands in for |t| so the pair pool and the
    exception bases are drawn reproducibly. Used by the coverage audit and by tests; real runs pass real scores."""
    g = CandidateGenerator(universe, {**(params or {}), "seed": seed})
    singles = g.stage_singles()
    if single_t is None:
        z = np.random.default_rng(seed + 1).random(len(singles))
        single_t = {c.text: float(v) for c, v in zip(singles, z)}
    pairs = g.stage_pairs(single_t)
    zp = np.random.default_rng(seed + 2).random(len(pairs))
    g.stage_unless({c.text: float(v) for c, v in zip(pairs, zp)})
    return g


# ---------------------------------------------------------------- coverage audit
def coverage_audit(universe: Universe, cset: CandidateSet, panel_columns: Iterable[str] = (),
                   candle_names: Iterable[str] = (), miner_features: Iterable[str] = (),
                   expected: Optional[Mapping[str, int]] = None) -> dict:
    """Prove reachability, per feature: does it appear as a single at all 5 levels, in at least one pair, and as a
    condition or exception in an 'unless' candidate?  Also cross-checks the universe against the REAL sources:
    the columns of the engine panel and the names candles.build produces, and reports what the miner reaches today
    (`miner_features`: features PatternMiner.fit would use = non-'m_' columns it is handed).

    `expected` = {'engine': 46, 'context': 8, 'candle': 25} sizes the universe must have (the blueprint's 54 =
    engine + context)."""
    panel = set(panel_columns) - {"date", "ticker"}
    candles = set(candle_names)
    miner = set(miner_features)
    singles, pairs, unlessf = {}, {}, {}
    for c in cset:
        e = c.expression
        if c.family == "single":
            f = e.base[0]
            singles.setdefault(f.feature, set()).add(f.level)
        for t in e.base:
            if c.family in ("pair", "unless"):
                (pairs if c.family == "pair" else unlessf).setdefault(t.feature, set()).add(c.id)
        for t in e.unless:
            unlessf.setdefault(t.feature, set()).add(c.id)
    rows = []
    for f in universe.all:
        g = universe.group_of(f)
        rows.append({"feature": f, "group": g, "single_levels": len(singles.get(f, ())),
                     "in_pairs": len(pairs.get(f, ())), "in_unless": len(unlessf.get(f, ())),
                     "in_panel": (f in panel) if panel else None,
                     "in_candle_build": (f in candles) if (candles and g == "candle") else None,
                     "miner_reaches_today": (f in miner) if miner else None})
    T = pd.DataFrame(rows)
    sizes = {g: len(getattr(universe, g)) for g in ("engine", "context", "candle")}
    problems = []
    if expected:
        for g, n in expected.items():
            if sizes.get(g) != n:
                problems.append(f"universe has {sizes.get(g)} {g} features, expected {n}")
    for _, r in T.iterrows():
        if r["single_levels"] < N_LEVELS:
            problems.append(f"{r['feature']}: only {r['single_levels']} of {N_LEVELS} single levels are candidates")
        if r["in_pairs"] == 0:
            problems.append(f"{r['feature']}: never appears in a pair candidate")
    if panel:
        for f in universe.engine + universe.context:
            if f not in panel:
                problems.append(f"{f}: in the universe but not a column of the engine panel")
        extra = sorted(panel - set(universe.engine) - set(universe.context))
        if extra:
            problems.append(f"engine panel columns not in the universe: {extra}")
    if candles:
        for f in universe.candle:
            if f not in candles:
                problems.append(f"{f}: named as a candle signal but engine.candles.build does not produce it")
        extra = sorted(candles - set(universe.candle))
        if extra:
            problems.append(f"candles.build produces signals not in the universe: {extra}")
    return {"sizes": sizes, "total_features": int(len(T)), "table": T, "problems": problems,
            "candidate_counts": cset.counts(), "n_candidates": len(cset), "digest": cset.digest(),
            "min_pairs_per_feature": int(T["in_pairs"].min()) if len(T) else 0,
            "median_pairs_per_feature": float(T["in_pairs"].median()) if len(T) else 0.0,
            "unreached_by_miner_today": sorted(T.loc[T["miner_reaches_today"] == False, "feature"]) if miner else [],
            "ok": not problems}


def audit_json(report: dict) -> str:
    """The audit without the table, as stable JSON (the table goes to CSV)."""
    d = {k: v for k, v in report.items() if k != "table"}
    return json.dumps(d, indent=1, sort_keys=True, default=str)


# ---------------------------------------------------------------- degenerate levels
def empty_levels(qm: QuantileMatrix, min_rows: int = 1) -> set:
    """(feature, level) pairs that hold fewer than `min_rows` rows. Flags and sparse counts (an event indicator is
    0 for nearly every stock) collapse to two occupied quintiles: the other three singles can never be tested."""
    out = set()
    for j, f in enumerate(qm.features):
        col = qm.Q[:, j]
        cnt = np.bincount(col[col >= 0], minlength=N_LEVELS)
        out |= {(f, q) for q in range(N_LEVELS) if cnt[q] < min_rows}
    return out


def prune_unoccupied(cset: CandidateSet, empty: set) -> tuple:
    """Split a candidate set into (kept, dropped) by whether any BASE term sits in an empty level. A dropped
    candidate would have been discarded by the miner's min_n gate anyway; dropping it up front keeps the candidate
    count honest (an exception on an empty level excludes nothing and is left alone)."""
    kept, dropped = CandidateSet(), []
    for c in cset:
        if any((t.feature, t.level) in empty for t in c.expression.base):
            dropped.append(c)
        else:
            kept.add(c)
    return kept, dropped
