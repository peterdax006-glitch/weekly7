"""Identity memorisation firewall (contract C62 section 29; checklist H10-H12, I20; section 62 tests 8 and 10).

Attack memorisation explicitly. Seven identity-preserving transformations of a (date, ticker) panel:
    ticker permutation     tickers relabelled by a random bijection (rows re-sorted, so name-order tie-breaks cannot survive)
    date permutation       dates relabelled by a random ORDER-PRESERVING map (year / month / weekday labels destroyed)
    year disguise          whole-week date shift (engine.learning_delta.make_presentation, the audited disguise)
    presentation disguise  opaque ticker codes + date shift, as a player would be shown a rerun
    stock substitution     a fraction of tickers swap identities pairwise
    sector substitution    sector labels (or one-hot sector columns) permuted
    episode substitution   blocks of dates swap CONTENT (features + returns) pairwise, labels stay
    sequence scrambling    date blocks are dealt back in a random order (content moves, date labels stay)
Every transform is verified to be identity-preserving: the multiset of feature rows and of returns is unchanged, the index
stays unique, and enough of the identity labels really changed (a no-op transform would pass trivially).

`IdentityHarness` runs ANY learner callable  learner(X_train, y_train, X_eval, seed) -> Series of scores  once on the original
data (twice, to prove determinism) and once per attack, in two modes:
    eval  only the evaluated rows are disguised: a memoriser keyed on the identity it saw in training collapses
    both  train and eval are relabelled consistently: any learner whose LOGIC depends on identity (absolute dates,
          name order, hard-coded tickers) changes, a memoriser does not
and flags COLLAPSE when skill (mean per-date rank IC) retention falls under the threshold with a bootstrap interval that
excludes 'unchanged'. Reference learners (a memoriser, per-ticker and per-date lookups, a legitimate ridge) are provided so
the harness can be shown to see memorisation (section 62 test 10). Reuses engine.learning_delta.make_presentation for renames/date shifts.

IMPLEMENTED - NOT VALIDATED."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from .core import FirewallBreach, stable_hash
from .firewalls import Finding, LayerName, fail, info, warn

L = LayerName.IDENTITY
Learner = Callable[[pd.DataFrame, pd.Series, pd.DataFrame, int], pd.Series]
MODES = ("eval", "both")


# ---------------------------------------------------------------- panel helpers
def _check_panel(X: pd.DataFrame, what: str = "X") -> None:
    if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels != 2:
        raise ValueError(f"{what} must be indexed by MultiIndex (date, ticker)")


def _dates(X) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(X.index.get_level_values(0))


def _tickers(X) -> pd.Index:
    return X.index.get_level_values(1)


def _reindex(X: pd.DataFrame, y: pd.Series | None, dates, tickers, sort: bool):
    """Return copies of X/y whose index is (dates, tickers); optionally re-sorted so row order follows the NEW labels."""
    idx = pd.MultiIndex.from_arrays([dates, tickers], names=X.index.names)
    X2 = X.copy()
    X2.index = idx
    y2 = None
    if y is not None:
        y2 = y.reindex(X.index).copy() if not y.index.equals(X.index) else y.copy()
        y2.index = idx
    if sort:
        order = idx.argsort()
        X2 = X2.iloc[order]
        y2 = y2.iloc[order] if y2 is not None else None
    return X2, y2


@dataclasses.dataclass(frozen=True)
class Transformed:
    kind: str
    X: pd.DataFrame
    y: pd.Series | None
    changed_frac: float                       # share of rows whose identity label (date or ticker) changed
    mapping: Mapping[Any, Any]                # old label -> new label (empty for content transforms)
    inverse_rows: np.ndarray | None = None    # for relabels: position in the ORIGINAL frame of each transformed row


# ---------------------------------------------------------------- the seven transforms
def _rng(seed: int, salt: str) -> np.random.Generator:
    return np.random.default_rng([seed, sum(map(ord, salt))])           # never hash(): str hashes change per process


def derangement(rng: np.random.Generator, n: int) -> np.ndarray:
    """A random permutation of range(n) with no fixed point (n >= 2), so a relabelling always relabels everything."""
    if n < 2:
        return np.arange(n)
    for _ in range(1000):
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p
    return np.roll(np.arange(n), 1)


def ticker_permutation(X, y=None, seed: int = 0, order_preserving: bool = False, universe: Sequence | None = None) -> Transformed:
    """Random bijection over `universe` (default: the tickers present) with no ticker keeping its name. Rows are re-sorted by
    the new names, so an alphabetical tie-break cannot smuggle the old identity through. With `order_preserving` the tickers
    are instead renamed to FRESH opaque codes in the same alphabetical order (rows keep their position), which isolates a
    dependence on the names themselves from a dependence on their order."""
    _check_panel(X)
    tk = pd.Index(sorted(set(universe) if universe is not None else set(_tickers(X))), dtype=object)
    rng = _rng(seed, "ticker")
    if order_preserving:
        nums = np.sort(rng.choice(10 ** 6, size=len(tk), replace=False))
        perm = pd.Index([f"Z{int(k):06d}" for k in nums])
    else:
        perm = tk[derangement(rng, len(tk))]
    tmap = dict(zip(tk, perm))
    new_t = _tickers(X).map(tmap)
    if new_t.isna().any():
        raise ValueError("tickers outside the permutation universe")
    X2, y2 = _reindex(X, y, _dates(X), new_t, sort=not order_preserving)
    orig_pos = pd.Series(np.arange(len(X)), index=pd.MultiIndex.from_arrays([_dates(X), new_t]))
    inv = orig_pos.reindex(X2.index).to_numpy()
    return Transformed("ticker_permutation", X2, y2, float(np.mean(_tickers(X) != new_t)), tmap, inv)


def date_permutation(X, y=None, seed: int = 0, span_years: tuple[int, int] = (30, 200), universe: Sequence | None = None) -> Transformed:
    """Order-preserving random relabelling of the dates: new dates are drawn from a random era, keep the original ORDER
    (so sequential logic still works) but have different gaps, weekdays, months and years."""
    _check_panel(X)
    ud = pd.DatetimeIndex(sorted(set(universe) if universe is not None else set(_dates(X))))
    rng = _rng(seed, "date")
    start = pd.Timestamp("1900-01-01") + pd.Timedelta(days=int(rng.integers(span_years[0], span_years[1]) * 365))
    gaps = rng.integers(1, 10, len(ud))
    new = start + pd.to_timedelta(np.cumsum(gaps), unit="D")
    dmap = dict(zip(ud, new))
    nd = pd.DatetimeIndex(_dates(X).map(dmap))
    X2, y2 = _reindex(X, y, nd, _tickers(X), sort=False)
    return Transformed("date_permutation", X2, y2, float(np.mean(_dates(X) != nd)), dmap, np.arange(len(X)))


def _panel_window(X: pd.DataFrame, wid: str = "panel"):
    """Wrap a panel's date x ticker skeleton as an engine.learning_delta.Window so the AUDITED disguise machinery
    (make_presentation: fresh order-preserving codes, fresh whole-week shift, bounds, never a shift already used) can
    be applied to it. Only the skeleton is used: no prices, snapshots or dividends."""
    from engine.learning_delta import Window
    d = sorted(set(_dates(X)))
    t = sorted(set(_tickers(X)))
    closes = pd.DataFrame(1.0, index=pd.DatetimeIndex(d), columns=t)
    return Window(wid, {}, closes, None, 0.0, {}, d[0], d[-1])


def presentation_disguise(X, y=None, seed: int = 0, order_preserving: bool = False, dates: bool = True,
                          names: bool = True) -> Transformed:
    """Disguise with engine.learning_delta.make_presentation (the audited disguise): opaque ticker codes and a whole-week
    date shift, values and row order untouched. `order_preserving=False` also scrambles alphabetical order, which is
    what exposes name-order tie-breaks; `dates`/`names` attribute a collapse to what the learner keyed on."""
    _check_panel(X)
    from engine.learning_delta import BlindnessError, make_presentation
    try:
        pres, rec = make_presentation(_panel_window(X), seed, order_preserving=order_preserving)
    except BlindnessError:
        # a panel still in the real era cannot reach the simulated era (year >= 2100) within the default +-900..1100 week
        # bounds; widen the upper bound, the lower bound is raised by make_presentation itself
        pres, rec = make_presentation(_panel_window(X), seed, order_preserving=order_preserving, shift_weeks=(0, 20000))
    delta = pd.Timedelta(days=rec.shift_days) if dates else pd.Timedelta(0)
    cmap = rec.code_map if names else {t: t for t in set(_tickers(X))}
    nd = _dates(X) + delta
    nt = _tickers(X).map(cmap)
    X2, y2 = _reindex(X, y, nd, nt, sort=False)
    changed = float(np.mean((_dates(X) != nd) | (np.asarray(_tickers(X)) != np.asarray(nt))))
    return Transformed("presentation_disguise" if names and dates else ("year_disguise" if dates else "ticker_codes"), X2, y2,
                       changed, {**{"shift_days": rec.shift_days}, **cmap}, np.arange(len(X)))


def year_disguise(X, y=None, seed: int = 0, shift_weeks: tuple[int, int] = (40, 400)) -> Transformed:
    """Whole-week date shift only (make_presentation's shift; the ticker codes are dropped). `shift_weeks` is kept for
    call compatibility: make_presentation chooses within its own audited bounds."""
    t = presentation_disguise(X, y, seed, order_preserving=True, dates=True, names=False)
    return dataclasses.replace(t, kind="year_disguise")


def stock_substitution(X, y=None, seed: int = 0, frac: float = 1.0) -> Transformed:
    """A fraction of the tickers swap identities pairwise (A's rows are labelled B and vice versa)."""
    _check_panel(X)
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")
    tk = sorted(set(_tickers(X)))
    rng = _rng(seed, "stock")
    n = max(2, int(round(frac * len(tk))) // 2 * 2)
    n = min(n, len(tk) // 2 * 2)
    pick = list(rng.permutation(tk)[:n]) if n else []
    tmap = {t: t for t in tk}
    for a, b in zip(pick[0::2], pick[1::2]):
        tmap[a], tmap[b] = b, a
    new_t = _tickers(X).map(tmap)
    X2, y2 = _reindex(X, y, _dates(X), new_t, sort=True)
    orig_pos = pd.Series(np.arange(len(X)), index=pd.MultiIndex.from_arrays([_dates(X), new_t]))
    return Transformed("stock_substitution", X2, y2, float(np.mean(_tickers(X) != new_t)),
                       {a: b for a, b in tmap.items() if a != b}, orig_pos.reindex(X2.index).to_numpy())


def sector_substitution(X, y=None, seed: int = 0, sector_columns: Sequence[str] | None = None,
                        groups: Mapping[str, str] | None = None) -> Transformed:
    """Permute sector identity. With one-hot `sector_columns` the columns' NAMES are permuted (values stay), so anything
    keyed to a sector's name is hit while numeric logic is not. With a ticker->sector `groups` mapping, the mapping
    returned is the permuted one and X is unchanged (the caller re-keys its own group lookup)."""
    _check_panel(X)
    rng = _rng(seed, "sector")
    if sector_columns:
        cols = list(sector_columns)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"sector columns not in X: {missing}")
        perm = [cols[i] for i in derangement(rng, len(cols))]
        cmap = dict(zip(cols, perm))
        X2 = X.rename(columns=cmap)[list(X.columns)]                  # same column ORDER, permuted meaning
        moved = float(np.mean([cmap[c] != c for c in cols]))
        return Transformed("sector_substitution", X2, y.copy() if y is not None else None, moved, cmap, np.arange(len(X)))
    if groups:
        secs = sorted(set(groups.values()))
        perm = [secs[i] for i in derangement(rng, len(secs))]
        smap = dict(zip(secs, perm))
        newg = {t: smap[s] for t, s in groups.items()}
        return Transformed("sector_substitution", X.copy(), y.copy() if y is not None else None,
                           float(np.mean([smap[s] != s for s in groups.values()])), newg, np.arange(len(X)))
    raise ValueError("sector substitution needs sector_columns or groups")


def _blocks(X, block: int) -> list[np.ndarray]:
    """Positional row indices of consecutive blocks of `block` distinct dates."""
    d = _dates(X)
    ud = pd.DatetimeIndex(sorted(set(d)))
    lab = pd.Series(np.arange(len(ud)) // max(1, block), index=ud)
    rowblock = lab.reindex(d).to_numpy()
    return [np.where(rowblock == b)[0] for b in sorted(set(rowblock))]


def _move_content(X, y, blocks: list[np.ndarray], order: Sequence[int], kind: str) -> Transformed:
    """Deal block `order[i]`'s CONTENT into the date slots of block i (per-date cross-sections stay intact). Blocks of
    different size or ticker sets are only moved onto slots they fit exactly; others stay put."""
    Xv = X.to_numpy(copy=True)
    yv = y.reindex(X.index).to_numpy(copy=True) if y is not None else None
    idx = X.index
    d = _dates(X)
    moved = np.zeros(len(X), dtype=bool)
    for i, j in enumerate(order):
        if i == j:
            continue
        a, b = blocks[i], blocks[j]
        da = pd.Series(np.arange(len(a)), index=pd.MultiIndex.from_arrays([d[a], _tickers(X)[a]]))
        db = pd.Series(np.arange(len(b)), index=pd.MultiIndex.from_arrays([d[b], _tickers(X)[b]]))
        ua, ub = sorted(set(d[a])), sorted(set(d[b]))
        if len(a) != len(b) or len(ua) != len(ub):
            continue
        # align date j-th of the source block with date j-th of the target block, tickers by name within the date
        src_pos: dict[Any, int] | None = {}
        for du_a, du_b in zip(ua, ub):
            ra = a[np.asarray(d[a] == du_a)]
            rb = b[np.asarray(d[b] == du_b)]
            ta = _tickers(X)[ra]
            tb = pd.Series(rb, index=_tickers(X)[rb])
            if len(ra) != len(rb) or set(ta) != set(tb.index) or tb.index.has_duplicates:
                src_pos = None
                break
            for r, t in zip(ra, ta):
                cast(dict, src_pos)[r] = int(tb[t])
        if not src_pos:
            continue
        for tgt, src in src_pos.items():
            Xv[tgt] = X.to_numpy()[src]
            if yv is not None:
                yv[tgt] = y.reindex(X.index).to_numpy()[src]
            moved[tgt] = True
    X2 = pd.DataFrame(Xv, index=idx, columns=X.columns).astype(X.dtypes.to_dict(), errors="ignore")
    y2 = pd.Series(yv, index=idx, name=y.name) if yv is not None else None
    return Transformed(kind, X2, y2, float(moved.mean()) if len(moved) else 0.0, {}, None)


def _size_groups(blocks: list[np.ndarray]) -> list[list[int]]:
    """Indices of blocks grouped by row count: only equal-size blocks can exchange content cell for cell."""
    by: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        by.setdefault(len(b), []).append(i)
    return [g for _, g in sorted(by.items())]


def episode_substitution(X, y=None, seed: int = 0, block: int = 20, frac: float = 0.5) -> Transformed:
    """Episodes (blocks of `block` consecutive dates) swap content pairwise, pairs drawn among equal-size blocks; the dates
    and tickers of each slot are kept, so a learner keyed on (date, ticker) sees familiar labels wrapped around another
    episode's market."""
    _check_panel(X)
    blocks = _blocks(X, block)
    rng = _rng(seed, "episode")
    order = list(range(len(blocks)))
    for grp in _size_groups(blocks):
        if len(grp) < 2:
            continue
        pick = [grp[i] for i in rng.permutation(len(grp))[:max(2, int(round(frac * len(grp))) // 2 * 2)]]
        for a, b in zip(pick[0::2], pick[1::2]):
            order[a], order[b] = order[b], order[a]
    return _move_content(X, y, blocks, order, "episode_substitution")


def sequence_scramble(X, y=None, seed: int = 0, block: int = 1) -> Transformed:
    """Deal all date blocks back in a random order: destroys temporal sequence while keeping every cross-section whole."""
    _check_panel(X)
    blocks = _blocks(X, block)
    rng = _rng(seed, "sequence")
    order = list(range(len(blocks)))
    for grp in _size_groups(blocks):
        if len(grp) > 1:
            for i, j in zip(grp, derangement(rng, len(grp))):
                order[i] = grp[j]
    return _move_content(X, y, blocks, order, "sequence_scramble")


ATTACKS: dict[str, Callable[..., Transformed]] = {
    "ticker_permutation": ticker_permutation,
    "date_permutation": date_permutation,
    "year_disguise": year_disguise,
    "presentation_disguise": presentation_disguise,
    "stock_substitution": stock_substitution,
    "episode_substitution": episode_substitution,
    "sequence_scramble": sequence_scramble,
}
DEFAULT_ATTACKS = tuple(ATTACKS) + ("sector_substitution",)


# ---------------------------------------------------------------- verifying a transform is identity-preserving
def _multiset_key(X: pd.DataFrame, y: pd.Series | None) -> str:
    cols = []
    for c in X.columns:
        v = X[c].to_numpy()
        cols.append(np.sort(v.astype(float)) if v.dtype.kind in "fiub" else np.sort(v.astype(str)))
    yk = np.sort(y.to_numpy(dtype=float)) if y is not None else np.array([])
    return stable_hash([[np.round(c, 10).tolist() if c.dtype.kind == "f" else c.tolist() for c in cols], np.round(yk, 10).tolist()], 24)


def verify_transform(orig_X: pd.DataFrame, orig_y: pd.Series | None, t: Transformed, min_changed: float = 0.05,
                     content_preserving: bool = True) -> list[Finding]:
    """A transform must not create or destroy information: same rows count, unique index, same multiset of values (unless
    it is a column-name relabel), and it must have changed enough labels to be a real attack."""
    out: list[Finding] = []
    if len(t.X) != len(orig_X):
        out.append(fail(L, "rowcount-changed", t.kind, f"{len(orig_X)} rows became {len(t.X)}"))
    if t.X.index.has_duplicates:
        out.append(fail(L, "index-not-unique", t.kind, "transformed index has duplicates"))
    if content_preserving and t.kind != "sector_substitution":
        if _multiset_key(orig_X, orig_y) != _multiset_key(t.X, t.y):
            out.append(fail(L, "content-not-preserved", t.kind, "multiset of feature/return values changed: not identity-preserving"))
    if t.changed_frac < min_changed:
        out.append(fail(L, "no-op-transform", t.kind, f"only {t.changed_frac:.1%} of rows changed identity: the attack tests nothing"))
    return out


# ---------------------------------------------------------------- scoring
def per_date_ic(scores: pd.Series, y: pd.Series, min_names: int = 5) -> pd.Series:
    """Spearman rank correlation of scores with realised returns, per date. Dates with fewer than `min_names` names, or whose
    returns have no dispersion, are dropped (nothing to rank). A date on which the SCORES are constant scores exactly 0: a
    learner that outputs one number for everybody has no skill there, it has not 'failed to be measured'."""
    df = pd.DataFrame({"s": scores.reindex(y.index), "y": y}).dropna()
    if not len(df):
        return pd.Series(dtype=float)
    out = {}
    for d, sub in df.groupby(level=0):
        if len(sub) >= min_names and sub["y"].nunique() > 1:
            out[d] = 0.0 if sub["s"].nunique() <= 1 else sub["s"].rank().corr(sub["y"].rank())
    return pd.Series(out, dtype=float).sort_index()


def skill_t(ic: np.ndarray) -> float:
    """t-statistic of a per-date IC series (mean over standard error). A zero-variance positive series is +inf."""
    ic = np.asarray(ic, dtype=float)
    ic = ic[np.isfinite(ic)]
    if len(ic) < 2:
        return float("nan")
    sd = ic.std(ddof=1)
    if sd == 0:
        return float("inf") if ic.mean() > 0 else (float("-inf") if ic.mean() < 0 else 0.0)
    return float(ic.mean() / (sd / np.sqrt(len(ic))))


def top_k_spread(scores: pd.Series, y: pd.Series, k: int = 5) -> float:
    """Mean return of the top-k scored names minus the cross-sectional mean, averaged over dates."""
    df = pd.DataFrame({"s": scores.reindex(y.index), "y": y}).dropna()
    vals = []
    for _, sub in df.groupby(level=0):
        if len(sub) > k and sub["s"].nunique() > 1:
            vals.append(sub.nlargest(k, "s")["y"].mean() - sub["y"].mean())
    return float(np.mean(vals)) if vals else float("nan")


def boot_mean(x: np.ndarray, rng: np.random.Generator, n: int = 400, block: int = 3) -> np.ndarray:
    """Moving-block bootstrap of the mean of a per-date series."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.array([])
    b = max(1, min(block, len(x)))
    starts_n = int(np.ceil(len(x) / b))
    out = np.empty(n)
    for i in range(n):
        st = rng.integers(0, max(1, len(x) - b + 1), starts_n)
        out[i] = np.concatenate([x[s:s + b] for s in st])[:len(x)].mean()
    return out


def retention_ci(base_ic: np.ndarray, att_ic: np.ndarray, seed: int = 0, n: int = 400, level: float = 0.9) -> tuple[float, float]:
    """Bootstrap interval for mean(attacked IC) / mean(base IC), resampling each date series independently."""
    rng = np.random.default_rng(seed)
    b, a = boot_mean(base_ic, rng, n), boot_mean(att_ic, rng, n)
    if not len(b) or not len(a):
        return float("nan"), float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(b) > 1e-12, a / b, np.nan)
    r = r[np.isfinite(r)]
    if not len(r):
        return float("nan"), float("nan")
    lo, hi = np.quantile(r, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------- verdicts and report
@dataclasses.dataclass(frozen=True)
class AttackVerdict:
    kind: str
    mode: str
    status: str                       # OK | COLLAPSE | NO_SKILL | INSUFFICIENT | NONDETERMINISTIC
    base_ic: float
    attacked_ic: float
    retention: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    base_topk: float = float("nan")
    attacked_topk: float = float("nan")
    n_dates: int = 0
    changed_frac: float = 0.0
    note: str = ""


@dataclasses.dataclass(frozen=True)
class IdentityReport:
    verdicts: tuple[AttackVerdict, ...]
    base_ic: float
    base_topk: float
    deterministic: bool
    seed: int
    findings: tuple[Finding, ...] = ()

    @property
    def collapsed(self) -> tuple[str, ...]:
        return tuple(sorted({f"{v.kind}[{v.mode}]" for v in self.verdicts if v.status == "COLLAPSE"}))

    @property
    def memorization_suspected(self) -> bool:
        """Collapse when only the evaluated rows are disguised = the learner recalls identities it saw in training."""
        return any(v.status == "COLLAPSE" and v.mode == "eval" for v in self.verdicts)

    @property
    def logic_identity_dependent(self) -> bool:
        return any(v.status == "COLLAPSE" and v.mode == "both" for v in self.verdicts)

    @property
    def passed(self) -> bool:
        return bool(self.verdicts) and self.deterministic and all(v.status == "OK" for v in self.verdicts) \
            and not any(f.is_fail for f in self.findings)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(v) for v in self.verdicts])

    def retention_by_kind(self, mode: str = "eval") -> pd.Series:
        return pd.Series({v.kind: v.retention for v in self.verdicts if v.mode == mode}, dtype=float)

    def digest(self) -> str:
        return stable_hash([[dataclasses.asdict(v) for v in self.verdicts], self.base_ic, self.deterministic, self.seed])

    def markdown(self) -> str:
        lines = [f"# Identity firewall report (seed {self.seed})", f"base IC {self.base_ic:.4f}, deterministic={self.deterministic}, "
                 f"passed={self.passed}", "", "| attack | mode | status | retention | 90% CI | changed |", "|---|---|---|---|---|---|"]
        for v in self.verdicts:
            lines.append(f"| {v.kind} | {v.mode} | {v.status} | {v.retention:.2f} | [{v.ci_low:.2f}, {v.ci_high:.2f}] | {v.changed_frac:.0%} |")
        if self.collapsed:
            lines.append("")
            lines.append("Collapsed under: " + ", ".join(self.collapsed) + (". Memorisation suspected." if self.memorization_suspected else "."))
        lines.append("")
        lines.append("IMPLEMENTED - NOT VALIDATED.")
        return "\n".join(lines)


# ---------------------------------------------------------------- the harness
class IdentityHarness:
    def __init__(self, learner: Learner, attacks: Sequence[str] = tuple(ATTACKS), seed: int = 0, collapse_ratio: float = 0.5,
                 min_skill: float = 0.01, min_dates: int = 8, k: int = 5, modes: Sequence[str] = MODES, boot: int = 400,
                 attack_kwargs: Mapping[str, Mapping] | None = None, min_t: float = 2.0):
        bad = [a for a in attacks if a not in ATTACKS and a != "sector_substitution"]
        if bad:
            raise ValueError(f"unknown attacks {bad}")
        if not set(modes) <= set(MODES):
            raise ValueError(f"modes must be within {MODES}")
        self.learner, self.attacks, self.seed = learner, tuple(attacks), seed
        self.ratio, self.min_skill, self.min_dates, self.k, self.modes, self.boot = collapse_ratio, min_skill, min_dates, k, tuple(modes), boot
        self.min_t = min_t
        self.kwargs = dict(attack_kwargs or {})

    def _run(self, Xt, yt, Xe) -> pd.Series:
        s = self.learner(Xt, yt, Xe, self.seed)
        if not isinstance(s, pd.Series):
            raise FirewallBreach("learner must return a pandas Series of scores")
        if len(s) != len(Xe) or not s.index.equals(Xe.index):
            raise FirewallBreach("learner scores must be indexed exactly like X_eval")
        return s.astype(float)

    def _apply(self, name: str, X, y, seed_off: int, universe=None, **extra) -> Transformed:
        kw = {**self.kwargs.get(name, {}), **extra}
        if name == "sector_substitution":
            return sector_substitution(X, y, self.seed + seed_off, **kw)
        return ATTACKS[name](X, y, self.seed + seed_off, **kw)

    def run(self, X_train, y_train, X_eval, y_eval, sector_columns: Sequence[str] | None = None,
            groups: Mapping[str, str] | None = None) -> IdentityReport:
        for X, nm in ((X_train, "X_train"), (X_eval, "X_eval")):
            _check_panel(X, nm)
        findings: list[Finding] = []
        s1, s2 = self._run(X_train, y_train, X_eval), self._run(X_train, y_train, X_eval)
        det = bool(np.array_equal(s1.to_numpy(), s2.to_numpy(), equal_nan=True))
        if not det:
            findings.append(fail(L, "nondeterministic", "learner", "two identical runs gave different scores: every comparison is meaningless"))
        ic0 = per_date_ic(s1, y_eval)
        base_ic = float(ic0.mean()) if len(ic0) else float("nan")
        base_topk = top_k_spread(s1, y_eval, self.k)
        verdicts: list[AttackVerdict] = []
        for ai, name in enumerate(self.attacks):
            if name == "sector_substitution" and not (sector_columns or groups):
                findings.append(warn(L, "attack-skipped", name, "no sector columns or groups supplied; sector substitution not run"))
                continue
            for mode in self.modes:
                verdicts.append(self._one(name, mode, ai, X_train, y_train, X_eval, y_eval, base_ic, ic0, base_topk, det,
                                          sector_columns, groups, findings))
        if not verdicts:
            findings.append(fail(L, "no-attacks-run", "harness", "no identity attack was run"))
        return IdentityReport(tuple(verdicts), base_ic, base_topk, det, self.seed, tuple(findings))

    def _one(self, name, mode, ai, Xt, yt, Xe, ye, base_ic, ic0, base_topk, det, sector_columns, groups, findings) -> AttackVerdict:
        extra = {}
        if name == "sector_substitution":
            extra = {"sector_columns": sector_columns, "groups": groups}
        if mode == "both" and name in ("date_permutation", "ticker_permutation"):
            union_key = "universe"
            u = sorted(set(_tickers(Xt)) | set(_tickers(Xe))) if name == "ticker_permutation" else \
                sorted(set(_dates(Xt)) | set(_dates(Xe)))
            extra[union_key] = u
        try:
            te = self._apply(name, Xe, ye, 100 + ai, **extra)
            findings.extend(verify_transform(Xe, ye, te))
            if mode == "both":
                tt = self._apply(name, Xt, yt, 100 + ai, **extra) if name in ("ticker_permutation", "date_permutation", "year_disguise", "presentation_disguise",
                                                                              "stock_substitution", "sector_substitution") \
                    else self._apply(name, Xt, yt, 700 + ai, **extra)
                Xt2, yt2 = tt.X, tt.y
            else:
                Xt2, yt2 = Xt, yt
            if name == "year_disguise" and mode == "both":
                pass                                                   # same seed => same shift on both sides (seed offset shared)
            s = self._run(Xt2, yt2, te.X)
        except (ValueError, FirewallBreach) as e:
            return AttackVerdict(name, mode, "INSUFFICIENT", base_ic, float("nan"), float("nan"), note=f"attack could not run: {e}")
        ic1 = per_date_ic(s, te.y)
        att_ic = float(ic1.mean()) if len(ic1) else float("nan")
        topk = top_k_spread(s, te.y, self.k)
        if not det:
            return AttackVerdict(name, mode, "NONDETERMINISTIC", base_ic, att_ic, float("nan"), n_dates=len(ic1), changed_frac=te.changed_frac)
        t_base = skill_t(ic0.to_numpy())
        if not np.isfinite(base_ic) or base_ic < self.min_skill or not (t_base >= self.min_t):
            return AttackVerdict(name, mode, "NO_SKILL", base_ic, att_ic, float("nan"), base_topk=base_topk, attacked_topk=topk,
                                 n_dates=len(ic1), changed_frac=te.changed_frac,
                                 note=f"baseline shows no significant skill to retain (IC {base_ic:.3f}, t {t_base:.1f})")
        if len(ic1) < self.min_dates or len(ic0) < self.min_dates:
            return AttackVerdict(name, mode, "INSUFFICIENT", base_ic, att_ic, float("nan"), n_dates=len(ic1),
                                 changed_frac=te.changed_frac, note=f"only {len(ic1)} scored dates")
        ret = att_ic / base_ic if np.isfinite(att_ic) else 0.0
        lo, hi = retention_ci(ic0.to_numpy(), ic1.to_numpy(), self.seed + ai, self.boot)
        if ret < self.ratio and (not np.isfinite(hi) or hi < 0.9):
            status = "COLLAPSE"
        elif ret < self.ratio:
            status = "INSUFFICIENT"
        else:
            status = "OK"
        return AttackVerdict(name, mode, status, base_ic, att_ic, float(ret), lo, hi, base_topk, topk, len(ic1), te.changed_frac)


# ---------------------------------------------------------------- reference learners (controls)
def memorizer_learner(X_train, y_train, X_eval, seed: int = 0) -> pd.Series:
    """POSITIVE CONTROL: a pure lookup of the training return for each exact (date, ticker); zero when unseen."""
    table = y_train.reindex(X_train.index)
    return table.reindex(X_eval.index).fillna(0.0).astype(float)


def ticker_mean_learner(X_train, y_train, X_eval, seed: int = 0) -> pd.Series:
    """Keyed on ticker identity only: predicts each ticker's average training return."""
    m = y_train.groupby(level=1).mean()
    return pd.Series(X_eval.index.get_level_values(1).map(m).to_numpy(dtype=float), index=X_eval.index).fillna(0.0)


def date_mean_learner(X_train, y_train, X_eval, seed: int = 0) -> pd.Series:
    """Keyed on date identity only: a date's average training return, broadcast to every name (no cross-section skill)."""
    m = y_train.groupby(level=0).mean()
    return pd.Series(X_eval.index.get_level_values(0).map(m).to_numpy(dtype=float), index=X_eval.index).fillna(0.0)


def ridge_learner(X_train, y_train, X_eval, seed: int = 0, alpha: float = 1.0) -> pd.Series:
    """LEGITIMATE control: closed-form ridge on standardised features; uses no identity at all."""
    cols = list(X_train.select_dtypes(include=[np.number]).columns)
    A = X_train[cols].to_numpy(dtype=float)
    yy = y_train.reindex(X_train.index).to_numpy(dtype=float)
    ok = np.isfinite(A).all(axis=1) & np.isfinite(yy)
    mu, sd = A[ok].mean(axis=0), A[ok].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (A[ok] - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(len(cols)), Z.T @ (yy[ok] - yy[ok].mean()))
    B = (np.nan_to_num(X_eval[cols].to_numpy(dtype=float), nan=0.0) - mu) / sd
    return pd.Series(B @ w, index=X_eval.index)


def random_learner(X_train, y_train, X_eval, seed: int = 0) -> pd.Series:
    """NULL control: seeded noise. Its IC is ~0 so the harness must report NO_SKILL, never COLLAPSE."""
    return pd.Series(np.random.default_rng([seed, 991, len(X_eval)]).normal(size=len(X_eval)), index=X_eval.index)


def distinguish_from_memorizer(candidate: IdentityReport, memorizer: IdentityReport, gap: float = 0.3) -> dict:
    """Section 62 test 10: the legitimate learner must stay distinguishable from a memoriser. Compare eval-mode retention
    profiles; the candidate should retain skill where the memoriser collapses."""
    rc, rm = candidate.retention_by_kind("eval"), memorizer.retention_by_kind("eval")
    common = sorted(set(rc.index) & set(rm.index))
    if not common:
        return {"distinguishable": False, "reason": "no common attacks", "gap": float("nan")}
    d = float(np.nanmean([rc[k] - rm[k] for k in common]))
    return {"distinguishable": bool(d >= gap and not candidate.memorization_suspected), "gap": d,
            "candidate_collapsed": list(candidate.collapsed), "memorizer_collapsed": list(memorizer.collapsed)}


# ---------------------------------------------------------------- structural identity probes
def identity_proxies(X: pd.DataFrame, eta_cut: float = 0.95, trend_cut: float = 0.99) -> list[Finding]:
    """Feature columns that are identity in disguise: constant within a ticker (a ticker id), or a monotone function of
    the date (a clock). Either lets a learner key on identity while looking like it uses a feature."""
    _check_panel(X)
    out: list[Finding] = []
    num = X.select_dtypes(include=[np.number])
    d = _dates(X).asi8.astype(float)
    tk = _tickers(X)
    for c in num.columns:
        v = num[c].to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 20 or np.nanstd(v) == 0:
            continue
        tot = np.nanvar(v)
        within = num[c].groupby(tk).transform(lambda s: s - s.mean()).to_numpy(dtype=float)
        eta = 1.0 - np.nanvar(within) / tot
        if eta >= eta_cut:
            out.append(fail(L, "ticker-proxy", c, f"{eta:.1%} of the variance is between tickers: the column identifies the stock", eta=float(eta)))
        r = np.corrcoef(v[ok], d[ok])[0, 1] if np.std(d[ok]) > 0 else 0.0
        if abs(r) >= trend_cut:
            out.append(fail(L, "date-proxy", c, f"correlation {r:.3f} with the calendar: the column is a clock", r=float(r)))
    return out


def label_only_sensitivity(learner: Learner, X_train, y_train, X_eval, seed: int = 0, min_changed: float = 0.5) -> dict:
    """How much do the scores change when ONLY the ticker labels of the eval rows are shuffled (values untouched)? An
    identity-free learner scores each row by its own features, so mapping the rows back must give identical scores."""
    base = learner(X_train, y_train, X_eval, seed).astype(float)
    t = ticker_permutation(X_eval, None, seed + 5)
    if t.changed_frac < min_changed:
        return {"sensitivity": float("nan"), "note": "too few tickers to shuffle"}
    s = learner(X_train, y_train, t.X, seed).astype(float)
    back = pd.Series(np.nan, index=X_eval.index)
    pos = cast(np.ndarray, t.inverse_rows)
    ok = ~pd.isna(pos)
    back.iloc[pos[ok].astype(int)] = s.to_numpy()[ok]
    both = pd.DataFrame({"a": base, "b": back}).dropna()
    if len(both) < 5 or both["a"].nunique() < 2:
        return {"sensitivity": float("nan"), "note": "scores have no variation"}
    corr = both["a"].rank().corr(both["b"].rank())
    return {"sensitivity": float(1.0 - corr), "rank_corr": float(corr), "n": int(len(both))}


def to_findings(report: IdentityReport) -> list[Finding]:
    """Translate an IdentityReport into firewall findings (COLLAPSE and NONDETERMINISTIC fail; inconclusive fails closed)."""
    out = list(report.findings)
    for v in report.verdicts:
        subj = f"{v.kind}[{v.mode}]"
        if v.status == "COLLAPSE":
            out.append(fail(L, "collapse", subj, f"retention {v.retention:.2f} (CI {v.ci_low:.2f}..{v.ci_high:.2f}) under {v.kind}, mode {v.mode}",
                            retention=v.retention))
        elif v.status == "NONDETERMINISTIC":
            out.append(fail(L, "nondeterministic", subj, "learner output not reproducible"))
        elif v.status in ("NO_SKILL", "INSUFFICIENT"):
            out.append(fail(L, "inconclusive", subj, f"{v.status}: {v.note or 'identity robustness not demonstrated'}"))
    if report.verdicts and not report.memorization_suspected and not report.logic_identity_dependent and report.passed:
        out.append(info(L, "identity-clean", "harness", "no attack reduced skill below the collapse ratio"))
    return out


# ---------------------------------------------------------------- skill significance
def ic_null_pvalue(scores: pd.Series, y: pd.Series, n_perm: int = 300, seed: int = 0, min_names: int = 5) -> dict:
    """Is the observed mean per-date rank IC distinguishable from chance? Scores are permuted WITHIN each date (the
    cross-section is what a stock picker is judged on), seed fixed. Returns mean IC, null mean/sd and a one-sided p-value.
    Used to decide whether a baseline has any skill worth retaining before an attack is blamed for destroying it."""
    df = pd.DataFrame({"s": scores.reindex(y.index), "y": y}).dropna()
    groups = [g for _, g in df.groupby(level=0) if len(g) >= min_names and g["s"].nunique() > 1 and g["y"].nunique() > 1]
    if not groups:
        return {"ic": float("nan"), "p": float("nan"), "n_dates": 0}
    ry = [g["y"].rank().to_numpy() for g in groups]
    rs = [g["s"].rank().to_numpy() for g in groups]

    def mean_ic(order_fn):
        vals = []
        for a, b in zip(rs, ry):
            bb = order_fn(b)
            vals.append(np.corrcoef(a, bb)[0, 1])
        return float(np.mean(vals))

    obs = mean_ic(lambda b: b)
    rng = np.random.default_rng(seed)
    null = np.array([mean_ic(lambda b: b[rng.permutation(len(b))]) for _ in range(n_perm)])
    p = float((1 + np.sum(null >= obs)) / (n_perm + 1))
    return {"ic": obs, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)), "p": p, "n_dates": len(groups)}


# ---------------------------------------------------------------- memorisation signatures
def seen_vs_unseen_gap(learner: Learner, X_train, y_train, X_eval, y_eval, seed: int = 0, k: int = 5) -> dict:
    """Split the evaluated rows into those whose exact (date, ticker) appeared in training and those that did not, and
    compare skill on each. A memoriser is brilliant on the first group and blind on the second; a learner with transferable
    knowledge performs alike on both. The gap is the memorisation signature of a plain rerun."""
    seen = X_eval.index.isin(X_train.index)
    if seen.all() or not seen.any():
        return {"gap": float("nan"), "note": "eval rows are all seen or all unseen: the comparison needs both", "n_seen": int(seen.sum()),
                "n_unseen": int((~seen).sum())}
    s = learner(X_train, y_train, X_eval, seed).astype(float)
    res = {}
    for name, m in (("seen", seen), ("unseen", ~seen)):
        ic = per_date_ic(s[m], y_eval[m])
        res[name] = float(ic.mean()) if len(ic) else float("nan")
        res[f"n_{name}"] = int(m.sum())
    res["gap"] = res["seen"] - res["unseen"] if np.isfinite(res["seen"]) and np.isfinite(res["unseen"]) else float("nan")
    return res


def rerun_linkability_panel(X_a: pd.DataFrame, X_b: pd.DataFrame, truth: Mapping[Any, Any], columns: Sequence[str] | None = None,
                            max_names: int = 300, seed: int = 0) -> dict:
    """Two disguised presentations of the SAME real window: can an observer re-identify names across them from the numbers
    alone? For each ticker in A the best-correlated feature history in B is taken as its match and compared with the true
    code map. Because a rerun replays identical values, this is ~1.0 on any panel: it documents that ANY numeric memory
    recognises a disguised rerun, which is why learned state must exclude the window's own first run (channel 4 / C56)."""
    _check_panel(X_a, "X_a")
    _check_panel(X_b, "X_b")
    cols = list(columns) if columns else list(X_a.select_dtypes(include=[np.number]).columns[:3])
    if not cols:
        return {"n_names": 0}

    def history(X):
        wide = X[cols[0]].unstack(level=1).sort_index()
        return wide

    A, B = history(X_a), history(X_b)
    common = [t for t in A.columns if truth.get(t) in B.columns]
    if not common:
        return {"n_names": 0}
    rng = np.random.default_rng(seed)
    if len(common) > max_names:
        common = list(rng.choice(common, size=max_names, replace=False))
    k = min(len(A), len(B))
    a = A[common].iloc[:k].fillna(0.0).to_numpy(dtype=float)
    b = B.iloc[:k].fillna(0.0).to_numpy(dtype=float)
    a = (a - a.mean(0)) / np.where(a.std(0) > 0, a.std(0), 1.0)
    b = (b - b.mean(0)) / np.where(b.std(0) > 0, b.std(0), 1.0)
    best = (a.T @ b).argmax(axis=1)
    cols_b = list(B.columns)
    hit = float(np.mean([cols_b[j] == truth[t] for t, j in zip(common, best)]))
    return {"n_names": len(common), "share_reidentified_by_values": hit,
            "note": "high share = a rerun is recognisable from numbers alone; learned state must not include the window's own first run"}


# ---------------------------------------------------------------- composing and attributing attacks
def compose(X, y, steps: Sequence[tuple[str, Mapping]], seed: int = 0) -> Transformed:
    """Apply several attacks in sequence (e.g. ticker permutation then date permutation). Each step gets its own derived
    seed. The result's `changed_frac` is the share of rows whose (date, ticker) label differs from the original."""
    if not steps:
        raise ValueError("compose needs at least one step")
    cur_X, cur_y = X, y
    orig_labels = list(zip(_dates(X), _tickers(X)))
    kinds = []
    for i, (name, kw) in enumerate(steps):
        if name not in ATTACKS:
            raise ValueError(f"unknown attack {name!r}")
        t = ATTACKS[name](cur_X, cur_y, seed + 31 * (i + 1), **dict(kw))
        cur_X, cur_y = t.X, t.y
        kinds.append(name)
    new_labels = dict(zip(zip(_dates(cur_X), _tickers(cur_X)), range(len(cur_X))))
    changed = float(np.mean([lab not in new_labels for lab in orig_labels]))
    return Transformed("+".join(kinds), cur_X, cur_y, changed, {}, None)


DIMENSION_OF_ATTACK = {"ticker_permutation": "ticker", "stock_substitution": "ticker", "presentation_disguise": "ticker+date",
                       "date_permutation": "date", "year_disguise": "date", "episode_substitution": "episode",
                       "sequence_scramble": "sequence", "sector_substitution": "sector"}


def attribute_keying(report: IdentityReport) -> dict[str, str]:
    """Which identity dimension does the learner key on? For each dimension: COLLAPSES if any eval-mode attack on it
    collapsed, HOLDS if every one held, UNTESTED otherwise. A learner that COLLAPSES on 'ticker' and HOLDS on 'date' is a
    ticker-memoriser; both COLLAPSE = a (ticker, date) lookup."""
    by: dict[str, list[str]] = {}
    for v in report.verdicts:
        if v.mode == "eval":
            by.setdefault(DIMENSION_OF_ATTACK.get(v.kind, v.kind), []).append(v.status)
    out = {}
    for dim in sorted(set(DIMENSION_OF_ATTACK.values())):
        st = by.get(dim)
        out[dim] = "UNTESTED" if not st else "COLLAPSES" if "COLLAPSE" in st else "HOLDS" if all(s == "OK" for s in st) else "INCONCLUSIVE"
    return out


# ---------------------------------------------------------------- many seeds
@dataclasses.dataclass(frozen=True)
class BatteryResult:
    reports: tuple[IdentityReport, ...]
    seeds: tuple[int, ...]

    def collapse_rate(self) -> pd.DataFrame:
        """Per (attack, mode): the share of seeds in which it collapsed. One collapse in ten seeds is noise; ten in ten is a finding."""
        rows = []
        for rep in self.reports:
            for v in rep.verdicts:
                rows.append({"kind": v.kind, "mode": v.mode, "collapsed": v.status == "COLLAPSE", "retention": v.retention})
        if not rows:
            return pd.DataFrame(columns=["kind", "mode", "n", "collapse_rate", "mean_retention"])
        g = pd.DataFrame(rows).groupby(["kind", "mode"])
        return pd.DataFrame({"n": g.size(), "collapse_rate": g["collapsed"].mean(), "mean_retention": g["retention"].mean()}).reset_index()

    def robust_collapse(self, min_rate: float = 0.7) -> list[str]:
        t = self.collapse_rate()
        return [f"{r.kind}[{r['mode']}]" for _, r in t.iterrows() if r["collapse_rate"] >= min_rate]

    @property
    def passed(self) -> bool:
        return bool(self.reports) and all(r.passed for r in self.reports)

    def digest(self) -> str:
        return stable_hash([r.digest() for r in self.reports])


def run_battery(learner: Learner, X_train, y_train, X_eval, y_eval, seeds: Sequence[int] = (0, 1, 2), **harness_kw) -> BatteryResult:
    """The whole harness under several seeds (different permutations, shifts and swaps). Attacks that are lucky or unlucky
    once are told apart from attacks that always bite. sector args pass through `harness_kw['run']` if needed."""
    run_kw = {k: harness_kw.pop(k) for k in ("sector_columns", "groups") if k in harness_kw}
    reps = []
    for sd in seeds:
        def _wrapped(a, b, c, s, _l=learner):
            return _l(a, b, c, s)
        h = IdentityHarness(_wrapped, seed=int(sd), **harness_kw)
        reps.append(h.run(X_train, y_train, X_eval, y_eval, **run_kw))
    return BatteryResult(tuple(reps), tuple(int(s) for s in seeds))


def report_to_json(report: IdentityReport) -> str:
    """Stable JSON of a report (for the ledger): verdicts, base metrics, determinism, digest."""
    import json
    return json.dumps({"seed": report.seed, "base_ic": report.base_ic, "base_topk": report.base_topk, "deterministic": report.deterministic,
                       "passed": report.passed, "digest": report.digest(), "verdicts": [dataclasses.asdict(v) for v in report.verdicts]},
                      sort_keys=True, default=str)


# ---------------------------------------------------------------- dose-response: how much identity does the learner use?
def substitution_curve(learner: Learner, X_train, y_train, X_eval, y_eval, fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
                       seed: int = 0, k: int = 5) -> pd.DataFrame:
    """Retention of skill as a growing fraction of the evaluated tickers swap identities. A memoriser's retention falls in
    proportion to the fraction changed (about 1 - f); a learner with transferable knowledge stays flat. The slope of this curve
    is a graded, threshold-free measure of memorisation."""
    s0 = learner(X_train, y_train, X_eval, seed).astype(float)
    ic0 = per_date_ic(s0, y_eval)
    base = float(ic0.mean()) if len(ic0) else float("nan")
    rows = []
    for f in fractions:
        t = stock_substitution(X_eval, y_eval, seed + 17, frac=float(f))
        s = learner(X_train, y_train, t.X, seed).astype(float)
        ic = per_date_ic(s, t.y)
        att = float(ic.mean()) if len(ic) else float("nan")
        rows.append({"fraction": float(f), "changed_frac": t.changed_frac, "ic": att,
                     "retention": att / base if np.isfinite(att) and np.isfinite(base) and abs(base) > 1e-12 else float("nan"),
                     "topk": top_k_spread(s, t.y, k)})
    return pd.DataFrame(rows, columns=["fraction", "changed_frac", "ic", "retention", "topk"])


def memorisation_slope(curve: pd.DataFrame) -> float:
    """Least-squares slope of retention against the fraction of names changed. About -1 for a lookup table, about 0 for a
    learner that does not use identity. NaN if fewer than two usable points."""
    d = curve.dropna(subset=["retention", "changed_frac"])
    if len(d) < 2 or d["changed_frac"].nunique() < 2:
        return float("nan")
    return float(np.polyfit(d["changed_frac"].to_numpy(dtype=float), d["retention"].to_numpy(dtype=float), 1)[0])


def name_versus_order(learner: Learner, X_train, y_train, X_eval, y_eval, seed: int = 0) -> dict:
    """Separate 'depends on the ticker NAMES' from 'depends on their ALPHABETICAL ORDER' (the tie-break leak). Fresh opaque
    codes that keep the alphabetical order are compared with codes that scramble it; a learner that only cares about order
    is hit by the second and not the first."""
    def skill(t):
        s = learner(X_train, y_train, t.X, seed).astype(float)
        ic = per_date_ic(s, t.y)
        return float(ic.mean()) if len(ic) else float("nan")
    base_s = learner(X_train, y_train, X_eval, seed).astype(float)
    base = per_date_ic(base_s, y_eval)
    b = float(base.mean()) if len(base) else float("nan")
    same_order = skill(ticker_permutation(X_eval, y_eval, seed + 3, order_preserving=True))
    scrambled = skill(ticker_permutation(X_eval, y_eval, seed + 3, order_preserving=False))
    ratio = lambda v: v / b if np.isfinite(v) and np.isfinite(b) and abs(b) > 1e-12 else float("nan")       # noqa: E731
    r_names, r_scr = ratio(same_order), ratio(scrambled)
    verdict = "NAMES" if np.isfinite(r_names) and r_names < 0.5 else "ORDER" if np.isfinite(r_scr) and r_scr < 0.5 else "NEITHER"
    return {"base_ic": b, "new_names_same_order": same_order, "scrambled": scrambled, "retention_new_names": r_names,
            "retention_scrambled": r_scr, "depends_on": verdict}


def attack_power(report: IdentityReport) -> pd.DataFrame:
    """Per attack: was it actually strong enough to matter? Rows changed, retention, and whether the baseline could be judged.
    An attack that changed few rows, or ran on a baseline with no skill, proves nothing either way."""
    rows = []
    for v in report.verdicts:
        rows.append({"kind": v.kind, "mode": v.mode, "changed_frac": v.changed_frac, "retention": v.retention, "status": v.status,
                     "informative": v.status in ("OK", "COLLAPSE") and v.changed_frac >= 0.25})
    return pd.DataFrame(rows, columns=["kind", "mode", "changed_frac", "retention", "status", "informative"])


# ---------------------------------------------------------------- cross-identity transfer
def cross_identity_transfer(learner: Learner, X: pd.DataFrame, y: pd.Series, seed: int = 0, train_frac: float = 0.5, k: int = 5) -> dict:
    """Train on half the tickers (and the first part of time), evaluate three ways: on the SAME tickers later (time
    transfer), on UNSEEN tickers over the same period (stock transfer), and on unseen tickers later (both). A learner with
    transferable knowledge scores alike in all three; a name-keyed one loses the stock transfer. Contract sections 26/29."""
    _check_panel(X)
    tk = sorted(set(_tickers(X)))
    rng = np.random.default_rng(seed)
    train_t = set(rng.permutation(tk)[:max(1, int(len(tk) * train_frac))])
    dates = sorted(set(_dates(X)))
    cut = dates[int(len(dates) * train_frac)]
    dmask = np.asarray(_dates(X) < cut)
    tmask = np.asarray(_tickers(X).isin(train_t))

    def part(rows):
        return X[rows], y.reindex(X.index)[rows]
    Xtr, ytr = part(dmask & tmask)
    out: dict[str, Any] = {"n_train": int(len(Xtr))}
    for name, rows in (("time", ~dmask & tmask), ("stock", dmask & ~tmask), ("both", ~dmask & ~tmask)):
        Xe, ye = part(rows)
        if not len(Xe):
            out[name] = float("nan")
            continue
        s = learner(Xtr, ytr, Xe, seed).astype(float)
        ic = per_date_ic(s, ye)
        out[name] = float(ic.mean()) if len(ic) else float("nan")
        out[f"{name}_topk"] = top_k_spread(s, ye, k)
    ref = out["time"]
    out["stock_retention"] = out["stock"] / ref if np.isfinite(out["stock"]) and np.isfinite(ref) and abs(ref) > 1e-12 else float("nan")
    out["transfers_across_stocks"] = bool(np.isfinite(out["stock_retention"]) and out["stock_retention"] >= 0.5)
    return out


def battery_markdown(bat: BatteryResult) -> str:
    """Report of a multi-seed battery: collapse frequency per attack and the attacks that bite in most seeds."""
    t = bat.collapse_rate()
    lines = [f"# Identity battery over seeds {list(bat.seeds)}", f"all seeds passed: {bat.passed}", "",
             "| attack | mode | seeds | collapse rate | mean retention |", "|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['kind']} | {r['mode']} | {int(r['n'])} | {r['collapse_rate']:.0%} | {r['mean_retention']:.2f} |")
    if bat.robust_collapse():
        lines.append("")
        lines.append("Robust collapse (memorisation suspected): " + ", ".join(bat.robust_collapse()))
    lines.append("")
    lines.append("IMPLEMENTED - NOT VALIDATED.")
    return "\n".join(lines)


# ---------------------------------------------------------------- can the instrument see memorisation? (harness self-check)
def harness_selfcheck(seed: int = 0, n_dates: int = 40, n_names: int = 18, boot: int = 80) -> dict:
    """Run the four reference learners through the harness on a synthetic panel with a real (linear) signal and a planted
    per-ticker effect, and check each lands where it must: the memoriser COLLAPSES on the rerun, the ticker lookup collapses on
    ticker attacks only, the legitimate ridge stays OK, the random learner has NO_SKILL. A harness that fails this cannot be
    trusted on a real learner (instruments lie in both directions)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-04", periods=n_dates * 5)[::5][:n_dates]
    tickers = [f"H{i:03d}" for i in range(n_names)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    X = pd.DataFrame({"f1": rng.normal(size=len(idx)), "f2": rng.normal(size=len(idx))}, index=idx)
    tk_effect = np.tile(np.linspace(-0.03, 0.03, n_names), n_dates)
    y = pd.Series(0.03 * X["f1"].to_numpy() + tk_effect + rng.normal(0, 0.03, len(idx)), index=idx, name="y")
    half = dates[n_dates // 2]
    tr = np.asarray(X.index.get_level_values(0) < half)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[~tr], y[~tr]
    kw: dict[str, Any] = dict(attacks=("ticker_permutation", "date_permutation"), modes=("eval",), boot=boot, seed=seed)
    out = {}
    out["memorizer"] = IdentityHarness(memorizer_learner, **kw).run(X, y, X, y)
    out["ticker_mean"] = IdentityHarness(ticker_mean_learner, **kw).run(Xtr, ytr, Xte, yte)
    out["ridge"] = IdentityHarness(ridge_learner, **kw).run(Xtr, ytr, Xte, yte)
    out["random"] = IdentityHarness(random_learner, **kw).run(Xtr, ytr, Xte, yte)

    def st(rep, kind):
        return next(v.status for v in rep.verdicts if v.kind == kind)
    expect = {"memorizer/ticker": (st(out["memorizer"], "ticker_permutation"), "COLLAPSE"),
              "memorizer/date": (st(out["memorizer"], "date_permutation"), "COLLAPSE"),
              "ridge/ticker": (st(out["ridge"], "ticker_permutation"), "OK"), "ridge/date": (st(out["ridge"], "date_permutation"), "OK"),
              "ticker_mean/date": (st(out["ticker_mean"], "date_permutation"), "OK"),
              "random/ticker": (st(out["random"], "ticker_permutation"), "NO_SKILL")}
    wrong = {k: {"got": g, "want": w} for k, (g, w) in expect.items() if g != w}
    return {"ok": not wrong, "wrong": wrong, "observed": {k: v[0] for k, v in expect.items()},
            "memorizer_distinguished": distinguish_from_memorizer(out["ridge"], out["memorizer"])["distinguishable"]}


def retention_summary(report: IdentityReport) -> dict:
    """Mean / worst retention per mode across attacks that could be judged, and how many attacks were informative."""
    out = {}
    for mode in MODES:
        r = [v.retention for v in report.verdicts if v.mode == mode and np.isfinite(v.retention)]
        out[mode] = {"mean": float(np.mean(r)) if r else float("nan"), "worst": float(np.min(r)) if r else float("nan"), "n": len(r)}
    return out


def is_identity_free(learner: Learner, X_train, y_train, X_eval, seed: int = 0, tol: float = 0.05) -> bool:
    """Quick structural test: shuffling only the ticker labels of the evaluated rows (values untouched) must not change the
    learner's scores once mapped back. True means the learner does not read identity at all."""
    res = label_only_sensitivity(learner, X_train, y_train, X_eval, seed)
    return bool(np.isfinite(res.get("sensitivity", float("nan"))) and res["sensitivity"] <= tol)


def informative_attacks(report: IdentityReport) -> list[str]:
    """Attacks (kind[mode]) whose verdict actually says something: judged OK or COLLAPSE on a baseline with skill."""
    return sorted(f"{v.kind}[{v.mode}]" for v in report.verdicts if v.status in ("OK", "COLLAPSE"))


def collapse_kinds(report: IdentityReport, mode: str | None = None) -> list[str]:
    """Attack kinds that collapsed, optionally restricted to one mode, sorted and de-duplicated."""
    return sorted({v.kind for v in report.verdicts if v.status == "COLLAPSE" and (mode is None or v.mode == mode)})


def any_collapse(report: IdentityReport) -> bool:
    return bool(collapse_kinds(report))
