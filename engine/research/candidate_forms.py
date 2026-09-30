"""Candidate representation and screen recall (C75 Phase 3A/3B 'search broadly, identify potential patterns, normalise, deduplicate',
3I interactions / XOR; canon C70 'identify ALL the potential patterns'; C63, C64). IMPLEMENTED - NOT VALIDATED.

F19 ranked two candidate-generation failures: (3) 240 of 900 real patterns can only be seen in their TRUE FORM (XOR, product
interaction, lagged, rare event, threshold cut, conditional) and the single-feature screen cannot express them; (4) only a third of the
real patterns were ever raised as candidates. This module makes every one of those forms EXPRESSIBLE, screens all of them cheaply,
counts every test it ran, and hands a short deduplicated list to the existing screen (volatility_lab.oriented_scan) and gate.

Forms (all point in time: a value at date t reads only rows dated <= t; nothing reads the outcome):
  lag{k}      xs-z of the atom k rows earlier for the same name, k from the pre-declared `lags`                (delayed effects)
  gt{q}       1[xs percentile rank > q], q from the pre-declared `cuts`                                          (threshold)
  zp{c}/zn{c} 1[+-(atom - mean of all EARLIER dates) / sd of earlier dates > c], c from `rare_z` (one-sided tails) (rare events)
  prod        xsz(a) * xsz(b)                                                                                  (interactive)
  xor         sign(rank(a) - .5) * sign(rank(b) - .5)                                                          (XOR)
  and{hh,hl,lh,ll}  1[a on side 1] * 1[b on side 2] (median split)                                             (conjunction)
  cond{hi,lo} xsz(a) * 1[b above / below its same-date median]                                                 (a only where b holds)
  reg{hi,lo}  xsz(a) * 1[market atom m above / below its expanding past median]                               (feature x regime)
A composite is a runtime derived feature named `ix__cf_<tag>__<a>[__<b>]` registered in vol_hypotheses.DERIVED (the registry the
screen, the evidence assembler and the benchmark already read); the `ix__` prefix keeps it OUT of every blanket scan
(loop.st_feature_screen, pattern_benchmark.scan_universe), so a composite is only ever scored when this module proposed it and
counted it.

Prescreen (the search): the outcome is residualised on date and name means (two-way: base-rate eras and persistent per-name rates -
what regime_corr / identity / autocorrelation traps latch onto - cannot rank a form), then every form is scored by a Fama-MacBeth t
over dates of the per-date sum yc * u. Pair forms are computed for EVERY pair with one BLAS product per date per form (the panel-scale,
per-date generalisation of targets._interaction_seeds' median-split quadrant screen, which needs no marginal association: an XOR
pair's members have none). The prescreen can be restricted to the earliest `select_frac` of dates so that the walk-forward screen
that follows scores the selected forms on dates the selection never saw (selection-honest confirmation).

Normalise / deduplicate: near-duplicate atoms are merged before the search (discovery.redundant_columns), each form has one canonical
spelling (symmetric pairs sorted, a quadrant stored once, a threshold and its complement are one form), and a proposed form whose
values correlate >= `dedup_rho` with a stronger proposal or with one of its own atoms is the same hypothesis reached two ways: it
is dropped and recorded as an alias of the survivor.

Multiplicity: `Proposal.scanned` counts every test per family (singles included); `multiplicity_cost` states the Bonferroni bar
(interactions.detectable_t) at that count and under a family-split alpha. The evaluation helpers at the bottom (`match`,
`score_patterns`, ...) read a benchmark answer key AFTER generation; the generator (`propose`) never takes one.

Public entries: `propose(F, feats, now, cfg)` (the generator), `screen_table(F, feats, now, lab, cfg)` (the screen entry the benchmark
calls instead of oriented_scan), `session()` / `register` / `unregister_all` (registry hygiene), `parse` / `family_of` /
`complexity` / `history_dependent`, `multiplicity_cost`, `raise_rule`, `match` / `score_patterns` / `recall_table`."""
from __future__ import annotations

import functools

import contextlib
import dataclasses as dc
import math
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from engine.research import vol_hypotheses as VH
from engine.research.core import FirewallBreach, as_date, stable_hash

LABEL = "IMPLEMENTED - NOT VALIDATED"
PFX = "ix__cf_"
EPS = VH.EPS
PAIR_FAMILIES = ("prod", "xor", "and", "cond")
ATOM_FAMILIES = ("lag", "gt", "rare", "reg")
FAMILIES = ("single",) + ATOM_FAMILIES + PAIR_FAMILIES
SEARCHED = ATOM_FAMILIES + ("prod", "xor", "cond")        # "and" is spellable but not searched (see _family_search)
AND_SIDES = ("hh", "hl", "lh", "ll")
_REGISTERED: set[str] = set()


class FormError(ValueError):
    """A form that cannot be built (unknown family, bad parameter, an atom name that would make the spelling ambiguous)."""


# ================================================================================================================ configuration
@dc.dataclass(frozen=True)
class FormConfig:
    lags: tuple = (1, 2, 3, 4)                         # pre-declared: up to a month of weekly rows
    cuts: tuple = (0.10, 0.25, 0.50, 0.75, 0.90)       # pre-declared cross-sectional percentile cuts
    rare_z: tuple = (1.2816, 1.6449, 1.9600)           # one-sided normal tails of 10 / 5 / 2.5 %, on the expanding past distribution
    rare_min_dates: int = 8                            # dates of history before an expanding cut exists
    families: tuple = SEARCHED
    k_per_family: int = 8                              # proposals kept per family (the operating point; see the recall runner)
    keep_top: int = 40                                 # ranked list kept per family for the trade-off curve
    t_pre: float = 2.5                                 # prescreen |t| a form needs before it can be proposed
    select_frac: float = 1.0 / 3.0                     # prescreen on the first third of dates: before the screen's first test fold
    dedup_rho: float = 0.90
    atom_dup_rho: float = 0.98
    y_col: str = "touch"
    min_names: int = 8                                 # a date with fewer scored names is not a date
    min_dates: int = 20
    max_pair_atoms: int = 1200                         # above this the pair families are skipped WITH A REASON, never sampled
    t_main: float = 6.0                                # a single this strong is raised on its own: its forms are not searched

    def validate(self) -> list[str]:
        errs = []
        if not self.lags or any(int(k) != k or k < 1 for k in self.lags):
            errs.append("lags must be positive integers")
        if not self.cuts or any(not 0 < q < 1 for q in self.cuts):
            errs.append("cuts must lie in (0, 1)")
        if not self.rare_z or any(c <= 0 for c in self.rare_z):
            errs.append("rare_z must be positive")
        if set(self.families) - set(SEARCHED):
            errs.append(f"unknown families {sorted(set(self.families) - set(SEARCHED))}")
        if self.k_per_family < 0 or self.keep_top < self.k_per_family:
            errs.append("0 <= k_per_family <= keep_top required")
        if not 0 < self.select_frac <= 1 or not 0 < self.dedup_rho <= 1 or not 0 < self.atom_dup_rho <= 1:
            errs.append("select_frac, dedup_rho, atom_dup_rho in (0, 1]")
        if self.min_names < 3 or self.min_dates < 5 or self.t_pre < 0:
            errs.append("min_names >= 3, min_dates >= 5, t_pre >= 0 required")
        return errs


# ================================================================================================================ one form
def _q_tag(q: float) -> str:
    return f"{int(round(q * 100)):02d}"


def _z_tag(c: float) -> str:
    return f"{int(round(c * 100)):03d}"


@dc.dataclass(frozen=True)
class FormSpec:
    """One candidate form in its canonical spelling. `param`: lag 'k', gt 'qq', rare 'p/n' + 'ccc', and 'hh/hl/lh/ll',
    cond / reg 'hi/lo'. Atoms: (a,) for atom families, (a, b) for pairs; reg's b is the market atom; cond's b is the conditioner."""
    family: str
    param: str
    atoms: tuple

    def __post_init__(self):
        errs = self.validate()
        if errs:
            raise FormError(f"{self.family}/{self.param}/{self.atoms}: " + "; ".join(errs))

    def validate(self) -> list[str]:
        errs = []
        n = 2 if self.family in PAIR_FAMILIES + ("reg",) else 1
        if self.family not in FAMILIES or self.family == "single":
            errs.append("unknown family")
        if len(self.atoms) != n:
            errs.append(f"needs {n} atom(s)")
        if any((not a) or "__" in a or a.startswith("ix__") for a in self.atoms):
            errs.append("atom names must be non-empty, contain no '__' and not be composites")
        if n == 2 and len(set(self.atoms)) != 2:
            errs.append("a pair needs two different atoms")
        ok = {"lag": lambda p: p.isdigit() and int(p) >= 1, "gt": lambda p: p.isdigit() and 0 < int(p) < 100,
              "rare": lambda p: len(p) == 4 and p[0] in "pn" and p[1:].isdigit(), "and": lambda p: p in AND_SIDES,
              "cond": lambda p: p in ("hi", "lo"), "reg": lambda p: p in ("hi", "lo"), "prod": lambda p: p == "",
              "xor": lambda p: p == ""}.get(self.family)
        if ok is not None and not ok(self.param):
            errs.append(f"bad parameter {self.param!r}")
        return errs

    @property
    def tag(self) -> str:
        return {"rare": "z"}.get(self.family, self.family) + self.param

    @property
    def name(self) -> str:
        return PFX + self.tag + "__" + "__".join(self.atoms)

    @property
    def complexity(self) -> int:
        """Free choices the form adds on top of one feature: a parameter for atom forms, a second feature (+ its side) for pairs."""
        return 1 if self.family in ATOM_FAMILIES and self.family != "reg" else 2

    @property
    def history_dependent(self) -> bool:
        """Needs earlier dates of the panel (a one-day frame cannot compute it) - cf. two_stage.HISTORY_DEPENDENT."""
        return self.family in ("lag", "rare", "reg")


def canonical(family: str, param: str, atoms: Sequence[str]) -> FormSpec:
    """The one spelling of a form: symmetric pairs sorted; a quadrant stored with its atoms sorted (sides swapped with them); a
    'below q' threshold is the complement of 'above q' (the same hypothesis, oriented later) and is spelled as 'gt'."""
    atoms = tuple(str(a) for a in atoms)
    if family in ("prod", "xor"):
        return FormSpec(family, "", tuple(sorted(atoms)))
    if family == "and":
        if len(atoms) == 2 and atoms[0] > atoms[1]:
            return FormSpec("and", param[::-1], (atoms[1], atoms[0]))
        return FormSpec("and", param, atoms)
    if family == "lt":
        return FormSpec("gt", param, atoms)
    return FormSpec(family, param, atoms)


def parse(name: str) -> FormSpec | None:
    """The FormSpec behind a composite name; None for a plain feature. A malformed composite name raises (fail closed)."""
    if not str(name).startswith(PFX):
        return None
    parts = str(name)[len(PFX):].split("__")
    tag, atoms = parts[0], tuple(parts[1:])
    for fam, head in (("lag", "lag"), ("gt", "gt"), ("rare", "z"), ("prod", "prod"), ("xor", "xor"), ("and", "and"), ("cond", "cond"),
                      ("reg", "reg")):
        if tag.startswith(head):
            spec = canonical(fam, tag[len(head):], atoms)
            if spec.name != name:
                raise FormError(f"{name!r} is not in canonical spelling ({spec.name!r})")
            return spec
    raise FormError(f"unknown composite tag in {name!r}")


def family_of(name: str) -> str:
    s = parse(name)
    return "single" if s is None else s.family


def complexity(name: str) -> int:
    s = parse(name)
    return 0 if s is None else s.complexity


def history_dependent(name: str) -> bool:
    s = parse(name)
    return bool(s is not None and s.history_dependent)


def base_atoms(name: str) -> tuple:
    s = parse(name)
    return (str(name),) if s is None else s.atoms


# ================================================================================================================ point-in-time primitives
def _sorted(F: pd.DataFrame | pd.Series):
    return F if F.index.is_monotonic_increasing else F.sort_index()


def xs_rank(s: pd.Series) -> pd.Series:
    return s.groupby(level=0).rank(pct=True)


def xs_z(s: pd.Series) -> pd.Series:
    return VH._xsz(s)


def lag_rows(s: pd.Series, k: int) -> pd.Series:
    """The same name's value k rows (dates) earlier; NaN for the first k rows of each name. Reads only earlier dates."""
    t = s.sort_index(level=[1, 0], sort_remaining=False)
    return t.groupby(level=1, sort=False).shift(int(k)).reindex(s.index)


def expanding_z(s: pd.Series, min_dates: int = 8) -> pd.Series:
    """(value - mean of every value on EARLIER dates) / sd of those values: the rarity of today's value against the past only (the
    current date's cross-section is excluded, so a whole-market jump today is still rare). NaN until `min_dates` dates of history."""
    s = pd.Series(np.asarray(s, float), index=s.index)
    g = s.groupby(level=0)
    agg = pd.DataFrame({"s1": g.sum(min_count=1), "s2": (s * s).groupby(level=0).sum(min_count=1), "n": g.count()}).sort_index()
    agg["has"] = (agg["n"] > 0).astype(float)
    cum = agg.fillna(0.0).cumsum().shift(1)
    mu = cum["s1"] / cum["n"]
    var = (cum["s2"] / cum["n"] - mu * mu) * cum["n"] / (cum["n"] - 1)
    sd = np.sqrt(var.clip(lower=0.0))
    ok = (cum["has"] >= min_dates) & (sd > EPS)
    d = s.index.get_level_values(0)
    mu_r, sd_r, ok_r = mu.reindex(d).to_numpy(), sd.reindex(d).to_numpy(), ok.reindex(d).fillna(False).to_numpy(bool)
    return pd.Series(np.where(ok_r, (s.to_numpy() - mu_r) / np.where(ok_r, sd_r, 1.0), np.nan), index=s.index)


def regime_side(m: pd.Series, side: str, min_periods: int = 8) -> pd.Series:
    """1 where the date-level atom is above (hi) / below (lo) its expanding past median (vol_hypotheses.expanding_pct, which reads the
    values seen up to and including that date); NaN before enough history. Broadcast back onto the rows."""
    d = m.groupby(level=0).first().sort_index()
    pct = VH.expanding_pct(d, min_periods=min_periods)
    ind = (pct > 0.5) if side == "hi" else (pct < 0.5)
    out = ind.astype(float).where(pct.notna())
    return out.reindex(m.index.get_level_values(0)).set_axis(m.index)


def _side(r: pd.Series, side: str) -> pd.Series:
    return ((r > 0.5) if side == "h" else (r < 0.5)).astype(float).where(r.notna())


def _atom(F: pd.DataFrame, a: str) -> pd.Series:
    if a not in VH.DERIVED:
        raise KeyError(f"unknown atom {a!r}")
    return pd.Series(np.asarray(VH.DERIVED[a][1](F), float), index=F.index).replace([np.inf, -np.inf], np.nan)


def _materialise_for(spec, cfg, F):
    """VH.DERIVED builder for a registered composite (a named partial instead of a default-argument lambda, for the type checker)."""
    return materialise(spec, F, cfg)


def materialise(spec: FormSpec, F: pd.DataFrame, cfg: FormConfig = FormConfig()) -> pd.Series:
    """The form's values on F's rows (the DERIVED function of the composite). Same definitions as the prescreen matrices."""
    a = _atom(F, spec.atoms[0])
    fam, p = spec.family, spec.param
    if fam == "lag":
        v = xs_z(lag_rows(a, int(p)))
    elif fam == "gt":
        r = xs_rank(a)
        v = (r > int(p) / 100.0).astype(float).where(r.notna())
    elif fam == "rare":
        z = expanding_z(a, cfg.rare_min_dates) * (1.0 if p[0] == "p" else -1.0)
        v = (z > int(p[1:]) / 100.0).astype(float).where(z.notna())
    else:
        b = _atom(F, spec.atoms[1])
        if fam == "prod":
            v = xs_z(a) * xs_z(b)
        elif fam == "xor":
            v = np.sign(xs_rank(a) - 0.5) * np.sign(xs_rank(b) - 0.5)
        elif fam == "and":
            v = _side(xs_rank(a), p[0]) * _side(xs_rank(b), p[1])
        elif fam == "cond":
            v = xs_z(a) * _side(xs_rank(b), p[0])
        else:
            v = xs_z(a) * regime_side(b, p, cfg.rare_min_dates)
    return pd.Series(np.asarray(v, float), index=F.index).replace([np.inf, -np.inf], np.nan)


# ================================================================================================================ registry hygiene
def register(specs: Iterable[FormSpec], cfg: FormConfig = FormConfig()) -> list[str]:
    """Add the forms to vol_hypotheses.DERIVED (idempotent) so the screen, the evidence assembler and the gate can derive them like any
    feature. Returns the names. Every name added here is removed by `unregister_all` / on leaving `session()`."""
    names = []
    for s in specs:
        for a in s.atoms:
            if a not in VH.DERIVED:
                raise KeyError(f"atom {a!r} of {s.name} is not a registered derived feature")
        if s.name not in VH.DERIVED:
            base = tuple(dict.fromkeys(c for a in s.atoms for c in VH.DERIVED[a][0]))
            VH.DERIVED[s.name] = (base, functools.partial(_materialise_for, s, cfg))
            _REGISTERED.add(s.name)
        names.append(s.name)
    return names


def unregister_all() -> int:
    n = 0
    for name in list(_REGISTERED):
        if VH.DERIVED.pop(name, None) is not None:
            n += 1
        _REGISTERED.discard(name)
    return n


@contextlib.contextmanager
def session() -> Iterator[None]:
    """Composites registered inside live only inside: planted column names repeat across benchmark worlds, and a composite left in
    the registry from one world would silently be derivable (and mean something else) in the next."""
    before = set(_REGISTERED)
    try:
        yield
    finally:
        for name in list(_REGISTERED - before):
            VH.DERIVED.pop(name, None)
            _REGISTERED.discard(name)


# ================================================================================================================ the atom panel
@dc.dataclass
class AtomPanel:
    """Everything the prescreen needs, built once per call from the frame (no outcome enters any matrix except yc)."""
    names: list                      # cross-sectional atoms after dedup (column order of every matrix)
    market: list                     # date-level atoms (regime conditioners)
    index: pd.MultiIndex
    starts: np.ndarray               # row offsets of each date (len = n_dates + 1)
    valid: np.ndarray                # per date: enough names
    yc: np.ndarray                   # two-way residual outcome
    R: np.ndarray                    # xs percentile rank (NaN kept)
    Z: np.ndarray                    # xs z (NaN -> 0)
    lagz: dict                       # k -> xs z of the k-row lag (NaN -> 0)
    zexp: np.ndarray                 # expanding past z (NaN kept)
    reg: dict                        # (market atom, side) -> per-row indicator (NaN -> 0)
    dup_atoms: dict                  # dropped near-duplicate atom -> its keeper
    constant: list                   # atoms with no variation (not screenable)
    select_dates: int                # dates the prescreen used
    single_t: np.ndarray = dc.field(default_factory=lambda: np.zeros(0))   # FM t of every atom on the two-way outcome
    main: tuple = ()                 # strong singles (|t| >= t_main): raised on their own, excluded from the form search


def _guard(F: pd.DataFrame, now, y_col: str) -> None:
    """Fail closed (rule 3 / C56): a row dated at/after `now`, or an outcome not matured strictly before `now`."""
    if len(F) == 0:
        return
    nd = np.datetime64(pd.Timestamp(as_date(now)))
    d = pd.to_datetime(F.index.get_level_values(0)).to_numpy()
    if (d >= nd).any():
        raise FirewallBreach(f"candidate_forms: {int((d >= nd).sum())} rows dated at/after now={as_date(now)}")
    if "end" in F.columns and y_col in F.columns:
        has = F[y_col].notna().to_numpy()
        end = pd.to_datetime(F["end"]).dt.normalize().to_numpy()
        if (has & (end >= nd)).any():
            raise FirewallBreach(f"candidate_forms: {int((has & (end >= nd)).sum())} outcomes not matured before now={as_date(now)}")


def _date_stats(D: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = D.groupby(level=0)
    return g.transform("mean"), g.transform("std")


def build_panel(F: pd.DataFrame, feats: Sequence[str], now, cfg: FormConfig = FormConfig()) -> AtomPanel:
    _guard(F, now, cfg.y_col)
    F = _sorted(F)
    F = F[F[cfg.y_col].notna()] if cfg.y_col in F.columns else F.iloc[0:0]
    atoms = [f for f in dict.fromkeys(feats) if f in VH.DERIVED and not str(f).startswith(("ix__", "ixnull")) and "__" not in f
             and not VH.missing_columns((f,), F.columns)]
    dates = F.index.get_level_values(0)
    codes, uniq = pd.factorize(dates)
    n_dates = len(uniq)
    sel = max(1, int(math.floor(cfg.select_frac * n_dates + 1e-9)))
    if sel < n_dates:
        F = F[codes < sel]
        dates = F.index.get_level_values(0)
        codes, uniq = pd.factorize(dates)
        n_dates = len(uniq)
    empty = np.zeros((len(F), 0), np.float32)
    if len(F) == 0 or not atoms:
        return AtomPanel([], [], F.index, np.zeros(1, int), np.zeros(0, bool), np.zeros(0), empty, empty, {}, empty, {}, {}, [], n_dates)
    D = VH.derive(F, atoms).astype(float)
    mean, std = _date_stats(D)
    within = std.fillna(0.0).abs().max()
    overall = D.std()
    constant = sorted(c for c in atoms if not np.isfinite(overall[c]) or overall[c] <= EPS)
    market = sorted(c for c in atoms if c not in constant and within[c] <= 1e-9)
    cross = [c for c in atoms if c not in constant and c not in market]
    dup: dict[str, str] = {}
    if len(cross) > 1:
        from engine.research.discovery import redundant_columns          # lazy: discovery is heavy and only needed here
        dup = redundant_columns(D[cross], threshold=cfg.atom_dup_rho)
        cross = [c for c in cross if c not in dup]
    starts = np.r_[np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]), len(F)]
    counts = np.diff(starts)
    valid = counts >= cfg.min_names
    y = F[cfg.y_col].to_numpy(float)
    yc = _two_way(y, codes, pd.factorize(F.index.get_level_values(1))[0])
    Dc = D[cross]
    R = Dc.groupby(level=0).rank(pct=True).to_numpy(np.float32)
    Z = ((Dc - mean[cross]) / (std[cross] + EPS)).fillna(0.0).to_numpy(np.float32)
    lagz = {}
    if "lag" in cfg.families:
        for k in cfg.lags:
            L = Dc.sort_index(level=[1, 0], sort_remaining=False).groupby(level=1, sort=False).shift(int(k)).reindex(Dc.index)
            m2, s2 = _date_stats(L)
            lagz[int(k)] = ((L - m2) / (s2 + EPS)).fillna(0.0).to_numpy(np.float32)
    zexp = np.column_stack([expanding_z(Dc[c], cfg.rare_min_dates).to_numpy(np.float32) for c in cross]) if "rare" in cfg.families \
        else np.zeros((len(F), 0), np.float32)
    reg = {}
    if "reg" in cfg.families:
        for m in market:
            for side in ("hi", "lo"):
                reg[(m, side)] = regime_side(D[m], side, cfg.rare_min_dates).fillna(0.0).to_numpy(np.float32)
    P = AtomPanel(cross, market, F.index, starts, valid, yc, R, Z, lagz, zexp, reg, dup, constant, n_dates)
    P.single_t = fm_t(Z, P)
    t = np.nan_to_num(np.abs(P.single_t), nan=0.0)
    P.main = tuple(sorted(c for c, v in zip(cross, t) if v >= cfg.t_main))
    return P


def _codes(P: AtomPanel) -> np.ndarray:
    return np.repeat(np.arange(len(P.starts) - 1), np.diff(P.starts))


def beyond_own(U: np.ndarray, Zo: np.ndarray, P: AtomPanel) -> np.ndarray:
    """Frisch-Waugh: column j of U with its per-date mean and its pooled projection on its OWN atom's z (column j of Zo) removed. The
    FM t of the result is the form's value BEYOND the atom's straight-line effect (a threshold beyond a slope, last week's value
    beyond this week's). Only the form's own atom is partialled out: projecting on OTHER strong features conditions on colliders
    (measured: a leak column, which mixes the outcome with volatility, made every volatility feature look predictive, t ~ -10)."""
    if U.shape[1] == 0:
        return U.astype(np.float64)
    c = _codes(P)
    cnt = np.maximum(np.bincount(c), 1)[:, None]
    Uc = U.astype(np.float64) - (np.add.reduceat(U.astype(np.float64), P.starts[:-1], axis=0) / cnt)[c]
    Zd = Zo.astype(np.float64)
    den = (Zd * Zd).sum(axis=0)
    beta = np.where(den > 1e-12, (Uc * Zd).sum(axis=0) / np.where(den > 1e-12, den, 1.0), 0.0)
    return Uc - beta[None, :] * Zd


def _two_way(y: np.ndarray, dcode: np.ndarray, tcode: np.ndarray, sweeps: int = 3) -> np.ndarray:
    """y minus date means and name means (alternating projections): what is left cannot be predicted by 'which week' or 'which name'."""
    r = y.astype(float).copy()
    for _ in range(sweeps):
        r -= (np.bincount(dcode, r) / np.maximum(np.bincount(dcode), 1))[dcode]
        r -= (np.bincount(tcode, r) / np.maximum(np.bincount(tcode), 1))[tcode]
    r -= (np.bincount(dcode, r) / np.maximum(np.bincount(dcode), 1))[dcode]
    return r


# ================================================================================================================ Fama-MacBeth statistics
def fm_t(U: np.ndarray, P: AtomPanel) -> np.ndarray:
    """Per column of U (rows x K): the t over valid dates of the per-date sum yc * u. NaN where the column never varies."""
    if U.shape[1] == 0 or not P.valid.any():
        return np.zeros(U.shape[1])
    X = np.nan_to_num(U.astype(np.float64), nan=0.0) * P.yc[:, None]
    s = np.add.reduceat(X, P.starts[:-1], axis=0)[P.valid]
    T = s.shape[0]
    if T < 3:
        return np.full(U.shape[1], np.nan)
    sd = s.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 1e-12, s.mean(axis=0) / sd * math.sqrt(T), np.nan)


def fm_pairs(A: np.ndarray, B: np.ndarray, P: AtomPanel) -> np.ndarray:
    """t for every (i, j) of the per-date sums yc * A_i * B_j: one (K x rows_t) @ (rows_t x K) product per date."""
    K = A.shape[1]
    S1 = np.zeros((K, K))
    S2 = np.zeros((K, K))
    T = 0
    for d in np.flatnonzero(P.valid):
        a, b = P.starts[d], P.starts[d + 1]
        m = (A[a:b].astype(np.float64) * P.yc[a:b, None]).T @ B[a:b].astype(np.float64)
        S1 += m
        S2 += m * m
        T += 1
    if T < 3:
        return np.full((K, K), np.nan)
    mu = S1 / T
    var = (S2 / T - mu * mu) * T / (T - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var > 1e-18, mu / np.sqrt(np.maximum(var, 1e-300)) * math.sqrt(T), np.nan)


def _sign0(R: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.sign(R - 0.5), nan=0.0).astype(np.float32)


def _ind(R: np.ndarray, side: str) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nan_to_num((R > 0.5) if side == "h" else (R < 0.5), nan=0.0).astype(np.float32)


def vector(P: AtomPanel, spec: FormSpec, cfg: FormConfig = FormConfig()) -> np.ndarray:
    """The prescreen's value of one form on the panel rows (NaN -> 0), the same definition `materialise` gives the screen."""
    ix = {c: i for i, c in enumerate(P.names)}
    i = ix[spec.atoms[0]]
    fam, p = spec.family, spec.param
    if fam == "lag":
        return P.lagz[int(p)][:, i]
    if fam == "gt":
        with np.errstate(invalid="ignore"):
            return np.nan_to_num(P.R[:, i] > int(p) / 100.0).astype(np.float32)
    if fam == "rare":
        with np.errstate(invalid="ignore"):
            return np.nan_to_num((P.zexp[:, i] * (1 if p[0] == "p" else -1)) > int(p[1:]) / 100.0).astype(np.float32)
    if fam == "reg":
        return P.Z[:, i] * P.reg[(spec.atoms[1], p)]
    j = ix[spec.atoms[1]]
    if fam == "prod":
        return P.Z[:, i] * P.Z[:, j]
    if fam == "xor":
        return _sign0(P.R[:, i]) * _sign0(P.R[:, j])
    if fam == "and":
        return _ind(P.R[:, i], p[0]) * _ind(P.R[:, j], p[1])
    return P.Z[:, i] * _ind(P.R[:, j], p[0])


# ================================================================================================================ the search
@dc.dataclass(frozen=True)
class Scored:
    spec: FormSpec
    t: float

    @property
    def name(self) -> str:
        return self.spec.name


def _top(t: np.ndarray, make, keep: int, t_min: float) -> list[Scored]:
    """The `keep` largest |t| entries of a flat array (NaN never qualifies); ties broken by the canonical NAME, never by position."""
    flat = np.abs(np.asarray(t, float).ravel())
    flat[~np.isfinite(flat)] = -1.0
    if flat.size == 0 or keep <= 0:
        return []
    m = min(flat.size, keep * 3 + 8)
    idx = np.argpartition(-flat, m - 1)[:m] if m < flat.size else np.arange(flat.size)
    out = []
    for k in idx:
        if flat[k] < t_min or flat[k] < 0:
            continue
        spec = make(int(k))
        if spec is not None:
            out.append(Scored(spec, float(np.asarray(t).ravel()[k])))
    out.sort(key=lambda s: (-round(abs(s.t), 9), s.name))
    return out[:keep]


def _rank(out: list[Scored], keep: int) -> list[Scored]:
    return sorted(out, key=lambda s: (-round(abs(s.t), 9), s.name))[:keep]


def welch_split(S: np.ndarray, a: np.ndarray, b: np.ndarray, min_n: int = 5) -> np.ndarray:
    """Per column: Welch t of mean(S[a]) - mean(S[b]) over dates (the per-date slope in one regime minus the other)."""
    if a.sum() < min_n or b.sum() < min_n:
        return np.full(S.shape[1], np.nan)
    x, y = S[a], S[b]
    se = np.sqrt(x.var(axis=0, ddof=1) / len(x) + y.var(axis=0, ddof=1) / len(y))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 1e-12, (x.mean(axis=0) - y.mean(axis=0)) / se, np.nan)


def _family_search(P: AtomPanel, fam: str, cfg: FormConfig) -> tuple[list[Scored], int]:
    """(ranked forms of one family, number of tests computed). Every test is counted, kept or not. Only atoms that are not strong
    singles are searched. Atom forms are scored beyond their own atom's straight line (`beyond_own`); pair forms are interaction
    contrasts that are orthogonal to additive main effects by construction (z_a z_b, s_a s_b, z_a s_b); a regime form is the
    difference of the atom's per-date slope between the regime's two sides. That is also why median-split AND quadrants are not
    searched: beyond the two additive sides a 2x2 split carries exactly one contrast, the xor (a quadrant form is the same hypothesis
    reached another way; it can still be spelled and materialised)."""
    free = [i for i, c in enumerate(P.names) if c not in set(P.main)]
    names = [P.names[i] for i in free]
    K = len(free)
    keep = cfg.keep_top
    if K == 0:
        return [], 0
    Z = P.Z[:, free]
    if fam == "lag":
        out, n = [], 0
        for k in cfg.lags:
            n += K
            t = fm_t(beyond_own(P.lagz[int(k)][:, free], Z, P), P)
            out += _top(t, lambda i, k=k: FormSpec("lag", str(int(k)), (names[i],)), keep, 0.0)
        return _rank(out, keep), n
    if fam == "gt":
        out, n = [], 0
        for q in cfg.cuts:
            with np.errstate(invalid="ignore"):
                U = np.nan_to_num(P.R[:, free] > q).astype(np.float32)
            n += K
            out += _top(fm_t(beyond_own(U, Z, P), P), lambda i, q=q: FormSpec("gt", _q_tag(q), (names[i],)), keep, 0.0)
        return _rank(out, keep), n
    if fam == "rare":
        out, n = [], 0
        for c in cfg.rare_z:
            for sg in ("p", "n"):
                with np.errstate(invalid="ignore"):
                    U = np.nan_to_num((P.zexp[:, free] * (1 if sg == "p" else -1)) > c).astype(np.float32)
                n += K
                out += _top(fm_t(beyond_own(U, Z, P), P), lambda i, c=c, sg=sg: FormSpec("rare", sg + _z_tag(c), (names[i],)), keep, 0.0)
        return _rank(out, keep), n
    if fam == "reg":
        out, n = [], 0
        S = np.add.reduceat(Z.astype(np.float64) * P.yc[:, None], P.starts[:-1], axis=0)
        first = P.starts[:-1]
        for m in sorted({m for m, _ in P.reg}):
            hi = (P.reg[(m, "hi")][first] > 0) & P.valid
            lo = (P.reg[(m, "lo")][first] > 0) & P.valid
            t = welch_split(S, hi, lo)
            n += K
            mh = np.abs(S[hi].mean(axis=0)) if hi.any() else np.zeros(K)
            ml = np.abs(S[lo].mean(axis=0)) if lo.any() else np.zeros(K)
            out += _top(t, lambda i, m=m: FormSpec("reg", "hi" if mh[i] >= ml[i] else "lo", (names[i], m)), keep, 0.0)
        return _rank(out, keep), n
    if K > cfg.max_pair_atoms or K < 2:
        return [], 0
    iu = np.triu(np.ones((K, K), bool), 1)
    off = ~np.eye(K, dtype=bool)
    if fam in ("prod", "xor"):
        M = Z if fam == "prod" else _sign0(P.R[:, free])
        t = np.where(iu, fm_pairs(M, M, P), np.nan)
        return _top(t, lambda k: canonical(fam, "", (names[k // K], names[k % K])), keep, 0.0), int(iu.sum())
    if fam == "cond":
        t = np.where(off, fm_pairs(Z, _sign0(P.R[:, free]), P), np.nan)
        lst = _top(t, lambda k: FormSpec("cond", "hi", (names[k // K], names[k % K])), keep, 0.0)
        return [_cond_side(P, s) for s in lst], int(off.sum())
    raise FormError(f"family {fam!r} is not searched")


def _cond_side(P: AtomPanel, s: Scored) -> Scored:
    """A slope contrast says the slope of a differs between b's sides, not on which side the effect lives: the side whose own
    FM t is larger is the proposed form (the contrast's t, which chose it, is kept)."""
    ix = {c: i for i, c in enumerate(P.names)}
    i, j = ix[s.spec.atoms[0]], ix[s.spec.atoms[1]]
    th = fm_t((P.Z[:, i] * _ind(P.R[:, j], "h"))[:, None], P)[0]
    tl = fm_t((P.Z[:, i] * _ind(P.R[:, j], "l"))[:, None], P)[0]
    side = "hi" if abs(np.nan_to_num(th)) >= abs(np.nan_to_num(tl)) else "lo"
    return Scored(FormSpec("cond", side, s.spec.atoms), s.t)


def _corr(u: np.ndarray, v: np.ndarray) -> float:
    u = u - u.mean()
    v = v - v.mean()
    d = math.sqrt(float(u @ u) * float(v @ v))
    return float(u @ v) / d if d > 0 else 0.0


def deduplicate(P: AtomPanel, ranked: Sequence[Scored], rho: float, cfg: FormConfig = FormConfig()) -> tuple[list[Scored], dict[str, str]]:
    """Greedy over |t| (ties by name): a form whose values correlate >= rho with one of its OWN atoms is that atom (the simpler
    hypothesis) and is dropped; one that correlates >= rho with an already kept form is the same pattern reached another way and
    becomes its alias. Structural duplicates were already removed by the canonical spelling."""
    kept: list[tuple[Scored, np.ndarray]] = []
    alias: dict[str, str] = {}
    ix = {c: i for i, c in enumerate(P.names)}
    for s in sorted(ranked, key=lambda s: (-round(abs(s.t), 9), s.name)):
        if s.name in alias or any(s.name == k.name for k, _ in kept):
            continue
        v = vector(P, s.spec, cfg).astype(np.float64)
        own = [a for a in s.spec.atoms if a in ix]
        hit = next((a for a in own if abs(_corr(v, P.Z[:, ix[a]].astype(np.float64))) >= rho), None)
        if hit is not None:
            alias[s.name] = hit
            continue
        dup = next((k for k, w in kept if abs(_corr(v, w)) >= rho), None)
        if dup is not None:
            alias[s.name] = dup.name
            continue
        kept.append((s, v))
    return [k for k, _ in kept], alias


def one_form_per_pair(ranked: Mapping[str, Sequence[Scored]]) -> tuple[dict[str, tuple], dict[str, str]]:
    """One pair of features is one hypothesis: a product interaction also shows up as a slope contrast in both directions and, more
    weakly, as an xor. Across prod / xor / cond only the form with the largest |t| represents the pair (it is also, measured on the
    benchmark's planted pairs, the true form); the others become its aliases and free their slots."""
    best: dict[frozenset, Scored] = {}
    for fam in ("prod", "xor", "cond"):
        for s in ranked.get(fam, ()):
            k = frozenset(s.spec.atoms)
            b = best.get(k)
            if b is None or (-round(abs(s.t), 9), s.name) < (-round(abs(b.t), 9), b.name):
                best[k] = s
    alias: dict[str, str] = {}
    out: dict[str, tuple] = {}
    for fam, lst in ranked.items():
        if fam not in ("prod", "xor", "cond"):
            out[fam] = tuple(lst)
            continue
        keep = []
        for s in lst:
            w = best[frozenset(s.spec.atoms)]
            if w.name == s.name:
                keep.append(s)
            else:
                alias[s.name] = w.name
        out[fam] = tuple(keep)
    return out, alias


@dc.dataclass(frozen=True)
class Proposal:
    now: str
    proposed: tuple                  # Scored, deduplicated, at the operating point (k_per_family, t_pre)
    ranked: Mapping[str, tuple]      # family -> Scored (deduplicated, up to keep_top) - for trade-off curves
    scanned: Mapping[str, int]       # family -> tests computed (singles included)
    aliases: Mapping[str, str]
    atoms: tuple
    market: tuple
    dup_atoms: Mapping[str, str]
    skipped: tuple                   # (family, reason)
    select_dates: int
    seconds: float
    config: str
    main: tuple = ()                 # strong singles the forms were screened beyond

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.proposed]

    @property
    def n_scanned(self) -> int:
        return int(sum(self.scanned.values()))

    def at(self, k: int, t_pre: float, families: Sequence[str] | None = None) -> list[Scored]:
        """The proposal at another operating point (a prefix of each family's ranked list: no re-search, no re-count)."""
        out = []
        for fam, lst in self.ranked.items():
            if families is not None and fam not in families:
                continue
            out += [s for s in lst if abs(s.t) >= t_pre][:k]
        return sorted(out, key=lambda s: (-round(abs(s.t), 9), s.name))

    def summary(self) -> dict:
        return {"now": self.now, "n_scanned": self.n_scanned, "scanned": dict(self.scanned), "n_proposed": len(self.proposed),
                "by_family": pd.Series([s.spec.family for s in self.proposed], dtype=object).value_counts().to_dict(),
                "aliases": len(self.aliases), "atoms": len(self.atoms), "market_atoms": list(self.market),
                "dup_atoms": len(self.dup_atoms), "main_effects": len(self.main), "skipped": list(self.skipped), "select_dates": self.select_dates, "seconds": self.seconds}


def propose(F: pd.DataFrame, feats: Sequence[str], now, cfg: FormConfig = FormConfig()) -> Proposal:
    """THE GENERATOR. Blind by construction: it sees the matured frame and the feature names, never an answer key. Every test it runs
    is counted in `scanned` (singles too: they are part of the same search). Raises FirewallBreach on rows at/after `now`."""
    errs = cfg.validate()
    if errs:
        raise FormError("invalid FormConfig: " + "; ".join(errs))
    t0 = time.monotonic()
    P = build_panel(F, feats, now, cfg)
    scanned: dict[str, int] = {"single": len(P.names) + len(P.market)}
    ranked: dict[str, tuple] = {}
    skipped: list[tuple[str, str]] = []
    n_valid = int(P.valid.sum())
    if n_valid < cfg.min_dates:
        skipped.append(("all", f"{n_valid} dates with >= {cfg.min_names} names (< {cfg.min_dates})"))
    else:
        for fam in cfg.families:
            if fam == "reg" and not P.market:
                skipped.append(("reg", "no date-level atom to condition on"))
                continue
            if fam in PAIR_FAMILIES and len(P.names) > cfg.max_pair_atoms:
                skipped.append((fam, f"{len(P.names)} atoms > max_pair_atoms={cfg.max_pair_atoms}"))
                continue
            lst, n = _family_search(P, fam, cfg)
            scanned[fam] = n
            ranked[fam] = tuple(lst)
    ranked, alias = one_form_per_pair(ranked)
    flat = [s for lst in ranked.values() for s in lst]
    kept, alias2 = deduplicate(P, flat, cfg.dedup_rho, cfg)
    alias.update(alias2)
    keep = {s.name for s in kept}
    ranked = {f: tuple(s for s in lst if s.name in keep) for f, lst in ranked.items()}
    prop = Proposal(str(as_date(now)), (), ranked, scanned, alias, tuple(P.names), tuple(P.market), dict(P.dup_atoms), tuple(skipped),
                    P.select_dates, round(time.monotonic() - t0, 2), stable_hash(dc.asdict(cfg), 12), tuple(P.main))
    return dc.replace(prop, proposed=tuple(prop.at(cfg.k_per_family, cfg.t_pre)))


# ================================================================================================================ the screen entry
def screen_table(F: pd.DataFrame, feats: Sequence[str], now, lab: Any, cfg: FormConfig = FormConfig(), *,
                 proposal: Proposal | None = None, singles: bool = True) -> tuple[pd.DataFrame, Proposal]:
    """The entry the benchmark / loop calls in place of `volatility_lab.oriented_scan(F, feats, now, lab)`: the same walk-forward table
    (feature, n_dates, auc, lo, hi, p, sign, q) over the singles AND the proposed forms, plus `form`, `complexity` and the search size
    in `attrs['n_scanned']` (the count the gate's multiplicity must use). Proposed forms are registered in vol_hypotheses.DERIVED so
    evidence.assemble can derive them; call it inside `session()` so they leave again."""
    from engine.research import volatility_lab as VL
    prop = proposal if proposal is not None else propose(F, feats, now, cfg)
    names = register([s.spec for s in prop.proposed], cfg)
    scan = (list(feats) if singles else []) + [n for n in names if n not in set(feats)]
    tab = VL.oriented_scan(F, scan, now, lab) if scan else pd.DataFrame(columns=["feature", "n_dates", "auc", "lo", "hi", "p", "sign", "q"])
    tab["form"] = [family_of(f) for f in tab["feature"]]
    tab["complexity"] = [complexity(f) for f in tab["feature"]]
    tab.attrs["n_scanned"] = prop.n_scanned
    tab.attrs["scanned"] = dict(prop.scanned)
    return tab, prop


def raise_rule(tab: pd.DataFrame, t_min: float, cap: int | None, already: Iterable[str] = ()) -> list[str]:
    """The benchmark's / loop's raise rule on a screen table: walk the rows by AUC, raise a feature when its t = (auc - .5) / se clears
    t_min, up to `cap` per call (None = no cap), skipping features raised before."""
    seen = set(already)
    out: list[str] = []
    if len(tab) == 0:
        return out
    s = tab.sort_values(["auc", "feature"], ascending=[False, True], na_position="last")
    auc, lo, hi = (s[c].to_numpy(float) for c in ("auc", "lo", "hi"))
    with np.errstate(invalid="ignore"):
        t = (auc - 0.5) / np.maximum((hi - lo) / (2 * 1.645), 1e-6)
    ok = np.isfinite(auc) & np.isfinite(lo) & np.isfinite(hi) & (t >= t_min)
    for f in s["feature"].to_numpy()[ok]:
        if cap is not None and len(out) >= cap:
            break
        if f not in seen:
            out.append(str(f))
            seen.add(str(f))
    return out


def screen_t(tab: pd.DataFrame) -> pd.Series:
    se = ((tab["hi"] - tab["lo"]) / (2 * 1.645)).clip(lower=1e-6)
    return ((tab["auc"] - 0.5) / se).set_axis(tab["feature"].to_numpy())


# ================================================================================================================ multiplicity
def multiplicity_cost(scanned: Mapping[str, int], alpha: float = 0.05, weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    """The |t| bar a single candidate must clear under Bonferroni: (a) singles only (the pre-F27 search), (b) pooled over everything
    scanned, (c) a family-split alpha (alpha * w_f spent on family f's n_f tests; default: half on singles, the rest equally). The
    step from (a) to (b) is the price of expressing the true forms; (c) is the honest way to pay less of it on the singles."""
    from engine.research.interactions import detectable_t
    fams = [f for f, n in scanned.items() if n > 0]
    total = int(sum(scanned[f] for f in fams))
    if not fams:
        return pd.DataFrame(columns=["family", "n_tests", "t_singles_only", "t_pooled", "alpha_share", "t_family_split"])
    if weights is None:
        rest = [f for f in fams if f != "single"]
        weights = {f: (0.5 if f == "single" else 0.5 / len(rest)) for f in fams} if "single" in fams and rest else {f: 1 / len(fams) for f in fams}
    wsum = sum(weights.get(f, 0.0) for f in fams)
    rows = []
    for f in fams:
        w = weights.get(f, 0.0) / wsum if wsum > 0 else 0.0
        rows.append({"family": f, "n_tests": int(scanned[f]), "t_singles_only": detectable_t(scanned.get("single", 1), alpha),
                     "t_pooled": detectable_t(total, alpha), "alpha_share": w,
                     "t_family_split": detectable_t(int(math.ceil(scanned[f] / w)), alpha) if w > 0 else float("inf")})
    return pd.DataFrame(rows)


# ================================================================================================================ evaluation (reads an answer key AFTER generation)
TRUE_FORMS = {"linear": ("single",), "changing": ("single",), "lifecycle": ("single",), "regime": ("reg",), "rare": ("rare", "gt"),
              "threshold": ("gt", "rare"), "conditional": ("cond", "prod"), "interactive": ("prod",), "xor": ("xor",), "delayed": ("lag",)}


def _pattern_columns(p: Mapping) -> list[str]:
    return list(p.get("columns_marginal") or p.get("columns") or [])


def context_of(p: Mapping, patterns: Sequence[Mapping]) -> str | None:
    """The context column a conditional pattern is conditioned on (the benchmark plants it as a NOISE 'context' child)."""
    return next((q["columns"][0] for q in patterns if q.get("kind") == "context" and q.get("parent") == p["pid"] and q.get("columns")), None)


def match(p: Mapping, patterns: Sequence[Mapping], name: str) -> str | None:
    """'true' when candidate `name` is the pattern in its TRUE form, 'any' when it involves the pattern's column(s) in another form,
    else None. Used only to SCORE (after generation)."""
    cols = _pattern_columns(p)
    if not cols:
        return None
    spec = parse(name)
    kind = str(p.get("kind") or "")
    if spec is None:
        if name not in cols:
            return None
        return "true" if "single" in TRUE_FORMS.get(kind, ()) else "any"
    atoms = set(spec.atoms)
    if kind in ("interactive", "xor"):
        if not set(cols) <= atoms:
            return None
        return "true" if spec.family in TRUE_FORMS[kind] else "any"
    if cols[0] != spec.atoms[0] and not (spec.family in PAIR_FAMILIES and cols[0] in atoms):
        return None
    fam = spec.family
    if kind == "delayed":
        return "true" if fam == "lag" and int(spec.param) == int(p.get("lag", -1)) else "any"
    if kind == "conditional":
        ctx = context_of(p, patterns)
        if ctx is not None and atoms == {cols[0], ctx} and (fam == "prod" or (fam == "cond" and spec.atoms[0] == cols[0])):
            return "true"
        return "any"
    if kind == "rare":                                    # planted as an upper-tail event of its column
        return "true" if (fam == "rare" and spec.param[0] == "p") or (fam == "gt" and int(spec.param) >= 75) else "any"
    if kind == "threshold":
        return "true" if fam in ("gt", "rare") else "any"
    return "true" if fam in TRUE_FORMS.get(kind, ()) else "any"


def score_patterns(key: Mapping, stages: Mapping[str, Iterable[str]]) -> list[dict]:
    """Per REAL pattern: for each stage (e.g. 'proposed', 'raised_before', 'raised_after') whether a candidate in that stage is the
    pattern in its true form / in any form. Plus, per stage, the candidates that match no real pattern (false candidates)."""
    pats = key["patterns"]
    real = [p for p in pats if p.get("label") == "REAL"]
    rows = []
    sets = {k: sorted(set(v)) for k, v in stages.items()}
    for p in real:
        r = {"world_id": key.get("world_id"), "tier": key.get("tier"), "pid": p["pid"], "kind": p["kind"], "band": p.get("band"),
             "status": p.get("status"), "oracle_power": p.get("oracle_power"), "marginal_power": p.get("marginal_power")}
        for st, names in sets.items():
            hits = [match(p, pats, n) for n in names]
            r[f"{st}_true"] = "true" in hits
            r[f"{st}_any"] = any(h is not None for h in hits)
        rows.append(r)
    return rows


def false_candidates(key: Mapping, names: Iterable[str]) -> dict:
    """Candidates (singles and forms) that match no REAL pattern in any form, split by family."""
    pats = key["patterns"]
    real = [p for p in pats if p.get("label") == "REAL"]
    out: dict[str, int] = {}
    total = 0
    for n in set(names):
        if not any(match(p, pats, n) for p in real):
            f = family_of(n)
            out[f] = out.get(f, 0) + 1
            total += 1
    return {"total": total, "by_family": out}


def recall_table(rows: Sequence[Mapping], stages: Sequence[str], by: Sequence[str] = ("kind", "band"),
                 detectable_only: bool = True) -> pd.DataFrame:
    """Recall per kind x band (by default only patterns whose TRUE form an oracle could detect: status != UNDETECTABLE)."""
    R = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if R.empty:
        return pd.DataFrame()
    if detectable_only and "status" in R:
        R = R[R["status"] != "UNDETECTABLE_IN_PRINCIPLE"]
    agg = {"n": ("pid", "size")}
    for st in stages:
        for m in ("true", "any"):
            c = f"{st}_{m}"
            if c in R:
                agg[c] = (c, "mean")
    return R.groupby(list(by)).agg(**agg).reset_index()
