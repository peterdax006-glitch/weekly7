"""F27 candidate representation and screen recall (C75 3A/3B/3I, C70): every true form planted ALONE is proposed in that form; a NULL
world's candidate count and false-candidate rate are reported and bounded; the search counts every test it runs; equivalent hypotheses
are one candidate (canonical spelling, near-duplicate atoms, a form that IS its atom, one form per pair); nothing reads past `now`;
the generator is blind (never sees a key) and name-order free. Synthetic panels only, < 90 s."""
from __future__ import annotations

import contextlib
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.learning.core import FirewallBreach                                # noqa: E402
from engine.research import candidate_forms as CF                              # noqa: E402
from engine.research import vol_hypotheses as VH                               # noqa: E402

warnings.filterwarnings("ignore")
T, N, NOISE = 120, 40, 14


@contextlib.contextmanager
def registered(cols):
    """Test columns enter the derived-feature registry the screen reads (as the benchmark's planted columns do), then leave."""
    added = []
    try:
        for c in cols:
            if c not in VH.DERIVED:
                VH.DERIVED[c] = ((c,), (lambda F, c=c: F[c]))
                added.append(c)
        yield added
    finally:
        for c in added:
            VH.DERIVED.pop(c, None)


def _ar(rng, phi, shape):
    x = np.zeros(shape)
    x[0] = rng.normal(size=shape[1:])
    for t in range(1, shape[0]):
        x[t] = phi * x[t - 1] + math.sqrt(1 - phi * phi) * rng.normal(size=shape[1:])
    return x


def world(kind: str | None, seed: int = 0, eff: float | None = None, n_noise: int = NOISE, names=None):
    """A panel with n_noise null atoms (x..), a date-level market atom 'mk', and at most one planted pattern in its true form on x00
    (and x01 for pairs / the context). Returns (frame, atom names, now)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2003-01-03", periods=T, freq="W-FRI")
    idx = pd.MultiIndex.from_product([dates, [f"T{i:02d}" for i in range(N)]], names=["date", "ticker"])
    X = {f"x{i:02d}": rng.normal(size=(T, N)) for i in range(n_noise)}
    mk = _ar(rng, 0.97, (T, 1))[:, 0]
    X["x01"] = X["x01"] if kind != "conditional" else np.repeat((rng.random(N) < 0.5)[None, :], T, 0).astype(float)
    if kind == "delayed":
        X["x00"] = _ar(rng, 0.5, (T, N))
    x0, x1 = X["x00"], X["x01"]
    if kind is None:
        form = np.zeros((T, N))
    elif kind == "delayed":
        form = np.vstack([np.zeros((2, N)), x0[:-2]])
    elif kind == "threshold":
        form = (x0 > 1.0).astype(float) - (x0 > 1.0).mean()
    elif kind == "rare":
        form = (x0 > 1.8).astype(float)
    elif kind == "interactive":
        form = x0 * x1
    elif kind == "xor":
        form = np.sign(x0) * np.sign(x1)
    elif kind == "conditional":
        form = x0 * x1
    elif kind == "regime":
        pct = VH.expanding_pct(pd.Series(mk), min_periods=8).fillna(0.5).to_numpy()
        form = x0 * (pct > 0.5)[:, None]
    else:
        raise ValueError(kind)
    # effects keep the single's own t below t_main where the form has a marginal (threshold / conditional / regime): a stronger single is
    # raised by itself and its forms are deliberately not searched (see test_strong_singles_are_raised_alone_not_searched_in_forms)
    e = {"delayed": 0.8, "threshold": 0.45, "rare": 1.2, "interactive": 0.8, "xor": 0.8, "conditional": 0.45, "regime": 0.35}.get(kind, 0.0)
    logit = -1.0 + (eff if eff is not None else e) * form
    y = (rng.random((T, N)) < 1 / (1 + np.exp(-logit))).astype(float)
    F = pd.DataFrame({k: v.ravel() for k, v in X.items()}, index=idx)
    F["mk"] = np.repeat(mk, N)
    F["touch"] = y.ravel()
    F["absmove"] = np.where(y.ravel() > 0, 0.12, 0.03)
    F["end"] = pd.to_datetime(F.index.get_level_values(0)) + pd.Timedelta(days=8)
    if names is not None:
        F = F.rename(columns=names)
    atoms = [c for c in F.columns if c not in ("touch", "end", "absmove")]
    return F, atoms, dates[-1] + pd.Timedelta(days=14)


KEY = {"delayed": {"kind": "delayed", "lag": 2}, "threshold": {"kind": "threshold"}, "rare": {"kind": "rare"},
       "interactive": {"kind": "interactive"}, "xor": {"kind": "xor"}, "conditional": {"kind": "conditional"}, "regime": {"kind": "regime"}}
SEED = {"regime": 4}


def toy_key(kind: str) -> list[dict]:
    """The planted pattern as a benchmark-style key entry (used only to SCORE the proposal, after generation)."""
    p = {"pid": "P0", "label": "REAL", "columns": ["x00"], **KEY[kind]}
    if kind in ("interactive", "xor"):
        p.update(columns=[], columns_marginal=["x00", "x01"])
    pats = [p]
    if kind == "conditional":
        pats.append({"pid": "P1", "label": "NOISE", "kind": "context", "columns": ["x01"], "parent": "P0"})
    return pats


# ============================================================================================================ each true form, planted alone
@pytest.mark.parametrize("kind", list(KEY))
def test_each_true_form_planted_alone_is_proposed(kind):
    F, atoms, now = world(kind, seed=SEED.get(kind, 3))
    with registered(atoms):
        prop = CF.propose(F, atoms, now)
    pats = toy_key(kind)
    true = [s for s in prop.proposed if CF.match(pats[0], pats, s.name) == "true"]
    assert true, (kind, [(s.name, round(s.t, 1)) for s in prop.proposed[:10]], prop.main)
    assert abs(true[0].t) >= 3.0
    assert "x00" not in prop.main                                    # its single alone would not have been a strong candidate
    top = max(prop.proposed, key=lambda s: abs(s.t))
    assert "x00" in top.spec.atoms                                   # the planted form is the strongest thing found


def test_the_single_feature_screen_cannot_see_the_pair_forms():
    """The defect F19 reported: an xor pair's members have no marginal association (so a single-feature screen never raises them)."""
    F, atoms, now = world("xor", seed=3)
    with registered(atoms):
        P = CF.build_panel(F, atoms, now)
    t = dict(zip(P.names, P.single_t))
    assert abs(t["x00"]) < 3 and abs(t["x01"]) < 3


# ============================================================================================================ null world, multiplicity
def test_null_world_false_candidates_are_reported_and_bounded():
    counts = []
    for seed in (1, 2, 3):
        F, atoms, now = world(None, seed=seed)
        with registered(atoms):
            prop = CF.propose(F, atoms, now)
        counts.append((len(prop.at(20, 2.0)), len(prop.at(8, 3.5)), len(prop.at(8, 4.5))))
    # reported: proposals at loose / default-ish / strict points; every one of them is a false candidate in a null world
    loose, mid, strict = np.array(counts).T
    assert strict.max() == 0                                         # Bonferroni over ~20k tests needs |t| ~ 4.5
    assert mid.mean() <= 3
    assert loose.mean() > mid.mean()                                 # the loose point floods: the curve is real


def test_every_test_is_counted():
    F, atoms, now = world(None, seed=5)
    cfg = CF.FormConfig()
    with registered(atoms):
        prop = CF.propose(F, atoms, now, cfg)
    K = len(prop.atoms) - len(prop.main)
    m = len(prop.market)
    assert prop.scanned["single"] == len(prop.atoms) + m
    assert prop.scanned["lag"] == K * len(cfg.lags)
    assert prop.scanned["gt"] == K * len(cfg.cuts)
    assert prop.scanned["rare"] == K * 2 * len(cfg.rare_z)
    assert prop.scanned["reg"] == K * m
    assert prop.scanned["prod"] == prop.scanned["xor"] == K * (K - 1) // 2
    assert prop.scanned["cond"] == K * (K - 1)
    assert prop.n_scanned == sum(prop.scanned.values())
    cost = CF.multiplicity_cost(prop.scanned).set_index("family")
    assert cost.loc["single", "t_pooled"] > cost.loc["single", "t_singles_only"]           # the price of the true forms, stated
    assert cost.loc["single", "t_family_split"] < cost.loc["single", "t_pooled"]           # a family split pays less of it on singles
    assert abs(cost["alpha_share"].sum() - 1.0) < 1e-9


# ============================================================================================================ normalise / deduplicate
def test_canonical_spelling_and_parse():
    a = CF.canonical("prod", "", ("x02", "x01"))
    assert a == CF.canonical("prod", "", ("x01", "x02")) and a.name == "ix__cf_prod__x01__x02"
    q = CF.canonical("and", "hl", ("x09", "x01"))
    assert q.atoms == ("x01", "x09") and q.param == "lh"
    assert CF.canonical("lt", "25", ("x01",)) == CF.FormSpec("gt", "25", ("x01",))
    for s in (a, q, CF.FormSpec("lag", "3", ("x01",)), CF.FormSpec("rare", "p164", ("x01",)), CF.FormSpec("reg", "lo", ("x01", "mk")),
              CF.FormSpec("cond", "hi", ("x02", "x01"))):
        assert CF.parse(s.name) == s
    assert CF.parse("x01") is None and CF.family_of("x01") == "single"
    with pytest.raises(CF.FormError):
        CF.parse("ix__cf_prod__x02__x01")                            # not canonical: the same pair spelled the other way
    with pytest.raises(CF.FormError):
        CF.FormSpec("prod", "", ("x01", "x01"))
    with pytest.raises(CF.FormError):
        CF.FormSpec("lag", "2", ("a__b",))
    with pytest.raises(CF.FormError):
        CF.parse("ix__cf_wobble__x01")
    assert CF.history_dependent("ix__cf_lag2__x01") and not CF.history_dependent("ix__cf_prod__x01__x02")
    assert CF.complexity("ix__cf_gt25__x01") == 1 and CF.complexity("ix__cf_xor__x01__x02") == 2


def test_duplicate_atoms_and_same_pattern_two_ways_are_one_candidate():
    F, atoms, now = world("interactive", seed=4)
    rng = np.random.default_rng(0)
    F["x00dup"] = F["x00"] + 1e-4 * rng.normal(size=len(F))        # the same feature twice
    atoms = atoms + ["x00dup"]
    with registered(atoms):
        prop = CF.propose(F, atoms, now)
    assert prop.dup_atoms.get("x00dup") == "x00" or prop.dup_atoms.get("x00") == "x00dup"
    pairs = [frozenset(s.spec.atoms) for s in prop.proposed if s.spec.family in ("prod", "xor", "cond")]
    assert len(pairs) == len(set(pairs))                               # one form per pair of features
    assert any(prop.aliases.get(n) == "ix__cf_prod__x00__x01" for n in prop.aliases)   # the pair's other forms point at the winner


def test_a_form_that_is_its_own_atom_is_dropped():
    F, atoms, now = world(None, seed=6)
    F["bin"] = np.repeat((np.random.default_rng(1).random(N) < 0.3)[None, :], T, 0).astype(float).ravel()
    atoms = atoms + ["bin"]
    with registered(atoms):
        P = CF.build_panel(F, atoms, now)
        s = CF.Scored(CF.FormSpec("gt", "50", ("bin",)), 9.0)            # 1[rank > .5] of a 0/1 column IS the column
        o = CF.Scored(CF.FormSpec("lag", "1", ("x03",)), 5.0)
        kept, alias = CF.deduplicate(P, [s, o], 0.9)
    assert alias == {s.name: "bin"} and [k.name for k in kept] == [o.name]


# ============================================================================================================ point in time, blindness
def test_rows_at_or_after_now_and_immature_outcomes_fail_closed():
    F, atoms, now = world(None, seed=7)
    with registered(atoms):
        last = pd.Timestamp(F.index.get_level_values(0).max())
        with pytest.raises(FirewallBreach):
            CF.propose(F, atoms, last)
        with pytest.raises(FirewallBreach):
            CF.propose(F, atoms, last + pd.Timedelta(days=3))       # dated before now, outcome not yet matured


@pytest.mark.parametrize("name", ["ix__cf_lag2__x00", "ix__cf_zp164__x00", "ix__cf_reghi__x00__mk", "ix__cf_gt75__x00",
                                  "ix__cf_condlo__x00__x01", "ix__cf_xor__x00__x01"])
def test_a_form_at_t_never_changes_when_the_future_is_scrambled(name):
    F, atoms, now = world(None, seed=8)
    spec = CF.parse(name)
    cut = F.index.get_level_values(0).unique()[70]
    G = F.copy()
    fut = G.index.get_level_values(0) > cut
    rng = np.random.default_rng(3)
    for c in atoms:
        G.loc[fut, c] = rng.permutation(G.loc[fut, c].to_numpy()) * 5 + 3
    with registered(atoms):
        a = CF.materialise(spec, F)
        b = CF.materialise(spec, G)
    past = ~fut
    assert np.allclose(a[past].to_numpy(), b[past].to_numpy(), equal_nan=True)


@pytest.mark.parametrize("name", ["ix__cf_lag1__x02", "ix__cf_gt25__x02", "ix__cf_zn128__x02", "ix__cf_prod__x02__x03",
                                  "ix__cf_xor__x02__x03", "ix__cf_andhl__x02__x03", "ix__cf_condhi__x02__x03", "ix__cf_reglo__x02__mk"])
def test_the_prescreen_scores_the_same_values_the_screen_derives(name):
    F, atoms, now = world(None, seed=9)
    cfg = CF.FormConfig(t_main=99.0)
    with registered(atoms):
        P = CF.build_panel(F, atoms, now, cfg)
        v = CF.vector(P, CF.parse(name), cfg)
        m = np.nan_to_num(CF.materialise(CF.parse(name), F.sort_index(), cfg).to_numpy(), nan=0.0)
    assert np.allclose(v, m, atol=1e-4)


def test_generator_is_name_order_free_and_blind():
    """Reordering the columns changes nothing; an order-preserving rename maps the proposal exactly; `propose` has no key argument."""
    F, atoms, now = world("xor", seed=10)
    with registered(atoms):
        a = CF.propose(F, atoms, now)
        b = CF.propose(F[list(reversed(F.columns))], list(reversed(atoms)), now)
    assert a.names == b.names and a.scanned == b.scanned
    ren = {c: "z" + c for c in atoms if c.startswith("x")}
    G, atoms2, _ = world("xor", seed=10, names=ren)
    with registered(atoms2):
        c = CF.propose(G, atoms2, now)
    back = {v: k for k, v in ren.items()}
    mapped = [CF.canonical(s.spec.family, s.spec.param, tuple(back.get(x, x) for x in s.spec.atoms)).name for s in c.proposed]
    assert sorted(mapped) == sorted(a.names)
    import inspect
    assert not {"key", "answers", "truth"} & set(inspect.signature(CF.propose).parameters)


# ============================================================================================================ searching beyond the known
def test_forms_are_judged_beyond_their_own_atom():
    """A purely LINEAR effect must not come back as a 'threshold' or a 'lag' of itself (the planted defect: without partialling out
    the atom's own line, every cut of a linear feature looks significant)."""
    rng = np.random.default_rng(11)
    F, atoms, now = world(None, seed=11)
    x = F["x05"].to_numpy()
    F["touch"] = (rng.random(len(F)) < 1 / (1 + np.exp(-(-1.0 + 0.35 * x)))).astype(float)
    cfg = CF.FormConfig(t_main=99.0)
    with registered(atoms):
        P = CF.build_panel(F, atoms, now, cfg)
        i = P.names.index("x05")
        U = np.nan_to_num(P.R[:, [i]] > 0.5).astype(np.float32)
        raw = CF.fm_t(U, P)[0]
        beyond = CF.fm_t(CF.beyond_own(U, P.Z[:, [i]], P), P)[0]
    assert abs(P.single_t[i]) > 4 and abs(raw) > 4                  # the defect is there to be caught ...
    assert abs(beyond) < 2.5                                        # ... and the beyond-own statistic does not fall for it


def test_strong_singles_are_raised_alone_not_searched_in_forms():
    F, atoms, now = world(None, seed=12)
    rng = np.random.default_rng(2)
    F["leak"] = F["touch"] * 2.0 + rng.normal(size=len(F))             # 'too good': it contains the outcome
    atoms = atoms + ["leak"]
    with registered(atoms):
        prop = CF.propose(F, atoms, now)
    assert "leak" in prop.main
    assert not any("leak" in s.spec.atoms for lst in prop.ranked.values() for s in lst)


# ============================================================================================================ registry, screen entry, raise rule
def test_screen_table_registers_inside_a_session_and_cleans_up():
    from engine.research import volatility_lab as VL
    F, atoms, now = world("xor", seed=3)
    before = set(VH.DERIVED)
    lab = VL.LabConfig(min_train_dates=40, test_step_dates=15, n_boot=50)
    with registered(atoms), CF.session():
        prop = CF.propose(F, atoms, now, CF.FormConfig(k_per_family=2))
        tab, p2 = CF.screen_table(F, atoms[:3], now, lab, proposal=prop)
        assert p2 is prop and tab.attrs["n_scanned"] == prop.n_scanned
        assert set(prop.names) <= set(tab["feature"]) and set(prop.names) <= set(VH.DERIVED)
        assert (tab.set_index("feature").loc["ix__cf_xor__x00__x01", "auc"]) > 0.55
        assert set(tab.loc[tab["feature"].isin(prop.names), "form"]) - set(CF.SEARCHED) == set()
        assert CF.register([s.spec for s in prop.proposed]) == prop.names          # idempotent
    assert set(VH.DERIVED) == before


def test_raise_rule_walks_by_auc_with_cap_and_memory():
    tab = pd.DataFrame({"feature": ["a", "b", "c", "d"], "auc": [0.60, 0.58, 0.51, np.nan], "lo": [0.57, 0.50, 0.505, np.nan],
                        "hi": [0.63, 0.66, 0.515, np.nan]})
    assert CF.raise_rule(tab, 2.0, None) == ["a", "c"]                 # b's interval is too wide (t < 2), d unscored
    assert CF.raise_rule(tab, 2.0, 1) == ["a"]
    assert CF.raise_rule(tab, 2.0, 1, already={"a"}) == ["c"]
    assert CF.raise_rule(tab.iloc[0:0], 2.0, 5) == []


# ============================================================================================================ scoring helpers (after generation)
def test_match_scores_true_form_any_form_and_misses():
    pats = [{"pid": "P0", "label": "REAL", "kind": "xor", "columns": [], "columns_marginal": ["a", "b"]},
            {"pid": "P1", "label": "REAL", "kind": "delayed", "columns": ["c"], "lag": 2},
            {"pid": "P2", "label": "REAL", "kind": "conditional", "columns": ["d"]},
            {"pid": "P3", "label": "NOISE", "kind": "context", "columns": ["e"], "parent": "P2"},
            {"pid": "P4", "label": "REAL", "kind": "linear", "columns": ["f"]}]
    m = lambda i, n: CF.match(pats[i], pats, n)                      # noqa: E731
    assert m(0, "ix__cf_xor__a__b") == "true" and m(0, "ix__cf_prod__a__b") == "any" and m(0, "a") == "any"
    assert m(0, "ix__cf_xor__a__c") is None
    assert m(1, "ix__cf_lag2__c") == "true" and m(1, "ix__cf_lag1__c") == "any" and m(1, "c") == "any"
    assert m(2, "ix__cf_condhi__d__e") == "true" and m(2, "ix__cf_prod__d__e") == "true" and m(2, "ix__cf_condhi__e__d") == "any"
    assert m(4, "f") == "true" and m(4, "ix__cf_gt50__f") == "any" and m(4, "g") is None
    key = {"world_id": "W", "patterns": pats}
    rows = CF.score_patterns(key, {"after": ["ix__cf_xor__a__b", "c", "zz"]})
    got = {r["pid"]: (r["after_true"], r["after_any"]) for r in rows}
    assert got == {"P0": (True, True), "P1": (False, True), "P2": (False, False), "P4": (False, False)}
    assert CF.false_candidates(key, ["ix__cf_xor__a__b", "zz", "ix__cf_prod__g__h"])["total"] == 2
    t = CF.recall_table(rows, ["after"], by=("kind",))
    assert set(t["kind"]) == {"xor", "delayed", "conditional", "linear"}


# ============================================================================================================ empty / degenerate
def test_empty_and_degenerate_inputs():
    F, atoms, now = world(None, seed=13)
    with registered(atoms):
        e = CF.propose(F.iloc[0:0], atoms, now)
        assert e.names == [] and e.n_scanned == 0 and e.skipped
        n = CF.propose(F, [], now)
        assert n.names == [] and n.atoms == ()
        short = CF.propose(F[F.index.get_level_values(0) < F.index.get_level_values(0).unique()[10]], atoms, now)
        assert short.names == [] and short.skipped[0][0] == "all"
        few = F[F.index.get_level_values(1).isin(["T00", "T01", "T02"])]
        assert CF.propose(few, atoms, now).names == []                 # 3 names a date: no date qualifies
    assert CF.multiplicity_cost({}).empty and CF.recall_table([], ["x"]).empty
    with pytest.raises(CF.FormError):
        CF.propose(F, atoms, now, CF.FormConfig(select_frac=0.0))
    assert CF.FormConfig(families=("and",)).validate()
