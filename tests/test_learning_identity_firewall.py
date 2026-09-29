"""Tests for engine/learning/identity_firewall.py (contract C62 section 29; section 62 tests 8 and 10).

Planted controls: a memoriser (keyed on exact date + ticker), a ticker-mean lookup, a learner with an absolute-date rule and a
learner that is nondeterministic must be caught; a legitimate ridge learner and the same learner under every identity attack must
NOT be flagged. Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from engine.learning import firewalls as F
from engine.learning import identity_firewall as I
from engine.learning.core import FirewallBreach


def make_panel(n_dates=48, n_names=20, seed=0, beta=0.03, first="2019-01-04"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(first, periods=n_dates * 5)[::5][:n_dates]
    tickers = [f"S{i:03d}" for i in range(n_names)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    X = pd.DataFrame({"f1": rng.normal(size=len(idx)), "f2": rng.normal(size=len(idx)), "f3": rng.normal(size=len(idx))}, index=idx)
    sec = np.tile(np.arange(n_names) % 4, n_dates)
    for k in range(4):
        X[f"sec_{k}"] = (sec == k).astype(float)
    y = pd.Series(beta * X["f1"].to_numpy() + rng.normal(0, 0.03, len(idx)), index=idx, name="y")
    return X, y


@pytest.fixture(scope="module")
def data():
    X, y = make_panel()
    return X, y


def rerun_split(X, y):
    """A rerun: the evaluated rows ARE the training rows (the memoriser's habitat)."""
    return X, y, X, y


# ---------------------------------------------------------------- the seven transforms
@pytest.mark.parametrize("name", list(I.ATTACKS))
def test_every_transform_is_identity_preserving_and_actually_changes_something(data, name):
    X, y = data
    kw = {"block": 8, "frac": 1.0} if name == "episode_substitution" else {}        # 6 equal blocks, so pairs can differ by seed
    t = I.ATTACKS[name](X, y, 3, **kw)
    assert I.verify_transform(X, y, t) == [], name
    assert t.changed_frac > 0.05 and len(t.X) == len(X) and not t.X.index.has_duplicates
    t2 = I.ATTACKS[name](X, y, 3, **kw)
    pd.testing.assert_frame_equal(t.X, t2.X)                      # deterministic in its seed
    other = I.ATTACKS[name](X, y, 4, **kw)
    assert not t.X.index.equals(other.X.index) or not np.array_equal(t.X.to_numpy(), other.X.to_numpy())


def test_ticker_permutation_relabels_by_a_bijection_and_resorts_rows(data):
    X, y = data
    t = I.ticker_permutation(X, y, 1)
    assert sorted(t.mapping) == sorted(t.mapping.values()) and set(t.X.index.get_level_values(1)) == set(X.index.get_level_values(1))
    assert t.X.index.is_monotonic_increasing                       # rows follow the NEW names, so name-order tie-breaks carry no identity
    pos = t.inverse_rows.astype(int)
    np.testing.assert_allclose(t.X.to_numpy(), X.to_numpy()[pos])  # each row is the original row, relabelled
    kept = I.ticker_permutation(X, y, 1, order_preserving=True)
    assert list(kept.X.index) != list(X.index) and kept.X.iloc[0].tolist() == X.iloc[0].tolist()


def test_date_permutation_keeps_order_but_destroys_calendar(data):
    X, y = data
    t = I.date_permutation(X, y, 2)
    d0, d1 = pd.DatetimeIndex(sorted(set(X.index.get_level_values(0)))), pd.DatetimeIndex(sorted(set(t.X.index.get_level_values(0))))
    assert len(d0) == len(d1) and d1.is_monotonic_increasing
    assert not set(d0) & set(d1)
    assert (d1.year != d0.year).all() or (d1.weekday != d0.weekday).any()
    np.testing.assert_allclose(t.X.to_numpy(), X.to_numpy())


def test_year_disguise_uses_the_audited_presentation_shift(data):
    X, y = data
    t = I.year_disguise(X, y, 5)
    shift = t.X.index.get_level_values(0)[0] - X.index.get_level_values(0)[0]
    assert shift.days % 7 == 0 and abs(shift.days) >= 8 * 7                     # whole weeks, at least min_abs_weeks away
    assert (t.X.index.get_level_values(1) == X.index.get_level_values(1)).all()  # tickers untouched
    p = I.presentation_disguise(X, y, 5)
    assert not set(p.X.index.get_level_values(1)) & set(X.index.get_level_values(1))          # opaque fresh codes
    assert p.kind == "presentation_disguise" and "shift_days" in p.mapping
    names_only = I.presentation_disguise(X, y, 5, dates=False)
    assert names_only.kind == "ticker_codes" and (names_only.X.index.get_level_values(0) == X.index.get_level_values(0)).all()


def test_stock_substitution_swaps_pairs_and_leaves_the_rest(data):
    X, y = data
    t = I.stock_substitution(X, y, 1, frac=0.5)
    assert all(t.mapping[t.mapping[a]] == a for a in t.mapping) and len(t.mapping) == 10
    assert 0.4 < t.changed_frac < 0.6
    with pytest.raises(ValueError):
        I.stock_substitution(X, y, 1, frac=0)


def test_sector_substitution_permutes_column_names_or_group_map(data):
    X, y = data
    cols = [f"sec_{k}" for k in range(4)]
    t = I.sector_substitution(X, y, 2, sector_columns=cols)
    assert list(t.X.columns) == list(X.columns) and not t.X["sec_0"].equals(X["sec_0"])
    moved = {c: X[[k for k, v in t.mapping.items() if v == c][0]] for c in cols}
    assert all(t.X[c].equals(moved[c]) for c in cols)
    g = I.sector_substitution(X, y, 2, groups={"A": "tech", "B": "energy", "C": "tech", "D": "util"})
    assert set(g.mapping.values()) <= {"tech", "energy", "util"}
    with pytest.raises(ValueError):
        I.sector_substitution(X, y, 2)
    with pytest.raises(ValueError):
        I.sector_substitution(X, y, 2, sector_columns=["nope"])


def test_episode_substitution_and_sequence_scramble_move_whole_cross_sections(data):
    X, y = data
    e = I.episode_substitution(X, y, 1, block=8, frac=1.0)
    s = I.sequence_scramble(X, y, 1, block=4)
    for t in (e, s):
        assert t.X.index.equals(X.index) and I.verify_transform(X, y, t) == []
        assert t.changed_frac > 0.3
        # each date's cross-section still holds one intact copy of some original date's values
        orig = {tuple(np.sort(g["f1"].to_numpy())) for _, g in X.groupby(level=0)}
        assert all(tuple(np.sort(g["f1"].to_numpy())) in orig for _, g in t.X.groupby(level=0))
    assert not np.array_equal(s.X.to_numpy(), X.to_numpy())


def test_a_no_op_transform_is_flagged_and_content_change_is_flagged(data):
    X, y = data
    noop = I.Transformed("ticker_permutation", X.copy(), y.copy(), 0.0, {})
    assert [f.check for f in I.verify_transform(X, y, noop)] == ["no-op-transform"]
    corrupt = I.Transformed("year_disguise", X.assign(f1=X["f1"] * 2), y, 1.0, {})
    assert "content-not-preserved" in {f.check for f in I.verify_transform(X, y, corrupt)}
    short = I.Transformed("year_disguise", X.iloc[:-3], y.iloc[:-3], 1.0, {})
    assert "rowcount-changed" in {f.check for f in I.verify_transform(X, y, short)}
    with pytest.raises(ValueError):
        I.ticker_permutation(X.reset_index(drop=True), None, 0)


# ---------------------------------------------------------------- scoring
def test_per_date_ic_and_topk_on_a_perfect_and_a_random_score(data):
    X, y = data
    perfect = I.per_date_ic(y, y)
    assert perfect.min() == pytest.approx(1.0) and len(perfect) == 48
    rnd = I.per_date_ic(pd.Series(np.random.default_rng(123).normal(size=len(y)), index=y.index), y)
    assert abs(rnd.mean()) < 0.1
    assert I.top_k_spread(y, y, 5) > 0 and abs(I.top_k_spread(-y, y, 5)) > 0 and I.top_k_spread(-y, y, 5) < 0
    assert I.per_date_ic(y.iloc[:0], y.iloc[:0]).empty and np.isnan(I.top_k_spread(y.iloc[:0], y.iloc[:0]))
    lo, hi = I.retention_ci(np.full(20, 0.1), np.full(20, 0.05), 0)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    assert np.isnan(I.retention_ci(np.array([]), np.array([0.1]))[0])


# ---------------------------------------------------------------- the harness (section 62 tests 8 and 10)
def harness(learner, **kw):
    kw.setdefault("attacks", tuple(I.ATTACKS) + ("sector_substitution",))
    kw.setdefault("boot", 100)
    return I.IdentityHarness(learner, **kw)


def run(learner, data, **kw):
    X, y, Xe, ye = rerun_split(*data)
    return harness(learner, **kw).run(X, y, Xe, ye, sector_columns=[c for c in X.columns if c.startswith("sec_")])


def train_test_split(data, cut=30):
    X, y = data
    d = X.index.get_level_values(0)
    dates = sorted(set(d))
    tr = np.asarray(d < dates[cut])
    return X[tr], y[tr], X[~tr], y[~tr]


def test_legitimate_ridge_learner_survives_every_identity_attack(data):
    """Section 62 test 8: change ticker identities; performance must not collapse solely because identities changed."""
    X, y, Xe, ye = train_test_split(data)
    rep = harness(I.ridge_learner).run(X, y, Xe, ye, sector_columns=["sec_0", "sec_1", "sec_2", "sec_3"])
    assert rep.deterministic and rep.base_ic > 0.1
    bad = [(v.kind, v.mode, v.status, round(v.retention, 2)) for v in rep.verdicts if v.status != "OK"]
    # sector columns are FEATURES ridge weights by name: renaming them in 'eval' mode changes what they mean, so only that attack may bite
    assert all(k == "sector_substitution" for k, *_ in bad), bad
    assert not rep.memorization_suspected or all(v.kind == "sector_substitution" for v in rep.verdicts if v.status == "COLLAPSE")
    ren = rep.retention_by_kind("eval")
    assert ren["ticker_permutation"] > 0.8 and ren["date_permutation"] > 0.8 and ren["presentation_disguise"] > 0.8


def test_memoriser_collapses_when_only_the_evaluated_identities_change(data):
    """The planted memoriser: brilliant on the rerun, blind once the labels are disguised."""
    rep = run(I.memorizer_learner, data)
    assert rep.base_ic > 0.99
    eval_ret = rep.retention_by_kind("eval")
    for kind in ("ticker_permutation", "date_permutation", "year_disguise", "presentation_disguise", "stock_substitution"):
        assert eval_ret[kind] < 0.3, kind
    assert rep.memorization_suspected and not rep.passed
    assert {"ticker_permutation[eval]", "presentation_disguise[eval]"} <= set(rep.collapsed)
    both = {(v.kind): v.status for v in rep.verdicts if v.mode == "both"}
    assert both["ticker_permutation"] == "OK" and both["year_disguise"] == "OK"       # relabelled consistently, a lookup still works


def test_episode_and_sequence_attacks_catch_a_date_keyed_lookup(data):
    rep = run(I.memorizer_learner, data, attacks=("episode_substitution", "sequence_scramble"))
    assert {v.kind for v in rep.verdicts if v.status == "COLLAPSE"} == {"episode_substitution", "sequence_scramble"}


def test_attribution_shows_which_identity_dimension_the_learner_keys_on(data):
    # a ticker-mean lookup has no skill on i.i.d. returns; plant a per-ticker effect it can memorise
    X, y = data
    y2 = y + X.index.get_level_values(1).map({t: (i - 10) * 0.004 for i, t in enumerate(sorted(set(X.index.get_level_values(1))))}).to_numpy()
    rep = harness(I.ticker_mean_learner, min_skill=0.01).run(X, y2, X, y2)
    att = I.attribute_keying(rep)
    assert att["ticker"] == "COLLAPSES" and att["date"] == "HOLDS" and att["sector"] == "UNTESTED"
    mem = I.attribute_keying(run(I.memorizer_learner, data))
    assert mem["ticker"] == "COLLAPSES" and mem["date"] == "COLLAPSES" and mem["sequence"] == "COLLAPSES"


def test_distinguishable_from_a_memoriser(data):
    """Section 62 test 10: give the learner a memoriser; the legitimate learner must remain distinguishable from it."""
    X, y, Xe, ye = train_test_split(data)
    legit = harness(I.ridge_learner, attacks=tuple(I.ATTACKS)).run(X, y, Xe, ye)
    mem = harness(I.memorizer_learner, attacks=tuple(I.ATTACKS)).run(X, y, X, y)
    d = I.distinguish_from_memorizer(legit, mem)
    assert d["distinguishable"] and d["gap"] > 0.5
    same = I.distinguish_from_memorizer(mem, mem)
    assert not same["distinguishable"]
    assert I.distinguish_from_memorizer(legit, I.IdentityReport((), 0.0, 0.0, True, 0))["reason"] == "no common attacks"


def test_a_learner_with_a_hardcoded_absolute_date_rule_breaks_under_consistent_relabelling(data):
    def clock_learner(Xt, yt, Xe, seed=0):
        sign = np.where(Xe.index.get_level_values(0).year < 2100, 1.0, -1.0)         # logic that depends on the calendar
        return pd.Series(Xe["f1"].to_numpy() * sign, index=Xe.index)
    X, y, Xe, ye = train_test_split(data)
    rep = harness(clock_learner, attacks=("year_disguise",)).run(X, y, Xe, ye)
    v = {x.mode: x for x in rep.verdicts}
    assert v["both"].status == "COLLAPSE" and v["eval"].status == "COLLAPSE" and rep.logic_identity_dependent


def test_null_and_nondeterministic_learners_are_told_apart_from_collapse(data):
    X, y, Xe, ye = train_test_split(data)
    rep = harness(I.random_learner, attacks=("ticker_permutation",), modes=("eval",)).run(X, y, Xe, ye)
    assert all(v.status == "NO_SKILL" for v in rep.verdicts) and not rep.passed and not rep.collapsed
    counter = {"n": 0}

    def flaky(Xt, yt, Xe_, seed=0):
        counter["n"] += 1
        return pd.Series(Xe_["f1"].to_numpy() + counter["n"] * 1e-3, index=Xe_.index)
    rep2 = harness(flaky, attacks=("ticker_permutation",), modes=("eval",)).run(X, y, Xe, ye)
    assert not rep2.deterministic and rep2.verdicts[0].status == "NONDETERMINISTIC" and not rep2.passed
    assert any(f.check == "nondeterministic" for f in rep2.findings)


def test_harness_input_validation_and_degenerate_cases(data):
    X, y, Xe, ye = train_test_split(data)
    with pytest.raises(ValueError):
        I.IdentityHarness(I.ridge_learner, attacks=("nope",))
    with pytest.raises(ValueError):
        I.IdentityHarness(I.ridge_learner, modes=("sideways",))
    with pytest.raises(ValueError):
        harness(I.ridge_learner).run(X.reset_index(drop=True), y, Xe, ye)
    with pytest.raises(FirewallBreach):
        harness(lambda a, b, c, s: [1, 2, 3], attacks=("ticker_permutation",)).run(X, y, Xe, ye)
    with pytest.raises(FirewallBreach):
        harness(lambda a, b, c, s: pd.Series(np.zeros(len(c)), index=range(len(c))), attacks=("ticker_permutation",)).run(X, y, Xe, ye)
    few = harness(I.ridge_learner, attacks=("ticker_permutation",), min_dates=100).run(X, y, Xe, ye)
    assert {v.status for v in few.verdicts} == {"INSUFFICIENT"}
    nothing = harness(I.ridge_learner, attacks=("sector_substitution",)).run(X, y, Xe, ye)          # no sector info supplied
    assert nothing.verdicts == () and any(f.check == "no-attacks-run" for f in nothing.findings) and not nothing.passed
    empty = harness(I.ridge_learner, attacks=("ticker_permutation",)).run(X, y, Xe.iloc[:0], ye.iloc[:0])
    assert not empty.passed and all(v.status in ("NO_SKILL", "INSUFFICIENT") for v in empty.verdicts)


def test_report_serialisation_markdown_and_findings(data):
    rep = run(I.memorizer_learner, data, attacks=("ticker_permutation", "date_permutation"))
    assert not rep.to_frame().empty and rep.digest() == run(I.memorizer_learner, data, attacks=("ticker_permutation", "date_permutation")).digest()
    md = rep.markdown()
    assert "Memorisation suspected" in md and "IMPLEMENTED - NOT VALIDATED" in md
    import json
    assert json.loads(I.report_to_json(rep))["passed"] is False
    fs = I.to_findings(rep)
    assert any(f.check == "collapse" and f.is_fail for f in fs)


def test_identity_report_feeds_the_identity_layer_of_the_gate(data):
    """Integration: a real report goes through firewalls.IdentityFirewall; collapse rejects, a clean report passes."""
    X, y, Xe, ye = train_test_split(data)
    good = harness(I.ridge_learner, attacks=("ticker_permutation", "date_permutation", "year_disguise")).run(X, y, Xe, ye)
    bad = harness(I.memorizer_learner, attacks=("ticker_permutation", "date_permutation")).run(X, y, X, y)
    layer = F.IdentityFirewall()
    ok = layer.run(F.GateContext(now="2030-01-01", identity_report=good))
    no = layer.run(F.GateContext(now="2030-01-01", identity_report=bad))
    assert ok.status == F.LayerStatus.PASS, [str(f) for f in ok.findings]
    assert no.status == F.LayerStatus.FAIL and any(f.check == "collapse" for f in no.findings)


# ---------------------------------------------------------------- structural probes
def test_identity_proxy_columns_are_found(data):
    X, y = data
    tk = X.index.get_level_values(1)
    proxy = X.assign(stock_id=pd.factorize(tk)[0].astype(float), clock=np.arange(len(X), dtype=float) // len(set(tk)))
    checks = {(f.subject, f.check) for f in I.identity_proxies(proxy)}
    assert ("stock_id", "ticker-proxy") in checks and ("clock", "date-proxy") in checks
    assert I.identity_proxies(X[["f1", "f2", "f3"]]) == []


def test_label_only_sensitivity_separates_identity_free_from_identity_using_learners(data):
    X, y, Xe, ye = train_test_split(data)
    free = I.label_only_sensitivity(I.ridge_learner, X, y, Xe)
    assert free["sensitivity"] == pytest.approx(0.0, abs=1e-9)

    def name_learner(Xt, yt, Xe_, seed=0):
        h = np.array([sum(map(ord, t)) % 17 for t in Xe_.index.get_level_values(1)], dtype=float)
        return pd.Series(h + Xe_["f1"].to_numpy() * 0.01, index=Xe_.index)
    assert I.label_only_sensitivity(name_learner, X, y, Xe)["sensitivity"] > 0.5
    few = Xe[Xe.index.get_level_values(1) == "S000"]
    assert "note" in I.label_only_sensitivity(I.ridge_learner, X, y, few)


def test_seen_vs_unseen_gap_and_null_pvalue(data):
    X, y = data
    half = X.index.get_level_values(0) < sorted(set(X.index.get_level_values(0)))[24]
    Xtr, ytr = X[half], y[half]
    mixed_X = X                                                             # first half seen, second half unseen
    gap_mem = I.seen_vs_unseen_gap(I.memorizer_learner, Xtr, ytr, mixed_X, y)
    gap_ridge = I.seen_vs_unseen_gap(I.ridge_learner, Xtr, ytr, mixed_X, y)
    assert gap_mem["gap"] > 0.5 and abs(gap_ridge["gap"]) < 0.25
    assert np.isnan(I.seen_vs_unseen_gap(I.ridge_learner, X, y, X, y)["gap"])
    s = I.ridge_learner(Xtr, ytr, X, 0)
    assert I.ic_null_pvalue(s, y, 100)["p"] < 0.05
    assert I.ic_null_pvalue(I.random_learner(Xtr, ytr, X, 0), y, 100)["p"] > 0.05
    assert np.isnan(I.ic_null_pvalue(y.iloc[:0], y.iloc[:0])["p"])


def test_rerun_linkability_shows_numeric_memory_recognises_a_disguised_rerun(data):
    """Documents the C54/C56 tension: a rerun replays identical values, so names are re-identified from numbers alone."""
    X, y = data
    p = I.presentation_disguise(X, y, 9, order_preserving=False)
    truth = {k: v for k, v in p.mapping.items() if k != "shift_days"}
    res = I.rerun_linkability_panel(X, p.X, truth, columns=["f1"])
    assert res["share_reidentified_by_values"] > 0.95 and res["n_names"] == 20
    assert I.rerun_linkability_panel(X, p.X, {})["n_names"] == 0


def test_compose_battery_and_domain_errors(data):
    X, y = data
    t = I.compose(X, y, [("ticker_permutation", {}), ("date_permutation", {})], seed=2)
    assert t.kind == "ticker_permutation+date_permutation" and t.changed_frac > 0.9 and I.verify_transform(X, y, t) == []
    with pytest.raises(ValueError):
        I.compose(X, y, [])
    with pytest.raises(ValueError):
        I.compose(X, y, [("sector_substitution", {})])
    Xt, yt, Xe, ye = train_test_split(data)
    bat = I.run_battery(I.ridge_learner, Xt, yt, Xe, ye, seeds=(0, 1), attacks=("ticker_permutation", "date_permutation"), boot=50)
    assert bat.passed and bat.robust_collapse() == [] and set(bat.collapse_rate()["kind"]) == {"ticker_permutation", "date_permutation"}
    mem = I.run_battery(I.memorizer_learner, Xt, yt, Xt, yt, seeds=(0, 1), attacks=("ticker_permutation",), modes=("eval",), boot=50)
    assert mem.robust_collapse() == ["ticker_permutation[eval]"] and not mem.passed
    assert bat.digest() != mem.digest()


# ---------------------------------------------------------------- second-wave additions
def test_substitution_curve_slope_separates_memoriser_from_legitimate_learner(data):
    X, y, Xe, ye = train_test_split(data)
    mem = I.substitution_curve(I.memorizer_learner, X, y, X, y, fractions=(0.25, 0.5, 1.0))
    ridge = I.substitution_curve(I.ridge_learner, X, y, Xe, ye, fractions=(0.25, 0.5, 1.0))
    assert I.memorisation_slope(mem) < -0.7 and abs(I.memorisation_slope(ridge)) < 0.35
    assert mem["retention"].iloc[-1] < 0.1 and ridge["retention"].iloc[-1] > 0.7
    assert np.isnan(I.memorisation_slope(mem.iloc[:1]))


def test_name_versus_order_and_cross_identity_transfer(data):
    X, y, Xe, ye = train_test_split(data)
    ridge = I.name_versus_order(I.ridge_learner, X, y, Xe, ye)
    assert ridge["depends_on"] == "NEITHER"
    Xd, yd = data
    tk_effect = pd.Series(Xd.index.get_level_values(1).map({t: (i - 10) * 0.006 for i, t in enumerate(sorted(set(Xd.index.get_level_values(1))))}).to_numpy(),
                          index=Xd.index)
    rid = I.cross_identity_transfer(I.ridge_learner, Xd, yd, seed=1)
    assert rid["transfers_across_stocks"] and rid["time"] > 0.05 and rid["n_train"] > 0
    mem = I.cross_identity_transfer(I.ticker_mean_learner, Xd, yd + tk_effect, seed=1)
    assert not mem["transfers_across_stocks"]


def test_harness_selfcheck_sees_memorisation_and_attack_power(data):
    sc = I.harness_selfcheck(seed=0)
    assert sc["ok"], sc["wrong"]
    assert sc["memorizer_distinguished"]
    rep = run(I.memorizer_learner, data, attacks=("ticker_permutation", "stock_substitution"), modes=("eval",))
    tab = I.attack_power(rep)
    assert tab["informative"].all() and set(tab["kind"]) == {"ticker_permutation", "stock_substitution"}
    X, y = data
    bat = I.run_battery(I.memorizer_learner, X, y, X, y, seeds=(0, 1), attacks=("ticker_permutation",), modes=("eval",), boot=40)
    assert "Robust collapse" in I.battery_markdown(bat) and "ticker_permutation" in I.battery_markdown(bat)


def test_retention_summary_and_identity_free_probe(data):
    X, y, Xe, ye = train_test_split(data)
    rep = harness(I.ridge_learner, attacks=("ticker_permutation", "date_permutation")).run(X, y, Xe, ye)
    summ = I.retention_summary(rep)
    assert summ["eval"]["n"] == 2 and summ["eval"]["worst"] > 0.8 and summ["both"]["mean"] > 0.8
    assert I.is_identity_free(I.ridge_learner, X, y, Xe)
    assert not I.is_identity_free(I.ticker_mean_learner, X, y, Xe)
