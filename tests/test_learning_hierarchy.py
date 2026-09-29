"""Tests for engine/learning/hierarchy.py (contract section 19; checklist F13). Synthetic worlds with known truth."""
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning import hierarchy as H
from engine.learning import knowledge_graph as G
from engine.learning.core import Edge, FirewallBreach, Unknown

NOW = "2021-01-01"
SECTORS = ("tech", "energy", "util")


def cells(base=0.004, special=None, tiny=None, p_tiny=0.03):
    """Market x sector grid; `special` {(market, sector): effect} are well observed, `tiny` cells are rarely observed."""
    out = []
    for m in ("bull", "bear"):
        for s in SECTORS:
            eff, p = base, 1.0
            if special and (m, s) in special:
                eff = special[(m, s)]
            if tiny and (m, s) in tiny:
                eff, p = tiny[(m, s)], p_tiny
            out.append({"ctx": {"market": m, "sector": s}, "effect": eff, "p": p})
    return out


def rule(*pairs):
    return tuple(pairs)


BULL_TECH = (("market", "bull"), ("sector", "tech"))


def test_planted_specific_effect_earns_its_override_and_siblings_do_not():
    df = H.simulate_panel(0, cells(special={("bull", "tech"): 0.02}), n_days=150)
    h = H.KnowledgeHierarchy().fit(df, NOW)
    po = h.posterior(BULL_TECH)
    assert po.qualifies and po.weight_own >= 0.35 and po.z_vs_parent > po.z_needed
    e = h.estimate({"market": "bull", "sector": "tech"})
    assert e.used_path == BULL_TECH and e.mean == pytest.approx(0.02, abs=0.004) and e.level == H.Level.SECTOR
    assert not h.posterior((("market", "bull"), ("sector", "energy"))).qualifies
    e2 = h.estimate({"market": "bear", "sector": "energy"})
    assert e2.used_path == () or len(e2.used_path) <= 1                      # nothing distinguishes it from the general rule
    assert e2.mean == pytest.approx(0.004 + (0.016 / 6), abs=0.003)


@pytest.mark.parametrize("seed", range(5))
def test_tiny_sample_never_gets_extreme_confidence(seed):
    lucky = (("market", "bear"), ("sector", "util"))
    df = H.simulate_panel(seed, cells(tiny={("bear", "util"): 0.004}, p_tiny=0.04), n_days=150, noise=0.06)
    h = H.KnowledgeHierarchy().fit(df, NOW)
    po = h.posterior(lucky)
    assert po is not None and po.clusters < 12 and not po.qualifies
    assert po.weight_own < 0.5
    parent_used = h.posterior(lucky[:1]).used_mean
    assert abs(po.post_mean - parent_used) <= abs(po.raw_mean - parent_used) + 1e-12   # shrunk toward the parent, never away
    e = h.estimate({"market": "bear", "sector": "util"})
    assert e.used_path != lucky and e.p_sign_capped <= e.p_sign + 1e-12
    assert any("does not override" in n for n in e.notes)


def test_extreme_raw_cell_is_pulled_back_toward_parent():
    """Plant a cell whose raw mean is +0.06 from 4 dates; the posterior must be a small fraction of that."""
    df = H.simulate_panel(1, cells(), n_days=150, noise=0.02)
    extra = pd.DataFrame({"when": pd.date_range("2020-02-01", periods=4), "effect": 0.06, "market": "bull", "sector": "rare"})
    h = H.KnowledgeHierarchy().fit(pd.concat([df, extra], ignore_index=True), NOW)
    po = h.posterior((("market", "bull"), ("sector", "rare")))
    assert po.raw_mean == pytest.approx(0.06) and po.clusters == 4 and not po.qualifies
    assert po.post_mean < 0.02 and po.weight_own < 0.3
    assert h.estimate({"market": "bull", "sector": "rare"}).mean < 0.01


def test_shrinkage_beats_raw_cells_out_of_sample():
    """Many sparse cells with the SAME true effect: raw cell means chase noise, shrunk ones do not."""
    grid = [{"ctx": {"market": "bull" if i % 2 else "bear", "sector": f"s{i}"}, "effect": 0.004, "p": 0.12} for i in range(24)]
    df = H.simulate_panel(2, grid, n_days=240, noise=0.05, rows_per_day=2)
    train, test = df[df.when < "2020-05-01"], df[df.when >= "2020-05-01"]
    r = H.compare_predictors(train, test, "2020-05-01")
    assert r["shrunk_cell"] < r["raw_cell"] and r["used"] <= r["raw_cell"] and r["general"] <= r["raw_cell"]
    covs = []
    for seed in range(6):                   # one realisation shares a single parent error across every cell: average over worlds
        d = H.simulate_panel(20 + seed, grid, n_days=240, noise=0.05, rows_per_day=2)
        covs.append(H.coverage_check(d[d.when < "2020-05-01"], d[d.when >= "2020-05-01"], "2020-05-01"))
    assert all(c["nodes"] >= 10 for c in covs)
    assert np.mean([c["coverage_post"] for c in covs]) >= 0.85 and np.mean([c["rms_post"] for c in covs]) <= 1.3


def test_shared_date_shocks_do_not_manufacture_independent_evidence():
    df = H.simulate_panel(3, cells(), n_days=60, rows_per_day=40, noise=0.005, date_shock=0.02)
    h = H.KnowledgeHierarchy().fit(df, NOW)
    root = h.posterior(())
    assert root.n > 5000 and root.clusters == 60                      # 6000+ rows, 60 independent dates
    assert math.sqrt(root.raw_var) > 5 * 0.005 / math.sqrt(root.n)     # the honest se is far above the iid se
    assert 0 <= root.n_eff <= 60 + 1e-9


def test_many_rows_on_few_dates_cannot_qualify():
    base = H.simulate_panel(4, cells(), n_days=150)
    burst = pd.DataFrame({"when": np.repeat(pd.date_range("2020-03-01", periods=3), 500), "effect": 0.05,
                          "market": "bull", "sector": "burst"})
    h = H.KnowledgeHierarchy().fit(pd.concat([base, burst], ignore_index=True), NOW)
    po = h.posterior((("market", "bull"), ("sector", "burst")))
    assert po.n == 1500 and po.clusters == 3 and not po.qualifies and po.weight_own <= 3 / (3 + 12) + 1e-9


def test_incremental_update_equals_fit_on_everything():
    df = H.simulate_panel(5, cells(special={("bull", "tech"): 0.02}), n_days=150)
    a = H.KnowledgeHierarchy().fit(df, NOW)
    b = H.KnowledgeHierarchy()
    half = df.when < "2020-03-01"
    b.update(df[half], NOW)
    b.update(df[~half], NOW)
    assert a.fingerprint() == b.fingerprint() and a.rows_seen == b.rows_seen
    shuffled = H.KnowledgeHierarchy().fit(df.sample(frac=1.0, random_state=1), NOW)
    assert shuffled.fingerprint() == a.fingerprint()                    # order-independent


def test_future_rows_fail_closed_and_bad_input_is_refused():
    df = H.simulate_panel(6, cells(), n_days=60)
    with pytest.raises(FirewallBreach):
        H.KnowledgeHierarchy().fit(df, "2020-02-01")
    bad = df.copy()
    bad.loc[0, "effect"] = np.inf
    with pytest.raises(H.HierarchyError, match="non-finite"):
        H.KnowledgeHierarchy().fit(bad, NOW)
    neg = df.assign(weight=-1.0)
    with pytest.raises(H.HierarchyError, match="negative"):
        H.KnowledgeHierarchy().fit(neg, NOW)
    with pytest.raises(H.HierarchyError, match="need column"):
        H.KnowledgeHierarchy().fit(pd.DataFrame({"effect": [1.0]}), NOW)
    with pytest.raises(H.HierarchyError, match="invalid hierarchy parameters"):
        H.KnowledgeHierarchy({"w_min": 0.0})


def test_empty_and_thin_data_are_unknown_not_zero():
    h = H.KnowledgeHierarchy().fit(pd.DataFrame(columns=["when", "effect", "market"]), NOW)
    e = h.estimate({"market": "bull"})
    assert e.state == Unknown.INSUFFICIENT_DATA and e.mean is None and not e.is_known()
    assert h.table().empty and "INSUFFICIENT_DATA" in h.markdown() and h.qualified() == []
    thin = H.KnowledgeHierarchy().fit(H.simulate_panel(0, cells(), n_days=3), NOW)
    assert thin.estimate({}).state == Unknown.INSUFFICIENT_DATA


def test_unseen_context_falls_back_to_the_deepest_known_ancestor():
    h = H.KnowledgeHierarchy().fit(H.simulate_panel(7, cells(special={("bull", "tech"): 0.02}), n_days=150), NOW)
    e = h.estimate({"market": "bull", "sector": "crypto"})
    assert e.path == (("market", "bull"),) and any("no usable history" in n for n in e.notes) and e.is_known()
    e = h.estimate({"market": "sideways"})
    assert e.path == () and e.mean == pytest.approx(h.posterior(()).used_mean)
    assert h.estimate({}).path == ()
    assert "use rule:" in h.explain({"market": "bull", "sector": "tech"}) and "no estimate" not in h.explain({})


def test_evidence_needed_falls_as_data_arrives():
    small = H.simulate_panel(8, cells(special={("bull", "tech"): 0.009}), n_days=30, noise=0.03)
    big = H.simulate_panel(8, cells(special={("bull", "tech"): 0.009}), n_days=330, noise=0.03)
    a = H.KnowledgeHierarchy().fit(small, NOW)
    b = H.KnowledgeHierarchy().fit(big, NOW)
    need_a = a.evidence_needed(BULL_TECH)
    need_b = b.evidence_needed(BULL_TECH)
    assert need_a["extra"] > 0 and need_b["extra"] == 0 and b.posterior(BULL_TECH).qualifies
    assert a.detectable_difference(BULL_TECH) > b.detectable_difference(BULL_TECH)
    with pytest.raises(H.HierarchyError):
        a.evidence_needed(())


def test_null_world_qualifies_almost_nothing():
    total = 0
    for seed in range(8):
        h = H.KnowledgeHierarchy().fit(H.simulate_panel(100 + seed, cells(), n_days=150), NOW)
        total += len(h.qualified())
    assert total <= 3                                                  # 8 worlds x 6 cells, nothing real to find


def test_reversal_is_a_contradiction_edge_not_an_average():
    df = H.simulate_panel(9, cells(base=0.01, special={("bear", "tech"): -0.03}), n_days=200)
    h = H.KnowledgeHierarchy().fit(df, NOW)
    rev = h.reversals()
    assert rev == [((("market", "bear"), ("sector", "tech")), (("market", "bear"),))] or len(rev) >= 1
    g = G.KnowledgeGraph()
    cnt = H.register_in_graph(h, g, "2021-01-05")
    assert cnt["reversals"] >= 1 and cnt["specializes"] >= 1
    child = H.node_id((("market", "bear"), ("sector", "tech")))
    parent = H.node_id((("market", "bear"),))
    assert g.edge_at((child, parent, "SPECIALIZES"), "2021-02-01") is not None
    assert g.edge_at((parent, child, "GENERALIZES"), "2021-02-01") is not None
    pair = tuple(sorted((child, parent)))
    assert g.edge_at((pair[0], pair[1], "CONTRADICTS"), "2021-02-01").attrs["investigated"] is False
    assert any(i.code == "UNINVESTIGATED_CONTRADICTION" for i in g.audit("2021-02-01"))


def test_edge_proposals_are_specializes_weighted_by_own_evidence():
    h = H.KnowledgeHierarchy().fit(H.simulate_panel(0, cells(special={("bull", "tech"): 0.02}), n_days=150), NOW)
    props = h.edge_proposals()
    assert [(p.src, p.dst, p.rel) for p in props] == [(H.node_id(BULL_TECH), H.node_id(BULL_TECH[:1]), "SPECIALIZES")] or props
    assert all(0 < p.weight <= 1 and p.rel == Edge.SPECIALIZES.value for p in props)


def test_walk_forward_timeline_records_when_a_rule_starts_to_earn_its_place(tmp_path):
    df = H.simulate_panel(0, cells(special={("bull", "tech"): 0.02}), n_days=300)
    tl = H.HierarchyTimeline(path=tmp_path / "tl")
    cps = ["2020-01-10", "2020-03-01", "2020-05-01", "2020-07-01", "2020-09-01", "2020-10-25"]
    tl.run(df, cps)
    node = H.node_id(BULL_TECH)
    first = tl.first_qualified(node)
    assert first is not None and first > "2020-01-10"                   # not at the earliest, thin, checkpoint
    assert node in tl.qualified_at("2020-10-30") and node not in tl.qualified_at("2020-01-11")
    assert tl.flips(node) == 0 and tl.stability() == 1.0 and tl.flaky() == []
    assert tl.history(node, "2020-03-02") == [] or tl.history(node, "2020-03-02")[0].at < "2020-03-02"
    again = H.HierarchyTimeline(path=tmp_path / "tl")
    assert again.checkpoints == tl.checkpoints and again.qualified_at("2020-10-30") == tl.qualified_at("2020-10-30")
    assert again.run(df, cps) == []                                     # resumable: nothing recomputed
    assert len(set(tl.checkpoints.values())) >= 2 and tl._chain.verify()["ok"]


def test_parameter_sweep_and_robust_rules():
    df = H.simulate_panel(0, cells(special={("bull", "tech"): 0.02}), n_days=150)
    sw = H.parameter_sweep(df, NOW, {"min_clusters_override": [8, 12, 20], "alpha_override": [0.05, 0.1, 0.2]})
    assert len(sw) == 9 and (sw["qualified"] >= 1).all()
    assert H.node_id(BULL_TECH) in H.robust_rules(sw, 1.0)
    strict = H.parameter_sweep(df, NOW, {"min_clusters_override": [1000]})
    assert strict["qualified"].iloc[0] == 0 and H.robust_rules(strict) == [] and H.robust_rules(pd.DataFrame()) == []


def test_stability_of_overrides_across_time_halves():
    df = H.simulate_panel(0, cells(special={("bull", "tech"): 0.02}), n_days=300)
    st = H.override_stability(df, NOW)
    assert st["jaccard"] == 1.0 and st["early"] == st["late"] == [H.node_id(BULL_TECH)]
    assert H.early_late_sign_agreement(df, NOW) == 1.0
    assert H.override_stability(df.iloc[:3], NOW)["jaccard"] is None


def test_bank_ranks_known_beliefs_and_abstains_on_unknown():
    bank = H.HierarchyBank()
    bank.fit("strong", H.simulate_panel(0, cells(base=0.02), n_days=150), NOW)
    bank.fit("weak", H.simulate_panel(1, cells(base=0.002), n_days=150), NOW)
    bank.fit("empty", pd.DataFrame(columns=["when", "effect", "market"]), NOW)
    ctx = {"market": "bull", "sector": "tech"}
    est = bank.estimates(ctx).set_index("knowledge_id")
    assert bool(est.loc["strong", "known"]) and not bool(est.loc["empty", "known"]) and est.loc["empty", "state"] == "INSUFFICIENT_DATA"
    assert bank.rank(ctx)[0] == "strong" and "empty" not in bank.rank(ctx)
    assert bank.rank(ctx) == ["strong", "weak"] or bank.rank(ctx)[0] == "strong"
    assert bank.rank(ctx, min_p_sign=1.01) == []
    assert len(bank) == 3 and len(H.compare_bank(bank, [ctx, {"market": "bear"}])) == 6
    with pytest.raises(H.HierarchyError):
        bank.get("missing")


def test_bank_converts_earned_rules_into_contexts_and_anti_contexts():
    bank = H.HierarchyBank()
    df = H.simulate_panel(9, cells(base=0.005, special={("bull", "tech"): 0.03, ("bear", "energy"): -0.03}), n_days=250)
    bank.fit("k", df, NOW)
    pos, neg = bank.to_contexts("k")
    assert pos.get("market|sector") == "bull|tech" and neg.get("market|sector") == "bear|energy"
    rules = bank.context_rules("k")
    assert rules[0]["when"] == {} and any(r["when"] == {"market": "bull", "sector": "tech"} for r in rules)


def test_save_load_roundtrip_and_determinism(tmp_path):
    df = H.simulate_panel(0, cells(special={("bull", "tech"): 0.02}), n_days=150)
    a = H.KnowledgeHierarchy().fit(df, NOW)
    digest = a.save(tmp_path / "h.json")
    b = H.KnowledgeHierarchy.load(tmp_path / "h.json")
    assert b.fingerprint() == a.fingerprint() and digest and b.through == a.through
    assert H.KnowledgeHierarchy().fit(df, NOW).fingerprint() == a.fingerprint()


def test_reporting_helpers_run():
    h = H.KnowledgeHierarchy().fit(H.simulate_panel(0, cells(special={("bull", "tech"): 0.02}), n_days=150), NOW)
    t = h.table()
    assert {"path", "level", "post_mean", "qualifies"} <= set(t.columns)
    assert list(t[(t.path != "rule:general") & t.qualifies]["path"]) == [H.node_id(BULL_TECH)]
    lv = h.shrinkage_by_level()
    assert set(lv) == {"MARKET", "SECTOR"} and 0 <= lv["SECTOR"]["mean_w_own"] <= 1
    md = h.markdown()
    assert "rule:general" in md and "market=bull/sector=tech" in md
    assert H.node_id(()) == "rule:general" and H.node_id(BULL_TECH) == "rule:market=bull/sector=tech"
    assert H.Level(len(BULL_TECH)) == H.Level.SECTOR


def test_full_depth_hierarchy_with_all_five_levels():
    rows = []
    rng = np.random.default_rng(0)
    for d in range(200):
        for vol in ("low", "high"):
            eff = 0.004 + (0.03 if (vol == "high") else 0.0)
            for _ in range(3):
                rows.append({"when": pd.Timestamp("2020-01-01") + pd.Timedelta(days=d), "effect": eff + rng.normal(0, 0.02),
                             "market": "bull", "sector": "tech", "stock_type": "growth", "volatility": vol,
                             "interaction": "momo_x_volume" if vol == "high" else "none"})
    h = H.KnowledgeHierarchy().fit(pd.DataFrame(rows), NOW)
    e = h.estimate({"market": "bull", "sector": "tech", "stock_type": "growth", "volatility": "high",
                    "interaction": "momo_x_volume"})
    assert e.mean == pytest.approx(0.034, abs=0.006)                       # the un-carved-out remainder IS the general rule
    assert [p.level for p in e.chain] == [H.Level(i) for i in range(len(e.chain))] and e.chain[-1].level == H.Level.INTERACTION
    low = h.estimate({"market": "bull", "sector": "tech", "stock_type": "growth", "volatility": "low"})
    assert low.mean == pytest.approx(0.004, abs=0.006)                          # whichever twin was carved out, both are right
    assert sum(1 for p in h.qualified() if p[-1][0] == "volatility") == 1


def test_explained_share_and_contrast_table():
    h = H.KnowledgeHierarchy().fit(H.simulate_panel(0, cells(special={("bull", "tech"): 0.02}), n_days=150), NOW)
    share = h.explained_share()
    assert 0.12 < share < 0.22                                        # one of six equally observed cells is carved out
    ct = h.contrast_table()
    row = ct[ct.rule == H.node_id(BULL_TECH)].iloc[0]
    assert row.qualified and row.extra_dates_to_qualify == 0 and row.residual_mean > 0.015 > row.general_mean
    assert (ct[~ct.qualified].extra_dates_to_qualify.dropna() >= 0).all()
    assert H.KnowledgeHierarchy().explained_share() is None


def test_parameter_validation_catches_safeguard_disabling_typos():
    assert H.validate_params(H.PARAMS) == []
    bad = H.validate_params({**H.PARAMS, "min_clusters_override": 0, "alpha_override": 0.9, "n_full": "40"})
    assert len(bad) == 3
    for override in ({"w_min": 0.0}, {"alpha_override": 0.0}, {"ci_z": 99}):
        with pytest.raises(H.HierarchyError):
            H.KnowledgeHierarchy(override)


def test_qualification_is_not_fooled_by_a_neighbours_outlier():
    """Two ordinary sectors sit next to one real outlier: only the outlier may qualify, not its ordinary neighbours."""
    df = H.simulate_panel(11, cells(special={("bull", "tech"): 0.03}), n_days=200)
    h = H.KnowledgeHierarchy().fit(df, NOW)
    assert h.qualified() == [BULL_TECH]
    est = h.estimate({"market": "bull", "sector": "energy"})
    assert est.mean == pytest.approx(0.004, abs=0.003)                # NOT dragged up by the tech outlier in its own market
    assert h.estimate({"market": "bull", "sector": "tech"}).mean == pytest.approx(0.03, abs=0.004)


def test_two_outliers_in_different_branches_are_both_found():
    df = H.simulate_panel(12, cells(special={("bull", "tech"): 0.03, ("bear", "util"): -0.02}), n_days=220)
    h = H.KnowledgeHierarchy().fit(df, NOW)
    assert set(h.qualified()) == {BULL_TECH, (("market", "bear"), ("sector", "util"))}
    assert h.estimate({"market": "bear", "sector": "util"}).mean < 0 < h.estimate({"market": "bear", "sector": "energy"}).mean


# ------------------------------------------------------------------ per-level rules (section 19: six levels)

def five_level_panel(seed, tiny_depth=None, n_days=160):
    """Every dimension takes values a/b at random; optionally one TINY cell (5 dates, raw effect +0.08) sits at `tiny_depth`."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for _ in range(3):
            v = {dim: rng.choice(["a", "b"]) for dim in H.DIM_ORDER}
            rows.append({"when": pd.Timestamp("2020-01-01") + pd.Timedelta(days=d), "effect": 0.004 + rng.normal(0, 0.02), **v})
    df = pd.DataFrame(rows)
    if tiny_depth:
        tiny = []
        for d in range(5):
            for _ in range(3):
                v = {dim: "a" for dim in H.DIM_ORDER}
                v[H.DIM_ORDER[tiny_depth - 1]] = "z"
                tiny.append({"when": pd.Timestamp("2020-02-01") + pd.Timedelta(days=d), "effect": 0.08 + rng.normal(0, 0.02), **v})
        df = pd.concat([df, pd.DataFrame(tiny)], ignore_index=True)
    return df


def test_each_level_has_its_own_rules_and_deeper_is_stricter():
    h = H.KnowledgeHierarchy()
    t = h.level_rules_table()
    assert list(t.level) == [l.name for l in H.Level if l != H.Level.GENERAL]
    assert t.min_clusters.is_monotonic_increasing and t.w_min.is_monotonic_increasing and t.n_full.is_monotonic_increasing
    assert t.alpha.is_monotonic_decreasing and t.min_clusters.iloc[-1] > t.min_clusters.iloc[0]
    flat = H.KnowledgeHierarchy({"min_clusters_override": 30})
    assert set(flat.level_rules_table().min_clusters) == {30}          # an explicit flat threshold applies to every level
    assert h.rule(2, "n_full") == 40.0 and h.rule(5, "alpha") == 0.05
    assert H.validate_params({**H.PARAMS, "level_rules": {7: {}}}) and H.validate_params({**H.PARAMS, "level_rules": {2: {"min_clusters": 1}}})


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_tiny_sample_never_gives_extreme_confidence_at_every_level(depth):
    df = five_level_panel(depth, tiny_depth=depth)
    h = H.KnowledgeHierarchy().fit(df, NOW)
    path = tuple((H.DIM_ORDER[i], "a" if i < depth - 1 else "z") for i in range(depth))
    po = h.posterior(path)
    assert po is not None and po.clusters == 5 and po.raw_mean > 0.06
    assert not po.qualifies and po.weight_own <= 5 / (5 + h.rule(depth, "min_clusters")) + 1e-9
    assert po.post_mean < 0.5 * po.raw_mean                            # pulled well back toward the parent at THIS level
    ctx = {H.DIM_ORDER[i]: ("a" if i < depth - 1 else "z") for i in range(depth)}
    e = h.estimate(ctx)
    assert e.mean < 0.03 and e.p_sign_capped <= e.p_sign + 1e-12 and e.used_path != path
    assert any("does not override" in n for n in e.notes)


def test_a_real_effect_at_each_level_is_only_accepted_with_that_levels_evidence():
    rng = np.random.default_rng(3)
    for depth in (1, 3, 5):
        df = five_level_panel(30 + depth, n_days=360)
        mask = np.ones(len(df), bool)
        for i in range(depth):
            mask &= (df[H.DIM_ORDER[i]] == "a").to_numpy()
        df.loc[mask, "effect"] += 0.03
        h = H.KnowledgeHierarchy().fit(df, NOW)
        path = tuple((H.DIM_ORDER[i], "a") for i in range(depth))
        po = h.posterior(path)
        assert po.clusters >= 0.9 * h.rule(depth, "min_clusters")
        twins = [q for q in h.qualified() if len(q) == depth and q[:-1] == path[:-1]]     # a or its "b" twin carries the split
        assert twins or po.reason.startswith("explained"), (depth, po.reason)


def test_level_verdicts_and_guard_report_show_which_level_refused():
    df = five_level_panel(4, tiny_depth=3)
    h = H.KnowledgeHierarchy().fit(df, NOW)
    ctx = {"market": "a", "sector": "a", "stock_type": "z"}
    lv = h.level_verdicts(ctx)
    assert list(lv.level) == ["MARKET", "SECTOR", "STOCK_TYPE"] and lv.iloc[-1]["dates"] == 5
    assert lv.iloc[-1]["dates_needed"] == h.rule(3, "min_clusters") == 16 and lv.iloc[-1]["verdict"] == "too few independent dates"
    assert h.level_verdicts({"market": "q"}).iloc[0]["verdict"] == "no history"
    g = h.guard_report()
    assert g["STOCK_TYPE"]["largest_refused_raw"] > 0.06 and g["STOCK_TYPE"]["its_shrunk_value"] < 0.5 * g["STOCK_TYPE"]["largest_refused_raw"]
    assert H.KnowledgeHierarchy().guard_report() == {}


def test_confidence_curve_is_cautious_when_small_and_rises_with_dates():
    df = H.simulate_panel(0, cells(special={("bull", "tech"): 0.02}), n_days=300)
    cur = H.confidence_curve(df, {"market": "bull", "sector": "tech"}, NOW, steps=30)
    assert cur.iloc[0].dates < 12 and not cur.iloc[0].overrides and cur.iloc[0].weight_own < 0.5
    assert bool(cur.iloc[-1].overrides) and cur.iloc[-1].dates > 100
    assert cur.dates.is_monotonic_increasing and cur.p_sign_capped.iloc[-1] >= cur.p_sign_capped.iloc[0]
    assert cur.iloc[0].p_sign_capped <= 0.9


def test_sibling_table_lists_the_comparison_the_selection_makes():
    h = H.KnowledgeHierarchy().fit(H.simulate_panel(0, cells(special={("bull", "tech"): 0.02}), n_days=150), NOW)
    t = h.sibling_table(BULL_TECH[:1])
    assert list(t.rule) == ["rule:market=bull/sector=energy", "rule:market=bull/sector=tech", "rule:market=bull/sector=util"]
    assert list(t.qualifies) == [False, True, False] and not t.thin.any() and (t.tau > 0).all()
    assert h.sibling_table(BULL_TECH).empty
