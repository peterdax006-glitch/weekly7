"""Tests for engine.pattern_memory_eval on synthetic weekly panels with planted patterns."""
import numpy as np
import pandas as pd
import pytest

from engine import pattern_memory as pm
from engine import pattern_memory_eval as ev
from engine.pattern_lifecycle import Panel, parse_key_named

FRI = pd.date_range("2013-01-04", "2024-12-27", freq="W-FRI")


def make_panel(seed=0, n_tick=120, era_end=2016, noise_sd=0.03):
    """f0 q4 (top quintile) pays +0.5%/week always; f1 q0 pays +0.7%/week until era_end then nothing; f2 is pure noise."""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([FRI, [f"s{i}" for i in range(n_tick)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.normal(size=(len(idx), 3)).astype("float32"), index=idx, columns=["f0", "f1", "f2"])
    vix = pd.Series(np.sin(np.arange(len(FRI)) / 40.0), index=FRI)
    X["m_vix"] = vix.reindex(idx.get_level_values(0)).values.astype("float32")
    r = X.groupby(level=0).rank(pct=True)
    y = rng.normal(0, noise_sd, len(idx))
    y += 0.005 * (r["f0"].values > 0.8)
    y += 0.007 * (r["f1"].values <= 0.2) * (idx.get_level_values(0).year.values <= era_end)
    y = pd.Series(y, index=idx)
    return Panel.build(X, y, FRI[-1] + pd.Timedelta(days=30))


def found(blocks, keys=("f0 q4", "f1 q0", "f2 q1"), effects=(0.005, 0.007, 0.001)):
    fr = pd.DataFrame({"key_named": list(keys), "effect": list(effects)})
    return {k: (fr, 300) for k in range(len(blocks))}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    panel = make_panel()
    blocks = ev.block_schedule("2015-01-02", "2024-07-01")
    mem, reg, cache, qs = ev.build_memory(tmp_path_factory.mktemp("m"), panel, blocks, found(blocks))
    sc = ev.Scorer(panel, reg, cache, blocks, seed=1, n_random=30)
    return panel, blocks, mem, reg, cache, qs, sc


def test_block_schedule():
    b = ev.block_schedule("2016-01-01", "2017-07-01")
    assert [str(x.date()) for x in b] == ["2016-01-01", "2016-07-01", "2017-01-01", "2017-07-01"]
    assert ev.block_schedule("2020-01-01", "2019-01-01") == []


def test_quarter_stats_flags_thin_and_constant_quarters(world):
    qs = ev.Quarters(world[0])
    panel = world[0]
    m = np.full(len(panel.dates), 0.01)
    eff, t, n = ev.quarter_stats(m, qs)
    assert np.isnan(t).all()                                   # zero variance: no defined t, never a fake 99
    m2 = np.random.default_rng(0).normal(0.004, 0.01, len(panel.dates))
    eff, t, n = ev.quarter_stats(m2, qs)
    assert np.isfinite(t).sum() > qs.nq * 0.9 and abs(np.nanmean(eff) - 0.004) < 0.002
    m3 = m2.copy()
    m3[qs.codes == 5] = np.nan
    assert np.isnan(ev.quarter_stats(m3, qs)[1][5]) and ev.quarter_stats(m3, qs)[2][5] == 0


def test_names_roundtrip_and_random_patterns_are_valid(world):
    panel = world[0]
    rng = np.random.default_rng(0)
    for names in ev.random_patterns(panel, 40, rng):
        assert panel.mask(names) is not None
        assert parse_key_named(ev.names_to_key(names)) == tuple(names)
    assert len({n[0] for n in ev.random_patterns(panel, 60, rng)}) == 3


def test_only_matured_post_registration_quarters_are_ingested(world):
    panel, blocks, mem, reg, *_ = world
    runs = {}
    for r in mem._recs:
        if r["kind"] == "run":
            runs[r["body"]["run_id"]] = pd.Timestamp(r["body"]["run_now"])
    assert len(runs) == len(blocks) and mem.cumulative_tries()["total_tries"] == 300 * len(blocks)
    for r in mem._recs:
        if r["kind"] != "obs":
            continue
        b = r["body"]
        assert pd.Timestamp(b["mature_date"]) <= runs[b["run_id"]]                   # labels matured at the run
        assert pd.Timestamp(b["obs_date"]) > reg.items[b["key"]]["T_reg"]            # strictly after registration
    ks = [len([o for o in mem.records() if o["key"] == "f0 q4" and o["run_id"] == f"block{i:03d}"]) for i in range(len(blocks))]
    assert sum(ks) == len({o["obs_date"] for o in mem.records() if o["key"] == "f0 q4"})   # each quarter ingested once


def test_memory_uses_the_planted_universal_pattern_and_drops_the_dead_era_one(world):
    panel, blocks, mem, reg, cache, qs, sc = world
    late = mem.view(pd.Timestamp("2024-01-03"), sc.ctx[len(blocks) - 3])
    assert late.weights["f0 q4"].mode == "universal" and late.weights["f0 q4"].weight > 0.4
    assert late.weights["f1 q0"].weight == 0.0                                           # era ended in 2016
    assert late.weights["f2 q1"].weight == 0.0                                           # noise
    early = mem.view(pd.Timestamp("2016-01-03"), sc.ctx[2])
    assert early.weights["f1 q0"].weight > 0                                             # inside its era it is used


def test_scorer_ranks_used_above_disregarded_and_random_and_audit_is_clean(world):
    panel, blocks, mem, reg, cache, qs, sc = world
    tab = sc.score(mem, {"use_fdr": False})
    s = ev.summarise(tab, min_use=1)
    assert s["blocks_with_use"] >= 10
    assert s["use"] > 0.003 and s["use"] > s["off"] and s["use"] > s["random"]
    assert s["use_minus_random"] > 0.002
    mid = pd.Timestamp("2019-07-02")
    assert pm.audit_prefix_invariance(mem, [mid, "2022-01-03"], sc.ctx[10]) == []


def test_scorer_restores_memory_parameters(world):
    _, _, mem, _, _, _, sc = world
    before = dict(mem.p)
    sc.score(mem, {"n_universal": 99, "use_fdr": False}, [3, 4])
    assert mem.p == before


def test_fit_thresholds_uses_early_blocks_only_and_reports_both(world):
    panel, blocks, mem, reg, cache, qs, sc = world
    n = len(blocks) - 1
    fit_b, judge_b = list(range(0, n // 2)), list(range(n // 2, n))
    res = ev.fit_thresholds(mem, sc, fit_b, judge_b, n_configs=12, seed=3)
    assert set(res["chosen"]) == set(ev.FIT_GRID) and set(res["defaults"]) == set(ev.FIT_GRID)
    assert res["chosen_active"] is None or res["fit_ranking_active"][0]["mean_n_use"] >= 5
    assert set(res["judge_tables"]) >= {"chosen", "defaults"} and res["configs_tried"] >= 12
    assert res["fit_objective_chosen"] >= res["fit_objective_defaults"] - 1e-12         # best on the fit blocks by construction
    assert set(res["judge_tables"]["chosen"]["block"]) == set(judge_b)
    assert set(res["judge_tables"]["defaults"]["block"]) == set(judge_b)
    assert "use_minus_random" in res["judge_chosen"]
    with pytest.raises(ValueError):
        ev.fit_thresholds(mem, sc, [0, 1, 2, 9], [5, 6, 7], n_configs=2)               # overlapping / not later
    with pytest.raises(ValueError):
        ev.fit_thresholds(mem, sc, [4, 5], [4, 6], n_configs=2)


def test_objective_gives_zero_credit_when_the_view_is_switched_off(world):
    _, _, mem, _, _, _, sc = world
    off = sc.score(mem, {"n_universal": 99, "t_min_local": 99.0, "use_fdr": False}, list(range(4, 12)))
    assert (off["n_use"] == 0).all() and ev.objective(off) == 0.0
    assert ev.objective(pd.DataFrame()) == 0.0


def test_a_planted_leak_in_the_scorer_would_be_visible(world):
    """If the memory had ingested next-window quarters the used-minus-off gap would jump; prove the guard against that
    (mature <= run date) blocks ingesting a quarter that had not matured."""
    panel, blocks, mem, reg, cache, qs, sc = world
    q = qs.nq - 1
    with pytest.raises(pm.LeakError):
        mem.add_observations("leak", blocks[3], [{"key": "f0 q4", "obs_date": qs.last_date[q], "effect": .01, "n": 12, "t": 3.0}])


def test_empty_found_blocks_give_an_empty_but_valid_memory(tmp_path):
    panel = make_panel(n_tick=40)
    blocks = ev.block_schedule("2016-01-01", "2018-01-01")
    mem, reg, cache, qs = ev.build_memory(tmp_path, panel, blocks, {})
    assert len(reg) == 0 and mem.keys() == [] and mem.verify()["ok"]
    sc = ev.Scorer(panel, reg, cache, blocks, n_random=10)
    tab = sc.score(mem)
    assert (tab["n_view"] == 0).all() and ev.summarise(tab)["blocks_with_use"] == 0
    assert ev.sample_configs(ev.FIT_GRID, 10 ** 6, 0) and len(ev.sample_configs(ev.FIT_GRID, 5, 0)) == 5
    assert ev.sample_configs(ev.FIT_GRID, 5, 0) == ev.sample_configs(ev.FIT_GRID, 5, 0)
