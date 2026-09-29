"""Tests for engine.pattern_memory (C55-C60): accumulation, strict past-only views, relevance rules, cumulative tries,
consistency (C59), within-year fade and the reliability gate (C60), chain integrity, planted leaks."""
import dataclasses
import os

import numpy as np
import pandas as pd
import pytest

from engine import pattern_memory as pm
from engine.pattern_memory import LeakError, PatternMemory


def month_ends(y0, y1):
    return [pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) for y in range(y0, y1 + 1) for m in range(1, 13)]


def plant(mem, key, sched, run_id="r", run_now=None, seed=0, base_ctx=None, ctx_drift=0.0, n=20, noise=0.25):
    """sched: {year: signed monthly t}. Stores 12 month-end observations per year (effect sign follows t)."""
    rng = np.random.default_rng(seed)
    obs = []
    for d in month_ends(min(sched), max(sched)):
        if d.year not in sched:
            continue
        t = sched[d.year] + rng.normal(0, noise)
        ctx = {"m_vix": (base_ctx or 0.0) + ctx_drift * (d.year - min(sched)) + rng.normal(0, 0.2),
               "m_breadth": rng.normal(0, 0.2)}
        obs.append({"key": key, "obs_date": d, "effect": 0.002 * np.sign(t) * max(abs(t), 0.1), "n": n, "t": t, "ctx": ctx})
    rn = run_now or (max(o["obs_date"] for o in obs) + pd.Timedelta(days=15))
    return mem.add_observations(run_id, rn, obs)


@pytest.fixture
def mem(tmp_path):
    return PatternMemory(tmp_path / "pm")


# ---------------------------------------------------------------- accumulation, persistence, integrity
def test_accumulates_across_runs_and_persists(tmp_path):
    m = PatternMemory(tmp_path / "pm")
    plant(m, "ret_20d_q5", {2010: 1.2, 2011: 1.2}, run_id="run1")
    plant(m, "vol_q4 & m_vix_q4", {2012: 1.2}, run_id="run2")
    plant(m, "ret_5d_q1", {2010: 1.0}, run_id="run3")
    assert m.keys() == ["ret_20d_q5", "ret_5d_q1", "vol_q4 & m_vix_q4"]
    tries = m.cumulative_tries()
    assert tries["runs"] == 3 and tries["total_tries"] == 24 + 12 + 12 and tries["distinct_keys"] == 3
    re = PatternMemory(tmp_path / "pm")                       # reopen: everything is still there
    assert len(re) == len(m) and re.keys() == m.keys()
    assert re.verify()["ok"] and re.verify()["records"] == len(m)
    v = re.view("2013-01-15")
    assert set(v.weights) == set(m.keys())


def test_second_writer_sees_first_writers_records(tmp_path):
    a, b = PatternMemory(tmp_path / "pm"), PatternMemory(tmp_path / "pm")
    plant(a, "ret_20d_q5", {2010: 1.2}, run_id="a")
    plant(b, "ret_5d_q1", {2010: 1.2}, run_id="b")            # b syncs before appending: the chain stays linear
    a.refresh()
    assert a.verify()["ok"] and a.keys() == b.keys() == ["ret_20d_q5", "ret_5d_q1"]


def test_tampering_breaks_the_chain(tmp_path):
    m = PatternMemory(tmp_path / "pm")
    plant(m, "ret_20d_q5", {2010: 1.2})
    path = m.path
    raw = open(path, "rb").read()
    open(path, "wb").write(raw.replace(b'"n":20', b'"n":99', 1))    # rewrite history
    assert not PatternMemory.verify(m)["ok"]
    with pytest.raises(pm.ChainCorrupt):
        PatternMemory(tmp_path / "pm")


def test_empty_store_is_harmless(mem):
    v = mem.view("2020-01-01", {"m_vix": 1.0})
    assert len(v) == 0 and v.active() == {} and v.frame().empty
    assert mem.cumulative_tries()["total_tries"] == 0
    assert mem.timeline_summary("ret_20d_q5", "2020-01-01") is None
    assert mem.verify()["ok"]


# ---------------------------------------------------------------- no future (C55, C56, C58)
def test_same_year_rerun_at_early_date_sees_only_matured_evidence(mem):
    plant(mem, "ret_20d_q5", {2019: 1.3, 2020: 1.3}, run_id="full_2020")      # the run saw all of 2020
    early = pd.Timestamp("2020-03-10")
    v = mem.view(early)
    used = [o for o in mem.records() if pd.Timestamp(o["mature_date"]) < early]
    assert mem.last_audit["n_used"] == len(used)
    assert pd.Timestamp(mem.last_audit["max_mature"]) < early
    assert v.weights["ret_20d_q5"].n_obs == len(used)
    late = mem.view("2021-01-30")
    assert late.weights["ret_20d_q5"].n_obs == 24 and v.weights["ret_20d_q5"].n_obs < 24


def test_h2_only_evidence_is_invisible_in_h1(mem):
    obs = [{"key": "vol_q5", "obs_date": d, "effect": 0.01, "n": 20, "t": 2.5, "ctx": {}} for d in month_ends(2020, 2020)[6:]]
    mem.add_observations("r", "2021-01-31", obs)
    assert len(mem.view("2020-06-30")) == 0                    # nothing of H2 leaks into June
    assert len(mem.view("2020-09-01")) == 1                    # July matured, August not yet


def test_maturity_is_strict(mem):
    d = pd.Timestamp("2020-03-02")                             # Monday; horizon 5 -> matures Monday 2020-03-09
    mem.add_observations("r", "2020-04-01", [{"key": "ret_5d_q1", "obs_date": d, "effect": 0.01, "n": 20, "t": 3.0}])
    assert len(mem.view("2020-03-09")) == 0                    # mature_date == real_now is NOT usable
    assert len(mem.view("2020-03-10")) == 1


def test_guard_blocks_a_planted_leak(mem, monkeypatch):
    plant(mem, "ret_20d_q5", {2020: 1.3})
    monkeypatch.setattr(mem, "_matured", lambda now: mem.records())      # a buggy filter that returns everything
    with pytest.raises(LeakError):
        mem.view("2020-06-01")
    with pytest.raises(LeakError):
        pm.check_no_future([{"key": "k", "mature_date": "2020-06-01"}], "2020-06-01")


def test_write_refuses_labels_not_yet_realised_at_run_date(mem):
    with pytest.raises(LeakError):
        mem.add_observations("r", "2020-03-04", [{"key": "ret_5d_q1", "obs_date": "2020-03-02", "effect": .01, "n": 5, "t": 2}])
    with pytest.raises(LeakError):                             # a claimed maturity earlier than obs + horizon
        mem.add_observations("r", "2020-06-01", [{"key": "ret_5d_q1", "obs_date": "2020-03-02", "effect": .01, "n": 5,
                                                  "t": 2, "mature_date": "2020-03-03"}])
    assert len(mem.records()) == 0


def test_no_dates_and_no_identities_reach_the_trader(mem):
    plant(mem, "ret_20d_q5", {2018: 1.3, 2019: 1.3, 2020: 1.3})
    v = mem.view("2020-06-01", {"m_vix": 0.1})
    pm.scan_for_dates(v)                                       # passes
    s = mem.timeline_summary("ret_20d_q5", "2020-06-01")
    pm.scan_for_dates(s)
    bad = dataclasses.replace(v.weights["ret_20d_q5"], reason="held until 2020-03-31")
    with pytest.raises(LeakError):
        pm.scan_for_dates(PatternWeightBox(bad))
    with pytest.raises(LeakError):
        pm.scan_for_dates({"x": pd.Timestamp("2020-01-01")})
    for k in ("AAPL & ret_5d_q1", "ret_5d_q1 & 2020-03-01", "vol_q5 in 2019"):
        with pytest.raises(pm.KeyError_):
            mem.add_observations("r", "2021-01-01", [{"key": k, "obs_date": "2020-03-02", "effect": .01, "n": 5, "t": 2}])
    with pytest.raises(pm.KeyError_):                          # caller-supplied identity list
        PatternMemory(mem.root + "_2", forbidden_identities=["acme"]).add_observations(
            "r", "2021-01-01", [{"key": "acme_flag", "obs_date": "2020-03-02", "effect": .01, "n": 5, "t": 2}])


@dataclasses.dataclass
class PatternWeightBox:
    w: object


# ---------------------------------------------------------------- repetition does not inflate evidence (C57)
def test_same_year_rerun_many_times_does_not_inflate_t(mem):
    plant(mem, "ret_20d_q5", {2019: 1.0, 2020: 1.0}, run_id="r0", seed=1)
    w0 = mem.view("2021-02-15").weights["ret_20d_q5"]
    for i in range(1, 40):
        plant(mem, "ret_20d_q5", {2019: 1.0, 2020: 1.0}, run_id=f"r{i}", seed=1)     # identical evidence, 39 more times
    w1 = mem.view("2021-02-15").weights["ret_20d_q5"]
    assert w1.n_obs == w0.n_obs == 24
    assert w1.pooled_t == pytest.approx(w0.pooled_t)
    assert mem.cumulative_tries()["runs"] == 40                # but every rerun is still counted as tries


# ---------------------------------------------------------------- relevance from the timeline
def test_relevance_ranks_recent_similar_above_distant_era(mem):
    plant(mem, "recent_q5", {2018: 1.2, 2019: 1.2}, base_ctx=1.0, seed=2, run_id="a")
    plant(mem, "ancient_q5", {2004: 1.2, 2005: 1.2}, base_ctx=-2.0, seed=3, run_id="b")
    v = mem.view("2020-03-02", {"m_vix": 1.0, "m_breadth": 0.0})
    a, b = v.weights["recent_q5"], v.weights["ancient_q5"]
    assert a.weight > 0 and a.mode == "local"
    assert a.weight > b.weight and b.weight == 0.0 and b.mode == "disregarded"
    assert a.components["ctx"] > 0.5 and a.components["time"] > 0.4


def test_context_similarity_raises_relevance(mem):
    plant(mem, "calm_q5", {2018: 1.2, 2019: 1.2}, base_ctx=-1.0, seed=4)
    calm = mem.view("2020-03-02", {"m_vix": -1.0, "m_breadth": 0.0}).weights["calm_q5"]
    storm = mem.view("2020-03-02", {"m_vix": 2.5, "m_breadth": 0.0}).weights["calm_q5"]
    assert calm.components["ctx"] > storm.components["ctx"]
    assert calm.weight > storm.weight


def test_more_distinct_periods_raise_relevance(mem):
    plant(mem, "one_q5", {2019: 1.2}, seed=5, run_id="a")
    plant(mem, "three_q5", {2017: 1.2, 2018: 1.2, 2019: 1.2}, seed=5, run_id="b")
    v = mem.view("2020-02-01")
    assert v.weights["three_q5"].components["persist"] > v.weights["one_q5"].components["persist"]
    assert v.weights["three_q5"].weight > v.weights["one_q5"].weight


# ---------------------------------------------------------------- multiple testing (C57)
def test_cumulative_tries_block_best_of_many_noise(mem):
    marginal = {2019: 0.95}                                    # single-period t ~ 3.3: passes if tried once
    plant(mem, "lucky_q5", marginal, seed=6, noise=0.0)
    assert mem.view("2020-02-01").weights["lucky_q5"].weight > 0
    for i in range(50):
        mem.record_tries(f"search{i}", "2020-01-20", 200)      # 10,000 further candidates tried over the same data
    w = mem.view("2020-02-01").weights["lucky_q5"]
    assert mem.cumulative_tries()["total_tries"] > 10000
    assert w.weight == 0 and w.reason == "not_significant_after_all_tries" and w.q_value > 0.1
    assert mem.cumulative_tries()["expected_best_null_t"] > 4.2


def test_a_real_pattern_survives_a_huge_try_count(mem):
    plant(mem, "real_q5", {y: 1.5 for y in range(2010, 2020)}, seed=7)
    mem.record_tries("big", "2020-01-20", 1_000_000)
    w = mem.view("2020-02-01").weights["real_q5"]
    assert w.mode == "universal" and w.weight > 0.5 and w.q_value < 0.01


def test_bh_adjust_is_monotone_and_conservative():
    q = pm.bh_adjust([0.001, 0.02, 0.5], 100)
    assert q[0] == pytest.approx(0.1) and q[1] > q[0] and q[2] == 1.0
    assert pm.bh_adjust([], 10).size == 0


# ---------------------------------------------------------------- C59: consistency across periods
def test_consistent_pattern_is_relevant_in_every_year_including_distant_eras(mem):
    plant(mem, "always_q5", {y: 1.3 for y in range(2000, 2020)}, seed=8, ctx_drift=0.5)
    weights = {}
    for yr in (2008, 2012, 2016, 2019):
        v = mem.view(pd.Timestamp(yr, 6, 15), {"m_vix": -9.0, "m_breadth": 5.0})       # an era unlike anything it saw
        w = v.weights["always_q5"]
        assert w.mode == "universal" and w.weight > 0.5, (yr, w)
        weights[yr] = w.weight
    assert max(weights.values()) - min(weights.values()) < 0.25       # no decay with era distance


def test_era_specific_pattern_is_used_only_where_it_held(mem):
    plant(mem, "era_q5", {2006: 1.3, 2007: 1.3, 2008: 1.3, 2009: 1.3}, seed=9, run_id="a")
    plant(mem, "era_q5", {y: 0.0 for y in range(2010, 2020)}, seed=10, run_id="b", noise=0.3)   # then nothing
    inside = mem.view("2008-06-15").weights["era_q5"]
    just_after = mem.view("2010-03-01").weights["era_q5"]
    far_after = mem.view("2016-06-15").weights["era_q5"]
    assert inside.weight > 0 and inside.mode == "local"
    assert just_after.weight > 0
    assert far_after.weight == 0 and far_after.mode == "disregarded"
    assert far_after.reason in ("lapsed", "stale_for_this_period", "weak_or_inconsistent")
    assert mem.timeline_summary("era_q5", "2016-06-15").effect.size == 12 * 10 + 5       # retained: 2006-2015 + Jan-May 2016
    assert "era_q5" in mem.keys()


def test_recent_break_excludes_the_pattern_for_now_but_keeps_it_and_it_can_requalify(mem):
    plant(mem, "break_q5", {y: 1.3 for y in range(2000, 2015)}, seed=11, run_id="hist")
    assert mem.view("2014-06-15").weights["break_q5"].mode == "universal"
    plant(mem, "break_q5", {2015: -1.3, 2016: -1.3}, seed=12, run_id="broke")
    w = mem.view("2016-06-15").weights["break_q5"]
    assert w.weight == 0 and w.reason in ("failed_recently", "weak_or_inconsistent", "broke_within_year")
    n_stored = sum(o["key"] == "break_q5" for o in mem.records())
    assert n_stored == 12 * 17                                                      # nothing deleted
    assert mem.timeline_summary("break_q5", "2016-06-15") is not None
    plant(mem, "break_q5", {2017: 1.3, 2018: 1.3, 2019: 1.3}, seed=13, run_id="restored")
    assert mem.view("2020-03-01").weights["break_q5"].weight > 0               # requalified from new matured evidence


def test_views_at_an_early_date_are_identical_whether_or_not_later_evidence_exists(tmp_path):
    a, b = PatternMemory(tmp_path / "a"), PatternMemory(tmp_path / "b")
    plant(a, "k_q5", {y: 1.3 for y in range(2000, 2015)}, seed=14, run_id="x", run_now="2015-01-20")
    plant(b, "k_q5", {y: 1.3 for y in range(2000, 2015)}, seed=14, run_id="x", run_now="2015-01-20")
    plant(b, "k_q5", {2015: -3.0, 2016: -3.0, 2017: -3.0}, seed=15, run_id="later")     # evidence from the future of 2014
    plant(b, "later_q1", {2016: 2.0}, seed=16, run_id="later2")
    for d in ("2013-06-15", "2014-06-15", "2014-12-15"):
        va, vb = a.view(d, {"m_vix": .3}).weights["k_q5"], b.view(d, {"m_vix": .3}).weights["k_q5"]
        assert va.weight == pytest.approx(vb.weight) and va.mode == vb.mode and va.pooled_t == pytest.approx(vb.pooled_t)
        assert "later_q1" not in b.view(d).weights


# ---------------------------------------------------------------- C60: within a year and the gate
def test_pattern_found_early_fades_within_the_year_as_the_break_matures(mem):
    plant(mem, "hist_q5", {y: 1.3 for y in range(2012, 2020)}, seed=17, run_id="h")
    sched = [(pd.Timestamp(2020, m, 1) + pd.offsets.MonthEnd(0), 1.3 if m <= 3 else -1.6) for m in range(1, 13)]
    mem.add_observations("y2020", "2021-01-31", [{"key": "hist_q5", "obs_date": d, "effect": .002 * np.sign(t), "n": 20, "t": t,
                                                  "ctx": {"m_vix": 0.0}} for d, t in sched])
    ws = [mem.view(pd.Timestamp(2020, m, 15)).weights["hist_q5"].weight for m in (4, 5, 6, 7, 10)]
    assert ws[0] > 0.3                                                      # break not yet matured in April
    assert ws[0] >= ws[1] > ws[2] or ws[2] == 0                             # fading as evidence arrives
    assert ws[-1] == 0 and mem.view("2020-10-15").weights["hist_q5"].reason in ("broke_within_year", "failed_recently")
    assert ws == sorted(ws, reverse=True)                                   # monotone loss of relevance through the year


def test_gate_scales_switches_off_and_never_deletes(mem):
    plant(mem, "g_q5", {y: 1.3 for y in range(2010, 2020)}, seed=18, base_ctx=0.0)
    seen = []

    def gate(key, ctx_now, summ):
        seen.append((key, dict(ctx_now), summ))
        return 0.0 if ctx_now.get("m_vix", 0) > 1.0 else 0.5

    calm = mem.view("2020-06-01", {"m_vix": 0.0}, gate=gate).weights["g_q5"]
    storm = mem.view("2020-06-01", {"m_vix": 2.0}, gate=gate).weights["g_q5"]
    plain = mem.view("2020-06-01", {"m_vix": 0.0}).weights["g_q5"]
    assert calm.weight == pytest.approx(plain.weight * 0.5) and calm.components["gate"] == 0.5
    assert storm.weight == 0 and storm.mode == "gated" and "g_q5" in mem.keys()
    key, ctx, summ = seen[0]
    assert key == "g_q5" and summ.ctx_cols == ("m_breadth", "m_vix") and summ.period_t.size == 10
    assert summ.ctx.shape == (120, 2) and summ.period_ctx.shape == (10, 2) and (summ.period_t > 0).all()


def test_gate_inputs_are_past_only_and_outputs_validated(mem):
    plant(mem, "g_q5", {y: 1.3 for y in range(2010, 2021)}, seed=19, run_now="2021-01-31")
    counts = []
    mem.view("2020-06-15", {"m_vix": 0.0}, gate=lambda k, c, s: counts.append(int(s.effect.size)) or 1.0)
    assert counts == [12 * 10 + 5]                                          # Jan-May 2020 only: June is not yet mature
    for bad in (1.5, -0.1, float("nan"), "x", None, True):
        with pytest.raises(pm.MemoryError_):
            mem.view("2020-06-15", gate=lambda k, c, s, b=bad: b)
    with pytest.raises(LeakError):                                          # a dated context cannot be smuggled in
        mem.view("2020-06-15", {"m_vix": pd.Timestamp("2020-06-20")}, gate=lambda k, c, s: 1.0)


def test_gate_is_not_called_for_disregarded_patterns(mem):
    plant(mem, "dead_q5", {2019: -0.2}, seed=20)
    calls = []
    v = mem.view("2020-03-01", gate=lambda k, c, s: calls.append(k) or 1.0)
    assert calls == [] and v.weights["dead_q5"].weight == 0


# ---------------------------------------------------------------- analytics
def test_timeline_analytics(mem):
    plant(mem, "t_q5", {2015: 1.3, 2016: 1.3, 2017: -1.3, 2018: 1.3}, seed=21)
    an = pm.Analytics(mem)
    fn = an.first_noticed("t_q5", "2019-06-01")
    assert fn["first_obs_date"] == "2015-01-31" and fn["first_run"] == "r"
    assert an.first_noticed("t_q5", "2014-01-01") is None                   # not yet noticed at that time
    hf = an.held_failed("t_q5", "2019-06-01")
    assert hf["held_years"] == [2015, 2016, 2018] and hf["failed_years"] == [2017]
    tl = an.timeline("t_q5", "2016-06-01")
    assert tl["year"].tolist() == [2015, 2016] and (tl["status"] == "held").all()      # bounded by as_of
    assert an.relevance_to_year("t_q5", 2016) is not None and an.relevance_to_year("t_q5", 2015) is None
    rep = an.report("2019-06-01")
    assert rep.loc[0, "key"] == "t_q5" and rep.loc[0, "first_noticed"] == "2015-01-31"
    assert an.report("2010-01-01").empty


# ---------------------------------------------------------------- feeding from a panel and from the miner
def test_observe_panel_finds_a_planted_effect_and_only_uses_matured_dates(mem):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2019-01-01", "2020-06-30")
    tick = [f"s{i}" for i in range(40)]
    idx = pd.MultiIndex.from_product([dates, tick], names=["date", "ticker"])
    x = pd.Series(rng.normal(size=len(idx)), index=idx)
    y = pd.Series(rng.normal(0, 0.03, len(idx)) + 0.02 * (x.values > 0.8), index=idx)
    noise_mask = pd.Series(rng.normal(size=len(idx)) > 0.8, index=idx)
    run_now = pd.Timestamp("2020-06-30")
    n1 = mem.observe_panel("feat_q5", x > 0.8, y, "p1", run_now)
    mem.observe_panel("junk_q5", noise_mask, y, "p1", run_now)
    assert n1 >= 15
    obs = [o for o in mem.records() if o["key"] == "feat_q5"]
    assert all(pd.Timestamp(o["mature_date"]) <= run_now for o in obs)
    assert np.mean([o["t"] for o in obs]) > 2.0
    junk = [o["t"] for o in mem.records() if o["key"] == "junk_q5"]
    assert abs(np.mean(junk)) < 1.0
    v = mem.view("2020-07-15")
    assert v.weights["feat_q5"].weight > 0 and v.weights["junk_q5"].weight == 0


def test_ingest_miner_frame(mem):
    frame = pd.DataFrame({"key_named": ["ret_20d_q5", "vol_q1 & m_vix_q4", "junk_q3"], "effect": [0.004, -0.003, 0.0001],
                          "t_conf": [3.5, -3.0, 0.4], "n_eff": [40, 40, 40],
                          "status": ["active", "rescoped", "rejected"]})
    assert mem.ingest_miner(frame, "miner1", pd.Timestamp("2020-06-30"), ctx={"m_vix": 0.2}) == 2
    assert mem.cumulative_tries()["total_tries"] == 3               # the rejected candidate still counts as a try
    assert mem.ingest_miner(pd.DataFrame(), "miner2", "2020-07-30", n_tried=500) == 0
    assert mem.cumulative_tries()["total_tries"] == 503
    v = mem.view("2020-08-30")
    assert v.weights["vol_q1 & m_vix_q4"].direction == -1


# ---------------------------------------------------------------- self-audit and reports
def test_prefix_invariance_audit_is_clean_on_a_correct_store_and_catches_a_leaky_view(mem, monkeypatch):
    plant(mem, "a_q5", {y: 1.3 for y in range(2005, 2018)}, seed=30, run_id="a")
    plant(mem, "b_q5", {2010: 1.3, 2011: 1.3, 2012: 1.3}, seed=31, run_id="b")
    dates = ["2008-03-01", "2011-06-15", "2012-12-31", "2016-09-01"]
    assert pm.audit_prefix_invariance(mem, dates, {"m_vix": 0.0}) == []
    real = PatternMemory._matured

    def leaky(self, now):                                        # sees 60 days past the date: a lookahead bug
        return real(self, pd.Timestamp(now) + pd.Timedelta(days=60))
    monkeypatch.setattr(PatternMemory, "_matured", leaky)
    with pytest.raises(LeakError):                               # the guard stops it first ...
        mem.view("2011-06-15")
    monkeypatch.setattr(pm, "check_no_future", lambda *a, **k: True)     # ... and with the guard off the audit still finds it
    monkeypatch.setattr(PatternMemory, "_matured", lambda self, now: real(self, pd.Timestamp(now) + pd.Timedelta(days=60))
                        if self is mem else real(self, now))
    assert pm.audit_prefix_invariance(mem, ["2011-06-15", "2012-12-31"], {"m_vix": 0.0}) != []


def test_era_breakdown_and_markdown(mem):
    plant(mem, "all_q5", {y: 1.3 for y in range(2000, 2020)}, seed=32, run_id="a")
    plant(mem, "early_q5", {y: 1.3 for y in range(2000, 2005)}, seed=33, run_id="b")
    an = pm.Analytics(mem)
    eb = an.era_breakdown("2020-03-01", {"early": (2000, 2004), "late": (2015, 2019)})
    got = {(r.key, r.era): (r.held, r.failed) for r in eb.itertuples()}
    assert got[("all_q5", "early")] == (5, 0) and got[("all_q5", "late")] == (5, 0)
    assert ("early_q5", "late") not in got
    md = an.markdown("2020-03-01", eras={"early": (2000, 2004)})
    assert "cumulative candidate tries" in md and "all_q5" in md and "universal" in md


def test_holiday_calendar_delays_maturity(tmp_path):
    m = PatternMemory(tmp_path / "h", holidays=["2020-03-04"])
    m.add_observations("r", "2020-04-01", [{"key": "ret_5d_q1", "obs_date": "2020-03-02", "effect": .01, "n": 20, "t": 3.0}])
    assert m.records()[0]["mature_date"] == "2020-03-10"        # Mon + 5 sessions, one of them a holiday
    assert len(m.view("2020-03-10")) == 0 and len(m.view("2020-03-11")) == 1


def test_conflicting_reobservation_uses_the_latest_and_keeps_history(mem):
    d = pd.Timestamp("2020-02-28")
    for i, t in enumerate((3.0, 0.5)):
        mem.add_observations(f"run{i}", "2020-06-01", [{"key": "ret_5d_q1", "obs_date": d, "effect": .01, "n": 20, "t": t}])
    assert len(mem.records()) == 2                               # append-only: both stay in the log
    assert mem.timeline_summary("ret_5d_q1", "2020-05-01").t.tolist() == [pytest.approx(0.5)]
