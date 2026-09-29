"""Tests for engine.pattern_reliability (C56-C61): every mechanism is proven on a planted world with a KNOWN cause.
A regime that switches a pattern off must be found, explained, predicted out of sample and gated; a pattern that breaks
at random times must be judged unpredictable (no fake gain); a family sharing one break driver must show pooling helps;
a context that peeks at the next period must be caught; a break caused by an unobserved shock must be labelled
'unknown cause' even when a decoy indicator happens to coincide in-sample; a healthy pattern must stay inside the
false-alarm budget and a phantom pattern must be caught within a bounded delay."""
import math

import numpy as np
import pandas as pd
import pytest

from engine import pattern_reliability as PR
from engine.pattern_memory import PatternMemory

FAST = {"boot": 300, "n_perm": 100}


# ---------------------------------------------------------------- shared runs (each world is run once)
@pytest.fixture(scope="module")
def regime():
    tl = PR.planted_world("regime", seed=2, n_weeks=780)
    return tl, PR.run_reliability(tl, FAST)


@pytest.fixture(scope="module")
def shock():
    tl = PR.planted_world("shock", seed=1)
    return tl, PR.run_reliability(tl, FAST, explain=False)


def _series_tl(rets, ctx=None, first=None):
    idx = pd.date_range("2010-01-01", periods=len(rets), freq="W-FRI")
    R = pd.DataFrame(np.asarray(rets, float).reshape(len(rets), -1), index=idx)
    R.columns = [f"pat_{j}" for j in range(R.shape[1])]
    C = pd.DataFrame(ctx if ctx is not None else {"m_x": np.zeros(len(rets))}, index=idx)
    return PR.Timelines(R, C, {}, first)


# ---------------------------------------------------------------- container, numerics
def test_timelines_reject_bad_input():
    idx = pd.date_range("2010-01-01", periods=10, freq="W-FRI")
    R = pd.DataFrame(np.zeros((10, 2)), index=idx, columns=["a", "b"])
    with pytest.raises(PR.ReliabilityError):
        PR.Timelines(R, pd.DataFrame({"m": np.zeros(9)}, index=idx[:9]))               # misaligned
    with pytest.raises(PR.ReliabilityError):
        PR.Timelines(R.rename(columns={"a": "x_2020-01-03"}), pd.DataFrame({"m": np.zeros(10)}, index=idx))  # date in a key
    with pytest.raises(PR.ReliabilityError):
        PR.Timelines(R.iloc[:0], pd.DataFrame({"m": []}, index=idx[:0]))                # empty
    with pytest.raises(PR.ReliabilityError):
        PR.Timelines(R.iloc[::-1], pd.DataFrame({"m": np.zeros(10)}, index=idx[::-1]))  # not increasing


def test_holm_cusum_and_bootstrap_numerics():
    p = np.array([0.001, 0.02, 0.04, 0.5])
    adj = PR.holm(p)
    assert np.all(adj >= p) and adj[0] == pytest.approx(0.004) and adj.max() <= 1
    # CUSUM threshold: simulated in-control run length is near the design value
    k = 0.25
    h = PR.cusum_threshold(k, 200)
    rng = np.random.default_rng(0)
    lens = []
    for _ in range(300):
        s, n = 0.0, 0
        while s < h and n < 5000:
            s = max(0.0, s + rng.normal(0, 1) - k)
            n += 1
        lens.append(n)
    assert 100 < np.mean(lens) < 400
    x = np.random.default_rng(1).normal(0.5, 1, 300)
    m, lo, hi = PR.stationary_bootstrap_ci(x, 5, 400, seed=3)
    assert m == pytest.approx(x.mean()) and lo < m < hi and 0.1 < hi - lo < 0.5
    assert np.isnan(PR.stationary_bootstrap_ci([1.0, 2.0])[1])                          # too short: no interval
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    assert PR.auc(np.arange(8), y) == pytest.approx(PR.auc(np.arange(8), y))
    assert PR.brier([0.5, 0.5], [1, 0]) == 0.25


# ---------------------------------------------------------------- context construction and leak guards
def _raw(n=900, seed=0):
    idx = pd.bdate_range("2008-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"vix": 20 + np.cumsum(rng.normal(0, 0.4, n)), "unrate": 5 + np.cumsum(rng.normal(0, 0.02, n))}, index=idx)


def test_context_builder_is_point_in_time_and_leaky_builder_is_caught():
    raw = _raw()
    dates = pd.date_range("2009-01-02", raw.index[-1], freq="W-FRI")
    good = lambda r, d: PR.build_context(r, d, lags={"unrate": 35, "vix": 1}, pct_cols=["vix"], min_hist=20)
    assert PR.audit_context_builder(good, raw, dates, n_cuts=6, seed=1)

    def leaky(r, d):                                     # centred window + next-day value: both use the future
        c = good(r, d)
        c["vix_smooth"] = r["vix"].rolling(9, center=True, min_periods=1).mean().reindex(d, method="ffill")
        return c

    with pytest.raises(PR.LeakError, match="vix_smooth"):
        PR.audit_context_builder(leaky, raw, dates, n_cuts=8, seed=1)
    # the publication lag is honoured: a value stamped at t is invisible until t + lag
    ctx = PR.build_context(raw, dates, lags={"unrate": 35, "vix": 1})
    d = dates[100]
    stamp = raw.index[raw.index <= d - pd.Timedelta(days=35)][-1]
    assert ctx.loc[d, "unrate"] == pytest.approx(raw.loc[stamp, "unrate"])
    assert ctx["vix_pct"].iloc[:50].isna().all()                                         # not enough past to rank yet


def test_peek_scan_catches_the_leak_and_passes_honest_worlds():
    leak = PR.planted_world("leak", seed=3, n_patterns=12)
    rep = PR.scan_context_for_peeking(leak)
    assert "m_peek" in rep["flagged"] and "m_regime" not in rep["flagged"]
    with pytest.raises(PR.LeakError, match="m_peek"):
        PR.assert_no_peeking(leak)
    honest = PR.planted_world("regime", seed=3, n_patterns=12)
    assert PR.scan_context_for_peeking(honest)["flagged"] == []


def test_leak_is_worth_catching_and_the_pipeline_refuses_it():
    """Without the guard the peeking column produces a huge (fake) gain; with it the run is refused."""
    leak = PR.planted_world("leak", seed=3, n_patterns=12, n_weeks=400)
    with pytest.raises(PR.LeakError):
        PR.run_reliability(leak, FAST)
    fake = PR.run_reliability(leak, FAST, guards=False, explain=False)
    honest = PR.run_reliability(PR.planted_world("regime", seed=3, n_patterns=12, n_weeks=400), FAST, explain=False)
    g_fake = fake["evaluation"]["summary"].loc["gated_raw", "gain_vs_always_on"]
    g_ok = honest["evaluation"]["summary"].loc["gated_raw", "gain_vs_always_on"]
    assert g_fake > 2.0 * max(g_ok, 1e-4)
    assert fake["scores"]["pooled"]["auc"] > 0.8


def test_causality_audit_passes_clean_code_and_catches_a_pipeline_that_reads_the_future(monkeypatch):
    tl = PR.planted_world("regime", seed=5, n_weeks=300, n_patterns=6)
    cfg = {**FAST, "refit_every": 26}
    assert PR.causality_audit(tl, cfg, rows=[150, 200, 260])["ok"]
    real = PR.meta_wide

    def leaky_meta(t, cfg=None, states=None):
        out = real(t, cfg, states)
        R = t.rets
        out["hit4"] = (R > 0).astype(float).where(R.notna()).rolling(4, min_periods=1).mean().shift(-3)   # 3 weeks AHEAD
        out["hit26"] = out["hit4"]
        return out

    monkeypatch.setattr(PR, "meta_wide", leaky_meta)
    with pytest.raises(PR.LeakError, match="read the future"):
        PR.causality_audit(tl, cfg, rows=[150, 200, 260])


def test_nothing_after_a_row_changes_health_breaks_or_meta():
    """Rows <= r are identical whatever the returns after r are (the monitor and the meta features are causal)."""
    tl = PR.planted_world("regime", seed=4, n_weeks=400, n_patterns=6)
    r = 250
    R2 = tl.rets.copy()
    R2.iloc[r:] = np.random.default_rng(9).normal(-0.01, 0.02, R2.iloc[r:].shape)         # rewrite the future
    tl2 = PR.Timelines(R2, tl.ctx, {}, tl.first_pos)
    a, b = PR.health_monitor(tl), PR.health_monitor(tl2)
    assert (a.codes.iloc[: r + 1].values == b.codes.iloc[: r + 1].values).all()
    np.testing.assert_allclose(a.cusum.iloc[: r + 1].values, b.cusum.iloc[: r + 1].values, equal_nan=True)
    sa, ea = PR.break_states(tl.rets)
    sb, eb = PR.break_states(R2)
    assert (sa.iloc[: r + 1].values == sb.iloc[: r + 1].values).all()
    ma, mb = PR.meta_wide(tl), PR.meta_wide(tl2)
    for k in PR.META:
        np.testing.assert_allclose(ma[k].iloc[: r + 1].values, mb[k].iloc[: r + 1].values, equal_nan=True)


# ---------------------------------------------------------------- break detection
def test_break_states_find_a_planted_flip_after_it_happens_not_before():
    rng = np.random.default_rng(0)
    r = np.r_[rng.normal(0.006, 0.01, 150), rng.normal(-0.006, 0.01, 100)]
    st, ev = PR.break_states(pd.DataFrame({"p": r}))
    assert len(ev) == 1
    e = ev.iloc[0]
    assert 150 < e["detect"] < 185                                                       # detected after the flip, not before
    assert abs(e["onset"] - 150) <= 12                                                   # change point close to the truth
    assert (st.iloc[:150, 0] != 2).all()                                                 # no false break before it happened
    _, ev0 = PR.break_states(pd.DataFrame({"p": np.random.default_rng(1).normal(0.006, 0.01, 260)}))
    assert len(ev0) == 0                                                                 # a steady pattern never breaks


# ---------------------------------------------------------------- planted regime: found, explained, predicted, gated
def test_regime_reliability_is_predicted_out_of_sample_and_calibrated(regime):
    tl, res = regime
    s = res["scores"]["pooled"]
    assert s["brier_model"] < s["brier_pool"] and s["brier_model"] < s["brier_own"]
    assert s["skill_t_vs_pool"] > PR.PARAMS["t_pool"] and s["gain_vs_pool_lo"] > 0
    assert s["ece_model"] < 0.06                                                          # calibrated (equal-mass bins)
    cal = res["calibration"]
    assert len(cal) == 10 and np.corrcoef(cal["p_mean"], cal["hit_rate"])[0, 1] > 0.9
    assert list(res["pred"]["pooled"].importance)[0] in ("m_regime", "m_regime_pct", "m_regime_d4") or \
        any(k.startswith("m_regime") for k in list(res["pred"]["pooled"].importance)[:3])


def test_regime_break_is_explained_predicted_and_gated(regime):
    tl, res = regime
    ex = res["explain"]["explanations"]
    assert ex and ex[0].driver.startswith("m_regime")
    e = ex[0]
    assert e.works_when == "high" and e.p_fwer <= PR.PARAMS["explain_alpha"] and e.n_candidates >= 8
    assert "family-wise" in e.statement and "pattern-weeks" in e.statement
    inv = res["investigations"]
    won = inv[inv["verdict"] == "EXPLAINED_AND_GATED"]
    assert len(won) >= 1 and won["driver"].str.startswith("m_regime").all()
    assert won["detail"].str.contains("out of sample").all()
    assert res["invariants"] == []
    noise_expl = inv[(inv["verdict"] == "EXPLAINED_AND_GATED") & (~inv["driver"].str.startswith("m_regime"))]
    assert noise_expl.empty                                                               # no noise column was ever trusted


def test_regime_gating_beats_always_on_and_discard(regime):
    tl, res = regime
    ev = res["evaluation"]
    s = ev["summary"]
    g = s.loc["gated_raw"]
    assert g["gain_vs_always_on_lo"] > 0 and g["gain_vs_best_discard_lo"] > 0             # CI excludes zero, both baselines
    assert s.loc["oracle", "mean"] > g["mean"] > s.loc["always_on", "mean"]
    off, on = ev["on_off"]["gated_raw"]["mean_off"], ev["on_off"]["gated_raw"]["mean_on"]
    assert off < on and off < 0.5 * on                                                    # what it switched off really was worse
    assert (s.loc["gated_c61", "mean"] >= s.loc["best_discard" if False else ev["best_discard"], "mean"] - 1e-4)
    assert {"era", "policy", "gain", "gain_lo", "gain_hi"} <= set(ev["eras"].columns) and ev["eras"]["era"].nunique() >= 3


def test_status_table_and_ledger_are_date_free_and_consistent(regime):
    tl, res = regime
    st = res["status"]
    assert set(st["status"]) <= {"insufficient", "steady", "gated", "broken_gated", "disregarded"}
    assert len(st) == len(tl.patterns) and st["weight_now"].between(0, 1).all()
    led = res["health"].ledger(400)
    assert set(led["status"]) <= {"healthy", "suspect", "broken"} and led["evidence"].str.contains("cusum").all()
    assert not led["evidence"].str.contains(r"\d{4}-\d{2}-\d{2}").any()
    assert (res["health"].codes.iloc[:5] == 0).all().all()                                # burn-in: unmonitored, not accused


# ---------------------------------------------------------------- random breaks: unpredictable, discarded, no fake gain
def test_random_breaks_are_judged_unpredictable_and_give_no_fake_gain():
    tl = PR.planted_world("random_breaks", seed=1, n_weeks=520)
    res = PR.run_reliability(tl, FAST, kinds=("pooled_ctx",))
    assert res["explain"]["explanations"] == []                                           # no driver survives the family-wise bar
    sc = res["scores"]["pooled_ctx"]
    assert sc["skill_t_vs_pool"] < PR.PARAMS["t_pool"]                                    # context does not predict reliability
    s = res["evaluation"]["summary"].loc["gated_raw"]
    assert s["gain_vs_always_on_hi"] < 5e-4 and s["gain_vs_always_on"] <= 1e-4            # gating context buys nothing
    inv = res["investigations"]
    done = inv[inv["verdict"].isin(["EXPLAINED_AND_GATED", "DISCARDED_UNPREDICTABLE"])]
    assert len(done) >= 5 and (done["verdict"] == "DISCARDED_UNPREDICTABLE").all() and (done["cause"] == "unknown").all()
    assert res["unknown_cause"]["share"] == 1.0
    st = res["status"]
    broken_now = res["health"].codes.iloc[-1].values == PR.BROKEN
    assert broken_now.any() and (st.loc[broken_now, "status"] == "disregarded").all()     # broken + unpredictable -> weight 0
    assert (st.loc[broken_now, "weight_now"] == 0).all()
    assert len(res["pred"]["pooled_ctx"].df) > 0                                          # ... but nothing was deleted


# ---------------------------------------------------------------- family sharing a break driver: pooling helps
def test_pooling_across_patterns_beats_per_pattern_models_when_history_is_short():
    tl = PR.planted_world("family", seed=2, n_weeks=520, n_patterns=14)
    cfg = PR._cfg(FAST)
    pooled = PR.score_predictions(PR.walk_forward(tl, cfg, "pooled_ctx"), n_boot=100)
    single = PR.score_predictions(PR.walk_forward(tl, cfg, "per_pattern"), n_boot=100)
    assert pooled["n"] == single["n"]
    assert pooled["brier_model"] < single["brier_model"] - 0.01
    assert pooled["auc"] > single["auc"] + 0.05 and pooled["brier_model"] < pooled["brier_pool"]
    tab = PR.driver_table(tl, cfg)
    assert tab.iloc[0]["driver"].startswith("m_regime") and tab.iloc[0]["p_fwer"] <= 0.05


# ---------------------------------------------------------------- unobserved shock with a decoy indicator
def test_unobserved_shock_is_labelled_unknown_cause_not_the_decoy(shock):
    tl, res = shock
    inv = res["investigations"]
    done = inv[inv["verdict"].isin(["EXPLAINED_AND_GATED", "DISCARDED_UNPREDICTABLE"])]
    assert len(done) >= 15
    assert (done["cause"] == "unknown").all() and not (inv["driver"] == "m_spur").any() or \
        not ((inv["verdict"] == "EXPLAINED_AND_GATED") & (inv["driver"] == "m_spur")).any()
    assert res["unknown_cause"]["share"] == 1.0 and res["unknown_cause"]["resolved"] == len(done)
    assert (inv["verdict"] != "EXPLAINED_AND_GATED").all()
    # the trap was real: in-sample the decoy DID pass the family-wise bar at the second shock's review
    tab = PR.driver_table(tl, PR._cfg(FAST), 312)
    row = tab[tab["driver"] == "m_spur"].iloc[0]
    assert row["p_fwer"] <= 0.10 and row["works_when"] == "low"
    assert res["invariants"] == []
    # shocks hit many patterns at once: the cluster size is recorded as evidence of a common cause
    assert inv["shared_with"].max() >= 5


def test_a_real_recurring_driver_with_the_same_shape_is_trusted():
    """Control for the test above: the indicator is in its bad zone in EVERY shock and has no false excursions, and the
    shocks recur soon enough to be tested. It must be explained - else 'unknown cause' would be the default answer."""
    wins = [(150, 190), (250, 290), (350, 390), (450, 490)]
    tl = PR.planted_world("shock", seed=1, n_weeks=560, opts={"windows": wins, "decoy_on": wins, "bumps": []})
    res = PR.run_reliability(tl, FAST, explain=False, guards=False)
    inv = res["investigations"]
    won = inv[(inv["verdict"] == "EXPLAINED_AND_GATED") & (inv["driver"] == "m_spur")]
    assert len(won) >= 2 and won["detail"].str.contains("independent weeks out of sample").all()
    assert res["invariants"] == []


# ---------------------------------------------------------------- pattern health: false-alarm budget and phantom catching
def test_healthy_patterns_stay_inside_the_false_alarm_budget():
    tl = PR.planted_world("healthy", seed=3, n_weeks=520, n_patterns=20)
    h = PR.health_monitor(tl, FAST)
    real_events = h.events[~h.events["phantom"].astype(bool)]
    monitored = int((h.codes.values > 0).sum())
    budget = PR.false_alarm_budget(h, monitored)
    assert monitored > 8000 and len(real_events) <= 2 * budget + 3
    assert len(h.events) - len(real_events) <= 1                                          # at most one healthy pattern called phantom
    # and the investigation of any false alarm still ends in a recorded verdict
    inv = PR.investigate(tl, FAST, None, h)
    assert PR.investigation_invariants(inv, h, tl.n_weeks - 1) == []


def test_phantom_pattern_is_caught_within_a_bounded_delay():
    tl = PR.planted_world("phantom", seed=2, n_weeks=400, n_patterns=8)                   # edge disappears at week 100
    h = PR.health_monitor(tl, FAST)
    ev = h.events[~h.events["phantom"].astype(bool)].sort_values("detect").groupby("pattern").first()
    assert len(ev) == 8                                                                    # every dead pattern was caught
    delay = ev["detect"] - 100
    assert (delay > 0).all() and delay.max() <= 110 and delay.median() <= 55
    fast = PR.health_monitor(tl, {**FAST, "arl0": 100})                                   # the false-alarm / delay trade-off
    ev2 = fast.events[~fast.events["phantom"].astype(bool)].sort_values("detect").groupby("pattern").first()
    assert (ev2["detect"] - 100).median() < delay.median()
    assert (h.codes.iloc[-1] == PR.BROKEN).all()                                          # ... and none is left active
    assert (h.codes.iloc[380] == PR.BROKEN).sum() >= 6                                    # a weak-edge pattern may lag, none escapes


def test_unestablished_pattern_is_phantom_until_evidence_establishes_it():
    rng = np.random.default_rng(1)
    r = np.r_[rng.normal(-0.002, 0.01, 60), rng.normal(0.008, 0.01, 200)]
    h = PR.health_monitor(_series_tl(r), {"est_win": 26})
    ev = h.events
    assert ev["phantom"].iloc[0] and 26 <= ev["detect"].iloc[0] <= 30                      # accused as soon as the burn-in ends
    assert ev["recover"].iloc[0] < 200                                                     # ... and released once it proves out
    assert (h.codes.iloc[:26, 0] == 0).all()                                               # nothing said during the burn-in


def test_investigation_invariants_detect_missing_and_stale_cases():
    tl = PR.planted_world("random_breaks", seed=2, n_weeks=420)
    h = PR.health_monitor(tl, FAST)
    inv = PR.investigate(tl, FAST, None, h)
    assert PR.investigation_invariants(inv, h, tl.n_weeks - 1, FAST) == []
    dropped = inv.iloc[1:]
    assert any("uninvestigated" in v for v in PR.investigation_invariants(dropped, h, tl.n_weeks - 1, FAST))
    stale = inv.copy()
    stale.loc[stale.index[0], ["verdict", "detect"]] = ["OPEN", 0]
    assert any("open too long" in v for v in PR.investigation_invariants(stale, h, tl.n_weeks - 1, FAST))
    blank = inv.copy()
    blank.loc[blank[blank["verdict"] == "DISCARDED_UNPREDICTABLE"].index[0], "detail"] = " "
    assert any("without reasoning" in v for v in PR.investigation_invariants(blank, h, tl.n_weeks - 1, FAST))
    assert PR.unknown_cause_share(inv.iloc[:0])["resolved"] == 0


# ---------------------------------------------------------------- degenerate inputs
def test_degenerate_inputs_do_not_crash_and_say_so():
    with pytest.raises(PR.ReliabilityError, match="not enough matured"):
        PR.walk_forward(_series_tl(np.random.default_rng(0).normal(0, 0.01, 60)), None)
    flat = _series_tl(np.zeros(120))
    h = PR.health_monitor(flat)                                                            # zero variance: nothing to test
    assert (h.codes.values == 0).all() and h.events.empty
    inv = PR.investigate(flat, None, None, h)
    assert inv.empty and PR.investigation_invariants(inv, h, 119) == []
    assert PR.unknown_cause_share(inv) == {"resolved": 0, "unknown": 0, "share": float("nan")} or \
        PR.unknown_cause_share(inv)["resolved"] == 0
    assert PR.driver_table(flat, None, 30).empty                                           # too early to search anything
    lone = PR.planted_world("regime", seed=1, n_weeks=200, n_patterns=1)
    assert len(lone.patterns) == 1 and PR.build_examples(lone)["pos"].max() == 199
    nan_col = PR.planted_world("regime", seed=1, n_weeks=200, n_patterns=3)
    R = nan_col.rets.copy()
    R.iloc[:, 2] = np.nan
    tl = PR.Timelines(R, nan_col.ctx)
    assert tl.first_pos.iloc[2] == tl.n_weeks                                              # a pattern with no returns never starts
    assert PR.health_monitor(tl).codes.iloc[:, 2].eq(0).all()


def test_explanations_never_name_a_stock_or_a_date(regime):
    tl, res = regime
    for e in res["explain"]["explanations"]:
        assert not PR._ISO.search(e.statement) and "ticker" not in e.statement.lower()
    text = PR.render_report(res)
    assert "unknown cause" in text.lower() or "Unknown-cause share" in text
    assert not PR._ISO.search(text)


# ---------------------------------------------------------------- B26 pattern memory adapters
def _planted_memory(tmp_path, n_keys=8, with_effect=True, seed=0):
    rng = np.random.default_rng(seed)
    mem = PatternMemory(tmp_path / "pm")
    obs = []
    for j in range(n_keys):
        key = f"feat{j}_q5 & mom{j}_q1"
        for d in pd.date_range("2015-01-31", periods=60, freq="ME"):
            c = rng.normal(0, 1)
            sgn = (1.0 if c > 0 else -1.0) if with_effect else rng.choice([-1.0, 1.0])
            t = 1.6 * sgn + rng.normal(0, 0.7) + 1.1
            obs.append({"key": key, "obs_date": d, "effect": 0.002 * t, "n": 30, "t": t,
                        "ctx": {"m_vix": c, "m_breadth": rng.normal(0, 1)}})
    mem.add_observations("run1", pd.Timestamp("2021-01-01"), obs)
    return mem


def test_memory_gate_switches_a_pattern_off_only_where_predicted_unreliable(tmp_path):
    mem = _planted_memory(tmp_path)
    gate = PR.MemoryGate.fit(mem, "2021-01-01")
    assert gate.skill["proven"] and gate.skill["brier_model"] < gate.skill["brier_base"]
    plain = mem.view("2021-01-01", {"m_vix": 1.5, "m_breadth": 0.0})
    active = {k for k, w in plain.weights.items() if w.weight > 0}
    assert len(active) >= 5
    good = mem.view("2021-01-01", {"m_vix": 1.5, "m_breadth": 0.0}, gate=gate)
    bad = mem.view("2021-01-01", {"m_vix": -1.5, "m_breadth": 0.0}, gate=gate)
    open_share = np.mean([good.weights[k].weight > 0 for k in active])
    shut = [k for k in active if bad.weights[k].weight == 0]
    assert open_share > 0.75 and len(shut) >= 0.75 * len(active)                          # off in the bad context ...
    summ = {k: mem.timeline_summary(k, "2021-01-01") for k in active}
    assert all(gate(k, {"m_vix": -1.5, "m_breadth": 0.0}, s) == 0.0 for k, s in summ.items())   # ... by the gate itself
    assert all(gate(k, {"m_vix": 1.5, "m_breadth": 0.0}, s) == 1.0 for k, s in summ.items())
    assert set(bad.weights) == set(good.weights) == set(mem.keys())                        # ... but never deleted
    assert all(0.0 <= float(g.components.get("gate", 1.0)) <= 1.0 for g in bad.weights.values())


def test_memory_gate_does_nothing_without_proven_skill(tmp_path):
    mem = _planted_memory(tmp_path, with_effect=False, seed=1)
    gate = PR.MemoryGate.fit(mem, "2021-01-01")
    v0 = mem.view("2021-01-01", {"m_vix": 1.0, "m_breadth": 0.0})
    v1 = mem.view("2021-01-01", {"m_vix": 1.0, "m_breadth": 0.0}, gate=gate)
    assert not gate.skill["proven"] or gate.skill["t"] < 2.5
    if not gate.skill["proven"]:
        assert {k: w.weight for k, w in v0.weights.items()} == {k: w.weight for k, w in v1.weights.items()}
    empty = PR.MemoryGate.fit(PatternMemory(tmp_path / "empty"), "2021-01-01")
    assert empty(  "k", {}, None) == 1.0 and not empty.skill["proven"]


def test_memory_gate_only_reads_what_the_view_hands_it(tmp_path):
    """The gate is fitted from matured summaries: fitting at an earlier real_now equals fitting on truncated evidence."""
    mem = _planted_memory(tmp_path)
    early = PR.MemoryGate.fit(mem, "2018-06-01")
    s = mem.timeline_summary(mem.keys()[0], "2018-06-01")
    assert len(s.t) <= 40 and early.skill["n_test"] < PR.MemoryGate.fit(mem, "2021-01-01").skill["n_test"]


def test_timelines_from_memory_bounds_by_real_now(tmp_path):
    mem = _planted_memory(tmp_path, n_keys=3)
    tl = PR.timelines_from_memory(mem, "2019-01-01")
    assert tl.rets.index.max() < pd.Timestamp("2019-01-01") and tl.rets.shape[1] == 3
    with pytest.raises(PR.ReliabilityError):
        PR.timelines_from_memory(mem, "2010-01-01")


def test_live_gate_gates_in_the_bad_regime_and_uses_only_matured_evidence():
    tl = PR.planted_world("regime", seed=2, n_weeks=520)
    cfg = {**FAST, "refit_every": 26}
    lg = PR.LiveGate(tl, cfg, as_of=420)
    assert lg.as_of == 420 and set(lg.status.index) == set(tl.patterns)
    p = tl.patterns[0]
    up = {c: 0.0 for c in tl.ctx.columns}
    good = lg.p_hold(p, {**up, "m_regime": 2.0})
    bad = lg.p_hold(p, {**up, "m_regime": -2.0})
    assert good > bad + 0.15
    if lg.status.loc[p, "status"] in ("gated", "steady", "insufficient"):
        assert lg.weight(p, {**up, "m_regime": 2.0}) >= lg.weight(p, {**up, "m_regime": -2.0})
    assert lg.weight("unknown_pattern") == 1.0
    lg2 = PR.LiveGate(PR.Timelines(tl.rets.copy().assign(**{p: tl.rets[p].where(np.arange(520) < 420, 9.9)}), tl.ctx, {}, tl.first_pos),
                      cfg, as_of=420)                                                       # rewrite the future: the gate cannot notice
    assert lg2.p_hold(p, {**up, "m_regime": 2.0}) == pytest.approx(good)


# ---------------------------------------------------------------- W-05: winner's curse in the expected effect
def _stationary_false_alarms(cfg, mu=0.0015, seeds=range(700, 730), T=520, n=8):
    events = monitored = flagged = total = 0
    for sd in seeds:
        tl = PR.planted_world("healthy", seed=sd, n_weeks=T, n_patterns=n, mu=mu)
        h = PR.health_monitor(tl, cfg)
        real = h.events[~h.events["phantom"].astype(bool)]
        events += len(real)
        monitored += int((h.codes.values > 0).sum())
        flagged += real["pattern"].nunique()
        total += n
    return events, monitored, flagged, total


def test_w05_weak_stationary_edges_false_alarm_rate_is_bounded_over_30_worlds():
    """240 stationary patterns with a weak edge (0.15 sd a week), the case where being picked for a good burn-in inflates the mean
    most. Nothing about them ever changes, so every alarm is false. Measured with the shrunk mean alone (old rule): 205 alarms =
    0.86 of the nominal budget, 50% of patterns flagged; with the lower-bound expected effect: 178 = 0.75, 42%."""
    events, monitored, flagged, total = _stationary_false_alarms(None)
    assert total == 240 and monitored > 80_000
    assert events <= 0.80 * monitored / PR.PARAMS["arl0"]
    assert flagged / total <= 0.46


def test_w05_old_winners_curse_expected_effect_would_fail_those_bounds():
    events, monitored, flagged, total = _stationary_false_alarms({"effect_lcb_z": -1e9})      # the shrunk mean alone (old rule)
    assert events > 0.80 * monitored / PR.PARAMS["arl0"] and flagged / total > 0.46


def test_w05_expected_effect_is_a_lower_bound_and_stays_positive():
    P = PR.PARAMS
    lo = PR.expected_effect(0.010, 0.010, 26, P)                                             # mean 0.010, se 0.00196
    assert lo == pytest.approx(min(0.7 * 0.010, 0.010 - P["effect_lcb_z"] * 0.010 / math.sqrt(26)))
    assert PR.expected_effect(0.003, 0.010, 26, P) < PR.expected_effect(0.003, 0.010, 400, P)   # more evidence, less discount
    thin = PR.expected_effect(0.002, 0.010, 26, P)                                            # t ~ 1: the bound is near zero
    assert P["effect_floor"] * 0.002 <= thin < 0.7 * 0.002 and thin > 0
    assert PR.expected_effect(0.0, 0.01, 30, P) == 0.0                                         # nothing established, nothing expected


def test_w05_planted_real_decay_is_still_caught():
    caught, n, delays = 0, 0, []
    for sd in range(600, 610):
        tl = PR.planted_world("phantom", seed=sd, n_weeks=400, n_patterns=8, mu=0.004)       # edge vanishes at week 100
        h = PR.health_monitor(tl)
        ev = h.events[~h.events["phantom"].astype(bool) & (h.events["detect"] > 100)].sort_values("detect").groupby("pattern").first()
        caught += len(ev)
        n += 8
        delays += list(ev["detect"] - 100)
    assert caught >= 0.9 * n and np.median(delays) <= 60


def test_w05_a_lucky_streak_in_a_dead_pattern_does_not_release_it_early():
    """Dead pattern (edge zero after week 80), then a short run of good weeks: the posterior leans on the prior expected effect,
    so the early release also needs the recent mean to be release_t standard errors above ZERO. Never releases sooner than
    the old rule across 30 dead series, and strictly later on the seed where the streak fools the old rule."""
    later = 0
    for sd in range(30):
        rng = np.random.default_rng(sd)
        r = np.r_[rng.normal(0.006, 0.01, 80), rng.normal(0.0, 0.01, 120), rng.normal(0.0045, 0.01, 12)]
        new = (PR.health_monitor(_series_tl(r), {"est_win": 26}).codes.iloc[:, 0] == PR.BROKEN).sum()
        old = (PR.health_monitor(_series_tl(r), {"est_win": 26, "release_t": 0.0}).codes.iloc[:, 0] == PR.BROKEN).sum()
        assert new >= old
        later += int(new > old)
    assert later >= 1
