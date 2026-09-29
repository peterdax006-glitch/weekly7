"""Bible Phase 9 (canon C34), the parts beyond the six weighting factors: era and modernity factors, capacity and
eviction, lessons with nothing answer-identifying in them, state round trips and fingerprints, and the diagnostics
that say WHY an estimate is what it is. Each mechanism has a planted case that would fail if the code were wrong,
plus the empty/degenerate case, plus a regression against the pre-Phase-9 arithmetic."""
import numpy as np
import pandas as pd
import pytest

from engine import memory as MM
from engine import memory_diagnostics as D
from engine.memory import CTX, Memory

Z = np.zeros(len(CTX))


def ctx(v):
    c = np.zeros(len(CTX)); c[0] = v
    return c


def stream(n=60, seed=0, arms=("a", "b"), regime_at=None, mu=0.01, sd=0.01, shift=-0.03):
    """A record stream [(arm, week, ctx, outcome)] with an optional planted regime change."""
    rng = np.random.default_rng(seed)
    out = []
    for w in range(n):
        for a in arms:
            m = mu + (shift if regime_at is not None and w >= regime_at and a == arms[0] else 0.0)
            out.append((a, float(w), ctx(rng.normal()), float(m + rng.normal(0, sd))))
    return out


# --- regression: with every new parameter at its default the arithmetic is the pre-Phase-9 arithmetic --------------
def reference_estimate(eps, breaks, p, week_now, ctx_now, scale):
    """The original weight/estimate formulas, written out independently."""
    ws, xs = [], []
    for arm, wk, c, y, src in eps:
        age = 0.0 if src else max(0.0, week_now - wk)
        w = 0.5 ** (age / p["mem_half_life"])
        d = (c - ctx_now) / scale
        w *= float(np.exp(-float(d @ d) / (2 * p["mem_bandwidth"] ** 2)))
        if src:
            w *= p["mem_prior_scale"]
        b = breaks
        if b is not None and not src and wk < b:
            w *= p["mem_shock_cut"]
        elif b is not None and src:
            w *= p["mem_shock_cut"]
        ws.append(w); xs.append(y)
    w, x = np.array(ws), np.array(xs)
    sw = w.sum()
    n_eff = sw ** 2 / (w ** 2).sum()
    mean = float((w * x).sum() / (sw + p["mem_shrink"] * w.mean()))
    var = float((w * (x - (w * x).sum() / sw) ** 2).sum() / sw)
    return mean, float(np.sqrt(max(var, 1e-8) / max(n_eff, 1.0))), float(n_eff)


def test_defaults_reproduce_the_original_arithmetic_including_breaks_and_long_term():
    lt = pd.DataFrame({"arm": [repr("a")] * 5, "ctx": [list(ctx(v)) for v in (-1, 0, 1, 2, -2)], "outcome": [0.01, -0.02, 0.03, 0.0, 0.02]})
    M = Memory(long_term=lt)
    for arm, wk, c, y in stream(50, seed=3, arms=("a",), regime_at=30):
        M.record(arm, wk, c, y)
    assert "a" in M.breaks
    got = M.estimate("a", 50.0, ctx(0.3))
    want = reference_estimate([e for e in M.ep if e[0] == "a"], M.breaks["a"], M.p, 50.0, ctx(0.3), M.scale)
    assert got == pytest.approx(want, rel=1e-12)


def test_half_life_constants_match_the_bible():
    assert MM.ADAPTER_HALF_LIFE_WEEKS == 8.0 and MM.MINER_HALF_LIFE_YEARS == 4.0
    assert MM.MEM_DEFAULT["mem_half_life"] == MM.ADAPTER_HALF_LIFE_WEEKS


# --- record validation -------------------------------------------------------------------------------------------
def test_nan_outcome_is_refused_and_counted_not_stored():
    M = Memory()
    assert M.record("a", 0, Z, float("nan")) is False
    assert M.record("a", 1, Z, float("inf")) is False
    assert len(M) == 0 and M.rejected == 2
    assert M.estimate("a", 2, Z)[2] == 0.0


def test_wrong_context_shape_raises():
    with pytest.raises(ValueError):
        Memory().record("a", 0, np.zeros(3), 0.01)


# --- era factor ---------------------------------------------------------------------------------------------------
def test_era_of_maps_dates_and_unknown():
    assert MM.era_of("2009-03-09") == "financial_crisis"
    assert MM.era_of("2023-05-01") == "rate_shock_2022_on"
    assert MM.era_of(None) == "unknown" and MM.era_of("not a date") == "unknown"


def test_era_factor_discounts_other_eras_and_is_neutral_by_default():
    def build(era_other):
        M = Memory({"mem_shrink": 0.0, "mem_shock_k": 1e9, "mem_era_other": era_other, "mem_half_life": 1e9})
        for w in range(6):
            M.record("a", w, Z, -0.03, date="2008-06-01")          # crisis-era lessons: it never works
        for w in range(6, 12):
            M.record("a", w, Z, +0.03, date="2021-06-01")          # pandemic-era lessons: it works
        M.set_clock("2021-09-01")
        return M
    neutral, era = build(1.0), build(0.1)
    assert neutral.estimate("a", 12, Z)[0] == pytest.approx(0.0, abs=1e-9)
    assert era.estimate("a", 12, Z)[0] > 0.015                        # today's era dominates once the factor is on


def test_era_factor_needs_a_known_era_on_both_sides():
    M = Memory({"mem_era_other": 0.0, "mem_shrink": 0.0})
    M.record("a", 0, Z, 0.02)                                          # no date, no era
    M.set_clock("2021-01-01")
    assert M.estimate("a", 1, Z)[0] > 0                                # an undated lesson is never zeroed by era


# --- modernity ----------------------------------------------------------------------------------------------------
def test_modernity_halves_weight_per_half_life_of_calendar_years():
    M = Memory({"mem_modern_half_life": 5.0, "mem_half_life": 1e12})
    M.record("a", 0, Z, 0.01, date="2010-01-01")
    M.record("a", 1, Z, 0.01, date="2020-01-01")
    e_old, e_new = M.ep
    now = M.date_now
    assert now == pd.Timestamp("2020-01-01")
    w_old, w_new = M.weight(e_old, 1, Z), M.weight(e_new, 1, Z)
    assert w_old / w_new == pytest.approx(0.5 ** (3652 / 365.25 / 5.0), rel=1e-6)


def test_modernity_off_by_default_leaves_old_and_new_equal():
    M = Memory({"mem_half_life": 1e12})
    M.record("a", 0, Z, 0.01, date="1990-01-01")
    M.record("a", 1, Z, 0.01, date="2020-01-01")
    assert M.weight(M.ep[0], 1, Z) == pytest.approx(M.weight(M.ep[1], 1, Z))


# --- shock ---------------------------------------------------------------------------------------------------------
def test_break_log_records_direction():
    M = Memory({"mem_shock_k": 1.0})
    rng = np.random.default_rng(0)
    for w in range(30):
        M.record("a", w, Z, 0.01 + rng.normal(0, 0.002))
    for w in range(30, 36):
        M.record("a", w, Z, -0.05)
    assert M.break_log and M.break_log[0][2] == "down"
    assert not D.shock_report(M).empty


def test_detect_lag_is_short_for_a_planted_break_and_none_without_one():
    rec = stream(80, seed=1, arms=("a",), regime_at=40, sd=0.005)
    lag = D.detect_lag(rec, "a", 40.0, {"mem_shock_k": 1.0})
    assert lag is not None and 0 <= lag <= 8
    calm = stream(80, seed=1, arms=("a",), regime_at=None, sd=0.005)
    assert D.detect_lag(calm, "a", 40.0) is None


# --- capacity and eviction ---------------------------------------------------------------------------------------
def test_total_capacity_is_enforced_and_evicts_the_oldest_window_episodes_first():
    M = Memory({"mem_capacity": 10, "mem_shock_k": 1e9})
    for w in range(25):
        M.record("a", w, Z, 0.01)
    assert len(M) == 10 and M.evicted == 15
    assert min(e[1] for e in M.ep) == 15.0                            # weeks 0..14 went, 15..24 stayed


def test_arm_capacity_protects_thin_arms():
    M = Memory({"mem_arm_capacity": 4, "mem_shock_k": 1e9})
    for w in range(20):
        M.record("busy", w, Z, 0.01)
    M.record("thin", 0, Z, 0.02)
    assert sum(1 for e in M.ep if e[0] == "busy") == 4 and sum(1 for e in M.ep if e[0] == "thin") == 1


def test_long_term_episodes_are_evicted_before_fresh_ones_of_equal_age_class():
    lt = pd.DataFrame({"arm": [repr("a")] * 3, "ctx": [list(Z)] * 3, "outcome": [0.01] * 3})
    M = Memory({"mem_capacity": 5}, long_term=lt)
    for w in range(4):
        M.record("a", w, Z, 0.02)
    assert len(M) == 5
    assert sum(1 for e in M.ep if e[4]) == 1 and sum(1 for e in M.ep if not e[4]) == 4    # 2 long-term lost, own all kept


def test_eviction_is_deterministic_and_keeps_the_index_consistent():
    def run():
        M = Memory({"mem_capacity": 12, "mem_arm_capacity": 5})
        for arm, wk, c, y in stream(30, seed=5, arms=("a", "b", "c")):
            M.record(arm, wk, c, y)
        return M
    a, b = run(), run()
    assert a.fingerprint() == b.fingerprint()
    for arm, ix in a._by_arm.items():
        assert all(a.ep[i][0] == arm for i in ix)                     # index never points at the wrong episode
    assert sum(len(v) for v in a._by_arm.values()) == len(a.ep)


def test_eviction_does_not_reset_shock_state():
    M = Memory({"mem_capacity": 8, "mem_shock_k": 1.0})
    rng = np.random.default_rng(0)
    for w in range(30):
        M.record("a", w, Z, 0.01 + rng.normal(0, 0.002))
    for w in range(30, 34):
        M.record("a", w, Z, -0.05)
    assert "a" in M.breaks and len(M) == 8


def test_unbounded_memory_never_evicts():
    M = Memory()
    for w in range(200):
        M.record("a", w, Z, 0.01)
    assert len(M) == 200 and M.evicted == 0


def test_capacity_report_counts_reasons():
    M = Memory({"mem_capacity": 5, "mem_arm_capacity": 3})
    for w in range(12):
        M.record("a", w, Z, 0.01)
    r = D.capacity_report(M)
    assert r["evicted"] == 9 and r["by_reason"].get("arm capacity", 0) == 9
    assert D.capacity_report(Memory())["evicted"] == 0


def test_capped_memory_regret_is_reported():
    rec = stream(60, seed=2, arms=("a",), regime_at=30)
    r = D.eviction_regret(rec, capacity=8, params={"mem_shock_k": 1.0})
    assert set(r) == {"skill_unbounded", "skill_capped", "regret", "capacity"}
    assert np.isfinite(r["regret"])


# --- lessons: Phase 9.1 fields, Phase 9.2 stripping ---------------------------------------------------------------
def build_lessons(tickers=("AAPL", "TSLA")):
    M = Memory()
    M.record(("knob", "k", 2, 3), 0, ctx(0.5), 0.0371, date="2019-08-14", source_experiment="e42", expected=0.01)
    M.record(("knob", "k", 2, 3), 1, ctx(-0.5), -0.0219, date="2019-08-21", source_experiment="e42", expected=0.01)
    M.record(("pick", "AAPL"), 2, ctx(0.0), 0.05, date="2020-03-02", source_experiment="e43", expected=-0.01)
    feats = {0: {"vol20": 0.0234567, "ticker": "TSLA", "note": "AAPL", "n": 3},
             1: {"symbol": "AAPL", "log_dv": 17.123456}, 2: {}}
    return M, feats, tickers


def test_lesson_has_every_phase_9_1_field():
    M, feats, tk = build_lessons()
    ls = M.lessons(feats, tk)
    assert len(ls) == 3
    need = {"fingerprint", "context", "features", "outcome_bin", "error_type", "source_experiment", "date", "era", "relevance",
            "reliability", "shock_state"}
    assert need <= set(MM.Lesson.__dataclass_fields__)
    l0 = ls[0]
    assert l0.error_type == "correct" and ls[1].error_type == "false_positive" and ls[2].error_type == "false_negative"
    assert l0.source_experiment == "e42" and l0.date == "2019Q3" and l0.era == "zero_rates_2010_19"
    assert set(l0.context) == set(CTX) and 0 < l0.reliability < 1 and l0.shock_state == "none"


def test_exported_lessons_carry_no_ticker_no_exact_outcome_no_exact_date():
    M, feats, tk = build_lessons()
    df = M.export_lessons(feats, tk)
    exact = [0.0371, -0.0219, 0.05]
    assert MM.scan_lessons(df, tickers=tk, window_ids=["r01a"], exact_outcomes=exact) == []
    assert "AAPL" not in df.to_json() and "TSLA" not in df.to_json()
    assert df["date"].str.fullmatch(r"\d{4}Q[1-4]").all()
    assert df["features"].iloc[0] == {"vol20": 0.023, "n": 3}          # ticker and note dropped, floats rounded


def test_leak_scan_can_fail_it_catches_planted_leaks():
    M, feats, tk = build_lessons()
    df = M.export_lessons(feats, tk)
    bad = df.copy()
    bad["ticker"] = ["AAPL", "TSLA", "AAPL"]
    bad["outcome"] = [0.0371, -0.0219, 0.05]
    bad.loc[0, "date"] = "2019-08-14"
    bad.loc[1, "source_experiment"] = "run r01a_2019"
    found = MM.scan_lessons(bad, tickers=tk, window_ids=["r01a"], exact_outcomes=[0.0371, -0.0219, 0.05])
    why = {w for _, _, w in found}
    assert {"forbidden field name", "contains a ticker", "exact date", "exact future outcome", "contains a test-window id"} <= why


def test_an_arm_naming_a_ticker_is_hashed_in_the_export():
    M, feats, tk = build_lessons()
    arms = set(M.export_lessons(feats, tk)["arm"])
    assert not any("AAPL" in a for a in arms) and any(a.startswith("arm:") for a in arms)
    assert D.leak_check(M, tickers=tk) == []


def test_outcome_bin_and_error_classification_edges():
    assert MM.outcome_bin(0.0) == "[+0.00,+0.02)" and MM.outcome_bin(-0.2).startswith("[-inf")
    assert MM.outcome_bin(0.5).endswith("+inf)") and MM.outcome_bin(float("nan")) == "nan"
    assert MM.classify_error(None, 0.1) == "unscored"
    assert MM.classify_error(0.02, 0.0005, noise=0.001) == "noise"
    assert MM.classify_error(-0.02, 0.03) == "false_negative" and MM.classify_error(-0.02, -0.03) == "correct"


def test_shock_state_labels_episodes_around_a_break():
    M = Memory({"mem_shock_k": 1.0})
    rng = np.random.default_rng(0)
    for w in range(30):
        M.record("a", w, Z, 0.01 + rng.normal(0, 0.002))
    for w in range(30, 36):
        M.record("a", w, Z, -0.05)
    states = {l.shock_state for l in M.lessons()}
    assert states == {"pre_break", "post_break"}


def test_empty_memory_exports_an_empty_lesson_table_with_all_columns():
    df = Memory().export_lessons()
    assert df.empty and list(df.columns) == list(MM.Lesson.__dataclass_fields__)


# --- state round trip and fingerprints ---------------------------------------------------------------------------
def test_state_round_trip_preserves_every_estimate():
    M = Memory({"mem_capacity": 40})
    for arm, wk, c, y in stream(30, seed=4, arms=(("knob", "k", 2, 3), "b"), regime_at=15):
        M.record(arm, wk, c, y, date=pd.Timestamp("2015-01-05") + pd.Timedelta(weeks=int(wk)))
    R = Memory.from_state(M.state_dict())
    assert R.fingerprint() == M.fingerprint()
    for arm in M.arms():
        assert R.estimate(arm, 31.0, ctx(0.2)) == pytest.approx(M.estimate(arm, 31.0, ctx(0.2)), rel=1e-12)
    assert R.breaks == M.breaks


def test_identical_inputs_give_identical_fingerprints_and_one_changed_outcome_does_not():
    recs = stream(20, seed=6)
    a, b, c = Memory(), Memory(), Memory()
    for i, (arm, wk, cx, y) in enumerate(recs):
        a.record(arm, wk, cx, y); b.record(arm, wk, cx, y)
        c.record(arm, wk, cx, y + (1e-9 if i == 7 else 0.0))
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()


def test_export_carries_date_and_era_back_into_a_new_memory():
    M = Memory()
    M.record("a", 0, Z, 0.02, date="2009-05-01")
    ex = M.export()
    M2 = Memory({"mem_modern_half_life": 3.0}, long_term=ex)
    assert M2.info[0]["era"] == "financial_crisis" and M2.info[0]["date"] == pd.Timestamp("2009-05-01")


# --- diagnostics ---------------------------------------------------------------------------------------------------
def test_episode_table_shares_sum_to_one_and_name_the_dominant_cut():
    M = Memory({"mem_shock_k": 1.0})
    rng = np.random.default_rng(0)
    for w in range(30):
        M.record("a", w, Z, 0.01 + rng.normal(0, 0.002))
    for w in range(30, 36):
        M.record("a", w, Z, -0.05)
    t = D.episode_table(M, "a", 36.0, Z)
    assert t["share"].sum() == pytest.approx(1.0)
    b = M.breaks["a"]
    assert (t.loc[t["week"] < b, "shock"] < 1).all() and (t.loc[t["week"] >= b, "shock"] == 1).all()
    assert t.loc[t["week"] == 0, "dominant_cut"].iloc[0] in ("recency", "shock")
    assert t.loc[t["week"] == b, "dominant_cut"].iloc[0] == "recency"      # the newest episode is cut by nothing but age


def test_factor_attribution_finds_the_factor_that_moved_the_estimate():
    """Old lessons say +3%, recent ones say -3%: removing recency must move the estimate up by a lot, and removing
    the (inactive) era/modernity factors must move it not at all."""
    M = Memory({"mem_half_life": 2.0, "mem_shrink": 0.0, "mem_shock_k": 1e9})
    for w in range(10):
        M.record("a", w, Z, +0.03)
    for w in range(10, 14):
        M.record("a", w, Z, -0.03)
    at = D.factor_attribution(M, "a", 14.0, Z).set_index("factor")["delta"]
    assert at["recency"] < -0.02                                       # without recency the estimate would be far higher
    assert at["era"] == pytest.approx(0.0) and at["modernity"] == pytest.approx(0.0) and at["shock"] == pytest.approx(0.0)


def test_explain_text_mentions_break_and_shrinkage_and_handles_unknown_arm():
    M = Memory({"mem_shock_k": 1.0})
    rng = np.random.default_rng(0)
    for w in range(30):
        M.record("a", w, Z, 0.01 + rng.normal(0, 0.002))
    for w in range(30, 36):
        M.record("a", w, Z, -0.05)
    ex = D.explain(M, "a", 36.0, Z)
    assert "structural break" in ex["text"] and ex["concentration"]["n_eff"] > 1
    none = D.explain(M, "missing", 36.0, Z)
    assert none["estimate"]["n_eff"] == 0.0 and none["table"].empty and none["attribution"].empty


def test_concentration_uniform_versus_one_dominant():
    u = D.concentration(np.ones(10))
    assert u["n_eff"] == pytest.approx(10) and u["gini"] == pytest.approx(0, abs=1e-9) and u["entropy"] == pytest.approx(1)
    d = D.concentration([100, 1, 1, 1])
    assert d["top1"] > 0.95 and d["n_eff"] < 1.2 and d["gini"] > 0.5
    assert D.concentration([])["n"] == 0 and D.concentration([0.0, 0.0])["n"] == 0


def test_memory_health_on_empty_and_populated():
    h = D.memory_health(Memory())
    assert h["summary"]["episodes"] == 0 and h["arms"].empty
    M = Memory({"mem_capacity": 100})
    for arm, wk, c, y in stream(10, arms=("a", "b")):
        M.record(arm, wk, c, y, expected=0.0)
    h = D.memory_health(M)
    assert h["summary"]["utilisation"] == pytest.approx(0.2) and len(h["arms"]) == 2
    assert set(h["summary"]["errors"]) <= set(MM.ERROR_TYPES)


def test_report_runs_and_names_arms():
    M = Memory()
    for arm, wk, c, y in stream(12, arms=("a", "b")):
        M.record(arm, wk, c, y)
    txt = D.report(M)
    assert "memory: 24 episodes" in txt and "arm 'a'" in txt


# --- does the memory predict? -------------------------------------------------------------------------------------
def regime_stream(n=120, seed=0):
    """An arm that wins in calm markets and loses in fear: only market similarity can predict it."""
    rng = np.random.default_rng(seed)
    out = []
    for w in range(n):
        c = rng.choice([-1.5, 1.5])
        out.append(("a", float(w), ctx(c), float(0.03 * np.sign(c) + rng.normal(0, 0.01))))
    return out


def test_walk_forward_skill_positive_when_similarity_matters_and_ablation_credits_it():
    rec = regime_stream()
    df, s = D.walk_forward_skill(rec, {"mem_bandwidth": 0.5, "mem_shrink": 1.0, "mem_shock_k": 1e9})
    assert s["n"] > 50 and s["skill"] > 0.5 and s["hit_rate"] > 0.9
    ab = D.factor_ablation(rec, {"mem_bandwidth": 0.5, "mem_shrink": 1.0, "mem_shock_k": 1e9}).set_index("removed")
    assert ab.loc["similarity", "contribution"] > 0.3                  # the factor that carries the signal
    assert abs(ab.loc["era", "contribution"]) < 1e-9 and abs(ab.loc["modernity", "contribution"]) < 1e-9   # dead factors show as dead


def test_walk_forward_skill_is_not_positive_on_pure_noise():
    rng = np.random.default_rng(0)
    rec = [("a", float(w), ctx(rng.normal()), float(rng.normal(0, 0.02))) for w in range(150)]
    _, s = D.walk_forward_skill(rec, {"mem_shrink": 6.0})
    assert s["skill"] < 0.05                                           # a memory of noise must not claim skill


def test_walk_forward_never_uses_the_outcome_it_is_predicting():
    """Plant an outcome so extreme that if it leaked into its own prediction the prediction would be huge."""
    rec = [("a", float(w), Z, 0.0) for w in range(10)] + [("a", 10.0, Z, 5.0)]
    df, _ = D.walk_forward_skill(rec, {"mem_shrink": 0.0}, warmup=3)
    assert df["pred"].abs().max() < 1e-9 and df["outcome"].iloc[-1] == 5.0


def test_sweep_finds_the_better_half_life_after_a_regime_change():
    rec = stream(90, seed=7, arms=("a",), regime_at=45, sd=0.003, shift=-0.05)
    t = D.sweep(rec, "mem_half_life", [1.0, 4.0, 1e6], {"mem_shrink": 0.0, "mem_shock_k": 1e9}).set_index("mem_half_life")
    assert t.loc[4.0, "skill"] > t.loc[1e6, "skill"]                   # forgetting helps when the world changed
    assert D.walk_forward_skill([], None)[1]["n"] == 0


# --- era distance, ranking arms, absorbing banks -------------------------------------------------------------------
def test_era_gap_counts_eras_apart_and_unknown_is_none():
    assert MM.era_gap("financial_crisis", "financial_crisis") == 0
    assert MM.era_gap("financial_crisis", "pandemic_2020_21") == 2
    assert MM.era_gap("unknown", "financial_crisis") is None and MM.era_gap(None, None) is None


def test_era_decay_makes_neighbouring_eras_count_more_than_distant_ones():
    M = Memory({"mem_era_decay": 0.5, "mem_half_life": 1e12, "mem_shrink": 0.0})
    M.record("a", 0, Z, 0.01, date="2005-06-01")        # expansion_2003_07: 3 eras from pandemic
    M.record("a", 1, Z, 0.01, date="2015-06-01")        # zero_rates: 1 era away
    M.record("a", 2, Z, 0.01, date="2021-06-01")        # pandemic: same era
    M.set_clock("2021-09-01")
    w = [M.weight(e, 2, Z) for e in M.ep]
    assert w[2] == pytest.approx(1.0) and w[1] == pytest.approx(0.5) and w[0] == pytest.approx(0.125)


def test_era_decay_overrides_era_other_and_never_discounts_unknown_eras():
    M = Memory({"mem_era_decay": 0.5, "mem_era_other": 0.01, "mem_half_life": 1e12})
    M.record("a", 0, Z, 0.01, date="2015-06-01", era="made_up_era")
    M.record("a", 1, Z, 0.01, date="2021-06-01")
    M.set_clock("2021-09-01")
    assert M.weight(M.ep[0], 1, Z) == pytest.approx(M.weight(M.ep[1], 1, Z))


def test_rank_arms_orders_by_z_lists_thin_arms_last_and_is_stable_on_ties():
    M = Memory({"mem_shrink": 0.0, "mem_shock_k": 1e9})
    rng = np.random.default_rng(0)
    for w in range(12):
        M.record("good", w, Z, 0.03 + rng.normal(0, 0.002))
        M.record("bad", w, Z, -0.03 + rng.normal(0, 0.002))
        M.record("tie_b", w, Z, 0.0 + (0.01 if w % 2 else -0.01))
        M.record("tie_a", w, Z, 0.0 + (0.01 if w % 2 else -0.01))
    M.record("thin", 0, Z, 0.5)
    t = M.rank_arms(["thin", "bad", "tie_b", "good", "tie_a", "unseen"], 12, Z, min_n_eff=4)
    assert t["arm"].iloc[0] == "good" and t["usable"].iloc[:4].all() and not t["usable"].iloc[4:].any()
    assert set(t["arm"].iloc[4:]) == {"thin", "unseen"} and (t["z"].iloc[4:] == 0).all()
    i, j = list(t["arm"]).index("tie_a"), list(t["arm"]).index("tie_b")
    assert abs(t["z"][i] - t["z"][j]) < 1e-9 and i < j                      # equal scores fall back to alphabetical


def test_absorb_is_idempotent_discounts_as_long_term_and_skips_bad_rows():
    src = Memory()
    for w in range(6):
        src.record(("knob", "k", 2, 3), w, ctx(w), 0.01 * w, date="2012-01-02")
    bank = src.export()
    bad = pd.concat([bank, pd.DataFrame({"arm": [repr("x")], "ctx": [[0.0, 1.0]], "outcome": [0.1]}),
                     pd.DataFrame({"arm": [repr("y")], "ctx": [list(Z)], "outcome": [float("nan")]})], ignore_index=True)
    M = Memory()
    assert M.absorb(bad) == 6 and M.absorb(bad) == 0                        # second absorb adds nothing
    assert all(e[4] == 1 for e in M.ep) and M.info[0]["era"] == "zero_rates_2010_19"
    assert M.estimate(("knob", "k", 2, 3), 0, Z)[2] > 0


def test_absorb_respects_capacity_and_per_arm_capacity():
    src = Memory()
    for w in range(30):
        src.record("a", w, Z, 0.01)
        src.record("b", w, Z, 0.02)
    M = Memory({"mem_arm_capacity": 5, "mem_capacity": 8})
    M.absorb(src.export())
    assert len(M) <= 8 and sum(1 for e in M.ep if e[0] == "a") <= 5


# --- statistics of the diagnostics ---------------------------------------------------------------------------------
def test_skill_ci_covers_the_point_estimate_and_excludes_zero_for_a_real_signal_not_for_noise():
    df, s = D.walk_forward_skill(regime_stream(150), {"mem_bandwidth": 0.5, "mem_shrink": 1.0, "mem_shock_k": 1e9})
    lo, hi = D.skill_ci(df, seed=1)
    assert lo < s["skill"] < hi and lo > 0.3
    rng = np.random.default_rng(3)
    noise = [("a", float(w), ctx(rng.normal()), float(rng.normal(0, 0.02))) for w in range(150)]
    dn, sn = D.walk_forward_skill(noise, {"mem_shrink": 6.0})
    lo2, hi2 = D.skill_ci(dn, seed=1)
    assert lo2 < 0.05 and np.isnan(D.skill_ci(dn.head(3))[0])                      # too short for a block bootstrap: NaN


def test_calibration_slope_positive_for_signal_and_table_is_ordered():
    df, _ = D.walk_forward_skill(regime_stream(200), {"mem_bandwidth": 0.5, "mem_shrink": 1.0, "mem_shock_k": 1e9})
    t, slope = D.calibration_table(df, bins=4)
    assert slope > 0.5 and t["n"].sum() == len(df)
    assert (t["outcome"].iloc[:2] < 0).all() and (t["outcome"].iloc[2:] > 0).all() and (t["share_positive"].iloc[-1] > 0.9)
    empty, s = D.calibration_table(df.head(3))
    assert empty.empty and np.isnan(s)


def test_nested_tune_helps_out_of_sample_when_the_world_changes_and_says_no_on_stationary_noise():
    changing = stream(120, seed=7, arms=("a",), regime_at=40, sd=0.003, shift=-0.05)
    r = D.nested_tune(changing, {"mem_half_life": [1.0, 3.0, 1e6]}, {"mem_shrink": 0.0, "mem_shock_k": 1e9, "mem_half_life": 1e6}, split=0.5)
    assert r["chosen"] in (1.0, 3.0) and r["helped"] and r["test_skill_chosen"] > r["test_skill_default"]
    rng = np.random.default_rng(11)
    noise = [("a", float(w), Z, float(rng.normal(0.0, 0.02))) for w in range(160)]
    r2 = D.nested_tune(noise, {"mem_half_life": [1.0, 2.0, 1e6]}, {"mem_half_life": 1e6}, split=0.5)
    assert r2["chosen"] == 1e6 and r2["helped"] is False                     # on noise the best memory is the longest; no fake gain


def test_arm_timeline_shows_belief_lagging_a_break_then_catching_up():
    rec = stream(80, seed=2, arms=("a",), regime_at=40, sd=0.003, shift=-0.05)
    t = D.arm_timeline(rec, "a", {"mem_shock_k": 1.0, "mem_shrink": 0.0})
    assert t["belief"].iloc[:35].mean() > 0.005 and t["belief"].iloc[-3:].mean() < t["belief"].iloc[:35].mean()
    assert t["broken"].iloc[-1] and set(t.columns) == {"week", "belief", "se", "n_eff", "broken", "outcome"}
    assert D.arm_timeline(rec, "nope").empty


def test_stability_by_block_separates_an_all_period_edge_from_a_one_period_edge():
    good = D.walk_forward_skill(regime_stream(240), {"mem_bandwidth": 0.5, "mem_shrink": 1.0, "mem_shock_k": 1e9})[0]
    t, share = D.stability_by_block(good, 4)
    assert share == 1.0 and len(t) == 4
    lopsided = good.copy()
    lopsided.loc[lopsided.index[len(good) // 2:], "pred"] = -lopsided.loc[lopsided.index[len(good) // 2:], "pred"]
    _, share2 = D.stability_by_block(lopsided, 4)
    assert share2 <= 0.5 and np.isnan(D.stability_by_block(good.head(5))[1])


# --- per-kind half-life ----------------------------------------------------------------------------------------------
def test_arm_kind_reads_the_family_of_a_tuple_arm():
    assert MM.arm_kind(("knob", "k", 2, 3)) == "knob" and MM.arm_kind(("ic", "frog")) == "ic"
    assert MM.arm_kind("plain") == "other" and MM.arm_kind(()) == "other" and MM.arm_kind((3, 4)) == "other"


def test_kind_half_life_fades_knob_and_ic_arms_at_different_speeds():
    M = Memory({"mem_half_life": 8.0, "mem_kind_half_life": {"knob": 2.0, "ic": 32.0}})
    for arm in (("knob", "k", 2, 3), ("ic", "frog"), "other"):
        M.record(arm, 0, Z, 0.01)
    w = {e[0]: M.weight(e, 8, Z) for e in M.ep}
    assert w[("knob", "k", 2, 3)] == pytest.approx(0.5 ** (8 / 2.0)) and w[("ic", "frog")] == pytest.approx(0.5 ** (8 / 32.0))
    assert w["other"] == pytest.approx(0.5)                                # arms with no override keep the global half-life


def test_kind_half_life_unset_is_identical_to_the_global_half_life():
    a, b = Memory({"mem_half_life": 5.0}), Memory({"mem_half_life": 5.0, "mem_kind_half_life": {}})
    for M in (a, b):
        for w in range(10):
            M.record(("knob", "k", 2, 3), w, Z, 0.01 * w)
    assert a.estimate(("knob", "k", 2, 3), 10, Z) == b.estimate(("knob", "k", 2, 3), 10, Z)


def test_kind_half_life_drives_eviction_too():
    M = Memory({"mem_capacity": 6, "mem_kind_half_life": {"knob": 1.0}, "mem_shock_k": 1e9})
    for w in range(6):
        M.record(("ic", "a"), w, Z, 0.01)
    for w in range(6, 9):
        M.record(("knob", "k", 2, 3), w, Z, 0.01)
    kinds = sorted(MM.arm_kind(e[0]) for e in M.ep)
    assert len(M) == 6 and kinds.count("knob") + kinds.count("ic") == 6
    assert D.factor_ablation(stream(30, seed=1), {"mem_kind_half_life": {"a": 1.0}}).shape[0] == 8


def test_ancient_evidence_does_not_underflow_to_nan():
    """Half-life 1 week and ~1,000 weeks of age: every weight is ~1e-300 (representable), but w**2 underflows to 0 and
    n_eff became 0/0. The estimate must stay finite and equal what the same data gives with a scale-free weighting."""
    M = Memory({"mem_half_life": 1.0, "mem_shrink": 0.0, "mem_shock_k": 1e9})
    for w in range(5):
        M.record("a", w, Z, 0.02 + 0.001 * w)
    d = M.estimate_detail("a", 1004.0, Z)
    assert 0 < d["sum_w"] < 1e-290
    assert np.isfinite(d["mean"]) and np.isfinite(d["se"]) and d["n_eff"] > 1
    assert 0.02 < d["mean"] < 0.024
    assert D._shrunk(np.array([1e-300, 2e-300, 1e-300]), np.array([1.0, 2.0, 3.0]), 0.0)[2] > 1


# --- validate(): the invariants, and proof that each one can fail ---------------------------------------------------
def test_a_healthy_memory_validates_through_capacity_eviction_absorb_and_state_round_trip():
    M = Memory({"mem_capacity": 30, "mem_arm_capacity": 12})
    for arm, wk, c, y in stream(40, seed=3, arms=("a", "b", "c")):
        M.record(arm, wk, c, y, date=pd.Timestamp("2018-01-01") + pd.Timedelta(weeks=int(wk)))
    assert M.validate() == []
    M.absorb(Memory().export())
    assert M.validate() == [] and Memory.from_state(M.state_dict()).validate() == []
    assert Memory().validate() == []


def test_validate_catches_each_planted_corruption():
    def healthy():
        M = Memory({"mem_capacity": 10})
        for w in range(6):
            M.record("a", w, ctx(w), 0.01 * w)
        return M
    M = healthy(); M.info.pop()
    assert any("out of step" in p for p in M.validate())
    M = healthy(); M._by_arm["a"].pop()
    assert any("arm index covers" in p for p in M.validate())
    M = healthy(); M._by_arm["a"][0] = 3; M._by_arm["a"][3] = 0
    assert M.validate() == []                                              # a permutation of correct entries is still correct
    M = healthy(); M.ep[2] = ("zzz",) + M.ep[2][1:]
    assert any("does not hold arm" in p for p in M.validate())
    M = healthy(); a = M.ep[1]; M.ep[1] = (a[0], a[1], a[2], float("nan"), a[4])
    assert any("non-finite" in p for p in M.validate())
    M = healthy(); M.info[1]["seq"] = M.info[0]["seq"]
    assert any("not unique" in p for p in M.validate())
    M = healthy(); M.p["mem_capacity"] = 3
    assert any("exceed capacity" in p for p in M.validate())
    M = healthy(); M.breaks["ghost"] = 4.0
    assert any("no CUSUM state" in p for p in M.validate())
    M = healthy(); M.scale = np.array([1.0, 0.0] + [1.0] * 5)
    assert any("context scale" in p for p in M.validate())


# --- lesson_summary / error_profile_by_arm ---------------------------------------------------------------------------
def scored_memory():
    """Arm 'wins' is predicted correctly every time; arm 'flip' is always predicted wrong."""
    M = Memory({"mem_shock_k": 1e9})
    for w in range(12):
        M.record("wins", w, Z, 0.02, date="2019-03-01", source_experiment="e1", expected=0.01)
        M.record("flip", w, Z, -0.02, date="2019-03-01", source_experiment="e2", expected=0.01)
    M.record("thin", 0, Z, 0.02, expected=0.01)
    M.record("unscored", 0, Z, 0.02)
    return M


def test_lesson_summary_counts_errors_eras_and_shares():
    s = D.lesson_summary(scored_memory())
    assert s["n"] == 26 and s["by_error"]["correct"] == 13 and s["by_error"]["false_positive"] == 12 and s["by_error"]["unscored"] == 1
    assert s["by_era"]["zero_rates_2010_19"] == 24 and s["by_experiment"] == {"e1": 12, "e2": 12, "unspecified": 2}
    assert s["false_positive_share"] == pytest.approx(12 / 25) and 0 < s["mean_reliability"] < 1
    e = D.lesson_summary(Memory())
    assert e["n"] == 0 and np.isnan(e["false_positive_share"])


def test_error_profile_separates_the_arm_memory_gets_right_from_the_one_it_gets_wrong():
    t = D.error_profile_by_arm(scored_memory()).set_index("arm")
    assert t.loc["'wins'", "hit_rate"] == 1.0 and t.loc["'flip'", "hit_rate"] == 0.0
    assert bool(t.loc["'thin'", "thin"]) and np.isnan(t.loc["'thin'", "hit_rate"]) and "'unscored'" not in t.index
    assert D.error_profile_by_arm(Memory()).empty
