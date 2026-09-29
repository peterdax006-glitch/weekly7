"""Tests for engine.learning.situation and engine.learning.similarity (contract C62 sections 15, 16; canon C56, C63).

Synthetic data only. Covers: unit (specs, blocks, builder), determinism, negative/control (planted identity leak, planted
similarity structure), anti-leakage (future data, unpublished macro), anti-memorization (ticker / date / year / sector scrambling,
learning_delta.make_presentation disguise, identity recoverability), empty and degenerate cases."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from engine import learning_delta as LD
from engine.learning import similarity as SM
from engine.learning import situation as ST
from engine.learning.core import FirewallBreach, Unknown

# ------------------------------------------------------------------------------------------------ helpers

BASE = {
    "market": dict(spy_ma200=0.04, spy_ma50=0.02, spy_r5=0.005, vix=15.0, vix_term=0.9, vix_chg5=0.0),
    "sector": dict(family="technology", strength20=0.6, strength60=0.6, rel20=0.01, rel60=0.02),
    "stock_type": dict(kind="ordinary", lottery_rank=0.4, skew60=0.1, persistence=0.0, event_load=0.0),
    "volatility": dict(vol_rank=0.5, atr_pct=0.02, vol_ratio=1.0, range_compress=0.8, state="stable"),
    "liquidity": dict(dv_rank=0.6, vol_surge1=1.0, vol_surge5=1.0, state="normal"),
    "trend": dict(r5=0.01, r20=0.03, r60=0.08, mom_12_1=0.1, dist_ma50=0.03, dist_ma200=0.08, dist_52wh=-0.05, state="up"),
    "breadth": dict(breadth=0.6, dispersion=0.01, state="healthy"),
    "macro": dict(rates=2.5, curve=0.5, credit=3.5, fin_stress=-0.2, recession="no", staleness_days=10.0, state="neutral"),
    "regime": dict(label="bull_calm", vol_regime="low", trend_regime="up"),
    "shock": dict(kind="none", gap_today=0.0, max20=0.03, min20=-0.03, days_since_earn=30.0, red_flag="no", market_shock=0.2),
    "correlation": dict(avg_pair_corr=0.3, top_eig_share=0.3, corr_to_mkt=0.6, beta=1.0, state="normal"),
    "horizon": dict(days=5.0, state="week"),
    "seasonality": dict(),
    "pattern_interaction": dict(n_active=1, agreement=None, top_score=1.0, score_spread=0.0, mix="single"),
    "position_risk": dict(n_positions=3, gross=0.5, drawdown=-0.01, port_vol=0.01, position_weight=0.1, risk_used=0.4, state="normal"),
}


def mk(patterns=(), **over):
    """A fully populated valid Situation; override with block__field=value."""
    blocks = {k: dict(v) for k, v in BASE.items()}
    for key, val in over.items():
        b, f = key.split("__")
        blocks[b][f] = val
    return ST.Situation(tuple(ST.Block.make(k, **blocks[k]) for k in ST.BLOCK_ORDER), tuple(sorted(set(patterns))))


COLS = ["r5", "r20", "r60", "mom_12_1", "dist_ma50", "dist_ma200", "dist_52wh", "vol20", "vol_ratio", "atr_pct", "range_compress",
        "log_dv", "vol_surge1", "vol_surge5", "max20", "min20", "skew60", "frog", "gap_today", "ind_mom20", "ind_mom60",
        "rel_ind20", "rel_ind60", "days_since_earn", "news5", "ev_red_flag"]


def panel(n_dates=6, n_tk=40, seed=0, start="2019-03-04"):
    """Random but in-range feature panel with m_ market columns that vary by date."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_dates)
    tk = [f"TK{i:03d}" for i in range(n_tk)]
    ix = pd.MultiIndex.from_product([dates, tk], names=["date", "ticker"])
    X = pd.DataFrame(index=ix)
    n = len(ix)
    for c, (m, s) in {"r5": (0, .03), "r20": (0, .06), "r60": (0, .12), "mom_12_1": (0, .25), "dist_ma50": (0, .05),
                      "dist_ma200": (0, .1), "dist_52wh": (-.1, .08), "skew60": (0, .7), "frog": (0, .05), "gap_today": (0, .015),
                      "ind_mom20": (0, .04), "ind_mom60": (0, .08), "rel_ind20": (0, .04), "rel_ind60": (0, .07),
                      "min20": (-.05, .02)}.items():
        X[c] = rng.normal(m, s, n)
    X["dist_52wh"] = -np.abs(X["dist_52wh"])
    X["min20"] = -np.abs(X["min20"])
    X["vol20"] = rng.uniform(.005, .05, n)
    X["vol_ratio"] = rng.uniform(.6, 1.6, n)
    X["atr_pct"] = rng.uniform(.005, .06, n)
    X["range_compress"] = rng.uniform(.4, 1.2, n)
    X["log_dv"] = rng.normal(16.5, 1.5, n)
    X["vol_surge1"] = rng.uniform(.4, 2.5, n)
    X["vol_surge5"] = rng.uniform(.6, 1.8, n)
    X["max20"] = np.abs(rng.normal(.03, .02, n))
    X["days_since_earn"] = rng.integers(0, 90, n).astype(float)
    X["news5"] = (rng.random(n) < .1).astype(float)
    X["ev_red_flag"] = (rng.random(n) < .02).astype(float)
    per_date = pd.DataFrame({"m_spy_ma200": rng.normal(.03, .05, n_dates), "m_spy_ma50": rng.normal(.01, .03, n_dates),
                             "m_spy_r5": rng.normal(0, .01, n_dates), "m_vix": rng.uniform(12, 30, n_dates),
                             "m_vix_term": rng.uniform(.85, 1.05, n_dates), "m_vix_chg5": rng.normal(0, .05, n_dates),
                             "m_breadth": rng.uniform(.3, .8, n_dates), "m_dispersion": rng.uniform(.008, .02, n_dates)}, index=dates)
    for c in per_date:
        X[c] = per_date[c].reindex(X.index.get_level_values(0)).values
    return X.astype("float64")


SIC = {f"TK{i:03d}": str(10 + 3 * (i % 25)) for i in range(40)}
NOW = pd.Timestamp("2019-03-11")


def sit_map(series, key_fn=lambda d, t: (d, t)):
    return {key_fn(*k): v for k, v in series.items()}


# ------------------------------------------------------------------------------------------------ specs & blocks

def test_specs_are_consistent_and_base_situation_valid():
    paths = ST.all_paths()
    assert len(paths) == len(set(paths)) > 60
    for kind, specs in ST.BLOCK_SPECS.items():
        names = [s.name for s in specs]
        assert len(names) == len(set(names)), kind
        for s in specs:
            assert list(s.bins) == sorted(s.bins)
    s = mk()
    assert s.validate() == []
    assert s.coverage() > 0.85


def test_block_rejects_unknown_fields_and_flags_out_of_range():
    with pytest.raises(ValueError):
        ST.Block.make("market", nonsense=1.0)
    with pytest.raises(ValueError):
        ST.Block.make("no_such_block")
    b = ST.Block.make("volatility", atr_pct=-0.5)          # below its sanity range
    assert any("atr_pct" in e for e in b.validate())
    b2 = ST.Block.make("regime", label="not_a_regime")
    assert any("label" in e for e in b2.validate())
    assert ST.Block.make("market", vix=float("nan")).get("vix") is None      # NaN never becomes a number
    assert ST.Block.make("market", vix=True).get("vix") is None


def test_fieldspec_bins_and_distance_semantics():
    spec = ST.spec_of("volatility.vol_rank")
    assert spec.bin_of(None) == "na"
    assert spec.bin_of(0.05) == "b0" and spec.bin_of(0.95) == "b4"
    assert spec.distance(None, 0.3) is None                 # unknown is not zero
    reg = ST.spec_of("regime.vol_regime")
    assert reg.distance("low", "crisis") > reg.distance("low", "mid")
    assert ST.spec_of("regime.label").distance("bull_calm", "stress") == 1.0


def test_classify_regime_and_sic_family():
    assert ST.classify_regime(0.05, 14.0)[0] == "bull_calm"
    assert ST.classify_regime(0.05, 25.0)[0] == "bull_volatile"
    assert ST.classify_regime(-0.05, 30.0)[0] == "bear_volatile"
    assert ST.classify_regime(0.05, 26.0, vix_term=1.1)[0] == "stress"
    assert ST.classify_regime(None, 20.0)[0] == "unknown"
    assert ST.sic_family("2834") == "manufacturing" and ST.sic_family("73") == "technology"
    assert ST.sic_family("junk") == "unknown" and ST.sic_family(None) == "unknown"


# ------------------------------------------------------------------------------------------------ builder

def test_builder_panel_valid_identity_free_and_deterministic():
    X = panel()
    b = ST.SituationBuilder()
    s1 = b.build_panel(X, NOW, sic=SIC)
    s2 = ST.SituationBuilder().build_panel(X, NOW, sic=SIC)
    assert len(s1) == len(X)
    assert all(s.validate() == [] for s in s1)
    assert [s.exact_id for s in s1] == [s.exact_id for s in s2]
    tks = sorted(set(X.index.get_level_values(1)))
    years = {2019}
    dates = [d for d in X.index.get_level_values(0).unique()]
    for s in list(s1)[:50]:
        assert ST.audit_identity_free(s, tickers=tks, years=years, dates=dates) == []
    assert all(s.rank_mode == "cross_section" for s in s1)


def test_planted_identity_leak_is_caught_by_the_audit():
    s = mk(patterns=("AAPL",))
    assert any("AAPL" in f for f in ST.audit_identity_free(s, tickers=["AAPL"]))
    leaky = ST.Situation(mk().blocks, ("2019-03-11",))
    found = ST.audit_identity_free(leaky, dates=[dt.date(2019, 3, 11)], years=[2019])
    assert any("date" in f for f in found) and any("2019" in f for f in found)


def test_ticker_scramble_with_reordering_leaves_situations_unchanged():
    X = panel(seed=1)
    Y, mapping = ST.scramble_tickers(X, seed=5)
    assert list(Y.index.get_level_values(1)[:5]) != list(X.index.get_level_values(1)[:5])       # rows really reordered
    sic2 = {mapping[t]: c for t, c in SIC.items()}
    a = ST.SituationBuilder().build_panel(X, NOW, sic=SIC)
    b = ST.SituationBuilder().build_panel(Y.sort_index(), NOW, sic=sic2)
    inv = {v: k for k, v in mapping.items()}
    for (d, t2), s in b.items():
        assert s.exact_id == a[(d, inv[t2])].exact_id


def test_date_and_year_scramble_invariance_and_legit_seasonality():
    X = panel(seed=2)
    now_far = pd.Timestamp("2030-01-01")
    base = ST.SituationBuilder().build_panel(X, now_far, sic=SIC)
    for Y in (ST.shift_dates(X, 400), ST.shift_years(X, 4), ST.shift_years(X, -3)):
        alt = ST.SituationBuilder().build_panel(Y, now_far, sic=SIC)
        assert [s.exact_id for s in alt] == [s.exact_id for s in base]
    # seasonality is the ONE legitimate calendar input: whole-year shifts keep it, a 45-day shift changes only that block
    cfg = ST.SituationConfig(include_seasonality=True)
    b0 = ST.SituationBuilder(cfg).build_panel(X, now_far, sic=SIC)
    b_year = ST.SituationBuilder(cfg).build_panel(ST.shift_years(X, 5), now_far, sic=SIC)
    b_days = ST.SituationBuilder(cfg).build_panel(ST.shift_dates(X, 45), now_far, sic=SIC)
    assert [s.exact_id for s in b0] == [s.exact_id for s in b_year]
    changed = 0
    for s0, s1 in zip(b0, b_days):
        d = ST.diff(s0, s1)
        assert all(p.startswith("seasonality.") for p, _, _ in d.changed)
        changed += d.n_changed > 0
    assert changed > 0


def test_sector_scramble_changes_only_the_sector_block():
    X = panel(seed=3)
    a = ST.SituationBuilder().build_panel(X, NOW, sic=SIC)
    b = ST.SituationBuilder().build_panel(X, NOW, sic=ST.scramble_sectors(SIC, seed=9))
    n_diff = 0
    for s0, s1 in zip(a, b):
        d = ST.diff(s0, s1)
        assert all(p == "sector.family" for p, _, _ in d.changed)
        n_diff += d.n_changed
    assert n_diff > 0                                       # the scramble is not a no-op


def test_learning_delta_disguise_leaves_situations_unchanged():
    """The audited disguise (new code names + whole-week date shift) must not change any situation."""
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2019-01-02", periods=120)
    codes = [f"K{i:03d}" for i in range(30)]
    closes = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .015, (120, 30)), axis=0)), index=dates, columns=codes)

    def feats(C):
        r = np.log(C / C.shift(1))
        F = {"r5": np.log(C / C.shift(5)), "r20": np.log(C / C.shift(20)), "r60": np.log(C / C.shift(60)),
             "vol20": r.rolling(20).std(), "dist_ma50": C / C.rolling(50).mean() - 1}
        X = pd.concat({k: v.stack() for k, v in F.items()}, axis=1).dropna()
        X.index.names = ["date", "ticker"]
        m = pd.DataFrame({"m_spy_ma200": .03, "m_vix": 18.0, "m_breadth": .55}, index=X.index.get_level_values(0).unique())
        for c in m:
            X[c] = m[c].reindex(X.index.get_level_values(0)).values
        return X
    w = LD.Window(id="w1", snaps={}, closes=closes, opens=None, bps=5.0, divs={}, real_start=dates[0], real_end=dates[-1])
    pres, rec = LD.make_presentation(w, seed=11, shift_weeks=(4000, 9000))
    Xa, Xp = feats(closes), feats(pres.closes)
    now = pd.Timestamp("2200-01-01")
    sa = ST.SituationBuilder().build_panel(Xa, now)
    sp = ST.SituationBuilder().build_panel(Xp, now)
    assert len(sa) == len(sp) > 500
    shift = pd.Timedelta(days=rec.shift_days)
    checked = 0
    for (d, t), s in sa.items():
        assert sp[(d + shift, rec.code_map[t])].exact_id == s.exact_id
        checked += 1
    assert checked == len(sa)
    assert set(Xp.index.get_level_values(1)).isdisjoint(codes)      # the presentation really renamed everything


def test_future_data_fails_closed_and_unpublished_macro_is_ignored():
    X = panel()
    with pytest.raises(FirewallBreach):
        ST.SituationBuilder().build_panel(X, X.index.get_level_values(0).max() - pd.Timedelta(days=1))
    with pytest.raises(FirewallBreach):
        ST.SituationBuilder().build({"r5": 0.0}, {}, now=None)
    with pytest.raises(FirewallBreach):
        ST.SituationBuilder().build({"r5": 0.0}, {}, now="2019-03-01", data_asof="2019-03-02")
    macro = [ST.MacroObs("BAMLH0A0HYM2", 9.0, dt.date(2019, 2, 1), dt.date(2019, 4, 1)),      # published AFTER now: must be invisible
             ST.MacroObs("BAMLH0A0HYM2", 3.0, dt.date(2019, 1, 1), dt.date(2019, 1, 8))]
    s = ST.SituationBuilder().build({}, {}, now="2019-03-01", macro=macro)
    assert s.get("macro.credit") == 3.0
    early = ST.SituationBuilder().build({}, {}, now="2018-12-01", macro=macro)
    assert early.get("macro.credit") is None and early.get("macro.state") == "unknown"
    assert ST.macro_obs_from_frame(pd.DataFrame({"UNRATE": [4.0]}, index=[pd.Timestamp("2019-01-01")]), "2019-01-20") == []
    assert len(ST.macro_obs_from_frame(pd.DataFrame({"UNRATE": [4.0]}, index=[pd.Timestamp("2019-01-01")]), "2019-03-20")) == 1
    with pytest.raises(ValueError):
        ST.SituationBuilder().build({}, {}, now="2019-03-01", macro=[ST.MacroObs("X", 1.0, dt.date(2019, 2, 1), dt.date(2019, 1, 1))])


def test_missing_inputs_stay_unknown_and_never_become_zero():
    s = ST.SituationBuilder().build({"r20": float("nan"), "r60": None}, {}, now="2019-03-01")
    assert s.get("trend.r20") is None and s.get("trend.state") is None
    assert s.coverage() < 0.3 and not s.usable(0.5)
    assert s.unknown_state(0.5) == Unknown.INSUFFICIENT_DATA
    assert s.get("regime.label") == "unknown"
    assert "trend.r20" in s.missing_paths()


def test_out_of_range_values_become_unknown_and_are_counted():
    b = ST.SituationBuilder()
    s = b.build({"atr_pct": -0.3, "vol_ratio": -2.0, "r5": 0.01}, {}, now="2019-03-01")
    assert s.get("volatility.atr_pct") is None and s.get("trend.r5") == 0.01
    assert b.dropped["volatility.atr_pct"] == 1 and b.dropped["volatility.vol_ratio"] == 1


def test_empty_and_degenerate_inputs():
    empty = ST.SituationBuilder().build_panel(panel().iloc[0:0], NOW)
    assert len(empty) == 0
    b = ST.SituationBuilder()
    s = b.build({}, {}, now="2019-03-01", patterns=[])
    assert s.pattern_ids == () and s.get("pattern_interaction.mix") == "none"
    cs = ST.CrossSection(pd.DataFrame({"a": [1.0, 2.0]}), min_n=20)
    assert not cs.has("a") and cs.pct("a", 1.0) is None       # too few names for a rank
    assert ST.SituationConfig(min_cs_n=1).validate()
    with pytest.raises(ValueError):
        ST.SituationBuilder(ST.SituationConfig(min_cs_n=1))
    with pytest.raises(ValueError):
        ST.ActivePattern("p", 1.0, direction=0)
    assert ST.correlation_structure(pd.DataFrame(), "a", "2019-01-01")["avg_pair_corr"] is None


def test_absolute_rank_fallback_is_recorded():
    s = ST.SituationBuilder().build({"vol20": 0.02, "log_dv": 16.5}, {}, now="2019-03-01")
    assert s.rank_mode == "absolute" and abs(s.get("volatility.vol_rank") - 0.5) < 1e-6
    t = ST.SituationBuilder().build({}, {}, now="2019-03-01")
    assert t.rank_mode == "cross_section"                   # nothing ranked, nothing to mix


def test_correlation_structure_and_future_rows():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2019-01-01", periods=80)
    common = rng.normal(0, .01, 80)
    R = pd.DataFrame({f"S{i}": .8 * common + .6 * rng.normal(0, .01, 80) for i in range(8)}, index=idx)
    R["MKT"] = common
    out = ST.correlation_structure(R, "S1", idx[-1], market_col="MKT")
    assert out["avg_pair_corr"] > 0.3 and out["corr_to_mkt"] > 0.5 and 0.5 < out["beta"] < 1.5
    assert ST.correlation_structure(R.iloc[:10], "S1", idx[-1], market_col="MKT")["avg_pair_corr"] is None
    with pytest.raises(FirewallBreach):
        ST.correlation_structure(R, "S1", idx[-2])


def test_pattern_interaction_block():
    b = ST.SituationBuilder()
    agree = b.build({}, {}, "2019-03-01", patterns=[ST.ActivePattern("p2", 1.5, 1), ST.ActivePattern("p1", 1.0, 1)])
    clash = b.build({}, {}, "2019-03-01", patterns=[ST.ActivePattern("p1", 1.0, 1), ST.ActivePattern("p2", 1.0, -1)])
    assert agree.pattern_ids == ("p1", "p2") and agree.get("pattern_interaction.mix") == "agree"
    assert clash.get("pattern_interaction.mix") == "conflict" and clash.get("pattern_interaction.agreement") == -1.0
    assert agree.situation_id != clash.situation_id


def test_position_risk_block_validates_state():
    b = ST.SituationBuilder()
    s = b.build({}, {}, "2019-03-01", portfolio=ST.PortfolioState(n_positions=4, gross=0.9, drawdown=-0.07, position_weight=0.1))
    assert s.get("position_risk.state") == "drawdown"
    with pytest.raises(ValueError):
        b.build({}, {}, "2019-03-01", portfolio=ST.PortfolioState(drawdown=0.2))


# ------------------------------------------------------------------------------------------------ utilities

def test_serialisation_roundtrip_and_hash_stability():
    s = mk(patterns=("a", "b"))
    r = ST.Situation.from_dict(s.to_dict())
    assert r.exact_id == s.exact_id and r.situation_id == s.situation_id
    assert ST.from_json(ST.to_json(s)).exact_id == s.exact_id
    assert mk(regime__label="stress").situation_id != s.situation_id


def test_same_bucket_same_situation_id_different_bucket_differs():
    a, b = mk(volatility__atr_pct=0.021), mk(volatility__atr_pct=0.024)
    assert a.situation_id == b.situation_id and a.exact_id != b.exact_id
    assert mk(volatility__atr_pct=0.05).situation_id != a.situation_id


def test_coarsen_marks_dropped_blocks_unknown_not_zero():
    c = ST.coarsen(mk(patterns=("p",)), keep=["regime", "volatility"])
    assert c.get("market.vix") is None and c.get("regime.label") == "bull_calm" and c.pattern_ids == ()
    with pytest.raises(ValueError):
        ST.coarsen(mk(), keep=["nope"])


def test_population_shift_detects_a_planted_regime_shift():
    a = [mk(regime__label="bull_calm") for _ in range(60)]
    b = [mk(regime__label="stress") for _ in range(60)]
    psi = ST.population_shift(a, a)
    assert max(psi.values()) < 1e-9
    shift = ST.population_shift(a, b)
    assert shift["regime.label"] > 1.0 and shift["volatility.vol_rank"] < 1e-9
    assert ST.population_shift(a[:5], b[:5]) == {}


def test_coverage_report_finds_degenerate_dimensions():
    sits = [mk(volatility__vol_rank=v) for v in np.linspace(0.05, 0.95, 40)]
    rep = ST.coverage_report(sits)
    assert "regime.label" in rep["degenerate"] and "volatility.vol_rank" not in rep["degenerate"]
    assert rep["entropy"]["volatility.vol_rank"] > 1.0
    assert ST.coverage_report([])["n"] == 0


def test_identity_recoverability_clean_vs_planted_leak():
    rng = np.random.default_rng(0)
    tks = [f"T{i}" for i in range(15)]
    clean, leaky, labels = [], [], []
    offs = {t: rng.uniform(0.05, 0.95) for t in tks}
    for _ in range(20):
        for t in tks:
            v = float(rng.uniform(.05, .95))
            clean.append(mk(volatility__vol_rank=v, liquidity__dv_rank=float(rng.uniform(.05, .95)), trend__r20=float(rng.normal(0, .06))))
            leaky.append(mk(volatility__vol_rank=v, liquidity__dv_rank=float(np.clip(offs[t] + rng.normal(0, .02), .02, .98)),
                            trend__r20=float(rng.normal(0, .06))))
            labels.append(t)
    r_clean = ST.identity_recoverability(clean, labels, seed=1)
    r_leak = ST.identity_recoverability(leaky, labels, seed=1)
    assert r_clean["p"] > 0.05 and r_clean["lift"] < 0.06
    assert r_leak["p"] < 0.02 and r_leak["lift"] > 0.15                     # the control can fail
    assert np.isnan(ST.identity_recoverability(clean[:5], labels[:5])["accuracy"])


def test_identity_proxy_audit_flags_ticker_constant_column():
    X = panel(n_dates=12, n_tk=20)
    X["log_dv"] = X.index.get_level_values(1).map({t: 10.0 + i for i, t in enumerate(sorted(set(X.index.get_level_values(1))))})
    proxies = ST.identity_proxy_audit(X)
    assert proxies.get("log_dv") == "ticker-constant"
    rep = ST.panel_input_report(X)
    assert "log_dv" in rep["identity_proxies"] and rep["columns_used"] >= 20


def test_library_tiny_bucket_report_and_deepest_rung():
    lib = ST.SituationLibrary()
    common = [mk(volatility__vol_rank=0.5) for _ in range(40)]
    rare = mk(regime__label="stress", trend__state="down", volatility__vol_rank=0.9)
    for i, s in enumerate(common):
        lib.add(s, f"c{i}")
    lib.add(rare, "r0")
    assert len(lib) == 41 and lib.distinct() == 2
    rep = lib.tiny_bucket_report(min_n=30)
    assert rep[0]["tiny"] == 0 and rep[1]["tiny"] == 1
    assert lib.deepest_supported_rung(common[0], 30) == len(ST.POOLING_LADDER) - 1
    assert lib.deepest_supported_rung(rare, 30) == 0
    assert lib.equivalents(common[0]) == tuple(f"c{i}" for i in range(40))
    with pytest.raises(IndexError):
        ST.ladder_key(rare, 99)
    bad = ST.Situation(mk().blocks[:3])
    with pytest.raises(ValueError):
        lib.add(bad, "x")


def test_situation_stream_orders_and_blocks_the_future():
    st = ST.SituationStream(maxlen=3)
    for d, r in (("2019-01-01", "stress"), ("2019-01-02", "bull_calm"), ("2019-01-03", "bull_calm"), ("2019-01-04", "stress")):
        st.push(d, mk(regime__label=r), now="2019-02-01")
    assert len(st) == 3 and st.history()[0].get("regime.label") == "stress" and len(st.history(2)) == 2
    with pytest.raises(ValueError):
        st.push("2019-01-04", mk(), now="2019-02-01")
    with pytest.raises(FirewallBreach):
        st.push("2019-03-01", mk(), now="2019-02-01")


# ------------------------------------------------------------------------------------------------ similarity

def test_similarity_self_symmetry_range_and_vector_agreement():
    rng = np.random.default_rng(0)
    cases = [mk(regime__label=str(rng.choice(["bull_calm", "bull_volatile", "correction_calm"])),
                volatility__vol_rank=float(rng.uniform(.05, .95)), trend__r20=float(rng.normal(0, .06)),
                liquidity__dv_rank=float(rng.uniform(.05, .95)), market__vix=float(rng.uniform(12, 30))) for _ in range(30)]
    rep = SM.consistency_report(cases, seed=1, n_pairs=60)
    assert rep["self_min"] > 0.999 and rep["asym_max"] < 1e-9 and rep["range_ok"] and rep["vector_gap_max"] < 1e-5
    assert SM.compare(cases[0], cases[0]).comparable


def test_similarity_components_are_stored_and_explained():
    a, b = mk(), mk(regime__label="stress", regime__vol_regime="crisis", volatility__vol_rank=0.95)
    r = SM.compare(a, b)
    assert [c.name for c in r.components] == list(SM.COMPONENTS)
    assert r.score("regime") < 0.5 and r.score("liquidity") > 0.99
    assert "regime" in r.vetoes and not r.comparable
    text = r.explain()
    for name in ("regime", "volatility", "VETO"):
        assert name in text
    assert "regime.label" in SM.contrast(a, b)
    d = r.as_dict()
    assert set(d["components"]) == set(SM.COMPONENTS)


def test_missing_data_lowers_coverage_and_can_make_pair_unknown():
    full = mk()
    sparse = ST.coarsen(mk(), keep=["regime"])
    r = SM.compare(full, sparse)
    assert r.coverage < 0.5 and r.unknown == Unknown.INSUFFICIENT_DATA and r.total is None and not r.comparable
    assert "not comparable" in r.explain()
    assert SM.compare(full, ST.coarsen(mk(), keep=["regime", "volatility", "market", "breadth", "macro", "trend"])).total is not None


def test_weights_validation_and_field_weights():
    with pytest.raises(ValueError):
        SM.SimilarityWeights.from_dict({c: 0.0 for c in SM.COMPONENTS})
    assert SM.SimilarityWeights(values=(("regime", 1.0),)).validate()
    assert SM.SimilarityWeights(field_weights=(("no.such", 1.0),)).validate()
    a, b = mk(), mk(volatility__vol_rank=0.95, volatility__atr_pct=0.05)
    heavy = SM.SimilarityWeights(field_weights=(("volatility.vol_rank", 10.0),))
    assert SM.compare(a, b, heavy).score("volatility") < SM.compare(a, b).score("volatility")
    M = SM.SituationMatrix([a, b])
    tot, _, _ = M.totals(a, heavy)
    assert abs(tot[1] - SM.compare(a, b, heavy).total) < 1e-5


def test_pattern_component_uses_jaccard_of_active_patterns():
    assert SM.jaccard(("a", "b"), ("b", "c")) == pytest.approx(1 / 3)
    assert SM.jaccard((), ()) is None
    same = SM.compare(mk(patterns=("p1", "p2")), mk(patterns=("p1", "p2"))).score("pattern")
    diff = SM.compare(mk(patterns=("p1", "p2")), mk(patterns=("p3",))).score("pattern")
    assert same > diff


def test_nearest_returns_the_planted_twin_first_and_is_deterministic():
    rng = np.random.default_rng(3)
    cases = [mk(volatility__vol_rank=float(rng.uniform(.05, .95)), market__vix=float(rng.uniform(12, 30)),
                trend__r20=float(rng.normal(0, .06))) for _ in range(40)]
    q = mk(volatility__vol_rank=0.33, market__vix=21.0, trend__r20=0.02)
    cases[17] = mk(volatility__vol_rank=0.33, market__vix=21.1, trend__r20=0.021)              # the planted twin
    a = SM.nearest(q, cases, k=5)
    b = SM.nearest(q, cases, k=5)
    assert a[0].index == 17 and [n.index for n in a] == [n.index for n in b]
    assert a[0].result.total > a[1].result.total
    assert SM.nearest(q, [], k=3) == [] and SM.nearest(q, cases, k=0) == []
    div = SM.nearest(q, cases + [cases[17]] * 3, k=4, diversity=0.9)
    assert len({cases_i.index for cases_i in div}) == len(div)
    with pytest.raises(ValueError):
        SM.nearest(q, cases, k=2, diversity=1.0)


def test_fit_weights_moves_toward_the_component_that_predicts_outcomes():
    rng = np.random.default_rng(1)
    cases, y = [], []
    for _ in range(160):
        lab = str(rng.choice(["bull_calm", "bull_volatile", "correction_calm", "bear_volatile"]))
        s = mk(regime__label=lab, regime__vol_regime=str(rng.choice(["low", "mid", "high"])),
               liquidity__dv_rank=float(rng.uniform(.05, .95)), volatility__vol_rank=float(rng.uniform(.05, .95)))
        cases.append(s)
        y.append({"bull_calm": .04, "bull_volatile": .01, "correction_calm": -.01, "bear_volatile": -.05}[lab] + rng.normal(0, .003))
    rows, agree = SM.pair_dataset(cases, y, n_pairs=800, seed=2)
    fit = SM.fit_weights(rows, agree, ridge=1.0)
    prior = SM.DEFAULT.normalised()
    assert fit.weights.normalised()["regime"] > prior["regime"] and fit.r2_after >= fit.r2_before - 1e-9
    same = SM.fit_weights(rows[:10], agree[:10])
    assert same.weights == SM.DEFAULT and same.n_pairs == 10                     # too little data: prior unchanged
    with pytest.raises(ValueError):
        SM.fit_weights(rows[:, :3], agree)


def test_calibrator_is_monotone_and_refuses_when_unfitted():
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 1, 400)
    a = (rng.random(400) < t).astype(float)
    cal = SM.SimilarityCalibrator().fit(t, a)
    grid = [cal.predict(x) for x in np.linspace(0.05, 0.95, 20)]
    assert all(g2 >= g1 - 1e-12 for g1, g2 in zip(grid, grid[1:])) and grid[-1] > grid[0] + 0.3
    assert SM.SimilarityCalibrator().predict(0.5) is None
    assert SM.SimilarityCalibrator().fit(t[:10], a[:10]).predict(0.5) is None
    assert cal.lift_over_base(0.9) > 0 > cal.lift_over_base(0.1)


def test_cluster_situations_recovers_planted_equivalence_classes():
    rng = np.random.default_rng(2)
    proto = [dict(regime__label="bull_calm", volatility__vol_rank=.2, market__vix=13.0),
             dict(regime__label="bear_volatile", regime__vol_regime="high", volatility__vol_rank=.9, market__vix=30.0, market__spy_ma200=-0.06),
             dict(regime__label="stress", regime__vol_regime="crisis", volatility__vol_rank=.95, market__vix=45.0, market__spy_ma200=-0.15)]
    cases, truth = [], []
    for k, p in enumerate(proto):
        for _ in range(15):
            q = dict(p)
            q["market__vix"] = p["market__vix"] + float(rng.normal(0, .3))
            q["volatility__vol_rank"] = float(np.clip(p["volatility__vol_rank"] + rng.normal(0, .01), .01, .99))
            cases.append(mk(**q))
            truth.append(k)
    order = rng.permutation(len(cases))
    cases = [cases[i] for i in order]
    truth = [truth[i] for i in order]
    cl = SM.cluster_situations(cases, threshold=0.97)
    assert len(cl) == 3 and all(len(c.members) == 15 for c in cl)
    for c in cl:
        assert len({truth[i] for i in c.members}) == 1 and c.medoid in c.members and c.cohesion > 0.97
    assert SM.cluster_situations([]) == []


def test_novelty_flags_a_situation_unlike_anything_stored():
    rng = np.random.default_rng(0)
    cases = [mk(market__vix=float(rng.uniform(12, 18)), volatility__vol_rank=float(rng.uniform(.3, .5)),
                trend__r20=float(rng.normal(0.03, .01))) for _ in range(60)]
    near = mk(market__vix=15.0, volatility__vol_rank=.4, trend__r20=0.03)
    far = mk(regime__label="stress", regime__vol_regime="crisis", market__vix=60.0, market__spy_ma200=-.3, volatility__vol_rank=.99,
             trend__r20=-.3, trend__state="down", breadth__breadth=.1, breadth__state="washed_out", liquidity__dv_rank=.05, liquidity__state="thin")
    assert SM.novelty(near, cases)["novel"] is False
    assert SM.novelty(far, cases)["novel"] is True
    assert SM.novelty(near, cases[:3])["novel"] is None                  # too few cases to say


def _skill_cases(n, sign, seed=0):
    rng = np.random.default_rng(seed)
    cases, y, t = [], [], []
    d0 = dt.date(2018, 1, 1)
    for i in range(n):
        v = float(rng.uniform(.05, .95))
        cases.append(mk(volatility__vol_rank=v, market__vix=float(rng.uniform(12, 30)), trend__r20=float(rng.normal(0, .06))))
        y.append(sign * (v - .5) * 0.08 + rng.normal(0, .01))
        t.append(d0 + dt.timedelta(days=i))
    return cases, y, t


def test_walk_forward_skill_is_positive_when_similarity_predicts_and_can_fail():
    good = SM.walk_forward_skill(*_skill_cases(260, +1), k=8, min_history=60)
    assert good["status"] in ("PROVEN", "UNPROVEN") and good["mse_skill"] > 0.05 and good["spearman"] > 0.2
    rng = np.random.default_rng(5)
    cases, _, t = _skill_cases(260, +1)
    noise = SM.walk_forward_skill(cases, list(rng.normal(0, .03, 260)), t, k=8, min_history=60)
    assert noise["status"] != "PROVEN"                                       # no structure -> never proven
    # structure that reverses halfway: the early neighbours now mislead, the report must show it, not hide it
    cases2, y2, t2 = _skill_cases(260, +1)
    y2 = [v if i < 130 else -v for i, v in enumerate(y2)]
    flip = SM.walk_forward_skill(cases2, y2, t2, k=8, min_history=60)
    assert flip["mse_skill"] < good["mse_skill"]
    assert SM.walk_forward_skill(cases[:20], [0.0] * 20, t[:20])["status"] == "INSUFFICIENT_EVIDENCE"
    with pytest.raises(ValueError):
        SM.walk_forward_skill(cases, [0.0], t)


def test_temporal_case_index_blocks_outcomes_from_the_future():
    idx = SM.TemporalCaseIndex()
    for i in range(6):
        idx.add(mk(volatility__vol_rank=0.1 * (i + 1)), matured=dt.date(2019, 1, 1 + i), outcome=0.01 * i, ref=f"c{i}", episode="e1" if i < 4 else "e2")
    q = mk(volatility__vol_rank=0.3)
    with pytest.raises(FirewallBreach):
        idx.query(q, "2019-01-04")                            # cases 3..5 matured on/after now
    drop = SM.TemporalCaseIndex(on_future="drop")
    for i in range(6):
        drop.add(mk(volatility__vol_rank=0.1 * (i + 1)), matured=dt.date(2019, 1, 1 + i), outcome=0.01 * i, ref=f"c{i}", episode="e1" if i < 4 else "e2")
    got = drop.query(q, "2019-01-04", k=6)
    assert {n.ref for n in got} == {"c0", "c1", "c2"}
    capped = drop.query(q, "2019-01-10", k=6, max_per_episode=1)
    assert len(capped) == 2 and {drop._cases[n.index][4] for n in capped} == {"e1", "e2"}
    ex = drop.expected_outcome(q, "2019-01-10", k=3, min_total=0.1)
    assert ex["n"] == 3 and ex["mean"] is not None and ex["n_eff"] > 1
    assert SM.TemporalCaseIndex().query(q, "2019-01-04") == []


def test_history_similarity_penalises_a_different_path():
    a, b = mk(), mk()
    same_path = [mk(trend__state="up", trend__r20=0.05)] * 3
    other_path = [mk(trend__state="down", trend__r20=-0.08, trend__r5=-0.04, shock__gap_today=-0.05, shock__max20=0.0, shock__min20=-0.15)] * 3
    r_same = SM.compare_with_history(a, b, same_path, same_path)
    r_diff = SM.compare_with_history(a, b, same_path, other_path)
    assert r_same.score("recent_history") > r_diff.score("recent_history")
    assert r_diff.total < r_same.total
    assert SM.compare_with_history(a, b, [], []).total == SM.compare(a, b).total
    with pytest.raises(ValueError):
        SM.compare_with_history(a, b, same_path, same_path, blend=2.0)


def test_component_diagnostics_run_and_are_bounded():
    rng = np.random.default_rng(0)
    cases = [mk(volatility__vol_rank=float(rng.uniform(.05, .95)), market__vix=float(rng.uniform(12, 30)),
                regime__label=str(rng.choice(["bull_calm", "correction_calm"]))) for _ in range(40)]
    cc = SM.component_correlation(cases, seed=1, n_pairs=200)
    assert all(-1.0 <= v <= 1.0 for v in cc.values())
    tv = SM.triangle_violation_rate(cases, n_triples=80, seed=1)
    assert 0 <= tv["rate"] <= 1
    y = [c.get("volatility.vol_rank") for c in cases]
    disc = SM.discrimination(cases, y, seed=1, n_pairs=300)
    assert disc["volatility"] > 0.2
    assert SM.component_correlation(cases[:2]) == {}


# ------------------------------------------------------------------------------------------------ reuse & extras

def test_memory_context_parity_with_engine_memory():
    from engine.memory import CTX, context_of
    X = panel(n_dates=2, n_tk=25, seed=3)
    day = X.xs(X.index.get_level_values(0)[0], level=0)
    row = day.iloc[[0]]
    s = ST.SituationBuilder().build(day.iloc[0].to_dict(), None, NOW, cross_section=ST.CrossSection(day, 20))
    par = ST.memory_context_parity(s, row)
    assert par["ok"] and par["columns_compared"] == len(CTX)
    assert np.allclose(ST.to_memory_context(s), context_of(row), atol=1e-5)
    blank = ST.SituationBuilder().build({}, {}, NOW)
    assert (ST.to_memory_context(blank) == 0.0).all()                      # unobserved fills 0.0 exactly as memory.context_of does
    assert ST.market_columns_missing({"m_vix": 15.0}) and not ST.market_columns_missing(dict.fromkeys(ST.market_columns(), 1.0))


def test_weight_registry_is_time_safe_and_append_only():
    reg = SM.WeightRegistry()
    w2 = SM.SimilarityWeights.from_dict({**SM.DEFAULT.as_dict(), "regime": 0.5})
    assert reg.register(w2, "2019-01-10", "refit on 2018 outcomes") == 1
    assert reg.in_effect("2019-01-10")[0] == 0 and reg.in_effect("2019-01-11")[0] == 1      # only strictly earlier registrations
    assert reg.in_effect("1990-01-01")[0] == 0
    with pytest.raises(ValueError):
        reg.register(w2, "2019-01-10", "same day")
    with pytest.raises(ValueError):
        reg.register(w2, "2019-02-01", "")
    assert len(reg) == 2 and reg.history()[1][3] == "refit on 2018 outcomes"
    assert "regime" in SM.weights_change(SM.DEFAULT, w2) and SM.weights_change(SM.DEFAULT, SM.DEFAULT) == {}
    assert "vetoes regime<0.40" in SM.describe_weights()


def test_field_importance_learns_which_dimension_matters_and_is_conservative():
    rng = np.random.default_rng(0)
    cases, y = [], []
    for _ in range(200):
        v = float(rng.uniform(.05, .95))
        cases.append(mk(volatility__vol_rank=v, liquidity__dv_rank=float(rng.uniform(.05, .95)), trend__r20=float(rng.normal(0, .06))))
        y.append(v * 0.1 + rng.normal(0, .002))
    w, corr = SM.fit_field_importance(cases, y, n_pairs=3000, seed=1)
    fw = dict(w.field_weights)
    assert corr["volatility.vol_rank"] > 0.3 and fw["volatility.vol_rank"] > fw["liquidity.dv_rank"]
    assert fw["liquidity.dv_rank"] >= 0.2                                                    # never silenced
    same, none = SM.fit_field_importance(cases[:5], y[:5])
    assert same == SM.DEFAULT and none == {}


def test_equivalent_situations_have_more_alike_outcomes_than_chance():
    rng = np.random.default_rng(1)
    cases, y = [], []
    for k in range(6):
        base = dict(volatility__vol_rank=.1 + .15 * k, market__vix=13.0 + 3 * k)
        for _ in range(12):
            cases.append(mk(volatility__vol_rank=float(np.clip(base["volatility__vol_rank"] + rng.normal(0, .002), .01, .99)),
                            market__vix=base["market__vix"] + float(rng.normal(0, .05))))
            y.append(0.01 * k + rng.normal(0, .001))
    good = SM.equivalent_transfer_check(cases, y, threshold=0.985, seed=2)
    assert good["clusters"] >= 4 and good["lift"] > 0 and good["p"] < 0.05
    noise = SM.equivalent_transfer_check(cases, list(rng.normal(0, .02, len(cases))), threshold=0.985, seed=2)
    assert noise["p"] > 0.05                                                                  # equivalence carries no information about noise
    assert np.isnan(SM.equivalent_transfer_check([], [])["lift"])


def test_weight_sensitivity_high_for_a_clear_twin_and_ranking_table_is_complete():
    rng = np.random.default_rng(2)
    cases = [mk(volatility__vol_rank=float(rng.uniform(.05, .95)), market__vix=float(rng.uniform(12, 30))) for _ in range(40)]
    q = mk(volatility__vol_rank=.5, market__vix=20.0)
    cases[3] = mk(volatility__vol_rank=.5, market__vix=20.05)
    sens = SM.weight_sensitivity(q, cases, k=3, trials=10, seed=1)
    assert sens["overlap"] > 0.6 and sens["trials"] == 10
    assert np.isnan(SM.weight_sensitivity(q, cases[:2], k=3)["overlap"])
    rows = SM.ranking_table(q, SM.nearest(q, cases, k=3))
    assert rows[0]["rank"] == 1 and rows[0]["ref"] == "3" and set(SM.COMPONENTS) <= set(rows[0])


def test_similarity_record_roundtrip_cache_and_component_views():
    a, b = mk(), mk(regime__label="stress", regime__vol_regime="crisis")
    r = SM.compare(a, b)
    again = SM.result_from_record(SM.result_to_record(r))
    assert again == r
    bad = SM.result_to_record(r)
    bad["components"] = bad["components"][:-1]
    with pytest.raises(ValueError):
        SM.result_from_record(bad)
    cache = SM.SimilarityCache(maxsize=2)
    assert cache.compare(a, b) == r and cache.compare(b, a) == r and cache.hits == 1 and cache.misses == 1
    heavy = SM.SimilarityWeights(field_weights=(("regime.label", 5.0),))
    assert cache.compare(a, b, heavy) != r                                                   # a different weighting is never served stale
    cache.compare(a, a)
    assert len(cache) == 2
    with pytest.raises(ValueError):
        SM.SimilarityCache(0)
    cases = [mk(volatility__vol_rank=v) for v in (0.1, 0.5, 0.9)] + [ST.coarsen(mk(), keep=["regime"])]
    top = SM.nearest_by_component(mk(volatility__vol_rank=0.52), cases, "volatility", k=3)
    assert top[0][0] == 1 and all(i != 3 for i, _ in top)                                    # the unobservable case is left out, not ranked last
    with pytest.raises(ValueError):
        SM.nearest_by_component(a, cases, "nonsense")
    part = SM.partition_by_veto(a, cases + [b])
    assert part["unknown"] == [3] and part["vetoed"] == [4] and part["comparable"] == [0, 1, 2]
    avail = SM.component_availability(cases[:3], n_pairs=20)
    assert avail["volatility"] == 1.0 and SM.component_availability(cases[:1]) == {}


def test_population_drift_monitor_alarms_on_a_shifted_regime_mix():
    ref = [mk(regime__label="bull_calm", volatility__vol_rank=float(v)) for v in np.linspace(.1, .9, 60)]
    mon = ST.PopulationDriftMonitor(ref, window=40)
    assert mon.report()["status"] == Unknown.INSUFFICIENT_DATA
    for v in np.linspace(.1, .9, 40):
        mon.push(mk(regime__label="bull_calm", volatility__vol_rank=float(v)))
    assert mon.report()["status"] == "STABLE" and not mon.report()["alarmed"]
    for v in np.linspace(.1, .9, 40):
        mon.push(mk(regime__label="stress", regime__vol_regime="crisis", volatility__vol_rank=float(v)))
    rep = mon.report()
    assert rep["status"] == "SHIFTED" and rep["alarmed"][0].startswith("regime.") and rep["n_recent"] == 40
    with pytest.raises(ValueError):
        ST.PopulationDriftMonitor(ref, window=5)
    with pytest.raises(ValueError):
        mon.push(ST.Situation(mk().blocks[:2]))


def test_pairwise_and_stratified_neighbours_and_record_roundtrips():
    rng = np.random.default_rng(4)
    cases = [mk(volatility__vol_rank=float(rng.uniform(.05, .95)), market__vix=float(rng.uniform(12, 30))) for _ in range(24)]
    P = SM.pairwise_totals(cases)
    assert P.shape == (24, 24) and np.allclose(np.nan_to_num(P), np.nan_to_num(P.T), atol=1e-6) and abs(P[0, 0] - 1.0) < 1e-9
    assert SM.pairwise_totals([]).shape == (0, 0)
    strata = ["a"] * 12 + ["b"] * 12
    q = mk(volatility__vol_rank=.5, market__vix=20.0)
    got = SM.stratified_nearest(q, cases, strata, per_stratum=2)
    assert len(got) == 4 and {strata[n.index] for n in got} == {"a", "b"} and [n.result.total for n in got] == sorted([n.result.total for n in got], reverse=True)
    with pytest.raises(ValueError):
        SM.stratified_nearest(q, cases, strata[:3])
    text = SM.explain_neighbours(q, SM.nearest(q, cases, k=2))
    assert text.startswith("#1 case") and "similarity" in text and SM.explain_neighbours(q, []) == "no comparable neighbours"
    w = SM.SimilarityWeights.from_dict({**SM.DEFAULT.as_dict(), "sector": 0.3}, field_weights=(("regime.label", 2.0),))
    assert SM.weights_from_record(SM.weights_to_record(w)) == w
    rec = SM.weights_to_record(w)
    rec["values"]["regime"] = 0.9
    with pytest.raises(ValueError):
        SM.weights_from_record(rec)                                            # edited content no longer matches its id
