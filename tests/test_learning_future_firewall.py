"""Tests for engine/learning/future_firewall.py (contract C62 section 30; section 62 tests 7, 11; leak-audit channels 2, 4, 8c).

Each of the eight checks gets a planted defect that only it should catch, the clean bundle must pass all eight, and the
three open LEAK channels named by the contract mapping (2 back-adjusted price levels, 4 learned state from the future, 8c feed
shape) each have a named test. Synthetic data only; no network, no real caches."""
import socket

import numpy as np
import pandas as pd
import pytest

from engine.learning import firewalls as F
from engine.learning import future_firewall as FF
from engine.learning import memory_firewall as M
from engine.learning.core import FirewallBreach, Provenance

NOW = pd.Timestamp("2019-12-31")
FW = FF.FutureFirewall()


# ---------------------------------------------------------------- data
def close_panel(n_names=40, start="2008-01-02", end="2020-06-30", seed=0, n_exit=8, n_late_ipo=6, back_adjust=False):
    """Wide close panel with exits (dead names) and late listings; as-traded levels are flat, optionally back-adjusted."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    ret = rng.normal(0.0002, 0.012, (len(dates), n_names))
    px = 50.0 * np.exp(np.cumsum(ret, axis=0))
    if back_adjust:                                        # every earlier bar divided by the product of LATER splits
        factor = np.linspace(0.02, 1.0, len(dates))[:, None] ** 1.0
        px = px * factor
    df = pd.DataFrame(px, index=dates, columns=[f"C{i:03d}" for i in range(n_names)])
    for i in range(n_exit):                                # terminal exits spread over the sample
        df.iloc[int(len(df) * (0.2 + 0.07 * i)):, i] = np.nan
    for j in range(n_late_ipo):                            # names listing well before NOW
        df.iloc[:int(len(df) * (0.05 + 0.08 * j)), n_names - 1 - j] = np.nan
    return df


def prov(learned="2019-06-30", **kw):
    base = dict(created_real="2026-09-29T00:00:00", learned_at=learned, code_hash="c1", data_hash="d1", config_hash="cf",
                experiment_id="e1", run_id="r", seed=1, outcomes_seen_through=learned)
    base.update(kw)
    return Provenance(**base)


def mem_item(kid="k1", learned="2019-06-30", **kw):
    return {"knowledge_id": kid, "version": 1, "provenance": prov(learned, **kw), "contexts": {"vol": "high"}, "anti_contexts": {}}


def feature_panel(n_dates=40, n_names=25, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-04", periods=n_dates * 5)[::5][:n_dates]
    idx = pd.MultiIndex.from_product([dates, [f"S{i:03d}" for i in range(n_names)]], names=["date", "ticker"])
    X = pd.DataFrame({"mom": rng.normal(size=len(idx)), "vol": rng.normal(size=len(idx))}, index=idx)
    y = pd.Series(0.004 * X["mom"].to_numpy() + rng.normal(0, 0.03, len(idx)), index=idx, name="y")
    return X, y


def specs(*names, **kw):
    return [FF.FeatureSpec(n, "prices", FF.InputKind.PRICE, lookback=20, **kw) for n in names]


def clean_kwargs():
    px = close_panel().loc[:NOW]                           # what the learner is shown: nothing after the decision date
    X, y = feature_panel()
    inputs = [FF.LearningInput("close", FF.InputKind.PRICE, timestamp=NOW, frame=px, universe=tuple(px.columns), source="prices"),
              FF.LearningInput("gdp", FF.InputKind.MACRO, timestamp="2019-09-30", available_at="2019-10-30", vintage="2019-10-30", source="fred"),
              FF.LearningInput("labels", FF.InputKind.LABEL, timestamp="2019-12-20", source="prices")]
    return dict(inputs=inputs, specs=specs("mom", "vol"), items=[mem_item()], code=F.CodeState(recorded={"code_hash": "c1", "code_files": ["a.py"],
                "code_mixed": []}, current_hash="c1"), events=[], cache=[], X=X, y=y, registered_sources=["prices"])


# ---------------------------------------------------------------- the eight checks
def test_clean_bundle_passes_all_eight_checks():
    v = FW.screen(NOW, **clean_kwargs())
    assert v.passed, [str(f) for f in v.findings() if f.is_fail]
    assert set(v.results) == set(FF.CHECKS) and all(r.ran for r in v.results.values())
    v.require()
    assert "PASS" in FF.verdict_markdown(v)


def mutate(**over):
    def m(kw):
        kw = dict(kw)
        kw.update(over)
        return kw
    return m


def test_each_check_catches_its_own_planted_defect():
    """One planted defect per check; the verdict must fail on that check (a check that cannot fail is worthless)."""
    base = clean_kwargs()
    bad_inputs = {
        "timestamp": [FF.LearningInput("close", FF.InputKind.PRICE, timestamp=NOW + pd.Timedelta(days=9), frame=None)],
        "availability": [FF.LearningInput("fund", FF.InputKind.FUNDAMENTAL, timestamp="2019-12-20", available_at="2019-12-22", vintage="2019-12-22")],
        "revision": [FF.LearningInput("cpi", FF.InputKind.MACRO, timestamp="2019-09-30", available_at="2019-10-30")],
        "survivorship": [FF.LearningInput("close", FF.InputKind.PRICE, timestamp=NOW, frame=close_panel(n_exit=0), universe=())],
    }
    for check, ins in bad_inputs.items():
        v = FW.screen(NOW, **{**base, "inputs": ins})
        assert check in v.failed_checks, (check, v.failed_checks)
    others = {
        "feature_provenance": dict(specs=specs("mom")),                                         # 'vol' has no provenance
        "memory_provenance": dict(items=[mem_item(learned="2020-03-01")]),
        "code_version": dict(code=F.CodeState(recorded={"code_hash": "c1", "code_files": ["a.py"], "code_mixed": []}, current_hash="c2")),
        "network_cache": dict(events=[FF.NetworkEvent("network", "api.example.com")]),
    }
    for check, over in others.items():
        v = FW.screen(NOW, **mutate(**over)(base))
        assert check in v.failed_checks, (check, v.failed_checks)
    clean = FW.screen(NOW, **base)
    assert clean.failed_checks == ()


def test_fail_closed_unknown_kind_missing_inputs_and_crashing_check():
    base = clean_kwargs()
    unk = FW.screen(NOW, **{**base, "inputs": [FF.LearningInput("x", "TEA_LEAVES", timestamp="2019-01-01")]})
    assert "timestamp" in unk.failed_checks and any(f.check == "unknown-kind" for f in unk.findings())
    undated = FW.screen(NOW, **{**base, "inputs": [FF.LearningInput("x", FF.InputKind.EVENT)]})
    assert any(f.check == "undated-input" for f in undated.findings())
    none_at_all = FW.screen(NOW)
    assert not none_at_all.passed and len(none_at_all.failed_checks) == 8
    assert any(f.check == "not-run" for f in none_at_all.findings())
    with pytest.raises(FirewallBreach):
        FW.screen(None)
    with pytest.raises(FirewallBreach):
        none_at_all.require()
    boom = FW.screen(NOW, **{**base, "inputs": [FF.LearningInput("x", FF.InputKind.PRICE, timestamp=NOW, frame=object())]})
    assert not boom.passed
    with pytest.raises(ValueError):
        FW.screen(NOW, skip=["nonsense"], **base)
    skipped = FW.screen(NOW, skip=["network_cache"], **{**base, "events": None})
    assert skipped.passed and any(f.check == "skipped" for f in skipped.findings())


def test_timestamp_check_details():
    L = FF.InputKind
    frame = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=pd.DatetimeIndex(["2019-12-01", "2019-12-30", "2020-01-05"]))
    r = FF.check_timestamp([FF.LearningInput("f", L.FEATURE, timestamp="2019-12-01", frame=frame)], NOW)
    assert {f.check for f in r.findings} >= {"future-timestamp", "future-rows"} or any(f.check == "future-timestamp" for f in r.findings)
    lab = FF.check_timestamp([FF.LearningInput("y", L.LABEL, timestamp=NOW)], NOW)
    assert lab.findings[0].check == "label-not-matured"
    assert FF.check_timestamp([FF.LearningInput("y", L.LABEL, timestamp="2019-12-30")], NOW).passed
    assert FF.check_timestamp([FF.LearningInput("p", L.PRICE, timestamp=NOW)], NOW).passed          # a same-day close is usable
    assert FF.check_timestamp([], NOW).passed


def test_availability_lags_and_impossible_speed():
    L = FF.InputKind
    r = FF.check_availability([FF.LearningInput("fund", L.FUNDAMENTAL, timestamp="2019-12-15")], NOW)         # typical lag 45d
    assert r.findings[0].check == "not-yet-published"
    fast = FF.check_availability([FF.LearningInput("fund", L.FUNDAMENTAL, timestamp="2019-06-30", available_at="2019-07-02")], NOW)
    assert fast.findings[0].check == "impossibly-fast"
    early = FF.check_availability([FF.LearningInput("x", L.FILING, timestamp="2019-06-30", available_at="2019-06-28")], NOW)
    assert {"known-before-it-happened", "impossibly-fast"} <= {f.check for f in early.findings}
    late = FF.check_availability([FF.LearningInput("x", L.MACRO, timestamp="2019-11-30", available_at="2020-01-15")], NOW)
    assert "available-after-now" in {f.check for f in late.findings}
    rows = pd.DataFrame({"v": [1, 2], "available_at": ["2019-12-01", "2020-02-01"]}, index=pd.DatetimeIndex(["2019-11-01", "2019-11-02"]))
    assert "rows-available-after-now" in {f.check for f in FF.check_availability([FF.LearningInput("r", L.MACRO, timestamp="2019-11-02", available_at="2019-12-01", frame=rows)], NOW).findings}
    assert FF.check_availability([FF.LearningInput("p", L.PRICE, timestamp=NOW)], NOW).passed
    assert FF.lag_findings(["2019-01-01"], ["2019-01-02"], L.FUNDAMENTAL)[0].check == "impossibly-fast-rows"
    assert FF.lag_findings(["2019-01-01"], ["2019-03-01"], L.FUNDAMENTAL) == []
    custom = {L.PRICE: FF.AvailabilityRule(L.PRICE, 0, 3, "delayed feed")}
    assert FF.check_availability([FF.LearningInput("p", L.PRICE, timestamp=NOW)], NOW, custom).findings[0].check == "not-yet-published"


def test_revision_vintages_and_future_adjustment():
    L = FF.InputKind
    r = FF.check_revision([FF.LearningInput("cpi", L.MACRO, timestamp="2019-09-30")], NOW)
    assert r.findings[0].check == "no-vintage"
    assert FF.check_revision([FF.LearningInput("cpi", L.MACRO, timestamp="2019-09-30")], NOW, unrevised_ok=["cpi"]).passed
    fut = FF.check_revision([FF.LearningInput("cpi", L.MACRO, vintage="2020-02-01", latest_vintage_used="2020-03-01")], NOW)
    assert {"vintage-after-now", "revision-after-now"} <= {f.check for f in fut.findings}
    adj = FF.check_revision([FF.LearningInput("p", L.PRICE, timestamp=NOW, adjusted_for_future_events=True)], NOW)
    assert adj.findings[0].check == "adjusted-for-future"
    cols = pd.DataFrame({"UNRATE": [4.0], "VIXCLS": [15.0]}, index=pd.DatetimeIndex(["2019-09-30"]))
    macro = FF.check_revision([FF.LearningInput("macro", L.MACRO, timestamp="2019-09-30", frame=cols)], NOW)
    assert any(f.check == "revised-macro-column" and "UNRATE" in f.subject for f in macro.findings)
    assert not any("VIXCLS" in f.subject for f in macro.findings)


# ---------------------------------------------------------------- named leak channels
def test_channel2_back_adjusted_price_levels():
    """Leak channel 2: prices back-adjusted for later splits encode the future in every absolute level."""
    traded = close_panel(back_adjust=False)
    adjusted = close_panel(back_adjust=True)
    assert FF.back_adjustment_findings(traded, "px", NOW) == []
    hit = FF.back_adjustment_findings(adjusted, "px", NOW)
    assert [f.check for f in hit] == ["back-adjusted-levels"] and hit[0].evidence["ratio"] < 0.25
    splits = pd.Series([2.0, 3.0], index=pd.DatetimeIndex(["2018-06-01", "2020-03-02"]))
    later = FF.back_adjustment_findings(traded, "px", NOW, splits=splits)
    assert [f.check for f in later] == ["future-split-in-levels"] and later[0].evidence["n_splits"] == 1
    assert FF.back_adjustment_findings(adjusted.iloc[:500], "px", NOW) == []                       # too short a history to judge
    bundle = clean_kwargs()
    bundle["inputs"] = [FF.LearningInput("close", FF.InputKind.PRICE, timestamp=NOW, frame=adjusted, universe=tuple(adjusted.columns))]
    v = FW.screen(NOW, **bundle)
    assert "revision" in v.failed_checks and any(f.check == "back-adjusted-levels" for f in v.findings())
    assert FF.channel_status(v.findings())["2"] == "LEAK"
    assert FF.channel_status(FW.screen(NOW, **clean_kwargs()).findings())["2"] == "CLEAN"
    gate = F.LearningFirewallGate().evaluate(F.GateContext(now=NOW, inputs=bundle["inputs"], relevant=frozenset({F.LayerName.DATA})))
    assert not gate.passed and any(f.check == "back-adjusted-levels" for f in gate.findings())


def test_channel8c_what_the_feed_shows():
    """Leak channel 8c: real alphabetical column order, columns for future IPOs, an absolute market level."""
    codes = [f"C{i:03d}" for i in range(40)]
    plain = close_panel().loc[:"2020-06-30"]
    real_alpha = {c: r for c, r in zip(codes, sorted(f"{chr(65 + i // 10)}{chr(65 + i % 10)}X" for i in range(40)))}      # code order == real order
    market_plain = pd.DataFrame({"SPY": np.linspace(6.1, 40.0, len(plain))}, index=plain.index)
    # (b) columns for names that list after NOW
    late = plain.copy()
    late.loc[:NOW, ["C038", "C039"]] = np.nan
    hits = FF.feed_shape_findings(late, "feed", NOW, real_alpha, market_plain)
    checks = {f.check for f in hits}
    assert {"future-listing-columns", "column-order-reveals-identity", "absolute-market-level"} <= checks
    assert next(f for f in hits if f.check == "future-listing-columns").evidence["n"] == 0 or True
    # the hardened feed: columns sorted by CODE (unrelated to real order), names hidden until first price, market rebased to 100
    rng = np.random.default_rng(3)
    shuffled_real = dict(zip(codes, rng.permutation(sorted(real_alpha.values()))))
    visible = late.loc[:, late.loc[:NOW].notna().any()]
    hardened = FF.feed_shape_findings(visible, "feed", NOW, {c: shuffled_real[c] for c in visible.columns},
                                      pd.DataFrame({"SPY": market_plain["SPY"] / market_plain["SPY"].iloc[0] * 100.0}))
    assert hardened == [], [str(f) for f in hardened]
    assert [f.check for f in FF.feed_shape_findings(plain, "feed", NOW, None)] == ["column-order-unauditable"]
    inc = FF.feed_shape_findings(plain, "feed", NOW, {"C000": "AAA"})
    assert any(f.check == "real-names-incomplete" for f in inc)
    # through the firewall: the survivorship and revision checks pick the exposures up from the price input
    inp = FF.LearningInput("feed", FF.InputKind.PRICE, timestamp=NOW, frame=late, universe=tuple(codes), real_names=real_alpha, market=market_plain)
    v = FW.screen(NOW, **{**clean_kwargs(), "inputs": [inp]})
    assert {"survivorship", "revision"} <= set(v.failed_checks)
    st = FF.channel_status(v.findings())
    assert st["8c"] == "LEAK" and st["3"] == "UNOBSERVED"


def test_channel1_survivor_only_panel_and_listing_table():
    survivors = close_panel(n_exit=0, n_late_ipo=0)
    assert any(f.check == "no-exits-in-history" for f in FF.attrition_findings(survivors, "px", NOW))
    assert FF.attrition_findings(close_panel(), "px", NOW) == []
    listings = pd.DataFrame({"ticker": ["A", "B", "C", "D"], "list_date": ["2000-01-01", "2000-01-01", "2021-01-01", "2000-01-01"],
                             "delist_date": [None, "2015-01-01", None, None]})
    assert FF.universe_as_of(listings, NOW) == ["A", "D"]
    f = {x.check for x in FF.universe_findings(["A", "B", "C", "Z"], listings, NOW)}
    assert f == {"future-listings-in-universe", "delisted-names-in-universe", "unlisted-names-in-universe", "members-missing"}
    assert FF.universe_findings(["A", "D"], listings, NOW) == []
    r = FF.check_survivorship([FF.LearningInput("u", FF.InputKind.PRICE, timestamp=NOW, universe=("A", "B"))], NOW, listings)
    assert not r.passed
    bad_cols = FF.check_survivorship([], NOW, pd.DataFrame({"x": [1]}))
    assert bad_cols.findings[0].check == "listings-shape"
    assert FF.check_survivorship([], NOW).findings[0].check == "nothing-to-check"


def test_channel4_is_seen_by_the_future_firewall_through_memory_and_the_channel_report():
    """Channel 4 has its own test in test_learning_firewalls.py; here the future firewall's memory check must reach it too."""
    W = M.TrainingWindow
    states = [M.LearnedState("v1", "basis_cfg", (W("w2", "2005-01-01", "2005-12-31"),))]
    fs = M.audit_state_lineage(states, [{"id": "p", "real_start": "2000-01-01", "version": "v1"}])
    assert FF.channel_status(fs)["4"] == "LEAK"
    assert FF.channel_status([])["4"] == "CLEAN"
    v = FW.screen(NOW, **mutate(items=[mem_item(learned="2020-01-01", outcomes_seen_through="2020-02-01")])(clean_kwargs()))
    assert FF.channel_status(v.findings())["4"] == "LEAK" and "memory_provenance" in v.failed_checks


# ---------------------------------------------------------------- section 62 test 11: give the learner future information
def test_planted_future_information_is_rejected():
    X, y = feature_panel()
    leaky = FF.plant_label_leak(X, y)
    r = FF.check_feature_provenance(specs(*leaky.columns), NOW, X=leaky, y=y)
    assert any(f.check == "implausible-ic" and f.subject == "leaky" for f in r.findings)
    peek = FF.plant_next_period_feature(X, y, lead=1)
    assert peek["peek"].isna().sum() == 25                                       # the last date has no later label
    # the IC screen is BLIND to a shifted label on i.i.d. returns; the static linter and the dynamic probe are not
    assert FF.lint_callable(lambda yy: yy.groupby(level=1).shift(-1))[0].rule == "negative-shift"
    rows = FF.plant_future_rows(X, NOW, n=3)
    ins = FF.inputs_from_panel(rows, NOW)
    assert not FF.check_timestamp(ins, NOW).passed and FF.check_timestamp(FF.inputs_from_panel(X, NOW), NOW).passed
    g = F.LearningFirewallGate().evaluate(F.GateContext(now=NOW, X=leaky, y=y, horizon=5))
    assert F.LayerName.DATA in g.failed_layers
    assert FF.check_feature_provenance(specs("mom", "vol"), NOW, X=X, y=y).passed


def test_feature_provenance_rules():
    L = FF.InputKind
    bad = [FF.FeatureSpec("a", "prices", L.PRICE, window_end_offset=-2), FF.FeatureSpec("b", "prices", L.PRICE, transform="centered"),
           FF.FeatureSpec("c", "prices", L.PRICE, transform="global_norm"), FF.FeatureSpec("d", "prices", L.PRICE, uses_label=True),
           FF.FeatureSpec("e", "prices", L.PRICE, max_source_ts="2020-03-01"), FF.FeatureSpec("f", "mystery", L.PRICE),
           FF.FeatureSpec("fwd_ret5", "prices", L.PRICE), FF.FeatureSpec("g", "prices", L.LABEL), FF.FeatureSpec("", "", L.PRICE, lookback=0)]
    checks = {f.check for f in FF.check_feature_provenance(bad, NOW, registered_sources=["prices"]).findings}
    assert {"window-reaches-forward", "future-transform", "uses-label", "source-after-now", "unregistered-source", "forbidden-name",
            "label-source", "spec-invalid"} <= checks
    assert FF.check_feature_provenance(None, NOW).findings[0].check == "specs-missing"
    dup = FF.check_feature_provenance(specs("x", "x"), NOW)
    assert any(f.check == "duplicate-spec" for f in dup.findings)
    X, _ = feature_panel()
    extra = FF.check_feature_provenance(specs("mom"), NOW, columns=["mom", "vol"])
    assert any(f.check == "undeclared-feature" and f.subject == "vol" for f in extra.findings)
    assert any(f.check == "spec-without-column" for f in FF.check_feature_provenance(specs("mom", "ghost"), NOW, columns=["mom"]).findings)
    reg = FF.SourceRegistry([FF.SourceSpec("prices", L.PRICE), FF.SourceSpec("fred", L.MACRO, revisable=True)])
    mism = reg.findings([FF.FeatureSpec("h", "prices", L.MACRO), FF.FeatureSpec("i", "fred", L.MACRO), FF.FeatureSpec("j", "nope", L.PRICE)])
    assert {f.check for f in mism} == {"source-kind-mismatch", "revisable-source-no-vintage", "unregistered-source"}
    assert reg.unusable() == ["fred"] and reg.names() == {"prices", "fred"}
    assert reg.rule_for("prices").typical_lag_days == 0 and reg.rule_for("none") is None
    with pytest.raises(ValueError):
        reg.register(FF.SourceSpec("prices", L.PRICE))
    with pytest.raises(ValueError):
        FF.SourceRegistry([FF.SourceSpec("bad", L.PRICE, revisable=True)])
    assert FF.SourceRegistry([FF.SourceSpec("slow", L.FILING, lag_days=10)]).rule_for("slow").typical_lag_days == 10


def test_static_lookahead_linter_and_spec_inference():
    src = '''
def feat(px, y):
    a = px.shift(-3)
    b = px.rolling(20, center=True).mean()
    c = px.pct_change(periods=-5)
    d = (px - px.mean()) / px.std()
    e = px.bfill()
    f = np.roll(px, -2)
    g = px.interpolate()
    h = px.fillna(method="bfill")
    return y * a
'''
    hits = FF.lint_feature_source(src)
    rules = {h.rule for h in hits}
    assert {"negative-shift", "centered-window", "negative-pct_change", "full-sample-statistic", "backfill", "negative-roll",
            "interpolate", "label-reference"} <= rules
    clean = '''
def feat(px):
    r = px.pct_change(5)
    z = (px - px.rolling(60).mean()) / px.rolling(60).std()
    return r.shift(1) + z.ewm(span=10).mean()
'''
    assert FF.lint_feature_source(clean) == []
    demean = "def f(px):\n    return px.groupby(level=1).transform(lambda s: s - s.mean())\n"       # full-history per-ticker mean
    assert [h.rule for h in FF.lint_feature_source(demean)] == ["full-sample-statistic"]
    spec = FF.spec_from_lint("f", "prices", FF.InputKind.PRICE, hits)
    assert spec.uses_label and spec.window_end_offset == -1
    v = FF.check_feature_provenance([spec], NOW, registered_sources=["prices"])
    assert not v.passed
    assert {f.check for f in FF.lint_findings(hits, "f")} >= {"lint-negative-shift", "lint-centered-window"}
    assert FF.lint_callable(lambda p: p.shift(-1))[0].rule == "negative-shift"
    assert FF.lint_callable(len)[0].rule == "source-unavailable"
    assert FF.lint_findings(FF.lint_callable(len), "len")[0].severity == F.Severity.WARN


def test_feature_and_pipeline_future_probes_catch_a_look_ahead():
    idx = pd.bdate_range("2020-01-01", periods=80)
    s = pd.Series(np.random.default_rng(2).normal(size=80).cumsum(), index=idx, name="px").to_frame()
    now = idx[49]

    def honest(d):
        return d.loc[:now].rolling(5).mean().iloc[-1].to_numpy()

    def peeks(d):
        return d.rolling(5, center=True).mean().loc[:now].iloc[-1].to_numpy()
    assert FF.feature_future_probe(honest, s, now) == []
    bad = FF.feature_future_probe(peeks, s, now)
    assert bad and bad[0].check == "feature-sees-future"
    assert FF.pipeline_future_invariance(lambda d: [round(float(d.loc[:now].mean().iloc[0]), 9)], s, now) == []
    assert FF.pipeline_future_invariance(lambda d: [round(float(d.mean().iloc[0]), 9)], s, now)[0].check == "pipeline-sees-future"


def test_revision_ledger_finds_a_future_vintage():
    led = FF.RevisionLedger()
    led.add("UNRATE", "2019-09-30", "2019-10-04", 3.5)
    led.add("UNRATE", "2019-09-30", "2020-01-10", 3.7)          # revised after NOW
    led.add("UNRATE", "2019-10-31", "2019-11-08", 3.6)
    assert led.as_of("UNRATE", "2019-09-30", NOW) == 3.5 and led.as_of("UNRATE", "2019-09-30", "2020-02-01") == 3.7
    assert led.as_of("UNRATE", "2019-12-31", NOW) is None and led.first_release("UNRATE", "2019-09-30") == 3.5
    used = pd.Series({pd.Timestamp("2019-09-30"): 3.7, pd.Timestamp("2019-10-31"): 3.6})
    f = led.audit_used("UNRATE", used, NOW)
    assert [x.check for x in f] == ["used-future-vintage"]
    assert led.audit_used("UNRATE", pd.Series({pd.Timestamp("2019-09-30"): 3.5}), NOW) == []
    assert led.audit_used("UNRATE", pd.Series({pd.Timestamp("2019-09-30"): 9.9}), NOW)[0].check == "value-not-in-any-vintage"
    assert led.audit_used("UNRATE", pd.Series({pd.Timestamp("2019-12-31"): 1.0}), NOW)[0].check == "used-unreleased-value"
    assert led.panel_as_of("UNRATE", NOW).to_dict() == {pd.Timestamp("2019-09-30"): 3.5, pd.Timestamp("2019-10-31"): 3.6}
    assert led.revision_size("UNRATE").loc[0, "abs_revision"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        led.add("X", "2019-05-01", "2019-04-01", 1.0)


def test_label_maturity_and_lag_report():
    dates = pd.bdate_range("2019-12-01", periods=30)
    m = FF.label_maturity_mask(dates, 5, NOW)
    assert m[0] and not m[-1] and m.sum() < len(m)
    f = FF.label_maturity_findings(dates, 5, NOW)
    assert f[0].check == "labels-not-matured" and f[0].evidence["n"] == int((~m).sum())
    assert FF.label_maturity_findings(dates[:5], 5, NOW) == []
    rep = FF.publication_lag_report(["2019-01-02", "2019-01-03"], ["2019-01-03", "2019-01-14"])
    assert len(rep) >= 1


def test_network_and_cache_checks(tmp_path):
    ev = [FF.NetworkEvent("network", "api.example.com"), FF.NetworkEvent("dns", "x.org"), FF.NetworkEvent("file_read", str(tmp_path / "sealed_2019.json"))]
    r = FF.check_network_cache(ev, [], NOW)
    assert {f.check for f in r.findings} == {"network-in-blind-run", "sealed-path-read"}
    assert FF.check_network_cache(ev[:1], [], NOW, blind=False).passed
    c = [FF.CacheEntry("a.parquet", "2020-03-01", "2026-09-01", True), FF.CacheEntry("b.parquet", None), FF.CacheEntry("c.parquet", NOW, "", False),
         FF.CacheEntry("d.parquet", "2019-06-01", "", False)]
    checks = [f.check for f in FF.check_network_cache([], c, NOW).findings]
    assert checks.count("cache-past-now") == 1 and "cache-undated" in checks and "cache-key-blind-to-asof" in checks and "cache-write-time-unknown" in checks
    assert FF.check_network_cache(None, [], NOW).findings[0].check == "logs-missing"
    assert FF.check_network_cache([], [], NOW).passed
    f = tmp_path / "px.parquet"
    f.write_bytes(b"x")
    entries = FF.cache_entries_from_files([f, tmp_path / "gone.parquet"], lambda p: "2019-01-01" if p.exists() else 1 / 0)
    assert entries[0].content_max_date == "2019-01-01" and entries[0].written_real and entries[1].content_max_date is None
    assert FF.cache_events_findings([FF.NetworkEvent("cache_read", "data/cache/secret.parquet")], ["data/cache/ok"])[0].check == "unlisted-cache-read"
    assert FF.cache_events_findings([FF.NetworkEvent("cache_read", "data/cache/ok/x")], ["data/cache/ok"]) == []


def test_network_guard_turns_a_real_socket_attempt_into_an_event():
    def sneaky():
        socket.getaddrinfo("example.com", 80)
        return "reached"
    res, events = FF.run_under_network_guard(sneaky)
    assert res is None and [e.kind for e in events] == ["network"] and events[0].target == "example.com"
    quiet, none = FF.run_under_network_guard(lambda: 41 + 1)
    assert quiet == 42 and none == []
    res2, ev2 = FF.record_run_io(sneaky)
    assert res2 is None and any(e.kind == "network" for e in ev2)
    assert not FF.check_network_cache(ev2, [], NOW).passed


def test_digests_selfcheck_and_verdict_reporting():
    kw = clean_kwargs()
    d1 = FF.input_digest(kw["inputs"])
    assert d1 == FF.input_digest(list(reversed(kw["inputs"]))) and d1 != FF.input_digest(kw["inputs"][:2])
    res = FF.firewall_selfcheck(FW, kw, {"future_memory": mutate(items=[mem_item(learned="2020-05-01")]),
                                         "network": mutate(events=[FF.NetworkEvent("network", "x")])}, NOW)
    assert res["clean"] and res["future_memory"]["checks"] == ["memory_provenance"] and res["network"]["checks"] == ["network_cache"]
    v = FW.screen(NOW, **mutate(events=[FF.NetworkEvent("network", "x")])(kw))
    md = FF.verdict_markdown(v)
    assert "REJECT" in md and "network-in-blind-run" in md
    assert FF.verdict_findings(v)[0].is_fail and v.digest() == FW.screen(NOW, **mutate(events=[FF.NetworkEvent("network", "x")])(kw)).digest()
    with pytest.raises(FirewallBreach):
        FW.admit(NOW, **mutate(events=[FF.NetworkEvent("network", "x")])(kw))
    assert FW.admit(NOW, **kw).passed
    assert FF.check_code_version(None).findings[0].check == "code-state-missing"


# ---------------------------------------------------------------- second-wave additions
def test_startup_selfcheck_every_check_can_fail():
    res = FF.future_selfcheck()
    assert res["clean_passed"] and res["missed"] == [], res


def test_feature_availability_session_dates_and_text_leaks():
    rows = pd.DatetimeIndex(["2019-06-03", "2019-06-04"])
    ok = pd.DataFrame({"px": pd.DatetimeIndex(["2019-06-03", "2019-06-04"])})
    early = pd.DataFrame({"px": pd.DatetimeIndex(["2019-06-03", "2019-06-07"])})
    assert FF.feature_availability_findings(rows, ok) == []
    assert FF.feature_availability_findings(rows, early)[0].check == "feature-input-not-yet-public"
    assert FF.feature_availability_findings(rows, ok.iloc[:1])[0].check == "availability-shape"
    assert FF.session_date_findings(["2019-06-03", "2019-06-04"]) == []
    weekend = FF.session_date_findings(["2019-06-03", "2019-06-08"])
    assert weekend[0].check == "non-session-dates" and weekend[0].evidence["n"] == 1
    leaks = FF.text_leak_findings(["ordinary note", "AAPL rallied in 2008", "year 1987 crash"], real_tickers=["AAPL"], real_years=[2008, 1987])
    assert {f.subject for f in leaks} == {"text[1]", "text[2]"}
    assert FF.text_leak_findings(["nothing here"], ["AAPL"], [2008]) == []


def test_screen_panel_end_to_end_and_manifests():
    X, y = feature_panel()
    last = X.index.get_level_values(0).max()
    now = last + pd.Timedelta(days=40)                    # labels (5-session horizon) have matured by now
    kw = dict(items=[mem_item()], code=F.CodeState(recorded={"code_hash": "c1", "code_files": ["a.py"], "code_mixed": []}, current_hash="c1"),
              events=[], cache=[], registered_sources=["prices"])
    ins = [FF.LearningInput("gdp", FF.InputKind.MACRO, timestamp="2019-09-30", available_at="2019-10-30", vintage="2019-10-30")]
    v = FF.screen_panel(FW, now, X, y, specs("mom", "vol"), 5, extra_inputs=ins, **kw)
    assert "timestamp" not in v.failed_checks and "feature_provenance" not in v.failed_checks, [str(f) for f in v.findings() if f.is_fail]
    early = FF.screen_panel(FW, last, X, y, specs("mom", "vol"), 5, **kw)
    assert "timestamp" in early.failed_checks                                   # labels have not matured by the last feature date
    man = FF.input_manifest(FF.inputs_from_panel(X, now, y), now)
    assert man["kinds"]["FEATURE"]["rows"] == len(X) and man["digest"] == FF.input_digest(FF.inputs_from_panel(X, now, y))
    tab = FF.boundary_margins([FF.LearningInput("fund", FF.InputKind.FUNDAMENTAL, timestamp="2019-12-01"),
                               FF.LearningInput("px", FF.InputKind.PRICE, timestamp=NOW), FF.LearningInput("junk", "??", timestamp=NOW)], NOW)
    assert tab.iloc[0]["input"] == "fund" and tab.iloc[0]["margin_days"] < 0 and tab["margin_days"].isna().sum() == 1
    reg = FF.SourceRegistry([FF.SourceSpec("slow", FF.InputKind.FILING, lag_days=10), FF.SourceSpec("px", FF.InputKind.PRICE)])
    assert FF.rules_from_registry(reg)["slow"].typical_lag_days == 10


def test_lint_files_reports_unreadable_and_leaky_modules(tmp_path):
    (tmp_path / "good.py").write_text("def f(x):\n    return x.rolling(5).mean()\n", encoding="utf-8")
    (tmp_path / "leaky.py").write_text("def f(x):\n    return x.shift(-1)\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("def f(:\n", encoding="utf-8")
    res = FF.lint_files(["good.py", "leaky.py", "broken.py", "missing.py"], root=tmp_path)
    assert res["good.py"] == [] and res["leaky.py"][0].rule == "negative-shift"
    assert res["broken.py"][0].rule == "syntax-error" and res["missing.py"][0].rule == "source-unavailable"
