"""F28 (C75 Phase 3 / Phase 10 repair loop; canon C70-C74; Firewalls 5, 6, 10): the F26 remainder. Each mechanism has a planted case
it must catch, a null case where it must find nothing, and the empty case. Synthetic data only.

  1 proxies          evidence.incremental + quality_gate.rival_check: a candidate that adds nothing beyond its strongest correlated
                     rival FAILS; the real pattern it imitates keeps its increment; mutual near-copies keep the stronger one
  2 identity units   evidence.name_units + quality_gate.name_units_check: a per-name artefact is tested with names as the units
  3 weak leaks       the per-candidate suspicion tier (UNKNOWN, never QUARANTINE) and documented_availability (a dated scheduled
                     event is not refused; a false record is itself a finding)
  4 defects a-d      calibration.platt_slope damped; complexity worst fold in each fold's own se; identity_firewall vectorised with
                     signed zeros folded; the IDENTITY firewall layer no longer fails a rule for having no skill
  also               reproducibility jitter keeps ties; the benchmark's counterfactual attribution and F28 tables"""
from __future__ import annotations

import dataclasses
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.learning import calibration as CAL                                # noqa: E402
from engine.learning import complexity as CX                                  # noqa: E402
from engine.learning import firewalls as FW                                   # noqa: E402
from engine.learning import identity_firewall as IDF                          # noqa: E402
from engine.learning import promotion as PR                                   # noqa: E402
from engine.learning.core import stable_hash                                  # noqa: E402
from engine.research import evidence as EV                                    # noqa: E402
from engine.research import pattern_benchmark as PB                           # noqa: E402
from engine.research import quality_gate as QG                                # noqa: E402
from engine.research import volatility_lab as VL                              # noqa: E402

warnings.filterwarnings("ignore")
POL = QG.QualityPolicy(code_hash="c")
EC = EV.EvidenceConfig()


def market(n_dates=120, n_names=48, seed=0, beta=0.8, name_sd=0.0, extra=None):
    """A weekly panel whose touch depends on a per-date feature x (strength beta) and a persistent per-name base rate (name_sd).
    extra(rng, x, u) -> {column: values} adds columns built from x and the per-name levels u."""
    rng = np.random.default_rng(seed)
    d = pd.date_range("2016-01-08", periods=n_dates, freq="W-FRI")
    idx = pd.MultiIndex.from_product([d, [f"N{i:02d}" for i in range(n_names)]], names=["date", "ticker"])
    x = rng.normal(size=(n_dates, n_names))
    u = rng.normal(size=n_names)
    p = 1 / (1 + np.exp(-(-1.4 + beta * x + name_sd * u[None, :])))
    y = (rng.random(p.shape) < p).astype(float)
    cols = {"x": x.ravel(), "touch": y.ravel()}
    for k, v in (extra(rng, x, u) if extra else {}).items():
        cols[k] = np.asarray(v, float).ravel()
    return pd.DataFrame(cols, index=idx)


def split(G, n_train=40):
    d = G.index.get_level_values(0)
    cut = d.unique()[n_train]
    return np.asarray(d < cut), np.asarray(d >= cut + pd.Timedelta(days=7))


# ============================================================================================================ 1 proxies
def _rival(G, feature, pool, sign=1.0, ec=EC):
    tr, te = split(G)
    score = sign * G[feature]
    y = G["touch"]
    eff_te = EV.per_date_effect(score[te], y[te], ec.min_names)
    spec = EV.FindingSpec("D_" + feature, "lv20", sign, seed=3)
    return EV.incremental(G, score, y, tr, te, eff_te, dataclasses.replace(spec, feature=feature), ec, EV.centered_ranks(G[pool]))


def test_a_proxy_fails_and_the_real_pattern_it_imitates_passes():
    G = market(seed=1, extra=lambda rng, x, u: {"proxy": 0.8 * x + 0.6 * rng.normal(size=x.shape),
                                                "null1": rng.normal(size=x.shape), "null2": rng.normal(size=x.shape)})
    pool = ["x", "proxy", "null1", "null2"]
    ev_p, parts_p = _rival(G, "proxy", pool)
    ev_x, _ = _rival(G, "x", pool)
    assert ev_p.rival == "x" and ev_x.rival == "proxy" and ev_p.corr == pytest.approx(0.8, abs=0.05)
    sp, tp, mp = QG.rival_check(ev_p, POL)
    sx, tx, _ = QG.rival_check(ev_x, POL)
    assert sp == QG.FAIL and "proxy" in tp and mp["rev_t"] > 2                  # planted proxy: caught
    assert sx == QG.PASS and "adds information" in tx                           # the real pattern keeps sqrt(1 - r^2) of its effect
    assert parts_p["rival"] == "x" and parts_p["increment_t"] < 2


def test_no_correlated_rival_and_the_empty_pool_pass_as_measured():
    G = market(seed=2, extra=lambda rng, x, u: {"null1": rng.normal(size=x.shape), "null2": rng.normal(size=x.shape)})
    ev, parts = _rival(G, "x", ["x", "null1", "null2"])
    assert ev.rival is None and abs(ev.corr) < 0.1 and ev.n_pool == 2 and parts["rival"] is None
    assert QG.rival_check(ev, POL)[0] == QG.PASS and "no scored feature" in QG.rival_check(ev, POL)[1]
    tr, te = split(G)
    empty, _ = EV.incremental(G, G["x"], G["touch"], tr, te, pd.Series(dtype=float), EV.FindingSpec("a", "lv20"), EC, None)
    assert empty.rival is None and empty.n_pool == 0
    assert QG.rival_check(None, POL)[0] == QG.PASS                              # a pre-F28 bundle is not checked (and says so)


def test_near_copies_keep_the_stronger_and_too_few_periods_wait():
    inc0 = PR.IncrementalEvidence(tuple(np.random.default_rng(1).normal(0, 0.02, 60)), 1)
    inc1 = PR.IncrementalEvidence(tuple(np.random.default_rng(2).normal(0, 0.02, 60)), 2)
    strong = QG.RivalEvidence("b", 0.97, 10, 0.3, inc0, inc1, 0.05, 0.04)
    weak = dataclasses.replace(strong, own_effect=0.04, rival_effect=0.05)
    assert QG.rival_check(strong, POL)[0] == QG.PASS and "stronger of the two" in QG.rival_check(strong, POL)[1]
    assert QG.rival_check(weak, POL)[0] == QG.FAIL
    short = dataclasses.replace(strong, incremental=PR.IncrementalEvidence(tuple(np.full(10, 0.01)), 1))
    assert QG.rival_check(short, POL)[0] == QG.MISSING
    assert QG.rival_check(dataclasses.replace(strong, incremental=None), POL)[0] == QG.MISSING


def test_the_out_of_sample_gate_fails_a_proxy_and_attributes_it():
    ev, _ = QG.reference_evidence()
    fn = QG.planted_defects()["proxy_of_a_rival"][0]
    d = QG.QualityGate(QG.reference_evidence()[1]).evaluate("p", fn(ev), "2021-06-01")
    o = d.outcome("out_of_sample")
    assert d.verdict.value == "FAILED" and o.state == QG.FAIL and o.measures["rival_state"] == QG.FAIL and "proxy" in o.detail
    clean = QG.QualityGate(QG.reference_evidence()[1]).evaluate("c", ev, "2021-06-01").outcome("out_of_sample")
    assert clean.state == QG.PASS and clean.measures["rival_checked"] and clean.measures["rival_state"] == QG.PASS


def test_rival_ranks_drop_market_level_columns_and_cache_per_frame():
    F = VL.planted_frame("null", n_dates=30, n_tickers=20, seed=1)
    R = EV.rival_ranks(F, ["lv20", "mkt_vol"])
    assert "lv20" in R.columns and "mkt_vol" not in R.columns                   # no within-date variation: correlates with nothing
    assert EV.rival_ranks(F, ["lv20", "mkt_vol"]) is R
    assert abs(float(R["lv20"].groupby(level=0).mean().abs().max())) < 1e-6
    assert EV.rival_ranks(F.iloc[:0], ["lv20"]).shape[1] == 0
    assert "lv20" in EV.rival_pool(F) and "lv20" not in EV.rival_pool(F, exclude=("lv20",))


# ============================================================================================================ 2 identity units
def _units(G, feature, n_tests=615):
    tr, te = split(G)
    ev, parts = EV.name_units(G, G[feature], G["touch"], tr, te, EC)
    return ev, parts, QG.name_units_check(ev, n_tests, POL)


def test_a_per_name_artefact_is_counted_in_names():
    """Base rates persist per name; the artefact is a per-name level that lines up with them by chance (r ~ 0.35 across 48 names).
    Counted in name-weeks it is overwhelming; counted in names it is not significant after the search."""
    G = market(seed=4, beta=0.0, name_sd=1.0,
               extra=lambda rng, x, u: {"artefact": np.repeat((0.35 * u + math.sqrt(1 - 0.35 ** 2) * rng.normal(size=u.size))[None, :],
                                                              x.shape[0], 0) + 0.3 * rng.normal(size=x.shape),
                                        "genuine": np.repeat(u[None, :], x.shape[0], 0) + 0.3 * rng.normal(size=x.shape)})
    tr, te = split(G)
    eff = EV.per_date_effect(G["artefact"][te], G["touch"][te], 8)
    t_dates = PR.t_stat(eff.to_numpy())
    assert t_dates > 2                                                          # counted in name-weeks it looks like evidence
    ev, parts, (state, text, m) = _units(G, "artefact")
    assert ev.applies and ev.between_share > 0.8 and ev.n_names == 48
    assert state == QG.FAIL and "names" in text and m["names_p_adj"] > POL.promotion.alpha            # planted: caught
    assert PR.one_sided_p(t_dates) < ev.name_p                                  # the name-weeks p overstated the evidence
    gen = _units(G, "genuine")
    assert gen[0].applies and gen[2][0] == QG.PASS                              # a strong genuine per-name predictor still passes


def test_a_per_date_feature_is_not_counted_in_names_and_empty_is_safe():
    G = market(seed=5, beta=0.8, name_sd=0.5)
    ev, parts, (state, text, _) = _units(G, "x")
    assert not ev.applies and ev.between_share < 0.1 and state == QG.PASS and "dates are the units" in text
    tr, te = np.zeros(10, bool), np.ones(10, bool)                             # no train rows: nothing to estimate name levels from
    e, _ = EV.name_units(G.iloc[:10], G["x"].iloc[:10], G["touch"].iloc[:10], tr, te, EC)
    assert not e.applies and e.n_names == 0
    assert QG.name_units_check(None, 10, POL)[0] == QG.PASS
    assert QG.name_units_check(QG.NameUnitsEvidence(0.9, True, 3), 10, POL)[0] == QG.MISSING       # applies, no test possible


def test_the_gate_fails_the_planted_name_level_defect():
    ev, pol = QG.reference_evidence()
    d = QG.QualityGate(pol).evaluate("n", QG.planted_defects()["name_level_artefact"][0](ev), "2021-06-01")
    assert d.verdict.value == "FAILED" and d.outcome("out_of_sample").measures["name_units_state"] == QG.FAIL


# ============================================================================================================ 3 weak leaks
@pytest.fixture(scope="module")
def h1():
    F = VL.planted_frame("H1", n_dates=110, n_tickers=48, seed=3).sort_index()
    return F


def _fd(F, col, ec=EC):
    spec = EV.FindingSpec("D", "lv20", 1.0, n_tests_searched=5, n_scanned=600, seed=1)
    return EV.future_dependence(F, F[col].astype(float), F["touch"].astype(float), dataclasses.replace(spec, feature=col), ec)


def test_the_suspicion_tier_catches_leaks_under_the_integrity_bar(h1):
    F = h1.copy()
    rng = np.random.default_rng(11)
    z = ((F["absmove"] - F["absmove"].mean()) / F["absmove"].std()).to_numpy()
    band = []
    for i, g in enumerate(np.linspace(0.02, 0.3, 15)):
        F[f"lk{i}"] = rng.normal(0, 1, len(F)) + g * z
        found, m = _fd(F, f"lk{i}")
        if m["z"] is None:
            continue
        if m["z"] >= m["bar"]:
            assert found and not m["suspicions"]                               # over the integrity bar: an accusation
        elif m["z"] >= m["suspect_bar"]:
            assert not found and m["suspicions"]                               # between the bars: a suspicion, no accusation
            band.append(g)
            off = _fd(F, f"lk{i}", dataclasses.replace(EC, f28_leak_suspect=False))[1]
            assert not off["suspicions"]                                       # the pre-F28 screen let it through
        else:
            assert not found and not m["suspicions"]
    assert band, "no planted leak landed between the two bars: the tier was never exercised"
    assert m["suspect_bar"] == pytest.approx(2.576, abs=0.001)


def test_a_genuine_predictor_and_a_null_raise_no_suspicion(h1):
    F = h1.copy()
    F["noise"] = np.random.default_rng(5).normal(size=len(F))
    from engine.research import vol_hypotheses as VH
    F["lv"] = VH.derive(F, ("lv20",))["lv20"]
    for c in ("noise", "lv"):
        found, m = _fd(F, c)
        assert not found and not m["suspicions"], (c, m["z"])
    assert EV.future_dependence(F.drop(columns=["absmove", "close"]), F["noise"], F["touch"], EV.FindingSpec("a", "lv20"), EC)[0] == []


def test_leak_gate_answers_unknown_for_a_suspicion_and_never_quarantines_it():
    ev, pol = QG.reference_evidence()
    d = QG.QualityGate(pol).evaluate("s", QG.planted_defects()["leak_suspicion"][0](ev), "2021-06-01")
    assert d.verdict.value == "UNKNOWN" and d.outcome("leakage").state == QG.UNKNOWN
    assert QG.gate_leakage(dataclasses.replace(ev.leak, suspicions=(), documented=("dated",)), True, "2021-06-01", pol).state == QG.PASS


def test_documented_availability_explains_a_scheduled_event_and_a_false_record_is_a_finding(h1):
    F = h1.copy()
    F["ev"] = (np.random.default_rng(2).random(len(F)) < 0.1).astype(float)
    d = pd.to_datetime(F.index.get_level_values(0))
    assert EV.documented_availability(F, "ev")[0] is None                      # no record: nothing documented
    F["ev" + EV.PUBLISHED_SUFFIX] = d - pd.Timedelta(days=3)
    ok, why = EV.documented_availability(F, "ev")
    assert ok is True and "strictly before" in why
    F.loc[F.index[5], "ev" + EV.PUBLISHED_SUFFIX] = d[5]                       # one value published AT its decision
    ok, why = EV.documented_availability(F, "ev")
    assert ok is False and "1 value" in why
    assert EV.documented_availability(F.iloc[:0], "ev")[0] is True             # no rows: nothing undocumented


def test_a_planted_scheduled_event_is_not_refused_when_documented():
    """The genuine new-information signal F26 flagged: on the outcome window it IS the leak signature; the dated record clears it, the
    same column without a record stays refused, and a planted magnitude leak with a false record is refused as well."""
    F = VL.planted_frame("null", n_dates=90, n_tickers=48, seed=6).sort_index()
    G = PB.plant_scheduled_event(F, 2, share=0.12, lift=2.0)
    G["leak"] = np.random.default_rng(2).normal(size=len(G)) + 0.6 * ((G["absmove"] - G["absmove"].mean()) / G["absmove"].std())
    G["leak" + EV.PUBLISHED_SUFFIX] = pd.to_datetime(G.index.get_level_values(0))
    spec = EV.FindingSpec("D", "lv20", 1.0, n_tests_searched=5, n_scanned=600, seed=1)
    y = G["touch"].astype(float)
    f_doc, m_doc = EV.future_dependence(G, G["ev_sched"], y, dataclasses.replace(spec, feature="ev_sched"), EC)
    f_raw, m_raw = EV.future_dependence(G.drop(columns=["ev_sched" + EV.PUBLISHED_SUFFIX]), G["ev_sched"], y,
                                        dataclasses.replace(spec, feature="ev_sched"), EC)
    f_leak, _ = EV.future_dependence(G, G["leak"], y, dataclasses.replace(spec, feature="leak"), EC)
    assert m_raw["z"] >= m_raw["suspect_bar"] and (f_raw or m_raw["suspicions"])            # the signature is there
    assert not f_doc and not m_doc["suspicions"] and m_doc["documented"]                    # documented: not refused
    assert any("published at/after" in f for f in f_leak)                                   # a false record is itself a finding
    with pytest.raises(PB.HeldOutAccess):
        PB.plant_scheduled_event(F, next(s for s in range(50) if PB.is_heldout(s)))
    assert "ev_sched" + EV.PUBLISHED_SUFFIX not in PB.plant_scheduled_event(F, 2, documented=False)


# ============================================================================================================ 4 defects a-d
def test_platt_slope_is_damped_and_reports_convergence():
    rng = np.random.default_rng(0)
    r = rng.random(2000).clip(0.01, 0.99)
    y = (rng.random(2000) < 0.25 + 0.2 * (r - 0.5)).astype(float)             # the input that diverged to |b| ~ 1e7
    s = CAL.platt_slope(r, y)
    f = QG.logistic_offset_fit(QG.logit(r), y)
    assert s["converged"] and abs(s["b"]) < 1 and s["b"] == pytest.approx(f["b"], abs=1e-5)
    p = rng.uniform(0.05, 0.95, 4000)
    yy = (rng.random(4000) < QG.expit(0.6 * QG.logit(p))).astype(float)
    t = CAL.platt_slope(p, yy)
    assert t["converged"] and t["b_lo"] < 0.6 < t["b_hi"]                       # the planted slope is recovered
    sep = CAL.platt_slope(np.r_[np.full(50, 0.2), np.full(50, 0.8)], np.r_[np.zeros(50), np.ones(50)])
    assert not sep["converged"] or sep["b"] > 5                                 # separable: no finite slope (flagged or huge, never NaN-silent)
    assert CAL.platt_slope([0.5] * 5, [1, 0, 1, 0, 1])["converged"] is False


def test_worst_fold_in_each_folds_own_standard_error():
    idx = pd.date_range("2018-01-05", periods=104, freq="W-FRI")
    folds = pd.Series([f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in idx], index=idx)
    spec_c, spec_s = CX.RuleSpec("c", n_features=1, n_free_params=1), CX.RuleSpec("s", n_features=0, n_free_params=0)
    passes = {"fold_se": 0, "pooled_se": 0, "null": 0}
    for seed in range(40):
        rng = np.random.default_rng(seed)
        base = CX.Candidate(spec_s, pd.Series(rng.normal(0, 0.13, 104), index=idx), folds)
        real = CX.Candidate(spec_c, pd.Series(0.06 + rng.normal(0, 0.13, 104), index=idx), folds)
        null = CX.Candidate(spec_c, pd.Series(rng.normal(0, 0.13, 104), index=idx), folds)
        for u in ("fold_se", "pooled_se"):
            wf = [f for f in CX.compare(base, real, dataclasses.replace(CX.DEFAULT_CCFG, worst_fold_units=u)).failed if f.startswith("worst fold")]
            passes[u] += not wf
        passes["null"] += CX.compare(base, null).verdict == CX.Verdict.COMPLEX
    assert passes["fold_se"] > passes["pooled_se"] and passes["fold_se"] >= 32  # a genuine t ~ 3.8 gain is no longer failed by noise
    assert passes["null"] <= 3                                                 # and a null rule still does not earn its place
    assert CX.validate_config(dataclasses.replace(CX.DEFAULT_CCFG, worst_fold_units="years"))


def _ic_loop(scores, y, min_names=5):
    df = pd.DataFrame({"s": scores.reindex(y.index), "y": y}).dropna()
    out = {}
    for d, sub in df.groupby(level=0):
        if len(sub) >= min_names and sub["y"].nunique() > 1:
            out[d] = 0.0 if sub["s"].nunique() <= 1 else sub["s"].rank().corr(sub["y"].rank())
    return pd.Series(out, dtype=float).sort_index()


def _top_loop(scores, y, k=5):
    df = pd.DataFrame({"s": scores.reindex(y.index), "y": y}).dropna()
    vals = [sub.nlargest(k, "s")["y"].mean() - sub["y"].mean() for _, sub in df.groupby(level=0) if len(sub) > k and sub["s"].nunique() > 1]
    return float(np.mean(vals)) if vals else float("nan")


def test_identity_firewall_scoring_is_vectorised_and_equal_to_the_loop():
    rng = np.random.default_rng(3)
    d = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    idx = pd.MultiIndex.from_product([d, [f"T{i:02d}" for i in range(25)]], names=["date", "ticker"])
    s = pd.Series(np.round(rng.normal(size=len(idx)), 1), index=idx)          # ties
    s[rng.random(len(idx)) < 0.05] = np.nan
    s.loc[d[3]] = 1.0                                                           # a constant date scores 0
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    a, b = IDF.per_date_ic(s, y), _ic_loop(s, y)
    assert a.index.equals(b.index) and np.allclose(a, b, atol=1e-12) and a.loc[d[3]] == 0.0
    assert IDF.top_k_spread(s, y, 5) == pytest.approx(_top_loop(s, y, 5), abs=1e-14)
    e = pd.Series(dtype=float)
    assert IDF.per_date_ic(e, e).empty and math.isnan(IDF.top_k_spread(e, e)) and math.isnan(IDF.top_k_spread(s, y, 100))


def test_multiset_key_is_stable_under_signed_zeros_and_sees_changed_content():
    X = pd.DataFrame({"f": np.r_[np.full(50, -0.0), np.zeros(50), np.arange(20.0)], "g": np.arange(120.0)})

    def old_key(X, y):                                                         # the pre-F28 key (canonical JSON of the sorted values)
        cols = [np.sort(X[c].to_numpy().astype(float)) for c in X.columns]
        return stable_hash([[np.round(c, 10).tolist() for c in cols], []], 24)
    perms = [X.sample(frac=1, random_state=k) for k in range(8)]
    assert len({old_key(P, None) for P in perms}) > 1                          # planted defect: a pure shuffle changed the old key
    assert len({IDF._multiset_key(P, None) for P in perms}) == 1               # fixed
    assert IDF._multiset_key(X, None) != IDF._multiset_key(X.assign(g=X["g"] * 1.001), None)
    y = pd.Series(np.arange(120.0))
    assert IDF._multiset_key(X, y) != IDF._multiset_key(X, y + 1) and IDF._multiset_key(X.iloc[:0], None) == IDF._multiset_key(X.iloc[:0], None)


def test_the_identity_layer_does_not_fail_a_rule_for_having_no_skill():
    def rep(status):
        return type("R", (), {"verdicts": [type("V", (), {"status": status, "kind": "ticker_permutation", "retention": 0.0})()]})()
    layer = FW.IdentityFirewall()
    ctx = type("C", (), {"identity_report": rep("NO_SKILL"), "subject": "s"})()
    found, n = layer.inspect(ctx)
    assert n == 1 and found and all(not f.is_fail for f in found) and found[0].check == "no-skill"
    ctx.identity_report = rep("COLLAPSE")
    assert any(f.is_fail for f in layer.inspect(ctx)[0])                       # a collapse is still an identity failure
    ctx.identity_report = rep("INSUFFICIENT")
    assert any(f.is_fail for f in layer.inspect(ctx)[0])


def test_reproducibility_jitter_keeps_ties_of_a_binary_feature():
    rng = np.random.default_rng(1)
    d = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    idx = pd.MultiIndex.from_product([d, range(30)], names=["date", "ticker"])
    s = pd.Series((rng.random(len(idx)) < 0.1).astype(float), index=idx)      # a rare binary flag: 90% ties
    y = pd.Series((rng.random(len(idx)) < 0.2 + 0.3 * s).astype(float), index=idx)
    base = EV.per_date_effect(s, y, 8)
    for seed in (1, 2, 3):
        assert (EV.per_date_effect(s, y, 8, jitter_seed=seed) - base).abs().max() < 1e-12
    cont = pd.Series(rng.normal(size=len(idx)), index=idx)
    assert (EV.per_date_effect(cont, y, 8, jitter_seed=2) - EV.per_date_effect(cont, y, 8)).abs().max() == 0.0


# ============================================================================================================ the benchmark
def test_config_accepts_f28_ablations_and_the_tables_attribute_each_fix():
    assert PB.BenchConfig(evidence_ablation=("f28_rival", "f28_name_units", "f28_leak_suspect")).validate() == []
    assert PB.BenchConfig(evidence_ablation=("f28_nope",)).validate()
    rows, summ = [], []
    for w, split in (("W1", "development"), ("W2", "heldout")):
        for pid, lab, kind, prom, cf in (("P0", "REAL", "linear", True, True), ("P1", "NOISE", "proxy", False, True),
                                         ("P2", "NOISE", "identity_null", False, False), ("P3", "NOISE", "null", False, False)):
            r = {"world_id": w, "split": split, "tier": 3, "eval_era": "calm", "pid": pid, "label": lab, "kind": kind, "band": "obvious",
                 "status": PB.DETECTABLE, "promoted": prom, "right": prom and lab == "REAL", "surfaced": True}
            for ab in PB.F28_ABLATIONS:
                r[f"promoted_{ab}"] = prom or (cf and ab in ("no_rival", "no_f28"))
                r[f"right_{ab}"] = r[f"promoted_{ab}"] and lab == "REAL"
            rows.append(r)
        summ.append({"world_id": w, "split": split, "tier": 3, "promoted_total": 1, "false_positives": 0})
    summ.append({"world_id": "W3", "split": "development", "tier": 0, "promoted_total": 0, "false_positives": 0})
    T = PB.f28_tables(pd.DataFrame(rows), pd.DataFrame(summ))
    at = T["attribution"]
    dev = at[at["split"] == "development"].set_index("configuration")
    assert dev.loc["with all F28 fixes", "proxy FP"] == 0 and dev.loc["no_rival", "proxy FP"] == 1 and dev.loc["no_name_units", "FP"] == 0
    assert set(T["family"]["family"]) >= {"correlated with a real pattern (proxy-shaped)", "per-name persistent (identity-shaped)"}
    assert int(T["null_worlds"]["worlds promoting anything"].sum()) == 0
    assert set(T["strength"].columns) >= {"band", "TP recall (detectable)"} and len(T["kind"]) == 2
    E = PB.f28_tables(pd.DataFrame(rows).drop(columns=[c for c in rows[0] if c.startswith(("promoted_", "right_"))]), pd.DataFrame(summ))
    assert E["attribution"].empty                                              # answers without counterfactuals: no attribution claimed


def test_small_world_records_rivals_and_counterfactual_verdicts():
    small = PB.BenchConfig(n_names=20, n_dates=90, frame_weeks=70, first_look=88, look_every=12, bands=(("obvious", 0.9),),
                           noise_counts=(("null", 4), ("proxy", 2), ("identity_null", 2)), real_kinds=("linear",), oracle_draws=8, mt_pool=4,
                           null_world_share=0.0, tiers=(1,), screen_calls=1)
    w = PB.make_world(4, small)
    ans = PB.run_system(w.frame, "W4", 4, small)
    gated = [g for c in ans["candidates"].values() for g in c["gate"]]
    assert gated and all(set(g["cf"]) == set(PB.F28_ABLATIONS) for g in gated)
    assert all("rival" in g and "between_share" in g for g in gated)
    order = ["PROMOTE", "NEEDS_MORE_EVIDENCE", "UNKNOWN", "FAILED", "QUARANTINED"]
    for g in gated:                                                            # removing a check can only make the verdict more permissive
        if g["verdict"] == "PROMOTE":
            assert all(v == "PROMOTE" for v in g["cf"].values())
    rows, summ = PB.score_world(w.key, ans)
    assert all(f"promoted_{ab}" in rows[0] for ab in PB.F28_ABLATIONS) and order
