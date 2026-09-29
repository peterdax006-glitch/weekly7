"""Tests for the Phase 25 library itself (engine/planted.py): the generator plants exactly what it claims, the name
parser and matcher are order-independent, the scorer counts true/false discoveries correctly on a hand-built miner
output, calibration metrics are right on known inputs, and the verdict fails closed on missing evidence."""
import numpy as np
import pandas as pd
import pytest

from engine import planted as PL


def small(plants, **kw):
    return PL.Scenario("t", plants, weeks=80, stocks=120, **kw)


def test_generator_is_deterministic():
    sc = small(PL.standard_plants())
    X1, y1, _ = PL.generate(sc, seed=3)
    X2, y2, _ = PL.generate(sc, seed=3)
    assert X1.equals(X2) and y1.equals(y2)


def test_planted_single_effect_is_present_at_the_right_size():
    sc = small([PL.Plant("s", "strong", (("f0", 4),), 0.03)], noise_sd=0.01, week_sd=0.0, sector_sd=0.0)
    X, y, truth = PL.generate(sc, seed=1)
    q = PL.quintiles(X[["f0"]])["f0"]
    diff = y[q == 4].mean() - y[q != 4].mean()
    assert diff == pytest.approx(0.03, abs=0.004)


def test_quintiles_match_the_miner():
    from engine.patterns import PatternMiner
    sc = small([])
    X, _, _ = PL.generate(sc, seed=2)
    F = X[[c for c in X if not c.startswith("m_")]]
    ours = PL.quintiles(F).values
    theirs = PatternMiner()._quintiles(F)
    assert (ours == theirs).all()


def test_regime_plant_only_fires_in_its_regime():
    p = PL.Plant("r", "regime", (("f3", 4),), 0.02, regime=("m_vix", 0.5, 99.0))
    X, _, truth = PL.generate(small([p]), seed=4)
    fired = truth["r"] != 0
    assert fired.any() and (X.loc[fired, "m_vix"] > 0.5).all()


def test_hallucinated_plant_only_in_first_half():
    p = PL.Plant("h", "hallucinated", (("f9", 0),), 0.02, active=(0.0, 0.5))
    X, _, truth = PL.generate(small([p]), seed=5)
    d = X.index.get_level_values(0)
    last = d.max(); first = d.min()
    frac = (d - first) / (last - first)
    fired = (truth["h"] != 0).values
    assert fired.any() and frac[fired].max() <= 0.5 + 1e-9


def test_unless_plant_is_cancelled_by_its_exception():
    p = PL.Plant("u", "unless", (("f6", 4), ("f7", 4)), 0.02, unless=("f8", 4))
    X, _, truth = PL.generate(small([p]), seed=6)
    Q = PL.quintiles(X[["f6", "f7", "f8"]])
    both = (Q["f6"] == 4) & (Q["f7"] == 4)
    assert (truth.loc[both & (Q["f8"] == 4), "u"] == 0).all()
    assert (truth.loc[both & (Q["f8"] != 4), "u"] == 0.02).all()


def test_correlated_pair_is_correlated():
    X, _, _ = PL.generate(small([], corr_pairs=((10, 11),)), seed=7)
    assert np.corrcoef(X["f10"], X["f11"])[0, 1] > 0.9


def test_parse_named_is_order_independent():
    assert PL.parse_named("f0 q4 & f1 q2") == PL.parse_named("f1 q2 & f0 q4")
    assert PL.parse_named("f0 q4 & f1 q2 unless f3 q0")[1] == "f3 q0"
    assert PL.parse_named("f10 q4") == PL.canon_key((("f10", 4),))


def fake_patterns(rows):
    return pd.DataFrame(rows, columns=["key_named", "status", "effect", "p_real"]).assign(scope=None)


def test_score_run_counts_true_and_false_discoveries():
    sc = PL.Scenario("standard", PL.standard_plants())
    P = fake_patterns([
        ("f0 q4", "active", 0.011, 0.99),               # strong: found
        ("f2 q4", "active", -0.007, 0.99),              # negative: found, right sign
        ("f5 q0 & f4 q4", "active", 0.02, 0.95),        # pair: found despite reversed order
        ("f9 q0", "active", 0.01, 0.9),                 # hallucinated: wrongly admitted
        ("f11 q2", "active", 0.004, 0.85),              # pure noise: false discovery
        ("f1 q4", "no_gain", 0.004, 0.9),               # weak: tested, not admitted
    ])
    r = PL.score_run(P, sc)
    pp = r["per_plant"]
    assert pp["strong"]["admitted"] and pp["negative"]["sign_ok"] and pp["pair"]["admitted"]
    assert not pp["weak"]["admitted"] and pp["weak"]["status"] == "no_gain"
    assert pp["hallucinated"]["admitted"]                               # recorded as a bad admission
    assert set(r["false_active"]) == {"f9 q0", "f11 q2"}               # both share no condition with a real plant
    assert r["planted_bad_admitted"] == ["f9 q0"]
    assert pp["strong"]["effect_ratio"] == pytest.approx(0.011 / 0.012)


def test_child_of_a_plant_is_not_a_false_discovery():
    sc = PL.Scenario("standard", PL.standard_plants())
    r = PL.score_run(fake_patterns([("f0 q4 & f7 q1", "active", 0.02, 0.9)]), sc)
    assert r["n_false_active"] == 0 and r["per_plant"]["strong"]["admitted_via_child"]


def test_calibration_table_on_known_inputs():
    pairs = [(0.95, True)] * 9 + [(0.95, False)] + [(0.1, False)] * 10
    c = PL.calibration_table(pairs)
    assert c["bins"]["(0.9, 1.0]"]["share_truly_real"] == pytest.approx(0.9)
    assert c["bins"]["(-0.01, 0.2]"]["share_truly_real"] == 0.0
    assert c["brier"] == pytest.approx((9 * 0.05 ** 2 + 0.95 ** 2 + 10 * 0.01) / 20)
    assert c["ece"] == pytest.approx((10 * 0.05 + 10 * 0.1) / 20)


def test_empty_calibration_is_none_not_perfect():
    c = PL.calibration_table([])
    assert c["brier"] is None and c["n"] == 0


def test_verdict_fails_closed_without_evidence():
    v = PL.verdict({"n_runs": 0, "by_scenario": {}})
    assert not v["validated"] and not any(c["pass"] for c in v["criteria"].values())


def test_verdict_passes_a_perfect_summary_and_fails_one_flaw():
    sc = PL.Scenario("standard", PL.standard_plants())
    P = fake_patterns([("f0 q4", "active", 0.012, 0.99), ("f2 q4", "active", -0.008, 0.99),
                       ("f1 q4", "active", 0.004, 0.95), ("f3 q4", "rescoped", 0.01, 0.95),
                       ("f4 q4 & f5 q0", "active", 0.02, 0.95), ("f6 q4 & f7 q4 unless f8 q4", "active", 0.02, 0.95),
                       ("f9 q0", "discarded", 0.0, 0.1), ("f1 q0", "discarded", 0.0, 0.1), ("f10 q4", "rejected", 0.0, 0.05)])
    runs = [PL.score_run(P, sc) for _ in range(3)]
    noise = PL.Scenario("noise_only", [PL.Plant("zero", "zero", (("f10", 4),), 0.0)])
    runs += [PL.score_run(fake_patterns([("f10 q4", "rejected", 0.0, 0.05)]), noise) for _ in range(3)]
    v = PL.verdict(PL.summarise(runs))
    assert v["validated"], v
    bad = PL.score_run(pd.concat([P, fake_patterns([("f9 q0", "active", 0.01, 0.9)])]).drop_duplicates("key_named", keep="last"), sc)
    v2 = PL.verdict(PL.summarise([bad] * 3 + runs[3:]))
    assert not v2["validated"] and not v2["criteria"]["hallucinated_rejected"]["pass"]


def test_exact_truth_counts_demeaning_side_effects_as_real_and_noise_as_false():
    """f0 top fifth +3%: after per-week demeaning the rest of f0 truly LOSES ~0.75% - a real effect the old
    condition-overlap rule called a false discovery. A noise feature's pattern must still count as false, and a real
    effect claimed with the WRONG sign must count as false (the check can fail)."""
    sc = small([PL.Plant("s", "strong", (("f0", 4),), 0.03)], noise_sd=0.01, week_sd=0.0, sector_sd=0.0)
    X, y, truth = PL.generate(sc, seed=21)
    P = fake_patterns([("f0 q4", "active", 0.029, 0.99),      # the plant
                       ("f0 q1", "active", -0.007, 0.95),     # demeaning side effect, right sign
                       ("f7 q2", "active", 0.004, 0.9),       # pure noise feature
                       ("f0 q2", "active", +0.007, 0.9)])     # real side effect, WRONG sign
    r = PL.score_run(P, sc, X=X, truth=truth)
    assert set(r["false_active_true"]) == {"f7 q2", "f0 q2"}
    assert "f0 q1" in r["false_active"]                        # the legacy rule got this one wrong
    te = PL.true_recent_effects([PL.parse_named("f0 q4"), PL.parse_named("f0 q1")], X, truth)
    assert te[0] == pytest.approx(0.03 * 0.8, abs=0.004) and te[1] == pytest.approx(-0.0075, abs=0.002)


def test_verdict_prefers_exact_truth_when_present():
    sc = PL.Scenario("standard", PL.standard_plants())
    X, y, truth = PL.generate(PL.Scenario("standard", PL.standard_plants(), weeks=80, stocks=120), seed=3)
    r = PL.score_run(fake_patterns([("f0 q4", "active", 0.012, 0.99)]), sc, X=X, truth=truth)
    v = PL.verdict(PL.summarise([r]))
    assert "exact truth" in v["criteria"]["fdr"]["rule"] and "fdr_by_condition" in v["criteria"]
