"""Tiered objective firewall (Phase 20): parity with the legacy scalar, lexicographic property tests, planted defect."""
import numpy as np
import pytest
from engine import objective as O


def legacy_oracle(rows):                       # verbatim copy of scripts/livesim_loop2.tiered (that module runs a loop on import)
    t1 = float(np.mean([r["in_band"] for r in rows]))
    over = float(np.mean([r["over_band"] for r in rows]))
    risk = float(np.mean([r["worst5"] for r in rows])) + float(min(r["max_dd"] for r in rows)) / 4
    t3 = float(np.mean([r["pos_in_band"] for r in rows]))
    reached = t1 >= 0.5
    return (100 * min(t1, 0.5) - 50 * over + (10 * (risk + 0.3) + t3 if reached else 0.0)), t1, risk, t3


def rand_row(rng):
    inb = rng.uniform(0, 1)
    over = rng.uniform(0, 1 - inb)
    return {"in_band": inb, "over_band": over, "worst5": -rng.uniform(0, 0.3), "max_dd": -rng.uniform(0, 0.8),
            "pos_in_band": rng.uniform(0, 1), "cat_rate": 0.0}


def mk(t1, risk_w5, t3, over=0.0, dd=-0.1):
    return [{"in_band": t1, "over_band": over, "worst5": risk_w5, "max_dd": dd, "pos_in_band": t3, "cat_rate": 0.0}]


def test_legacy_parity():
    rng = np.random.default_rng(1)
    for _ in range(200):
        rows = [rand_row(rng) for _ in range(rng.integers(1, 6))]
        assert O.tiered_legacy(rows) == pytest.approx(legacy_oracle(rows))


def test_week_row_known_series():
    w = [0.07, -0.06, 0.02, 0.12, -0.30, 0.08]
    r = O.week_row(w)
    assert r["in_band"] == pytest.approx(3 / 6) and r["over_band"] == pytest.approx(2 / 6) and r["under_band"] == pytest.approx(1 / 6)
    assert r["pos_in_band"] == pytest.approx(2 / 3) and r["cat_rate"] == pytest.approx(1 / 6) and r["plus10"] == pytest.approx(1 / 6)
    eq = np.cumprod(1 + np.array(w))
    assert r["max_dd"] == pytest.approx((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min())
    assert r["dir_precision"] == pytest.approx(3 / 5)


def test_empty_and_degenerate():
    assert O.week_row([])["in_band"] == 0.0
    s = O.evaluate([])
    assert s.scalar == -9.0 and s.wiped and not O.firewall(O.evaluate(mk(0.1, -0.1, 0.5)), s)[0]
    assert O.firewall(s, O.evaluate(mk(0.1, -0.1, 0.5)))[0]
    assert O.compare([], mk(0.1, -0.1, 0.5)) == 1
    with pytest.raises(ValueError):
        O.week_row([0.01, float("nan")])
    with pytest.raises(KeyError):
        O.evaluate([{"in_band": 0.5}])
    with pytest.raises(ValueError):
        O.evaluate([{**mk(0.5, -0.1, 0.5)[0], "worst5": float("inf")}])


def test_wipeout_floor():
    s = O.evaluate(mk(0.9, -0.05, 0.9, dd=-0.995))
    assert s.wiped and s.scalar == -9.0 and s.soft == -9.0
    assert O.firewall(O.evaluate(mk(0.0, -0.5, 0.0)), s)[0] is False


def test_tier1_dominates_property():
    """Any candidate with a higher tier-1 bucket wins, however much worse its risk and direction are."""
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(500):
        a = mk(rng.uniform(0, 0.48), -rng.uniform(0, 0.02), rng.uniform(0.8, 1))            # low t1, excellent below
        b = mk(min(0.49, a[0]["in_band"] + rng.uniform(0.04, 0.3)), -rng.uniform(0.2, 0.3), rng.uniform(0, 0.2), dd=-0.7)
        sa, sb = O.evaluate(a), O.evaluate(b)
        if sb.key[0] > sa.key[0]:
            checked += 1
            assert O.firewall(sa, sb)[0] and not O.firewall(sb, sa)[0] and sb.scalar > sa.scalar
    assert checked > 100


def test_tier2_dominates_tier3_once_reached():
    rng = np.random.default_rng(3)
    for _ in range(300):
        t1 = rng.uniform(0.5, 0.9)
        safe = O.evaluate(mk(t1, -0.02, rng.uniform(0, 0.3), dd=-0.05))
        risky = O.evaluate(mk(t1, -0.25, rng.uniform(0.9, 1), dd=-0.6))
        assert safe.reached and safe.scalar > risky.scalar
        assert O.firewall(risky, safe)[0] and not O.firewall(safe, risky)[0]


def test_risk_ignored_before_tier1_reached():
    a, b = O.evaluate(mk(0.3, -0.01, 0.5)), O.evaluate(mk(0.3, -0.29, 0.5, dd=-0.6))
    assert a.key == b.key and not O.firewall(a, b)[0] and not O.firewall(b, a)[0]


def test_monotone_in_each_tier():
    rng = np.random.default_rng(4)
    for _ in range(300):
        r = rand_row(rng)
        base = O.evaluate([r])
        up1 = O.evaluate([{**r, "in_band": min(0.5, r["in_band"] + 0.1)}])
        up2 = O.evaluate([{**r, "worst5": r["worst5"] + 0.05, "max_dd": min(0.0, r["max_dd"] + 0.05)}])
        up3 = O.evaluate([{**r, "pos_in_band": min(1.0, r["pos_in_band"] + 0.2)}])
        assert up2.key >= base.key and up3.key >= base.key
        if r["over_band"] == 0:
            assert up1.key >= base.key


def test_order_is_transitive_and_matches_scalar():
    rng = np.random.default_rng(5)
    sc = [O.evaluate([rand_row(rng)]) for _ in range(120)]
    order_key = sorted(range(len(sc)), key=lambda i: sc[i].key)
    order_scalar = sorted(range(len(sc)), key=lambda i: sc[i].scalar)
    assert [sc[i].key for i in order_key if not sc[i].wiped] == [sc[i].key for i in order_scalar if not sc[i].wiped]


def test_planted_defect_naive_sum_would_be_fooled():
    """A candidate that buys tier-3 direction with a catastrophic tier-2 tail must lose to a safe one of equal tier 1."""
    safe = O.evaluate(mk(0.6, -0.04, 0.55, dd=-0.10))
    greedy = O.evaluate(mk(0.6, -0.30, 1.0, dd=-0.75))
    naive = lambda s: 10 * s.t1 + s.t3 + 0.5 * s.risk                    # a weighted-sum objective, the thing we must not be
    assert naive(greedy) > naive(safe) - 1.0                              # sum nearly (or fully) forgives it ...
    assert not O.firewall(safe, greedy)[0] and O.firewall(greedy, safe)[0]  # ... the firewall does not


def test_catastrophic_weeks_hurt_tier2():
    calm = O.week_row(np.r_[np.full(50, 0.06), 0.02, 0.03])
    boom = O.week_row(np.r_[np.full(50, 0.06), -0.35, 0.03])
    assert boom["cat_rate"] > 0 and O.evaluate([boom]).risk < O.evaluate([calm]).risk


def test_paired_bootstrap_detects_and_rejects():
    rng = np.random.default_rng(6)
    assert O.paired_bootstrap(rng.normal(0.5, 0.2, 20)) > 0
    assert O.paired_bootstrap(rng.normal(0.0, 1.0, 20)) < 0.3
    assert O.paired_bootstrap([]) == float("-inf")
    d = rng.normal(0.5, 0.2, 20)
    assert O.paired_bootstrap(d, seed=3) == O.paired_bootstrap(d, seed=3)


def _rows_from(sd, n, seed):
    rng = np.random.default_rng(seed)
    return [O.week_row(sd * rng.standard_normal(52)) for _ in range(n)]


def test_decisive_diffs_picks_the_deciding_tier():
    a = [{**mk(0.3, -0.05, 0.5)[0]} for _ in range(6)]
    b = [{**mk(0.4, -0.25, 0.9)[0], "max_dd": -0.7} for _ in range(6)]        # tier 1 better, everything below worse
    tier, d = O.decisive_diffs(a, b)
    assert tier == 0 and (d > 0).all()
    c = [{**mk(0.6, -0.05, 0.5)[0]} for _ in range(6)]
    e = [{**mk(0.6, -0.02, 0.1)[0]} for _ in range(6)]                        # tier 1 tied, risk better, direction worse
    tier, d = O.decisive_diffs(c, e)
    assert tier == 1 and (d > 0).all()
    assert O.decisive_diffs(c, c)[0] is None and not O.decisive_diffs(c, c)[1].any()
    with pytest.raises(ValueError):
        O.decisive_diffs(a, b[:3])


def test_decisive_diffs_sign_matches_firewall():
    rng = np.random.default_rng(7)
    for _ in range(100):
        a = [rand_row(rng) for _ in range(5)]
        b = [rand_row(rng) for _ in range(5)]
        tier, d = O.decisive_diffs(a, b)
        sa, sb = O.evaluate(a), O.evaluate(b)
        if tier is not None and not sa.wiped and not sb.wiped:
            assert O.firewall(sa, sb)[0] == (sb.key > sa.key)


def test_by_group_reports_each_era_and_never_drops_one():
    calm, wild = _rows_from(0.02, 4, 1), _rows_from(0.07, 4, 2)
    g = O.by_group(calm + wild, ["calm"] * 4 + ["wild"] * 4)
    assert set(g) == {"calm", "wild"} and g["wild"]["in_band"] > g["calm"]["in_band"] and g["calm"]["n_windows"] == 4
    with pytest.raises(ValueError):
        O.by_group(calm, ["x"])
    assert O.by_group([], []) == {}


def test_summary_is_json_safe():
    import json
    json.dumps(O.summary(O.evaluate(_rows_from(0.06, 3, 3))))
    assert O.summary(O.evaluate([]))["wiped"] is True


def test_rank_correlation_of_tiers_detects_planted_conflict():
    """Planted: as weekly sd grows, in-band share rises but tail risk worsens. The diagnostic must see that tier 1 and
    tier 2 pull against each other; on identical scores it must not invent a number."""
    scores = [O.evaluate(_rows_from(sd, 6, int(sd * 1000))) for sd in np.linspace(0.02, 0.09, 12)]
    assert O.rank_correlation_of_tiers(scores) < -0.5
    assert np.isnan(O.rank_correlation_of_tiers([scores[0]] * 5))
    assert np.isnan(O.rank_correlation_of_tiers(scores[:2]))


def test_overshoot_costs_tier1_even_when_band_share_is_equal():
    calm = {**mk(0.4, -0.05, 0.5, over=0.0)[0]}
    wild = {**mk(0.4, -0.05, 0.5, over=0.5)[0]}
    assert O.evaluate([wild]).key[0] < O.evaluate([calm]).key[0]
