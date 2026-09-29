"""Bible Phase 34: ablation with paired block-bootstrap CIs; the six-arm feature test must accept a real feature and
reject noise, a look-ahead evaluator and a redundant copy."""
import numpy as np
import pandas as pd
import pytest

from engine import ablation as ab
from engine import antioverfit as ao


def make_panel(seed=0, n_dates=260, n_tk=30):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"T{i:03d}" for i in range(n_tk)]], names=["date", "ticker"])
    X = pd.DataFrame(rng.standard_normal((len(idx), 6)), index=idx, columns=[f"f{i}" for i in range(6)])
    X["m_vix"] = pd.Series(rng.standard_normal(n_dates), index=dates).reindex(idx.get_level_values(0)).values
    y = (0.5 * X["f0"] + 0.3 * X["f1"]) * 0.05 + 0.05 * rng.standard_normal(len(idx))
    return X, pd.Series(y.values, index=idx)


NOW = pd.Timestamp("2016-01-01") + pd.offsets.BDay(259)


# ------------------------------------------------------------ statistics
def test_paired_ci_detects_a_real_gain_and_not_a_null():
    rng = np.random.default_rng(0)
    up = ab.paired_ci(pd.Series(rng.standard_normal(300) * 0.02 + 0.01), pd.Series(rng.standard_normal(300) * 0.02),
                      np.random.default_rng(1))
    assert up["lo"] > 0 and not up["insufficient"]
    z = ab.paired_ci(pd.Series(rng.standard_normal(300)), pd.Series(rng.standard_normal(300)), np.random.default_rng(1))
    assert z["lo"] < 0 < z["hi"] and 0.05 < z["p_le0"] < 0.95


def test_block_bootstrap_widens_the_interval_for_autocorrelated_periods():
    e = np.random.default_rng(3).standard_normal(400)
    ar = pd.Series(np.convolve(e, np.ones(10) / 10, mode="same"))            # strongly autocorrelated
    zero = pd.Series(0.0, index=range(400))
    iid = ab.paired_ci(ar, zero, np.random.default_rng(0), block=1)
    blk = ab.paired_ci(ar, zero, np.random.default_rng(0), block=20)
    assert (blk["hi"] - blk["lo"]) > 1.5 * (iid["hi"] - iid["lo"])


def test_paired_ci_insufficient_and_misaligned():
    s = pd.Series([1.0] * 5)
    r = ab.paired_ci(s, s, np.random.default_rng(0))
    assert r["insufficient"] and np.isnan(r["mean"])
    a, b = pd.Series(1.0, index=range(50)), pd.Series(0.0, index=range(40, 90))
    assert ab.paired_ci(a, b, np.random.default_rng(0), min_periods=10)["n"] == 10      # only shared periods count


# ------------------------------------------------------------ layer 1
def _contrib_eval(effects, n=200, seed=0):
    noise = np.random.default_rng(seed).standard_normal(n) * 0.02
    return lambda active: pd.Series(sum(effects[c] for c in active) + noise, index=range(n))


def test_ablate_remove_classifies_helpful_useless_and_harmful():
    ev = _contrib_eval({"good": 0.03, "inert": 0.0, "bad": -0.03})
    rep = ab.ablate(ev, ["good", "inert", "bad"], seed=1, n_boot=500)
    v = rep["table"].set_index("component")["verdict"].to_dict()
    # removing 'good' lowers the score (delta>0 => earns keep); removing 'bad' RAISES it (delta<0 => harmful)
    assert v == {"good": "earns_keep", "inert": "no_effect", "bad": "harmful"}
    assert rep["keep"] == ["good"] and set(rep["drop"]) == {"inert", "bad"}


def test_ablate_add_mode_and_caching():
    calls = []
    inner = _contrib_eval({"good": 0.03, "inert": 0.0, "given": 0.01})

    def ev(active):
        calls.append(frozenset(active))
        return inner(active)
    rep = ab.ablate(ev, ["good", "inert", "given"], mode="add", baseline=("given",), n_boot=300)
    assert list(rep["table"]["component"]) == ["good", "inert"]        # baseline members are not re-added
    assert rep["keep"] == ["good"]
    assert len(calls) == len(set(calls)) == 3                          # baseline + two additions, nothing twice


def test_ablate_validates_inputs_and_handles_no_data():
    ev = _contrib_eval({"a": 0.0})
    with pytest.raises(ValueError):
        ab.ablate(ev, [])
    with pytest.raises(ValueError, match="duplicate"):
        ab.ablate(ev, ["a", "a"])
    with pytest.raises(ValueError, match="mode"):
        ab.ablate(ev, ["a"], mode="sideways")
    with pytest.raises(ValueError, match="per-period"):
        ab.ablate(lambda a: 1.0, ["a"])
    short = ab.ablate(lambda a: pd.Series([0.1] * 4), ["a"], n_boot=100)
    assert short["table"].loc[0, "verdict"] == "insufficient"


# ------------------------------------------------------------ layer 2
@pytest.fixture(scope="module")
def panel():
    return make_panel()


KW = dict(n_rep=3, n_boot=400, seed=5)


def test_real_feature_earns_deployment_consideration(panel):
    X, y = panel
    rep = ab.feature_ablation(X, y, NOW, "f0", baseline_cols=["f2", "f3", "f4", "f5"], **KW)
    assert rep["deploy"], (rep["reasons"], rep["arm_means"])
    m = rep["arm_means"]
    assert m["baseline_plus_feature"] > m["baseline"] + 0.03 and m["feature_alone"] > 0.03
    assert m["baseline_plus_shuffled"] < m["baseline_plus_feature"] - 0.03
    assert rep["checks"]["future_leak_check"]["status"] == "pass"
    assert "EARNS" in ab.render_text(rep).upper()


def test_noise_feature_is_rejected(panel):
    X, y = panel
    noise = pd.Series(np.random.default_rng(9).standard_normal(len(X)), index=X.index)
    rep = ab.feature_ablation(X, y, NOW, noise, baseline_cols=["f2", "f3", "f4", "f5"], **KW)
    assert not rep["deploy"] and rep["feature"] == "candidate" and rep["reasons"]


def test_redundant_feature_is_rejected(panel):
    """A copy of a baseline feature adds nothing incremental."""
    X, y = panel
    X2 = X.copy()
    X2["f0_copy"] = X2["f0"]
    rep = ab.feature_ablation(X2, y, NOW, "f0_copy", baseline_cols=["f0", "f2", "f3"], **KW)
    assert not rep["deploy"]


def test_lookahead_evaluator_blocks_deployment(panel):
    X, y = panel

    def leaky(X, y, now):
        return ao.wf_evaluate(X, y, X.index.get_level_values(0).max())
    rep = ab.feature_ablation(X, y, NOW, "f0", baseline_cols=["f2", "f3"], evaluate=leaky, **KW)
    assert not rep["deploy"] and any("look-ahead" in r for r in rep["reasons"])


def test_future_arm_can_be_marked_not_applicable(panel):
    X, y = panel
    rep = ab.feature_ablation(X, y, NOW, "f0", baseline_cols=["f2", "f3"], future_applicable=False, **KW)
    assert "baseline_plus_future_scrambled" not in rep["arm_means"] and "future_leak_check" not in rep["checks"]


def test_interaction_feature_lifts_the_score_over_baseline():
    """y depends on f0*f1: the product column carries signal that f2/f3 do not."""
    X, _ = make_panel(seed=4)
    y = pd.Series(0.05 * X["f0"].values * X["f1"].values + 0.05 * np.random.default_rng(4).standard_normal(len(X)),
                  index=X.index)
    X["prod"] = X["f0"] * X["f1"]
    rep = ab.feature_ablation(X, y, NOW, "prod", baseline_cols=["f2", "f3"], **KW)
    assert rep["arm_means"]["baseline_plus_feature"] > rep["arm_means"]["baseline"]


def test_feature_ablation_input_validation_and_determinism(panel):
    X, y = panel
    with pytest.raises(ValueError, match="not in X"):
        ab.feature_ablation(X, y, NOW, "nope")
    with pytest.raises(ValueError, match="baseline"):
        ab.feature_ablation(X[["f0", "m_vix"]], y, NOW, "f0")
    with pytest.raises(ValueError, match="index"):
        ab.feature_ablation(X, y, NOW, pd.Series([1.0, 2.0]))
    a = ab.feature_ablation(X, y, NOW, "f0", baseline_cols=["f2", "f3"], **KW)
    b = ab.feature_ablation(X, y, NOW, "f0", baseline_cols=["f2", "f3"], **KW)
    assert a["arm_means"] == b["arm_means"] and a["checks"]["incremental"] == b["checks"]["incremental"]
