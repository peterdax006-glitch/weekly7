"""Bible PHASE 32 level 2 (with levels 4, 6, 7): features -> pattern miner -> session snapshot, end to end on a small
synthetic market with a planted forward effect (Phases 2, 3, 25; checklist A1, A2, A13, T10, T17).

What is proven, each with a control that can fail:
  * the miner finds the planted effect THROUGH the real candle features, and the found score ranks stocks by their
    realised future return out of sample; a market with no effect yields (almost) no patterns;
  * the panel is point-in-time: rows computed with data up to t equal the same rows computed with more data;
  * training labels are realised before the decision date, in every refit;
  * scrambling all prices after the decision date leaves the snapshot bit-identical for the honest pipeline and
    CHANGES it for the two planted look-ahead defects (so the scramble test is able to fail);
  * with too little history the pipeline degrades to a defined no-signal snapshot instead of crashing."""
import numpy as np
import pandas as pd
import pytest

from engine import adaptive as A, policy
from pipeline_world import CFG, HORIZON, Pipeline, decision_days, make_world, scramble_after


@pytest.fixture(scope="module")
def world():
    return make_world(seed=0)


@pytest.fixture(scope="module")
def fitted(world):
    P = Pipeline(world)
    return P, P.fit(world["Close"].index[-1])


def active(M):
    P = M.patterns
    return P[P["status"].isin(["active", "rescoped"])]


def test_planted_effect_found_through_real_candle_features(fitted):
    _, M = fitted
    A_ = active(M)
    row = A_[A_["key_named"] == "cd_pos q4"]
    assert len(row), sorted(A_["key_named"])
    assert (row["effect"] > 0).all()


def test_planted_pattern_outranks_unrelated_features(fitted):
    """cd_pos q4 carries the effect; features unrelated to it (streak q2, gap q2) must not be stronger."""
    _, M = fitted
    P = M.patterns.set_index("key_named")
    assert abs(P.loc["cd_pos q4", "effect"]) > abs(P.loc["streak q2", "effect"])
    assert P.loc["cd_pos q4", "p_real"] > P.loc["gap q2", "p_real"]


def test_score_ranks_out_of_sample_future_returns(world):
    """Walk-forward: fit at day 500, then score each later decision day and correlate with what happened next."""
    P = Pipeline(world)
    C = world["Close"]
    as_of = C.index[499]
    M = P.fit(as_of)
    ics = []
    for d in decision_days(C, C.index[500])[:-2]:
        j = C.index.get_loc(d)
        fwd = C.iloc[j + HORIZON] / C.iloc[j] - 1
        s = M.score(P.feature_row(d)[0])
        ics.append(s.corr(fwd, method="spearman"))
    assert np.nanmean(ics) > 0.08, np.nanmean(ics)
    assert np.mean(np.array(ics) > 0) > 0.7


def test_noise_only_world_invents_almost_nothing():
    w = make_world(seed=3, eff=0.0)
    M = Pipeline(w).fit(w["Close"].index[-1])
    assert len(active(M)) <= 2, list(active(M)["key_named"])


def test_panel_is_point_in_time():
    """Rows computed at as_of must equal the same rows computed later with more data appended (truncation test)."""
    w = make_world(seed=4, n_days=400, n_tickers=20)
    P = Pipeline(w)
    early = P.panel(w["Close"].index[249])[0]
    late = P.panel(w["Close"].index[399])[0].loc[early.index]
    pd.testing.assert_frame_equal(early, late)


def test_training_labels_are_realised_before_the_decision_date(world):
    P = Pipeline(world)
    for i in (150, 300, 450, 600):
        P.fit(world["Close"].index[i])
    fitted_only = [(a, last) for a, last, n in P.fit_log if last is not None]
    assert len(fitted_only) >= 3
    for as_of, last_label_day in fitted_only:
        assert last_label_day <= as_of, (as_of, last_label_day)


def _snap_and_miner(stocks, day, leak=None):
    P = Pipeline(stocks, leak=leak)
    M = P.fit(day)
    return P.snapshot(day, M), M


def test_future_scramble_leaves_the_honest_snapshot_identical(world):
    day = decision_days(world["Close"], world["Close"].index[500])[3]
    s1, m1 = _snap_and_miner(world, day)
    s2, m2 = _snap_and_miner(scramble_after(world, day, seed=9), day)
    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(active(m1)[["key_named", "effect"]].reset_index(drop=True),
                                  active(m2)[["key_named", "effect"]].reset_index(drop=True))


@pytest.mark.parametrize("leak", ["labels", "peek"])
def test_scramble_exposes_planted_lookahead(world, leak):
    """Control: the same scramble gate must FAIL on a pipeline that reads the future (unrealised labels / next returns)."""
    day = decision_days(world["Close"], world["Close"].index[500])[3]
    s1, _ = _snap_and_miner(world, day, leak=leak)
    s2, _ = _snap_and_miner(scramble_after(world, day, seed=9), day, leak=leak)
    assert (s1["mu_raw"] - s2["mu_raw"]).abs().max() > 1e-9


def test_reproducible_fit_and_snapshot(world):
    day = world["Close"].index[600]
    s1, m1 = _snap_and_miner(world, day)
    s2, m2 = _snap_and_miner(world, day)
    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(m1.patterns.drop(columns=["key"]), m2.patterns.drop(columns=["key"]))


def test_too_little_history_gives_defined_no_signal_snapshot(world):
    P = Pipeline(world)
    day = world["Close"].index[99]
    assert P.fit(day) is None                                  # < MIN_TRAIN_WEEKS realised weeks
    snap = P.snapshot(day, None)
    assert snap["mu_raw"].between(0, 1e-5).all() and np.isfinite(snap.select_dtypes("number").to_numpy()).all()


def test_snapshot_is_consumable_by_the_session_selection_rule(fitted, world):
    """Interface contract between the pipeline and engine.adaptive.pick: right columns, sane target weights."""
    P, M = fitted
    snap = P.snapshot(world["Close"].index[-1], M)
    cfg = {**CFG, "ew": policy.default_evidence_weights()}
    target = A.pick(snap, cfg, [], {})
    assert 0 < len(target) <= cfg["k"] and (target >= 0).all() and target.sum() <= 1.0 + 1e-9
    assert set(target.index) <= set(snap.index)
    # the highest pattern score is among the picks' universe: top of mu_raw is selected
    assert snap["mu_raw"].idxmax() in target.index
