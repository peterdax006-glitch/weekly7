"""Tests for engine/research/expectations.py, outcomes.py, prediction_error.py (C68 checklists A, B, C). Synthetic data only."""
import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest

from engine.learning.core import FirewallBreach
from engine.learning.surprise import SurpriseTracker
from engine.research import expectations as EX
from engine.research import outcomes as OC
from engine.research import prediction_error as PE
from engine.research.two_stage import DayDecision, Funnel

DATES = ("2021-03-01", "2021-03-02", "2021-03-03", "2021-03-04", "2021-03-05")
NOW = "2021-03-10"


def mk_exp(**kw):
    base = dict(subject="AAA", decided_at="2021-02-26", entry_at="2021-03-01", timestamp="2021-02-26T16:00:00",
                model_version="m1", predicted_return=0.10, distribution=((0.1, 0.02), (0.5, 0.10), (0.9, 0.17)),
                direction=1, confidence=0.7, predicted_volatility=0.02, time_to_peak=2, exit_window=(1, 3), holding_period=3,
                mfe=0.12, mae=-0.03, market_regime="UP_CALM", sector_regime="UP", patterns=("mom", "gap"),
                pattern_strengths={"mom": 0.6, "gap": 0.3}, interactions={"mom x gap": 0.2},
                prob_distribution=((-1.0, 0.0, 0.1), (0.0, 0.05, 0.2), (0.05, 0.10, 0.4), (0.10, 0.5, 0.3)),
                reasons_selected=("highest p_move", "direction confidence 0.7"), reasons_rejected={"BBB": "lower confidence"},
                uncertainty={"aleatoric": 0.03, "epistemic": 0.01}, alternatives=({"hypothesis": "reversal", "probability": 0.2,
                                                                                 "predicted_return": -0.02},),
                feature_state={"vol20": 0.02, "mom5": 0.04}, knowledge_state={"digest": "abc", "n_items": 12},
                information_set={"prices": "2021-02-26", "knowledge": "2021-02-25"}, n_competitors=1)
    base.update(kw)
    return EX.Expectation(**base)


def mk_path(closes=(115, 111, 107, 106, 105), opens=None, highs=None, lows=None, exit_at="2021-03-03", **kw):
    opens = opens or (100, closes[0], closes[1], closes[2], closes[3])
    highs = highs or tuple(max(o, c) + 1 for o, c in zip(opens, closes))
    lows = lows or tuple(min(o, c) - 1 for o, c in zip(opens, closes))
    return OC.PathData(dates=DATES, open=tuple(opens), high=tuple(highs), low=tuple(lows), close=tuple(closes), exit_at=exit_at, **kw)


def setup(exp=None, root=None, path=None):
    led = EX.ExpectationLedger(root, code_hash="t")
    exp = exp or mk_exp()
    led.record(exp, "2021-02-26")
    h = led.meta(exp.prediction_id)["content_hash"]
    return led, exp, h, OC.reconstruct(exp, h, path or mk_path(), NOW)


# ------------------------------------------------------------------------------------------------ A: expectations
def test_record_get_and_verify():
    led = EX.ExpectationLedger(code_hash="t")
    e = mk_exp()
    pid = led.record(e, "2021-02-26")
    assert led.get(pid) == e and len(led) == 1 and led.verify()["ok"]
    assert led.frozen_before(pid, "2021-03-01")
    with pytest.raises(FirewallBreach):
        led.get(pid, now="2021-02-26")                       # recorded ON now is not yet visible
    assert led.get(pid, now="2021-02-27") == e
    assert led.ids(now="2021-02-26") == [] and led.ids() == [pid]


def test_rewrite_refused_and_identical_rerecord_idempotent():
    led = EX.ExpectationLedger(code_hash="t")
    e = mk_exp()
    pid = led.record(e, "2021-02-26")
    assert led.record(mk_exp(), "2021-02-26") == pid and len(led.lane) == 1
    with pytest.raises(EX.ExpectationRewrite):
        led.record(mk_exp(predicted_return=0.11, mfe=0.13), "2021-02-26")   # hindsight edit under the same prediction id
    with pytest.raises(EX.ExpectationRewrite):
        led.update(pid, predicted_return=0.05)
    with pytest.raises(EX.ExpectationRewrite):
        led.delete(pid)
    with pytest.raises(EX.ExpectationRewrite):
        led.amend(pid)
    assert led.get(pid).predicted_return == 0.10 and led.verify()["ok"]


def test_expectation_is_frozen_object():
    e = mk_exp()
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.predicted_return = 0.5
    with pytest.raises(TypeError):
        e.pattern_strengths["mom"] = 9.0                      # mappings are read-only views


def test_timing_rules():
    led = EX.ExpectationLedger(code_hash="t")
    with pytest.raises(FirewallBreach):
        led.record(mk_exp(), "2021-02-25")                    # claims a decision not yet taken
    with pytest.raises(EX.LateExpectation):
        led.record(mk_exp(), "2021-03-02")                    # written after the entry session began
    led.record(mk_exp(), "2021-03-01")                        # on the entry day (before the open) is allowed


@pytest.mark.parametrize("kw,needle", [
    (dict(distribution=((0.1, 0.05), (0.5, 0.02), (0.9, 0.17))), "crossing"),
    (dict(distribution=((0.5, 0.1), (0.9, 0.2))), "quantiles"),
    (dict(prob_distribution=((-1.0, 0.0, 0.5), (0.0, 1.0, 0.4))), "sum to 1"),
    (dict(prob_distribution=((-1.0, 0.1, 0.5), (0.0, 1.0, 0.5))), "overlap"),
    (dict(mfe=0.05), "excursions"),
    (dict(mae=0.02), "excursions"),
    (dict(exit_window=(4, 6)), "outside the predicted exit window"),
    (dict(confidence=0.4), "confidence"),
    (dict(direction=0), "direction"),
    (dict(predicted_volatility=0.0), "predicted_volatility"),
    (dict(predicted_return=float("nan")), "finite"),
    (dict(pattern_strengths={"mom": 0.6}), "pattern_strengths"),
    (dict(interactions={"mom x zzz": 0.1}), "not in patterns"),
    (dict(reasons_selected=()), "reasons_selected"),
    (dict(reasons_rejected={}, n_competitors=2), "rejection reasons"),
    (dict(alternatives=()), "alternative"),
    (dict(uncertainty={"aleatoric": 0.1}), "uncertainty.epistemic"),
    (dict(knowledge_state={}), "digest"),
    (dict(information_set={}), "information_set"),
    (dict(entry_at="2021-02-25"), "entry_at must be after"),
    (dict(timestamp="2021-03-01T09:00:00"), "timestamp"),
])
def test_incomplete_or_inconsistent_expectations_refused(kw, needle):
    e = mk_exp(**kw)
    assert any(needle in m for m in e.validate()), e.validate()
    with pytest.raises(EX.ExpectationInvalid):
        EX.ExpectationLedger(code_hash="t").record(e, "2021-02-26")


def test_future_information_and_outcome_columns_refused():
    late = mk_exp(information_set={"prices": "2021-02-26", "news": "2021-03-02"})
    assert any("after the decision session" in m for m in late.validate())
    leak = mk_exp(feature_state={"vol20": 0.02, "fwd_ret_5": 0.09})
    assert any("outcome/future column" in m for m in leak.validate())
    leak2 = mk_exp(patterns=("mom", "y_label"), pattern_strengths={"mom": 1.0, "y_label": 1.0}, interactions={})
    assert any("outcome/future column" in m for m in leak2.validate())


def test_valid_baseline_has_no_errors():
    assert mk_exp().validate() == []


def test_distribution_functions():
    d = EX.ReturnDistribution(((0.1, 0.02), (0.5, 0.10), (0.9, 0.17)))
    assert d.median == pytest.approx(0.10)
    assert d.cdf(0.10) == pytest.approx(0.5) and d.cdf(0.02) == pytest.approx(0.1)
    assert d.cdf(-1) == 0.0005 and d.cdf(1) == 0.9995        # never-predicted outcomes are extreme but finite
    assert d.pinball(0.10) < d.pinball(0.30)                  # a proper score: nearer the centre is better
    assert 0 < d.prob_within(0.05, 0.10) < 1
    assert mk_exp().prob_in_band(0.05, 0.10) == pytest.approx(0.4)


def test_persistent_reload_and_tamper_detection(tmp_path):
    root = tmp_path / "chain"
    led = EX.ExpectationLedger(root, code_hash="t")
    pid = led.record(mk_exp(), "2021-02-26")
    anchor = led.anchor()
    led2 = EX.ExpectationLedger(root, code_hash="t")
    assert led2.get(pid) == mk_exp() and led2.verify(anchors=[anchor])["ok"]
    f = root / "chain.jsonl"
    txt = f.read_text(encoding="utf-8")
    assert '"predicted_return":0.1' in txt
    f.write_text(txt.replace('"predicted_return":0.1', '"predicted_return":0.15', 1), encoding="utf-8")   # rewrite history
    with pytest.raises(Exception):
        EX.ExpectationLedger(root, code_hash="t")             # reopening the chain fails closed
    rep = led.verify()                                        # the live ledger re-reads the medium and sees the edit too
    assert not rep["ok"] and rep["problems"]
    with pytest.raises(EX.LedgerTampered):
        led.assert_intact()


def test_in_memory_tamper_and_missing_anchor():
    led = EX.ExpectationLedger(code_hash="t")
    led.record(mk_exp(), "2021-02-26")
    assert led.verify(anchors=["f" * 64])["ok"] is False      # an anchor that is not part of the chain
    line = led.lane.chain._all()[0]
    line["body"]["expectation"]["predicted_return"] = 0.5
    rep = led.verify()
    assert not rep["ok"]


def test_consistent_rewrite_caught_by_stored_anchor():
    led = EX.ExpectationLedger(code_hash="t")
    led.record(mk_exp(), "2021-02-26")
    anchor = led.anchor()
    other = EX.ExpectationLedger(code_hash="t")               # a forger rebuilds a self-consistent chain with other content
    other.record(mk_exp(predicted_return=0.2, mfe=0.25), "2021-02-26")
    assert other.verify()["ok"]                               # internally consistent ...
    assert not other.verify(anchors=[anchor])["ok"]           # ... but it is not the chain the anchor was taken from


def test_coverage_reports_missing_days():
    led = EX.ExpectationLedger(code_hash="t")
    led.record(mk_exp(), "2021-02-26")
    cov = led.coverage(["2021-02-26", "2021-03-01"])
    assert cov["missing"] == ["2021-03-01"] and cov["with_expectations"] == 1


def test_expectations_from_two_stage_day():
    idx = pd.MultiIndex.from_tuples([("2021-02-26", "AAA"), ("2021-02-26", "BBB"), ("2021-02-26", "CCC")])
    table = pd.DataFrame({"eligible": True, "p_move": [0.8, 0.7, 0.2], "mover": [True, True, False], "p_up": [0.72, 0.5, 0.5],
                          "p_down": [0.28, 0.5, 0.5], "side": [1, 0, 0],
                          "reason": ["POSITION", "LOW_CONFIDENCE", "NOT_PREDICTED_MOVER"]}, index=idx)
    f = Funnel()
    f.add("positions", 2, 1, "abstain on the rest")
    day = DayDecision("2021-02-26", table, f, {"open": True, "verdict": "OPEN"}, "dig")
    traj = mk_exp()
    fields = {k: getattr(traj, k) for k in EX.TRAJECTORY_FIELDS}
    ctx = dict(entry_at="2021-03-01", timestamp="2021-02-26T16:00:00", market_regime="UP_CALM", sector_regime="UP",
               patterns=traj.patterns, pattern_strengths=dict(traj.pattern_strengths), interactions=dict(traj.interactions))
    today = pd.DataFrame({"vol20": [0.02, 0.03, 0.01], "mom5": [0.04, 0.0, 0.0], "fwd_ret_5": [0.1, 0.2, 0.3]}, index=idx)
    out = EX.expectations_from_day(day, lambda row, s: fields, ctx, "2021-02-26", model_version="m1",
                                   information_set={"prices": "2021-02-26"}, today=today)
    assert set(out[0].feature_state) == {"vol20", "mom5"}                       # the outcome column never becomes a feature
    assert len(out) == 1 and out[0].subject == "AAA" and out[0].confidence == pytest.approx(0.72)
    assert "BBB" in out[0].reasons_rejected and "LOW_CONFIDENCE" in out[0].reasons_rejected["BBB"]
    assert out[0].knowledge_state["digest"] == "dig"
    with pytest.raises(EX.ExpectationInvalid):
        EX.expectations_from_day(day, lambda row, s: {"predicted_return": 0.1}, ctx, "2021-02-26", model_version="m1",
                                 information_set={"prices": "2021-02-26"})


def test_empty_ledger():
    led = EX.ExpectationLedger(code_hash="t")
    assert len(led) == 0 and led.ids() == [] and led.verify()["ok"] and led.coverage([])["days"] == 0


# ------------------------------------------------------------------------------------------------ B: outcomes
def test_checklist_example_path_is_fully_reconstructed():
    led, exp, h, out = setup()
    assert out.exit_return == pytest.approx(0.07) and out.max_return == pytest.approx(0.15) and out.t_max == 1
    assert out.min_return == pytest.approx(0.07) and out.t_min == 3 and out.holding_days == 3
    assert out.trajectory == pytest.approx((0.15, 0.11, 0.07))
    assert out.regret == pytest.approx(0.08) and out.best_exit["t"] == 1
    assert out.mfe >= out.max_return >= out.exit_return and out.mae <= 0
    assert out.actual_direction == 1 and out.matured_at == "2021-03-03"
    assert out.realized_vol > 0 and out.vol_method in ("parkinson", "close_to_close")
    assert out.post_exit_best is None                          # exit window ends at session 3: nothing after it is judged


def test_short_position_signs():
    e = mk_exp(direction=-1, predicted_return=0.05, mfe=0.08, mae=-0.02, exit_window=(1, 3),
               distribution=((0.1, 0.0), (0.5, 0.05), (0.9, 0.09)))
    led, exp, h, out = setup(e, path=mk_path(closes=(97, 94, 95, 96, 97), opens=(100, 97, 94, 95, 96),
                                             highs=(101, 98, 95, 96, 97), lows=(96, 93, 93, 94, 95), exit_at="2021-03-03"))
    assert out.exit_return == pytest.approx(0.05) and out.max_return == pytest.approx(0.06) and out.t_max == 2
    assert out.mfe == pytest.approx(0.07) and out.mae == pytest.approx(-0.01)
    assert out.market_move is None and out.excess_over_market is None


def test_maturity_and_future_bars_fail_closed():
    exp = mk_exp()
    h = "x" * 8
    with pytest.raises(FirewallBreach):
        OC.reconstruct(exp, h, mk_path(), "2021-03-03")        # exits ON now: not yet known
    with pytest.raises(FirewallBreach):
        OC.reconstruct(exp, h, mk_path(), "2021-03-05")        # a bar dated 03-05 is not strictly before now
    OC.reconstruct(exp, h, mk_path(), "2021-03-06")


def test_unavailable_paths_raise_not_guess():
    exp = mk_exp()
    bad_entry = OC.PathData(dates=DATES[1:], open=(1,) * 4, high=(1,) * 4, low=(1,) * 4, close=(1,) * 4, exit_at="2021-03-03")
    with pytest.raises(OC.OutcomeUnavailable):
        OC.reconstruct(exp, "h", bad_entry, NOW)
    with pytest.raises(OC.OutcomeUnavailable):
        OC.reconstruct(exp, "h", OC.PathData(dates=(), open=(), high=(), low=(), close=(), exit_at="2021-03-03"), NOW)
    ragged = dataclasses.replace(mk_path(), close=(1, 2))
    with pytest.raises(OC.OutcomeUnavailable):
        OC.reconstruct(exp, "h", ragged, NOW)
    nan_open = mk_path(opens=(float("nan"), 115, 111, 107, 106))
    with pytest.raises(OC.OutcomeUnavailable):
        OC.reconstruct(exp, "h", nan_open, NOW)
    done, skipped = OC.reconstruct_many([(exp, "h", mk_path()), (exp, "h", bad_entry)], NOW)
    assert len(done) == 1 and len(skipped) == 1


def test_nan_close_is_reported_not_zeroed():
    closes = (115, float("nan"), 107, 106, 105)
    _, _, _, out = setup(path=mk_path(closes=closes, opens=(100, 115, 111, 107, 106), highs=(116, 116, 112, 108, 107),
                                       lows=(99, 110, 106, 105, 104)))
    assert any("close" in g for g in out.data_gaps) and math.isfinite(out.max_return)


def test_swings_outliers_and_events_are_found():
    v = [0.0, 0.06, 0.02, 0.09, 0.03]
    legs = OC.zigzag(v, 0.03)
    assert [round(l["change"], 2) for l in legs] == [0.06, -0.04, 0.07, -0.06]
    assert OC.zigzag([0.0], 0.03) == [] and OC.zigzag([0, 0.001, 0.0], 0.03) == []
    assert OC.outlier_days([0.001, 0.08, -0.002], 0.01)[0]["t"] == 2
    ev = OC.event_indicators(np.array([100, 108, 110.0]), np.array([100, 100, 108.0]), None, ["a", "b", "c"], 0.01, [])
    assert [e["kind"] for e in ev] == ["gap", "gap"] and ev[0]["t"] == 2
    calm = OC.event_indicators(np.full(3, 100.0), np.full(3, 100.0), None, ["a", "b", "c"], 0.01, [])
    assert calm == []
    vol = np.array([10, 10, 10, 10, 60.0])
    spikes = OC.event_indicators(np.full(5, 100.0), np.full(5, 100.0), vol, list("abcde"), 0.01, [])
    assert [e["kind"] for e in spikes] == ["volume_spike"] and spikes[0]["t"] == 5


def test_market_sector_peers_patterns_and_events_in_reconstruction():
    path = mk_path(market_close=(101, 102, 103, 104, 105), market_base=100.0, sector_close=(100.5, 101, 101, 100, 99),
                   sector_base=100.0, peers={"P1": (10.1, 10.2, 10.3, 10.4, 10.5), "P2": (20, 20, 20, 20, 20)},
                   peer_base={"P1": 10.0, "P2": 20.0}, pattern_series={"mom": (0.6, 0.3, -0.1, -0.2, -0.3), "gap": (0.3, 0.3, 0.3, 0.3, 0.3)},
                   pattern_realized={"mom": 0.2, "gap": 0.3}, interaction_realized={"mom x gap": 0.05},
                   volume=(100, 100, 100, 100, 900), events=({"date": "2021-03-02", "kind": "earnings", "size": 1.0},))
    _, _, _, out = setup(path=path)
    assert out.market_move == pytest.approx(0.03) and out.sector_move == pytest.approx(0.01)
    assert out.peer_moves["P1"] == pytest.approx(0.03) and out.peer_mean_move == pytest.approx(0.015)
    assert out.excess_over_market == pytest.approx(0.07 - 0.03)
    assert out.pattern_behavior["mom"]["flipped"] and [c["pattern"] for c in out.pattern_changes] == ["mom"]
    assert any(e["kind"] == "earnings" and e["t"] == 2 for e in out.external_events)
    assert not any(e["kind"] == "volume_spike" for e in out.external_events)     # the spike is after the exit


def test_post_exit_best_only_within_window_and_before_now():
    e = mk_exp(exit_window=(1, 5), holding_period=3)
    _, _, _, out = setup(e, path=mk_path(closes=(115, 111, 107, 118, 105)))
    assert out.post_exit_best["return"] == pytest.approx(0.18) and out.post_exit_best["t"] == 4
    assert out.post_exit_best["sessions_after_exit"] == 2


def test_outcome_ledger_rules(tmp_path):
    led, exp, h, out = setup()
    ol = OC.OutcomeLedger(led, root=None)
    assert ol.pending(NOW) == [exp.prediction_id]
    ol.add(out, NOW)
    assert ol.add(out, NOW) == exp.prediction_id and len(ol) == 1 and ol.pending(NOW) == []
    other = dataclasses.replace(out, exit_return=0.5)
    with pytest.raises(OC.OutcomeRewrite):
        ol.add(other, NOW)                                     # a different outcome for the same prediction
    with pytest.raises(OC.OutcomeRewrite):
        ol.update(out)
    with pytest.raises(FirewallBreach):
        ol.add(out, "2021-03-03")                              # not matured before now
    stale = dataclasses.replace(out, expectation_hash="deadbeef")
    ol2 = OC.OutcomeLedger(led)
    with pytest.raises(OC.OutcomeRewrite):
        ol2.add(stale, NOW)                                    # reconstructed against other expectation content
    assert ol.matured("2021-03-03") == [] and len(ol.matured(NOW)) == 1
    assert ol.verify()["ok"]
    ol.lane.chain._all()[-1]["body"]["outcome"]["exit_return"] = 0.9
    assert not ol.verify()["ok"]


def test_outcome_needs_recorded_expectation():
    led = EX.ExpectationLedger(code_hash="t")
    exp = mk_exp()
    out = OC.reconstruct(exp, "hh", mk_path(), NOW)            # never recorded
    with pytest.raises(OC.OutcomeRewrite):
        OC.OutcomeLedger(led).add(out, NOW)


# ------------------------------------------------------------------------------------------------ C: errors
def test_checklist_example_separate_errors_and_diagnosis():
    led, exp, h, out = setup()
    rep = PE.compute_errors(exp, h, out, NOW)
    assert set(rep.components) == set(PE.COMPONENTS) and len(PE.COMPONENTS) == 10
    r = rep["return"]
    assert r.error == pytest.approx(-0.03) and r.detail["peak_vs_prediction"] == pytest.approx(0.05)
    assert rep["direction"].error == 0 and rep["direction"].actual == 1
    assert rep["exit"].error == pytest.approx(-0.08) and rep["exit"].detail["best_t"] == 1
    assert rep["timing"].error == pytest.approx(1 - 2)         # peaked at session 1, predicted 2
    for tag in ("DIRECTION_CORRECT", "EXIT_RETURN_BELOW_PREDICTION", "OPPORTUNITY_LARGER_THAN_PREDICTED", "EXIT_LATE"):
        assert tag in rep.diagnosis, rep.diagnosis
    assert not hasattr(rep, "total") and not hasattr(rep, "score")
    assert len(set(round(z, 6) for z in rep.vector().values() if z is not None)) > 3   # ten different numbers, not one
    assert rep.severity(1)[0][1] == max(abs(z) for z in rep.vector().values() if z is not None)


def test_exit_error_is_independent_of_the_return_predictor():
    led, exp, h, out = setup()
    a = PE.compute_errors(exp, h, out, NOW)
    exp2 = mk_exp(predicted_return=0.06, mfe=0.1, distribution=((0.1, 0.0), (0.5, 0.06), (0.9, 0.11)), confidence=0.9,
                  direction=1, model_version="m2")
    led2 = EX.ExpectationLedger(code_hash="t")
    led2.record(exp2, "2021-02-26")
    h2 = led2.meta(exp2.prediction_id)["content_hash"]
    out2 = OC.reconstruct(exp2, h2, mk_path(), NOW)
    b = PE.compute_errors(exp2, h2, out2, NOW)
    assert a["return"].error != pytest.approx(b["return"].error)             # return predictor differs ...
    assert a["exit"].error == b["exit"].error and a["exit"].z == b["exit"].z    # ... the exit is judged the same
    c = PE.compute_errors(exp, h, out, NOW, PE.ErrorConfig(tol=0.2))
    assert c["exit"] == a["exit"]                                             # the +-tol target never reaches the exit error


def test_perfect_prediction_reads_as_no_surprise():
    e = mk_exp(predicted_return=0.07, mfe=0.16, distribution=((0.1, 0.03), (0.5, 0.07), (0.9, 0.12)), time_to_peak=1,
               exit_window=(1, 3), holding_period=3)
    led, exp, h, out = setup(e)
    rep = PE.compute_errors(exp, h, out, NOW)
    assert abs(rep["return"].error) < 1e-9 and "EXIT_RETURN_WITHIN_TOLERANCE" in rep.diagnosis and not rep.confident_wrong
    assert abs(rep["timing"].error) < 1e-9


def test_wrong_direction_confident_versus_unconfident():
    down = dict(closes=(95, 92, 90, 90, 90), opens=(100, 95, 92, 90, 90), highs=(101, 96, 93, 91, 91), lows=(94, 91, 89, 89, 89))
    led, exp, h, out = setup(mk_exp(confidence=0.9), path=mk_path(**down))
    rep = PE.compute_errors(exp, h, out, NOW)
    assert rep["direction"].error == 1 and rep.confident_wrong and "CONFIDENT_WRONG" in rep.diagnosis
    assert rep["direction"].z < -2 and rep["confidence"].error < 0 and rep["confidence"].detail["overconfident"]
    led2, exp2, h2, out2 = setup(mk_exp(confidence=0.55), path=mk_path(**down))
    rep2 = PE.compute_errors(exp2, h2, out2, NOW)
    assert rep2["direction"].error == 1 and not rep2.confident_wrong
    assert abs(rep2["direction"].z) < abs(rep["direction"].z)                 # a confident miss is a bigger surprise


def test_regime_errors_and_unscorable_labels():
    path = mk_path(market_close=(99, 98, 97, 96, 95), market_base=100.0, sector_close=(101, 102, 103, 104, 105), sector_base=100.0)
    led, exp, h, out = setup(path=path)
    rep = PE.compute_errors(exp, h, out, NOW)
    assert rep["market_regime"].detail["actual_label"].startswith("DOWN") and "direction" in rep["market_regime"].detail["mismatched"]
    assert rep["market_regime"].error > 0 and rep["sector_regime"].applicable
    weird = mk_exp(market_regime="goldilocks", sector_regime="rotation")
    led2, exp2, h2, out2 = setup(weird, path=path)
    rep2 = PE.compute_errors(exp2, h2, out2, NOW)
    assert not rep2["market_regime"].applicable and rep2["market_regime"].error is None       # left unscored, not scored 0 or 1
    none = PE.compute_errors(*setup()[1:3], setup()[3], NOW)
    assert not none["market_regime"].applicable and "unknowable" in none["market_regime"].note


def test_pattern_and_interaction_errors():
    path = mk_path(pattern_realized={"mom": 0.1, "gap": 0.3}, interaction_realized={"mom x gap": -0.1})
    _, exp, h, out = setup(path=path)
    rep = PE.compute_errors(exp, h, out, NOW)
    p = rep["pattern_strength"]
    assert p.detail["per_pattern"]["mom"]["error"] == pytest.approx(-0.5) and p.detail["worst_pattern"] == "mom"
    assert rep["interaction"].error == pytest.approx(-0.3)
    _, exp2, h2, out2 = setup(path=mk_path(pattern_realized={"mom": 0.6}))
    rep2 = PE.compute_errors(exp2, h2, out2, NOW)
    assert rep2["pattern_strength"].detail["unobserved"] == ["gap"]
    _, exp3, h3, out3 = setup()
    rep3 = PE.compute_errors(exp3, h3, out3, NOW)
    assert not rep3["pattern_strength"].applicable and not rep3["interaction"].applicable   # unknowable, not zero
    _, exp4, h4, out4 = setup(mk_exp(interactions={}), path=mk_path(interaction_realized={"mom x gap": 0.4}))
    assert PE.compute_errors(exp4, h4, out4, NOW)["interaction"].error == pytest.approx(0.4)


def test_volatility_error_scale_and_unestimable():
    _, exp, h, out = setup()
    v = PE.compute_errors(exp, h, out, NOW)["volatility"]
    assert v.expected == 0.02 and v.detail["ratio"] == pytest.approx(out.realized_vol / 0.02)
    flat = mk_path(closes=(100, 100, 100, 100, 100), opens=(100,) * 5, highs=(100,) * 5, lows=(100,) * 5)
    _, exp2, h2, out2 = setup(path=flat)
    assert not PE.compute_errors(exp2, h2, out2, NOW)["volatility"].applicable


def test_future_information_cannot_enter_error_analysis():
    _, exp, h, out = setup()
    with pytest.raises(FirewallBreach):
        PE.compute_errors(exp, h, out, "2021-03-03")           # outcome matures ON now
    other = dataclasses.replace(out, prediction_id="X" + "0" * 14)
    with pytest.raises(ValueError):
        PE.compute_errors(exp, h, other, NOW)
    with pytest.raises(ValueError):
        PE.compute_errors(exp, "wrong-hash", out, NOW)
    # scrambling everything after the exit changes no error
    a = PE.compute_errors(exp, h, out, NOW)
    scrambled = OC.reconstruct(exp, h, mk_path(closes=(115, 111, 107, 300, 1)), NOW)
    b = PE.compute_errors(exp, h, scrambled, NOW)
    assert a.vector() == b.vector()


def test_engine_step_is_incremental_deterministic_and_feeds_the_tracker(tmp_path):
    led, exp, h, out = setup()
    ol = OC.OutcomeLedger(led)
    eng = PE.ErrorEngine(tracker=SurpriseTracker())
    assert eng.step(ol, NOW) == []                             # nothing matured yet
    ol.add(out, NOW)
    assert eng.step(ol, "2021-03-03") == []                    # matured ON now is invisible
    new = eng.step(ol, NOW)
    assert len(new) == 1 and eng.step(ol, NOW) == [] and len(eng.tracker) == 5      # return, direction, volatility, timing, exit: confidence is not double counted; the rest are unscorable here
    twin = PE.ErrorEngine().step(ol, NOW)
    assert twin[0].report_hash == new[0].report_hash           # deterministic replay
    assert eng.verify()["ok"] and eng.summary(NOW)["n"] == 1
    eng.lane.chain._all()[-1]["body"]["report"]["confident_wrong"] = True
    assert not eng.verify()["ok"]
    with pytest.raises(EX.LedgerTampered):
        eng.assert_intact()


def test_engine_persists_and_reloads(tmp_path):
    led, exp, h, out = setup()
    ol = OC.OutcomeLedger(led)
    ol.add(out, NOW)
    eng = PE.ErrorEngine(root=tmp_path / "err")
    rep = eng.step(ol, NOW)[0]
    again = PE.ErrorEngine(root=tmp_path / "err")
    assert len(again) == 1 and again.report(exp.prediction_id).report_hash == rep.report_hash and again.step(ol, NOW) == []


def _series(n, under=True):
    """n predictions decided on distinct days, all under-predicting a strong move by the same margin (a systematic error)."""
    reports, exps = [], {}
    base = pd.Timestamp("2021-01-04")
    for i in range(n):
        d = (base + pd.tseries.offsets.BDay(i * 6)).date()
        e = base + pd.tseries.offsets.BDay(i * 6 + 1)
        days = [(e + pd.tseries.offsets.BDay(k)).date().isoformat() for k in range(5)]
        exp = mk_exp(subject=f"S{i}", decided_at=d.isoformat(), entry_at=days[0], timestamp=d.isoformat() + "T16:00:00",
                     information_set={"prices": d.isoformat()})
        c = 113 + 0.4 * (i % 4)
        path = OC.PathData(dates=tuple(days), open=(100, c - 1, c, c, c), high=(c + 1,) * 5, low=(99,) * 5,
                           close=(c - 1, c, c, c, c), exit_at=days[2])
        exps[exp.prediction_id] = exp
        reports.append((exp, path, days[2]))
    return reports


def test_repeated_underprediction_escalates_research_priority():
    led = EX.ExpectationLedger(code_hash="t")
    ol = OC.OutcomeLedger(led)
    tr = SurpriseTracker()
    eng = PE.ErrorEngine(tracker=tr)
    for exp, path, _ in _series(16):
        led.record(exp, exp.decided_at)
        h = led.meta(exp.prediction_id)["content_hash"]
        ol.add(OC.reconstruct(exp, h, path, "2022-12-30"), "2022-12-30")
    eng.step(ol, "2022-12-30")
    assert len(eng) == 16
    prof = PE.error_profile(eng.reports(), "return")
    assert prof["bias"] > 0.02 and prof["share_positive"] == 1.0 and prof["t_bias"] > 5        # systematic under-prediction
    pri = tr.research_priority("2022-12-30")
    assert pri and pri[0].score > 0 and "err=" in pri[0].cell
    groups = PE.by_group(eng.reports(), {e.prediction_id: e for e, _, _ in _series(16)}, lambda x: x.market_regime)
    assert groups["UP_CALM"]["n"] == 16


def test_honest_tolerance_measurement():
    _, exp, h, out = setup()
    hit = PE.compute_errors(exp, h, out, NOW, PE.ErrorConfig(tol=0.05))      # -3pp error, inside 5pp
    miss = PE.compute_errors(exp, h, out, NOW)                                # outside 1pp
    assert hit["return"].detail["within_tol"] and not miss["return"].detail["within_tol"]
    few = PE.honest_tolerance([hit] * 10)
    assert few["verdict"] == "INSUFFICIENT_SAMPLE" and few["rate"] == 1.0
    mixed = PE.honest_tolerance([miss] * 100)
    assert mixed["verdict"] == "TARGET_NOT_MET" and mixed["rate"] == 0.0
    good = PE.honest_tolerance([hit] * 200)
    assert good["verdict"] == "TARGET_MET"
    penalised = PE.honest_tolerance([hit] * 200, n_unscored=200)
    assert penalised["worst_case_rate"] == pytest.approx(0.5)                # unreconstructable predictions count as misses
    gamble = PE.honest_tolerance([hit] * 45 + [miss] * 15)
    assert gamble["verdict"] in ("NOT_DEMONSTRATED", "TARGET_NOT_MET") and gamble["wilson_low"] < 0.8
    assert PE.honest_tolerance([])["rate"] is None


def test_confidence_reliability_detects_overconfidence():
    down = dict(closes=(95, 92, 90, 90, 90), opens=(100, 95, 92, 90, 90), highs=(101, 96, 93, 91, 91), lows=(94, 91, 89, 89, 89))
    reps = []
    for i in range(10):
        p = down if i % 2 else {}
        _, exp, h, out = setup(mk_exp(confidence=0.9, subject=f"S{i}"), path=mk_path(**p))
        reps.append(PE.compute_errors(exp, h, out, NOW))
    rel = PE.confidence_reliability(reps)
    assert rel["n"] == 10 and rel["overconfidence"] == pytest.approx(0.4) and rel["ece"] > 0.3
    assert PE.confidence_reliability([]) == {"n": 0, "bins": [], "ece": None, "overconfidence": None}


def test_empty_and_degenerate_cases():
    eng = PE.ErrorEngine()
    led = EX.ExpectationLedger(code_hash="t")
    assert eng.step(OC.OutcomeLedger(led), NOW) == [] and eng.summary(NOW)["n"] == 0
    assert PE.error_profile([], "return") == {"component": "return", "n": 0, "unscored": 0}
    assert PE.all_profiles([])["exit"]["n"] == 0
    assert PE.wilson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        PE.ErrorEngine(PE.ErrorConfig(tol=0.9))
    assert PE.parse_regime("no idea") == {} and PE.parse_regime("down-volatile") == {"direction": "DOWN", "vol": "VOLATILE"}


def test_error_record_feeds_error_research_by_schema():
    from engine.research import error_research as ER
    led, exp, h, out = setup(path=mk_path(pattern_realized={"mom": 0.2, "gap": 0.3}))
    ol = OC.OutcomeLedger(led)
    ol.add(out, NOW)
    eng = PE.ErrorEngine()
    eng.step(ol, NOW)
    recs = eng.records(ol, NOW)
    assert len(recs) == 1 and eng.records(ol, "2021-03-03") == []
    rec = recs[0]
    assert set(dataclasses.asdict(rec)) == set(PE.ERROR_RECORD_SCHEMA)
    obs = ER.obs_from_record(rec)                                   # object form
    assert obs.obs_id == exp.prediction_id and obs.expected == pytest.approx(0.10) and obs.realised == pytest.approx(0.07)
    assert obs.pattern == "mom" and obs.regime == "UP_CALM" and obs.stock_type == "vol_mid"
    assert obs.exit_regret == pytest.approx(0.8) and obs.scale > 0 and obs.matured_at > obs.decided_at
    assert ER.obs_from_record(dataclasses.asdict(rec)).obs_id == obs.obs_id      # mapping form
    assert obs.check() == []
    with pytest.raises(ER.ErrorResearchError):
        ER.obs_from_record({k: v for k, v in dataclasses.asdict(rec).items() if k != "realised"})
